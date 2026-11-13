"""SQLite ledger: atomic reservations, replay cache, and charged failures."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .contracts import CalibrationResult, EvaluationRecord, EvaluationRequest


class BudgetExhausted(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    """Campaign bounds are persisted and cannot silently change on restart.

    In-flight work reserves its full timeout. Terminal work is charged elapsed
    wall time, including failed work. An abandoned worker keeps its reservation
    until an operator marks it failed; unknown elapsed time charges the timeout.
    Termination and controller cleanup can add small wall-time overhead beyond
    a timeout. Leave headroom in time allocations; attempt counts remain exact.
    Optional stage_limits keep cheap screening from spending the allocations
    assigned to posterior calibration and confirmation.
    """

    def __init__(self, path, max_evaluations=None, max_seconds=None, stage_limits=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if stage_limits is not None:
            stage_limits = json.loads(json.dumps(stage_limits, allow_nan=False))
            if not stage_limits:
                raise ValueError("stage_limits must contain at least one stage")
            for name, limits in stage_limits.items():
                if not isinstance(name, str) or not name or set(limits) != {"max_evaluations", "max_seconds"}:
                    raise ValueError("each named stage must declare max_evaluations and max_seconds")
                if (not isinstance(limits["max_evaluations"], int) or isinstance(limits["max_evaluations"], bool)
                        or limits["max_evaluations"] < 0):
                    raise ValueError("stage max_evaluations must be a nonnegative integer")
                if not math.isfinite(limits["max_seconds"]) or limits["max_seconds"] < 0:
                    raise ValueError("stage max_seconds must be finite and nonnegative")
            total_count = sum(value["max_evaluations"] for value in stage_limits.values())
            total_seconds = sum(value["max_seconds"] for value in stage_limits.values())
            if max_evaluations is not None and max_evaluations < total_count:
                raise ValueError("global evaluation budget must cover all reserved stage allocations")
            if max_seconds is not None and max_seconds < total_seconds:
                raise ValueError("global wall-time budget must cover all reserved stage allocations")
        if max_evaluations is not None and (
            not isinstance(max_evaluations, int) or isinstance(max_evaluations, bool) or max_evaluations < 0
        ):
            raise ValueError("max_evaluations must be a nonnegative integer")
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds < 0):
            raise ValueError("max_seconds must be nonnegative and finite")
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("""CREATE TABLE IF NOT EXISTS evaluations (
                cache_key TEXT PRIMARY KEY, status TEXT NOT NULL,
                reserved_seconds REAL NOT NULL, charge_seconds REAL NOT NULL,
                record_json TEXT NOT NULL)""")
            for key, value in (("max_evaluations", max_evaluations), ("max_seconds", max_seconds),
                               ("stage_limits", stage_limits)):
                row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
                if row is None:
                    if value is None and stage_limits is not None:
                        if key == "max_evaluations":
                            value = total_count
                        elif key == "max_seconds":
                            value = total_seconds
                    conn.execute("INSERT INTO settings VALUES (?, ?)", (key, json.dumps(value)))
                elif value is not None and json.loads(row[0]) != value:
                    raise ValueError(f"persisted campaign {key} differs; use a new ledger for a new budget")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    @staticmethod
    def _decode(row):
        return EvaluationRecord.from_dict(json.loads(row[0])) if row else None

    def get(self, cache_key):
        with self._connect() as conn:
            return self._decode(conn.execute(
                "SELECT record_json FROM evaluations WHERE cache_key=?", (cache_key,)
            ).fetchone())

    def records(self):
        with self._connect() as conn:
            return [self._decode(row) for row in conn.execute(
                "SELECT record_json FROM evaluations ORDER BY rowid"
            ).fetchall()]

    @staticmethod
    def _usage(conn):
        rows = conn.execute(
            "SELECT status, COUNT(*), SUM(reserved_seconds), SUM(charge_seconds) FROM evaluations GROUP BY status"
        ).fetchall()
        stage_usage = {}
        for record_json, status, reserved, charged in conn.execute(
            "SELECT record_json,status,reserved_seconds,charge_seconds FROM evaluations"
        ):
            name = json.loads(record_json)["request"]["stage"]["name"]
            usage = stage_usage.setdefault(name, {
                "evaluation_count": 0, "status_counts": {}, "reserved_seconds": 0.0, "charged_seconds": 0.0
            })
            usage["evaluation_count"] += 1
            usage["status_counts"][status] = usage["status_counts"].get(status, 0) + 1
            usage["reserved_seconds"] += reserved
            usage["charged_seconds"] += charged
        return {
            "evaluation_count": sum(row[1] for row in rows),
            "status_counts": {row[0]: row[1] for row in rows},
            "reserved_seconds": sum(row[2] for row in rows),
            "charged_seconds": sum(row[3] for row in rows),
            "stage_usage": stage_usage,
        }

    def usage(self):
        with self._connect() as conn:
            usage = self._usage(conn)
            usage.update({key: json.loads(value) for key, value in conn.execute("SELECT key,value FROM settings")})
            return usage

    def reserve(self, request: EvaluationRequest, artifact_dir: str):
        """Return (record, claimed). Only the claimant may execute this work."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = self._decode(conn.execute(
                "SELECT record_json FROM evaluations WHERE cache_key=?", (request.cache_key,)
            ).fetchone())
            if existing is not None:
                return existing, False
            limits = {key: json.loads(value) for key, value in conn.execute("SELECT key,value FROM settings")}
            usage = self._usage(conn)
            if limits["max_evaluations"] is not None and usage["evaluation_count"] >= limits["max_evaluations"]:
                raise BudgetExhausted("campaign evaluation budget exhausted")
            reserved_total = usage["reserved_seconds"] + usage["charged_seconds"] + request.stage.max_seconds
            if limits["max_seconds"] is not None and reserved_total > limits["max_seconds"]:
                raise BudgetExhausted("insufficient campaign wall-time budget for this stage reservation")
            if limits["stage_limits"] is not None:
                if request.stage.name not in limits["stage_limits"]:
                    raise BudgetExhausted("stage is not in the campaign's reserved stage allocations")
                stage_limit = limits["stage_limits"][request.stage.name]
                stage_usage = usage["stage_usage"].get(request.stage.name, {
                    "evaluation_count": 0, "reserved_seconds": 0.0, "charged_seconds": 0.0
                })
                if stage_usage["evaluation_count"] >= stage_limit["max_evaluations"]:
                    raise BudgetExhausted(f"{request.stage.name} stage evaluation budget exhausted")
                stage_reserved = stage_usage["reserved_seconds"] + stage_usage["charged_seconds"] + request.stage.max_seconds
                if stage_reserved > stage_limit["max_seconds"]:
                    raise BudgetExhausted(f"insufficient {request.stage.name} stage wall-time allocation")
            now = _now()
            record = EvaluationRecord(request.cache_key, request, "reserved", str(artifact_dir), created_at=now, updated_at=now)
            conn.execute("INSERT INTO evaluations VALUES (?, ?, ?, 0, ?)", (
                record.cache_key, record.status, request.stage.max_seconds, json.dumps(record.to_dict())
            ))
            return record, True

    def _transition(self, cache_key, allowed, status, *, result=None, error=None, elapsed_seconds=0.0):
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            record = self._decode(conn.execute(
                "SELECT record_json FROM evaluations WHERE cache_key=?", (cache_key,)
            ).fetchone())
            if record is None or record.status not in allowed:
                raise ValueError(f"invalid evaluation transition to {status}")
            value = record.to_dict()
            terminal = status in {"completed", "failed"}
            value.update(status=status, result=result.to_dict() if result else None, error=error,
                         elapsed_seconds=elapsed_seconds, charge_seconds=elapsed_seconds if terminal else 0.0,
                         updated_at=_now())
            updated = EvaluationRecord.from_dict(value)
            conn.execute("UPDATE evaluations SET status=?,reserved_seconds=?,charge_seconds=?,record_json=? WHERE cache_key=?", (
                status, 0.0 if terminal else record.request.stage.max_seconds,
                updated.charge_seconds, json.dumps(updated.to_dict()), cache_key
            ))
            return updated

    def start(self, cache_key):
        return self._transition(cache_key, {"reserved"}, "running")

    def complete(self, cache_key, result: CalibrationResult, elapsed_seconds):
        return self._transition(cache_key, {"running"}, "completed", result=result, elapsed_seconds=elapsed_seconds)

    def fail(self, cache_key, error, elapsed_seconds):
        return self._transition(cache_key, {"reserved", "running"}, "failed", error=str(error), elapsed_seconds=elapsed_seconds)

    def fail_abandoned(self, cache_key, reason, elapsed_seconds=None):
        """Call only after independently stopping/checking the old worker."""
        record = self.get(cache_key)
        if record is None:
            raise KeyError(cache_key)
        return self.fail(cache_key, reason, record.request.stage.max_seconds if elapsed_seconds is None else elapsed_seconds)
