"""Versioned optional hard guidelines with explicit, candidate-specific approval.

This is an audit and approval protocol, not a natural-language verifier or an
identity/security service. Only the trusted operator/evaluator may write this
book or call approval/review methods. Keep it outside author-writeable paths.
The actor field is an audit label; it does not authenticate a caller.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import uuid


class ConstraintPending(RuntimeError):
    """This candidate needs trusted review or explicit permission; others may run."""


class ConstraintIntegrityError(RuntimeError):
    """The append-only constraint history failed its integrity checks."""


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("candidate_sha must be a lowercase SHA256 digest")
    return value


def _text(value: str, field: str, *, blank: bool = False) -> str:
    if not isinstance(value, str) or (not blank and not value.strip()):
        raise ValueError(f"{field} must be a {'possibly empty ' if blank else 'nonempty '}string")
    if len(value) > 100_000:
        raise ValueError(f"{field} is too long")
    return value


@dataclass(frozen=True)
class ConstraintSnapshot:
    version: int
    sha256: str
    text: str
    active: bool

    @property
    def prompt_text(self) -> str:
        if not self.active:
            return ""
        return (
            f"\n\nMandatory user guidelines (version {self.version}; SHA256 {self.sha256}):\n"
            f"{self.text}\n"
            "Follow these guidelines. Do not waive, reinterpret away, or change them. "
            "If a candidate needs an exception or a guideline needs changing, submit a permission request "
            "for trusted human review. A pending request is not permission. Continue other compliant "
            "research while it is pending. Model structure is otherwise open.\n"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _snapshot(version: int, text: str) -> ConstraintSnapshot:
    return ConstraintSnapshot(version, _digest({"version": version, "text": text}), text, bool(text.strip()))


class ConstraintBook:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    @contextmanager
    def _locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def _read(self) -> list[dict]:
        directory = self.root / "events"
        files = sorted(directory.glob("*.json")) if directory.exists() else []
        head_path = self.root / "head.json"
        if not files:
            if head_path.exists():
                raise ConstraintIntegrityError("Constraint event history is missing")
            return []
        if not head_path.exists():
            raise ConstraintIntegrityError("Constraint history head is missing")
        previous = "0" * 64
        events = []
        try:
            for sequence, path in enumerate(files, 1):
                event = json.loads(path.read_text())
                digest = event.pop("sha256")
                if event["sequence"] != sequence or event["previous_sha256"] != previous or _digest(event) != digest:
                    raise ConstraintIntegrityError("Constraint event hash or history order changed")
                if path.name != f"{sequence:08d}-{digest}.json":
                    raise ConstraintIntegrityError("Constraint event filename changed")
                event["sha256"] = digest
                events.append(event)
                previous = digest
            head = json.loads(head_path.read_text())
            if head != {"sequence": len(events), "sha256": previous}:
                raise ConstraintIntegrityError("Constraint history does not match its recorded head")
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise ConstraintIntegrityError("Invalid constraint history") from error
        return events

    def _append(self, events: list[dict], kind: str, payload: dict) -> dict:
        event = {
            "sequence": len(events) + 1,
            "previous_sha256": events[-1]["sha256"] if events else "0" * 64,
            "kind": kind, "payload": payload,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        event["sha256"] = _digest(event)
        directory = self.root / "events"
        directory.mkdir(exist_ok=True)
        destination = directory / f"{event['sequence']:08d}-{event['sha256']}.json"
        with destination.open("x") as stream:
            json.dump(event, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary = self.root / f".head-{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps({"sequence": event["sequence"], "sha256": event["sha256"]}) + "\n")
        temporary.replace(self.root / "head.json")
        return event

    @staticmethod
    def _current(events: list[dict]) -> ConstraintSnapshot:
        current = _snapshot(0, "")
        for event in events:
            if event["kind"] in {"initialized", "change_approved"}:
                current = _snapshot(event["payload"]["version"], event["payload"]["text"])
        return current

    def snapshot(self) -> ConstraintSnapshot:
        if not self.root.exists():
            return _snapshot(0, "")
        with self._locked():
            return self._current(self._read())

    def events(self) -> list[dict]:
        if not self.root.exists():
            return []
        with self._locked():
            return self._read()

    def initialize(self, text: str = "") -> ConstraintSnapshot:
        text = _text(text, "text", blank=True)
        with self._locked():
            events = self._read()
            current = self._current(events)
            if current.version:
                if current.text == text:
                    return current
                raise ValueError("Guidelines already exist; request_change and explicit approval are required")
            if not text.strip():
                return current
            self._append(events, "initialized", {"version": 1, "text": text, "actor": "initial_user_guidelines"})
            return _snapshot(1, text)

    def request_change(self, text: str, reason: str) -> dict:
        text, reason = _text(text, "text", blank=True), _text(reason, "reason")
        with self._locked():
            events = self._read()
            current = self._current(events)
            request = {"id": uuid.uuid4().hex, "type": "change", "text": text,
                       "reason": reason, "constraint_sha256": current.sha256,
                       "constraint_version": current.version}
            self._append(events, "requested", request)
            return dict(request)

    def request_exception(self, candidate_sha: str, reason: str, constraint_refs=()) -> dict:
        candidate_sha, reason = _sha(candidate_sha), _text(reason, "reason")
        if isinstance(constraint_refs, str):
            raise ValueError("constraint_refs must be a sequence of references")
        refs = [_text(value, "constraint reference") for value in constraint_refs]
        with self._locked():
            events = self._read()
            current = self._current(events)
            if not current.active:
                raise ValueError("There are no active guidelines requiring an exception")
            request = {"id": uuid.uuid4().hex, "type": "exception", "candidate_sha": candidate_sha,
                       "reason": reason, "constraint_refs": refs,
                       "constraint_sha256": current.sha256, "constraint_version": current.version}
            self._append(events, "requested", request)
            return dict(request)

    @staticmethod
    def _pending(events: list[dict], request_id: str) -> dict:
        request = next((event["payload"] for event in events
                        if event["kind"] == "requested" and event["payload"]["id"] == request_id), None)
        if request is None:
            raise ValueError("Unknown constraint request")
        if any(event["kind"] in {"change_approved", "exception_approved", "rejected"}
               and event["payload"]["request_id"] == request_id for event in events):
            raise ValueError("Constraint request is already resolved")
        return request

    def approve(self, request_id: str, actor: str = "human") -> dict:
        actor = _text(actor, "actor")
        with self._locked():
            events = self._read()
            request = self._pending(events, request_id)
            current = self._current(events)
            if request["constraint_sha256"] != current.sha256:
                raise ValueError("Request is stale: the active guideline version changed")
            approval = {**request, "request_id": request_id, "actor": actor}
            if request["type"] == "change":
                approval["version"] = current.version + 1
                kind = "change_approved"
            else:
                kind = "exception_approved"
            self._append(events, kind, approval)
            return dict(approval)

    def reject(self, request_id: str, actor: str = "human", reason: str = "Declined") -> dict:
        actor, reason = _text(actor, "actor"), _text(reason, "reason")
        with self._locked():
            events = self._read()
            self._pending(events, request_id)
            decision = {"request_id": request_id, "actor": actor, "reason": reason}
            self._append(events, "rejected", decision)
            return decision

    def review_compliance(self, candidate_sha: str, compliant: bool, evidence: str, actor: str,
                          *, expected_constraint_sha: str | None = None) -> dict:
        candidate_sha, evidence, actor = _sha(candidate_sha), _text(evidence, "evidence"), _text(actor, "actor")
        if not isinstance(compliant, bool):
            raise ValueError("compliant must be a boolean")
        with self._locked():
            events = self._read()
            current = self._current(events)
            if expected_constraint_sha is not None and current.sha256 != expected_constraint_sha:
                raise ConstraintPending("Guidelines changed during compliance review; review the current version before calibration.")
            review = {"candidate_sha": candidate_sha, "compliant": compliant, "evidence": evidence,
                      "actor": actor, "constraint_sha256": current.sha256, "constraint_version": current.version}
            self._append(events, "compliance_review", review)
            return dict(review)

    def record_proposal(self, snapshot: ConstraintSnapshot, *, generation,
                        system_prompt: str, user_prompt: str, **context) -> dict:
        """Keep a receipt for the exact immutable constraints shown to an author."""
        if not isinstance(snapshot, ConstraintSnapshot) or snapshot != _snapshot(snapshot.version, snapshot.text):
            raise ValueError("Invalid constraint snapshot")
        payload = {
            "proposal_id": uuid.uuid4().hex,
            "constraints_version": snapshot.version,
            "constraints_sha256": snapshot.sha256,
            "constraints_active": snapshot.active,
            "constraints_text": snapshot.text,
            "generation": generation,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "prompt_sha256": hashlib.sha256(json.dumps({
                "system_prompt": system_prompt, "user_prompt": user_prompt,
                "msg_history": context.get("msg_history", []),
            }, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
            "context": context,
        }
        payload["receipt_sha256"] = _digest(payload)
        with self._locked():
            events = self._read()
            historical = [_snapshot(0, "")] + [
                _snapshot(event["payload"]["version"], event["payload"]["text"])
                for event in events if event["kind"] in {"initialized", "change_approved"}
            ]
            if snapshot not in historical:
                raise ValueError("Snapshot does not belong to this constraint history")
            directory = self.root / "proposals"
            directory.mkdir(exist_ok=True)
            path = directory / f"{payload['proposal_id']}.json"
            with path.open("x") as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.write("\n")
        return {key: payload[key] for key in (
            "proposal_id", "constraints_version", "constraints_sha256", "constraints_active", "prompt_sha256", "receipt_sha256"
        )} | {"constraints_receipt_path": str(path)}

    def assert_allowed(self, candidate_sha: str, *, expected_constraint_sha: str | None = None) -> dict:
        candidate_sha = _sha(candidate_sha)
        if not self.root.exists():
            current = _snapshot(0, "")
            if expected_constraint_sha is not None and current.sha256 != expected_constraint_sha:
                raise ConstraintPending("The expected guideline version is no longer active.")
            return {"allowed": True, "reason": "no_active_guidelines", **current.to_dict()}
        with self._locked():
            events = self._read()
            current = self._current(events)
            if expected_constraint_sha is not None and current.sha256 != expected_constraint_sha:
                raise ConstraintPending("Guidelines changed before the calibration gate; review the current version.")
            if not current.active:
                return {"allowed": True, "reason": "no_active_guidelines", **current.to_dict()}
            matching = [event for event in events if event["payload"].get("candidate_sha") == candidate_sha
                        and event["payload"].get("constraint_sha256") == current.sha256]
            exception = next((event for event in reversed(matching) if event["kind"] == "exception_approved"), None)
            if exception is not None:
                return {"allowed": True, "reason": "approved_exception", "evidence_event": exception["sha256"], **current.to_dict()}
            review = next((event for event in reversed(matching) if event["kind"] == "compliance_review"), None)
            if review is not None and review["payload"]["compliant"]:
                return {"allowed": True, "reason": "trusted_compliance_review", "evidence_event": review["sha256"], **current.to_dict()}
            if review is None and not any(event["kind"] == "review_requested" for event in matching):
                self._append(events, "review_requested", {
                    "id": uuid.uuid4().hex, "candidate_sha": candidate_sha,
                    "constraint_sha256": current.sha256, "constraint_version": current.version,
                    "reason": "Trusted compliance review or explicit exception approval is required before calibration.",
                })
            reason = "failed trusted compliance review; request an exception" if review else "needs a trusted compliance review or an approved exception"
            raise ConstraintPending(f"Candidate {candidate_sha[:12]} {reason} under guideline version {current.version}. Other candidates may continue.")
