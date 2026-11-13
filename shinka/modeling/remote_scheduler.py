"""Connect Shinka's proposal queue to durable remote calibration jobs."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
import uuid

from .constraints import ConstraintPending
from .contracts import EvaluationRecord
from .pipeline import EvaluationInProgress
from .shinka_adapter import _bind_archive_protocol, author_feedback
from .store import BudgetExhausted

logger = logging.getLogger(__name__)


class RemoteCalibrationScheduler:
    """A scheduler compatible with ShinkaEvolveRunner's evaluation interface.

    Factories and the pipeline belong to the trusted controller. The author may
    supply candidate source, never evaluator code, data access or score rules.
    Use LocalJobConfig(time=None): remote queue/runtime limits are enforced by
    the remote controller, rather than Shinka's generic submission-time clock.
    """

    job_type = "remote_calibration"

    def __init__(self, pipeline, build_request, build_inputs, spec, command,
                 state_dir, poll_interval=2.0):
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")
        if not command or any(not isinstance(arg, str) or not arg for arg in command):
            raise ValueError("command must contain trusted nonempty argv strings")
        self.pipeline = pipeline
        self.build_request = build_request
        self.build_inputs = build_inputs
        self.spec = spec
        self.command = tuple(command)
        self.root = Path(state_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "scheduler-state.json"
        self.poll_interval = float(poll_interval)
        self._closing = threading.Event()
        identity = {"spec": spec.to_dict(), "command": list(command),
                    "ledger": str(Path(pipeline.ledger.path).resolve()),
                    "remote_root": str(Path(pipeline.controller.root).resolve())}
        with self._locked():
            if self.state_path.exists():
                if self._load()["identity"] != identity:
                    raise ValueError("Remote scheduler campaign identity changed on restart")
            else:
                self._save({"schema_version": 1, "identity": identity, "jobs": {}, "bindings": {}})

    @contextmanager
    def _locked(self):
        with (self.root / ".scheduler.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _load(self):
        value = json.loads(self.state_path.read_text())
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported remote scheduler state")
        return value

    def _save(self, state):
        temporary = self.state_path.with_name(".scheduler-" + uuid.uuid4().hex + ".tmp")
        with temporary.open("w") as stream:
            json.dump(state, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.state_path)

    @staticmethod
    def _capture_record(entry, record):
        entry["record"] = record.to_dict()
        entry["status"] = record.status
        entry["terminal"] = record.status in {"completed", "failed"}
        entry["error"] = None

    def _advance(self, state, key):
        entry = state["jobs"][key]
        if entry["terminal"]:
            return entry
        from .contracts import EvaluationRequest
        request = EvaluationRequest.from_dict(entry["request"])
        try:
            job = self.pipeline.job_for(key)
            record = self.pipeline.ledger.get(key)
            if entry["cancel_requested"]:
                if job is None:
                    if record is not None and record.status in {"reserved", "running"}:
                        # No known job is not sufficient proof after an ambiguous
                        # submit. Leave its reservation for reconciliation.
                        entry["status"] = "cancel_pending_reconciliation"
                    else:
                        entry.update(status="cancelled", terminal=True)
                else:
                    self._capture_record(entry, self.pipeline.cancel(key))
            elif job is not None or (record is not None and record.status in {"completed", "failed"}):
                self._capture_record(entry, self.pipeline.poll(key))
            else:
                source = Path(request.candidate.source_path)
                if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != request.candidate.source_sha:
                    raise ValueError("Candidate source changed before remote submission")
                inputs = self.build_inputs(request)
                self._capture_record(entry, self.pipeline.submit(request, self.spec, inputs, self.command))
            job = self.pipeline.job_for(key)
            if job is not None:
                entry["remote_work_key"] = job.key
                entry["remote_status"] = job.status
                entry["evaluation_started_at"] = getattr(job, "started_at", None)
        except ConstraintPending as exc:
            entry.update(status="pending_constraints", error=str(exc))
        except BudgetExhausted as exc:
            # A rejected allocation consumes no remote work. It is an explicit
            # unsuccessful proposal, not a score or a reason to stop the engine.
            entry.update(status="budget_exhausted", error=str(exc), terminal=True)
        except Exception as exc:
            record = self.pipeline.ledger.get(key)
            if record is not None and record.status in {"completed", "failed"}:
                self._capture_record(entry, record)
            else:
                entry.update(status="unresolved", error=str(exc))
            logger.warning("Remote evaluation %s requires attention: %s", key, exc)
        entry["updated_at"] = time.time()
        self._save(state)
        return entry

    def submit_async(self, exec_fname, results_dir):
        """Synchronous submission step; the async interface delegates to a thread."""
        if self._closing.is_set():
            raise RuntimeError("Remote scheduler is shutting down; cancellation state is preserved")
        source = Path(exec_fname).resolve()
        request = self.build_request(str(source))
        if request.stage.confirmation:
            raise ValueError("Confirmation results cannot enter the author search archive")
        if Path(request.candidate.source_path).resolve() != source:
            raise ValueError("Request factory must bind the submitted candidate source path")
        if hashlib.sha256(source.read_bytes()).hexdigest() != request.candidate.source_sha:
            raise ValueError("Request factory returned the wrong candidate source SHA")
        _bind_archive_protocol(self.root / "archive-protocol.json", request)
        destination = str(Path(results_dir).resolve())
        key = request.cache_key
        with self._locked():
            if self._closing.is_set():
                raise RuntimeError("Remote scheduler began shutdown before submission")
            state = self._load()
            binding = {"key": key, "source_path": str(source), "source_sha": request.candidate.source_sha}
            if destination in state["bindings"] and state["bindings"][destination] != binding:
                raise ValueError("Shinka result directory is already bound to a different evaluation")
            state["bindings"][destination] = binding
            if key not in state["jobs"]:
                state["jobs"][key] = {
                    "request": request.to_dict(), "status": "pending", "terminal": False,
                    "cancel_requested": False, "record": None, "error": None,
                    "created_at": time.time(), "updated_at": time.time(),
                }
            self._save(state)
            self._advance(state, key)
        return key

    async def submit_async_nonblocking(self, exec_fname, results_dir):
        return await asyncio.to_thread(self.submit_async, exec_fname, results_dir)

    def _poll(self, key):
        with self._locked():
            state = self._load()
            return dict(self._advance(state, key))

    def check_job_status(self, job):
        entry = self._poll(str(job.job_id))
        if entry.get("evaluation_started_at") is not None:
            job.evaluation_started_at = entry["evaluation_started_at"]
        return not entry["terminal"]

    async def batch_check_status_async(self, jobs):
        return await asyncio.gather(*(
            asyncio.to_thread(self.check_job_status, job) for job in jobs
        ), return_exceptions=True)

    def get_job_results(self, key, results_dir):
        """Read only cached, verified terminal results; never perform transport."""
        # State is atomically replaced and terminal records do not revert. Read
        # the committed snapshot without waiting on another job's transport lock;
        # Shinka gives result retrieval only 30 seconds.
        state = self._load()
        destination = str(Path(results_dir).resolve())
        if state["bindings"].get(destination, {}).get("key") != key:
            raise ValueError("Unrecognized remote result directory binding")
        entry = state["jobs"][key]
        if not entry["terminal"]:
            raise EvaluationInProgress("Remote evaluation has no verified terminal result")
        if entry["record"] is not None:
            record = EvaluationRecord.from_dict(entry["record"])
            metrics = author_feedback(record)
        else:
            metrics = {"combined_score": 0.0, "public": {
                "calibration_status": entry["status"], "calibration_reliable": False,
                "evidence_key": key,
            }}
        correct = bool(metrics["public"]["calibration_reliable"])
        payload = {"correct": {"correct": correct, "error": None if correct else
                   "Remote calibration failed, was cancelled, or is unreliable; inspect the private evaluator ledger."},
                   "metrics": metrics, "stdout_log": "", "stderr_log": ""}
        destination_path = Path(destination)
        destination_path.mkdir(parents=True, exist_ok=True)
        for name in ("correct", "metrics"):
            temporary = destination_path / f".{name}-{uuid.uuid4().hex}.tmp"
            temporary.write_text(json.dumps(payload[name], indent=2, allow_nan=False) + "\n")
            temporary.replace(destination_path / (name + ".json"))
        return payload

    async def get_job_results_async(self, key, results_dir):
        return await asyncio.to_thread(self.get_job_results, str(key), results_dir)

    def run(self, exec_fname, results_dir):
        """Generation zero waits in Shinka's executor until its seed is evaluated."""
        started = time.monotonic()
        key = self.submit_async(exec_fname, results_dir)
        while True:
            entry = self._poll(key)
            if entry["terminal"]:
                return self.get_job_results(key, results_dir), time.monotonic() - started
            if self._closing.is_set():
                raise RuntimeError("Remote seed evaluation remains unresolved; cancellation intent is saved")
            time.sleep(self.poll_interval)

    def _cancel(self, key):
        with self._locked():
            state = self._load()
            entry = state["jobs"][key]
            if entry["terminal"]:
                return True
            entry["cancel_requested"] = True
            self._save(state)
            return bool(self._advance(state, key)["terminal"])

    async def cancel_job_async(self, key):
        return await asyncio.to_thread(self._cancel, str(key))

    def shutdown(self):
        """Record all stop intents, then make one bounded cancellation pass.

        Unconfirmed jobs stay in durable state; a restarted controller can keep
        polling them. This method does not claim that scancel means terminated.
        """
        self._closing.set()
        with self._locked():
            state = self._load()
            keys = [key for key, entry in state["jobs"].items() if not entry["terminal"]]
            for key in keys:
                state["jobs"][key]["cancel_requested"] = True
            self._save(state)
        for key in keys:
            try:
                self._poll(key)
            except Exception as exc:
                logger.warning("Remote cancellation remains unresolved for %s: %s", key, exc)
