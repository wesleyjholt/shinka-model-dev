"""Connect asynchronous remote jobs to the existing scientific fit ledger."""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from .constraints import ConstraintPending, ConstraintSnapshot
from .pipeline import CalibrationPipeline
from .remote import TERMINAL
from .remote_worker import file_identity


class RemoteCalibrationPipeline:
    """Submit/poll without waiting through a scheduler queue.

    The local ledger counts calibration attempts and compute wall time. The
    remote controller separately reserves and charges CPU/GPU allocation time
    for every job, including intermediate probes. Use per-tier controllers when
    CPU/GPU allocations must also be reserved separately for each fidelity.
    """

    def __init__(self, ledger, controller, *, constraints=None, constraint_reviewer=None):
        self.ledger = ledger
        self.controller = controller
        self.local = CalibrationPipeline(ledger, controller.root / "calibration-records",
                                         constraints=constraints, constraint_reviewer=constraint_reviewer)

    def _gate(self, request):
        book = self.local.constraints
        if book is None:
            return None
        snapshot = book.snapshot()
        if snapshot.version == 0:
            raise ValueError("Configured user constraint book is missing or uninitialized")
        if snapshot.active and self.local.constraint_reviewer is not None:
            review = self.local.constraint_reviewer(request.candidate, snapshot)
            if review is not None:
                book.review_compliance(request.candidate.source_sha,
                                       expected_constraint_sha=snapshot.sha256, **review)
                if not review["compliant"]:
                    book.request_exception(request.candidate.source_sha, review["evidence"])
        permission = book.assert_allowed(request.candidate.source_sha)
        return ConstraintSnapshot(permission["version"], permission["sha256"], permission["text"], permission["active"])

    def _reference_path(self, cache_key):
        return self.local.artifact_root / cache_key / "remote-reference.json"

    @contextmanager
    def _locked(self, cache_key):
        import fcntl
        path = self.local.artifact_root / cache_key
        path.mkdir(parents=True, exist_ok=True)
        with (path / ".bridge.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def _save_reference(self, request, job, snapshot):
        path = self._reference_path(request.cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"work_key": job.key, "request_key": request.cache_key,
                   "constraints": snapshot.to_dict() if snapshot else None}
        if path.exists() and json.loads(path.read_text())["work_key"] != job.key:
            raise ValueError("This calibration already has a different remote work identity")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(path)

    def job_for(self, cache_key):
        path = self._reference_path(cache_key)
        if path.exists():
            return self.controller.get(json.loads(path.read_text())["work_key"])
        # Recover a crash after remote submission but before the pointer write.
        matches = [job for job in self.controller.records()
                   if job.packet["request_key"] == cache_key and job.packet["task_kind"] == "calibration"]
        if len(matches) > 1:
            raise ValueError("Ambiguous remote jobs for one calibration; reconcile before continuing")
        return matches[0] if matches else None

    def submit(self, request, spec, files, command):
        with self._locked(request.cache_key):
            return self._submit(request, spec, files, command)

    def _submit(self, request, spec, files, command):
        snapshot = self._gate(request)
        if request.stage.max_seconds > spec.runtime_seconds:
            raise ValueError("Remote allocation must cover the numerical stage timeout plus suitable setup headroom")
        artifact_dir = self.local.artifact_root / request.cache_key
        record, claimed = self.ledger.reserve(request, str(artifact_dir))
        if not claimed and record.status in {"completed", "failed"}:
            return record
        if record.status == "reserved":
            self.ledger.start(request.cache_key)
        previous = self.job_for(request.cache_key)
        if previous is not None:
            # Never create another allocation for an in-flight calibration.
            if previous.packet["spec"] != spec.to_dict() or tuple(previous.packet["command"]) != tuple(command):
                raise ValueError("In-flight remote calibration resources/command differ; reconcile the existing job")
            manifest = [file_identity(Path(path), name) for name, path in sorted(files.items())]
            if manifest != [dict(item) for item in previous.packet["files"]]:
                raise ValueError("In-flight remote inputs changed; retain the original files or create a new request identity")
            job = previous
        else:
            try:
                job = self.controller.submit(request, spec, files, command)
            except Exception:
                # The controller persists ambiguous submissions and returns an
                # uncertain job. An exception here precedes a recorded job.
                if self.job_for(request.cache_key) is None:
                    self.ledger.fail(request.cache_key, "Remote submission rejected before a job was recorded", 0)
                raise
        self._save_reference(request, job, snapshot)
        return self._poll(request.cache_key)

    def poll(self, cache_key):
        with self._locked(cache_key):
            return self._poll(cache_key)

    def _poll(self, cache_key):
        record = self.ledger.get(cache_key)
        if record is None:
            raise KeyError(cache_key)
        if record.status in {"completed", "failed"}:
            return record
        job = self.job_for(cache_key)
        if job is None:
            return record
        job = self.controller.poll(job.key)
        if job.status not in TERMINAL:
            return self.ledger.get(cache_key)
        elapsed = job.runtime_seconds or 0.0
        if job.status == "completed" and job.result is not None:
            result = job.result
            path = self._reference_path(cache_key)
            snapshot = json.loads(path.read_text()).get("constraints") if path.exists() else None
            note = f"Remote work {job.key}; compute accounting: {job.accounting_basis}; queue time excluded."
            if snapshot:
                note += f" User constraints at submission: version={snapshot['version']}, sha256={snapshot['sha256']}."
            result = replace(result, notes=result.notes + (note,))
            return self.ledger.complete(cache_key, result, elapsed)
        return self.ledger.fail(cache_key, job.error or f"Remote job ended as {job.status}", elapsed)

    def cancel(self, cache_key):
        job = self.job_for(cache_key)
        if job is None:
            raise KeyError(cache_key)
        self.controller.cancel(job.key)
        return self.poll(cache_key)

    def submit_probe(self, request, spec, files, command, *, task_kind, parent_evaluation_key):
        if task_kind == "calibration" or not parent_evaluation_key:
            raise ValueError("Intermediate work requires its own task kind and parent evaluation identity")
        self._gate(request)
        return self.controller.submit(request, spec, files, command, task_kind=task_kind,
                                      parent_evaluation_key=parent_evaluation_key)
