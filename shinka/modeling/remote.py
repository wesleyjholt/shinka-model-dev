"""Durable remote work accounting; scheduler transports never supply scores.

No SSH, environment bootstrap, submission, or waiting is built into this module.
Backends implement bounded transport calls through a configured scheduler. An
ambiguous submission stays reserved until authoritative reconciliation succeeds.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Protocol

from .contracts import CalibrationResult, EvaluationRequest, _freeze
from .store import BudgetExhausted


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value):
    path = Path(value)
    if not isinstance(value, str) or not value or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("artifact paths must be normalized relative paths")
    return path


@dataclass(frozen=True)
class RemoteSpec:
    profile: str
    cpus: int = 1
    gpus: int = 0
    runtime_seconds: float = 300
    queue_timeout_seconds: float = 3600
    memory_mb: int = 1024

    def __post_init__(self):
        if not isinstance(self.profile, str) or not self.profile:
            raise ValueError("an explicitly configured scheduler profile is required")
        for name, minimum in (("cpus", 1), ("gpus", 0), ("memory_mb", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"invalid {name}")
        for name in ("runtime_seconds", "queue_timeout_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, float(value))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Reconciliation:
    # 'absent' means authoritative proof that the previous submit did not
    # create a job; an empty squeue response alone is never that proof.
    state: str
    job_id: str | None = None


@dataclass(frozen=True)
class JobStatus:
    state: str
    runtime_seconds: float | None = None
    error: str | None = None
    allocated_cpus: int | None = None
    allocated_gpus: int | None = None


class RemoteBackend(Protocol):
    def stage(self, packet_dir: Path, key: str) -> str: ...
    def submit(self, handle: str, key: str, spec: RemoteSpec, command: tuple[str, ...]) -> str: ...
    def reconcile(self, key: str) -> Reconciliation: ...
    def status(self, job_id: str) -> JobStatus: ...
    def cancel(self, job_id: str) -> None: ...
    def fetch(self, handle: str, job_id: str, destination: Path) -> None: ...


TERMINAL = frozenset({"completed", "failed", "cancelled", "invalid_result"})


@dataclass(frozen=True)
class RemoteJob:
    key: str
    status: str
    packet: dict
    created_at: float
    updated_at: float
    handle: str | None = None
    job_id: str | None = None
    submitted_at: float | None = None
    started_at: float | None = None
    error: str | None = None
    result: CalibrationResult | None = None
    output_paths: dict | None = None
    charged_cpu_seconds: float = 0
    charged_gpu_seconds: float = 0
    accounting_final: bool = False
    cancel_intent: bool = False
    runtime_seconds: float = 0
    accounting_basis: str | None = None

    @property
    def request(self):
        return EvaluationRequest.from_dict(self.packet["request"])

    def to_dict(self):
        value = asdict(self)
        return json.loads(_json(value))

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["packet"] = _freeze(value["packet"])
        if value.get("result") is not None:
            value["result"] = CalibrationResult.from_dict(value["result"])
        if value.get("output_paths") is not None:
            value["output_paths"] = _freeze(value["output_paths"])
        return cls(**value)


class RemoteController:
    """Advance one lifecycle step per call; never wait for queued/running jobs.

    Limits are immutable campaign settings. In-flight work reserves the complete
    declared CPU/GPU allocation. Terminal work uses scheduler elapsed allocation
    time; missing accounting conservatively charges the full reservation. Queue
    time consumes an in-flight slot but is not billed as compute. A transport's
    submit must make exactly one scheduler attempt, with no automatic repairs.
    """

    def __init__(self, state_dir, backend: RemoteBackend, *, max_jobs=None, max_inflight=None,
                 max_cpu_seconds=None, max_gpu_seconds=None, clock=time.time):
        self.root = Path(state_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend, self.clock = backend, clock
        self.state_path = self.root / "remote-state.json"
        supplied = dict(max_jobs=max_jobs, max_inflight=max_inflight,
                        max_cpu_seconds=max_cpu_seconds, max_gpu_seconds=max_gpu_seconds)
        with self._locked():
            if self.state_path.exists():
                state = self._load()
                if any(value is not None and value != state["limits"][key] for key, value in supplied.items()):
                    raise ValueError("remote campaign limits cannot change on restart")
            else:
                for name, value in supplied.items():
                    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                        raise ValueError("new remote campaigns require finite nonnegative limits")
                if type(max_jobs) is not int or type(max_inflight) is not int or max_inflight < 1:
                    raise ValueError("max_jobs/max_inflight must be integers, with max_inflight positive")
                self._save({"schema_version": 1, "limits": supplied, "jobs": {}})

    @contextmanager
    def _locked(self):
        with (self.root / ".remote.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _load(self):
        value = json.loads(self.state_path.read_text())
        if value.get("schema_version") != 1:
            raise ValueError("unknown remote state schema")
        return value

    def _save(self, state):
        temporary = self.state_path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            stream.write(_json(state))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.state_path)
        descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _update(self, state, job, **changes):
        job.update(changes, updated_at=self.clock())
        self._save(state)

    @staticmethod
    def _usage(state):
        jobs = list(state["jobs"].values())
        pending = [job for job in jobs if not job["accounting_final"]]
        return {"job_count": len(jobs), "inflight": len(pending),
                "reserved_cpu_seconds": sum(job["packet"]["spec"]["cpus"] * job["packet"]["spec"]["runtime_seconds"] for job in pending),
                "reserved_gpu_seconds": sum(job["packet"]["spec"]["gpus"] * job["packet"]["spec"]["runtime_seconds"] for job in pending),
                "charged_cpu_seconds": sum(job["charged_cpu_seconds"] for job in jobs),
                "charged_gpu_seconds": sum(job["charged_gpu_seconds"] for job in jobs),
                "limits": state["limits"]}

    def usage(self):
        with self._locked():
            return self._usage(self._load())

    def get(self, key):
        with self._locked():
            value = self._load()["jobs"].get(key)
            return RemoteJob.from_dict(value) if value else None

    def records(self):
        with self._locked():
            return [RemoteJob.from_dict(value) for value in self._load()["jobs"].values()]

    def submit(self, request: EvaluationRequest, spec: RemoteSpec, files, command,
               *, task_kind="calibration", parent_evaluation_key=None):
        if request.stage.max_seconds > spec.runtime_seconds:
            raise ValueError("stage timeout exceeds the remote compute allocation")
        if not task_kind or not isinstance(task_kind, str) or not command or any(not isinstance(arg, str) or not arg for arg in command):
            raise ValueError("a named work item and nonempty command argv are required")
        entries = []
        for name, source in sorted(files.items()):
            _relative(name)
            source = Path(source)
            if not source.is_file():
                raise ValueError("packet inputs must be explicit local files")
            entries.append({"path": name, "sha256": _sha(source), "bytes": source.stat().st_size})
        candidate_source = next((entry["path"] for entry in entries if entry["sha256"] == request.candidate.source_sha), None)
        if task_kind == "calibration" and candidate_source is None:
            raise ValueError("calibration packet must contain the exact candidate source")
        packet = {"schema_version": 1, "request": request.to_dict(), "request_key": request.cache_key,
                  "protocol_key": request.protocol_key, "source_sha": request.candidate.source_sha,
                  "task_kind": task_kind, "parent_evaluation_key": parent_evaluation_key,
                  "spec": spec.to_dict(), "command": list(command), "files": entries,
                  "candidate_source": candidate_source}
        key = hashlib.sha256(_json(packet).encode()).hexdigest()
        packet["work_key"] = key
        with self._locked():
            state = self._load()
            job = state["jobs"].get(key)
            if job is None:
                usage, limits = self._usage(state), state["limits"]
                if usage["job_count"] >= limits["max_jobs"] or usage["inflight"] >= limits["max_inflight"]:
                    raise BudgetExhausted("remote job count or in-flight limit exhausted")
                for resource in ("cpu", "gpu"):
                    allocation = getattr(spec, resource + "s") * spec.runtime_seconds
                    if usage[f"reserved_{resource}_seconds"] + usage[f"charged_{resource}_seconds"] + allocation > limits[f"max_{resource}_seconds"]:
                        raise BudgetExhausted(f"remote {resource.upper()} allocation budget exhausted")
                now = self.clock()
                job = RemoteJob(key, "reserved", packet, now, now).to_dict()
                state["jobs"][key] = job
                self._save(state)
            if job["status"] in {"submitting", "uncertain"}:
                self._reconcile(state, job)
            if job["cancel_intent"] and job["status"] == "staged":
                self._update(state, job, status="cancelled", accounting_final=True,
                             accounting_basis="confirmed_not_submitted")
            elif job["cancel_intent"] and job["job_id"] and job["status"] not in TERMINAL:
                self._cancel(state, job)
            if job["status"] == "reserved":
                directory = self.root / "packets" / key
                try:
                    if not (directory / "files").resolve().is_relative_to(self.root):
                        raise ValueError("packet directory escapes remote controller state")
                    (directory / "files").mkdir(parents=True, exist_ok=True)
                    for entry in entries:
                        target = directory / "files" / entry["path"]
                        if target.is_symlink() or not target.resolve().is_relative_to((directory / "files").resolve()):
                            raise ValueError("packet target escapes its immutable input directory")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(files[entry["path"]], target)
                        if _sha(target) != entry["sha256"] or target.stat().st_size != entry["bytes"]:
                            raise ValueError("input changed while snapshotting remote packet")
                    (directory / "packet.json").write_text(_json(packet))
                    handle = self.backend.stage(directory, key)
                    if not isinstance(handle, str) or not handle:
                        raise ValueError("backend did not return a staging handle")
                    self._update(state, job, status="staged", handle=handle, error=None)
                except Exception as exc:
                    self._update(state, job, status="failed", accounting_final=True, error=f"staging failed: {exc}")
            if job["status"] == "staged":
                self._update(state, job, status="submitting", submitted_at=self.clock(), error=None)
                try:
                    job_id = self.backend.submit(job["handle"], key, spec, tuple(command))
                    if not isinstance(job_id, str) or not job_id:
                        raise ValueError("submission returned no scheduler job identity")
                    self._update(state, job, status="queued", job_id=job_id)
                except Exception as exc:
                    self._update(state, job, status="uncertain", error=f"submission requires reconciliation: {exc}")
            return RemoteJob.from_dict(job)

    def _reconcile(self, state, job):
        try:
            answer = self.backend.reconcile(job["key"])
            if answer.state == "found" and answer.job_id:
                self._update(state, job, status="queued", job_id=answer.job_id, error=None)
            elif answer.state == "absent":
                self._update(state, job, status="staged", error="previous submission authoritatively absent; safe to submit again")
            elif answer.state == "unknown":
                self._update(state, job, status="uncertain")
            else:
                raise ValueError("invalid reconciliation response")
        except Exception as exc:
            self._update(state, job, status="uncertain", error=f"reconciliation unavailable: {exc}")

    def reconcile(self, key):
        return self.poll(key)

    def _charge(self, job, status):
        spec = job["packet"]["spec"]
        runtime = status.runtime_seconds
        if runtime is None:
            runtime = spec["runtime_seconds"]
        if not math.isfinite(runtime) or runtime < 0:
            raise ValueError("invalid scheduler compute accounting")
        counts = {}
        for name in ("cpus", "gpus"):
            observed = getattr(status, "allocated_" + name)
            if observed is not None and (type(observed) is not int or observed < (1 if name == "cpus" else 0)):
                raise ValueError("invalid scheduler allocation counts")
            counts[name] = max(spec[name], observed if observed is not None else spec[name])
        job.update(accounting_final=True, runtime_seconds=runtime,
                   accounting_basis="scheduler" if status.runtime_seconds is not None else "reserved_upper_bound",
                   charged_cpu_seconds=runtime * counts["cpus"],
                   charged_gpu_seconds=runtime * counts["gpus"])

    def _fetch(self, state, job):
        destination = self.root / "results" / job["key"]
        if not destination.resolve().is_relative_to(self.root):
            self._update(state, job, status="invalid_result", error="result directory escapes controller state")
            return
        destination.mkdir(parents=True, exist_ok=True)
        try:
            self.backend.fetch(job["handle"], job["job_id"], destination)
        except Exception as exc:
            self._update(state, job, status="fetch_pending", error=f"result retrieval pending: {exc}")
            return
        try:
            manifest = json.loads((destination / "result-manifest.json").read_text())
            packet, request = job["packet"], job["packet"]["request"]
            identity = {"schema_version": 1, "work_key": job["key"],
                        **{key: packet[key] for key in ("request_key", "protocol_key", "source_sha", "task_kind", "parent_evaluation_key")},
                        **{key: request[key] for key in ("stage", "dataset_id", "split_id", "objective_id", "evaluator_id")}}
            if any(manifest.get(key) != value for key, value in identity.items()):
                raise ValueError("result manifest does not match immutable request identity")
            paths = {}
            for entry in manifest["artifacts"]:
                relative = _relative(entry["path"])
                path = destination / relative
                if (entry["path"] in paths or path.is_symlink() or not path.is_file()
                        or not path.resolve().is_relative_to(destination.resolve())
                        or path.stat().st_size != entry["bytes"] or _sha(path) != entry["sha256"]):
                    raise ValueError("artifact integrity or containment check failed")
                paths[entry["path"]] = str(path.resolve())
            result = None
            if packet["task_kind"] == "calibration":
                result_path = manifest.get("result_path")
                if result_path not in paths:
                    raise ValueError("calibration result must be a checksummed artifact")
                result = CalibrationResult.from_dict(json.loads(Path(paths[result_path]).read_text()))
                if request["stage"]["method"].lower() == "map" and result.posterior_kind not in {"none", "point_estimate"}:
                    raise ValueError("MAP cannot claim posterior uncertainty")
                if any(value not in paths for value in result.artifacts.values()):
                    raise ValueError("result references an unverified artifact")
                value = result.to_dict()
                value["artifacts"] = {name: paths[value] for name, value in result.artifacts.items()}
                result = CalibrationResult.from_dict(value)
            elif manifest.get("result_path") is not None:
                raise ValueError("intermediate work cannot masquerade as a calibration score")
            self._update(state, job, status="completed", result=result.to_dict() if result else None,
                         output_paths=paths, error=None)
        except Exception as exc:
            self._update(state, job, status="invalid_result", error=f"result verification failed: {exc}")

    def poll(self, key):
        with self._locked():
            state = self._load()
            job = state["jobs"][key]
            if job["status"] in TERMINAL:
                return RemoteJob.from_dict(job)
            if job["status"] in {"submitting", "uncertain"}:
                self._reconcile(state, job)
                if job["cancel_intent"] and job["job_id"]:
                    self._cancel(state, job)
                elif job["cancel_intent"] and job["status"] == "staged":
                    self._update(state, job, status="cancelled", accounting_final=True,
                                 accounting_basis="confirmed_not_submitted")
            elif job["status"] == "fetch_pending":
                self._fetch(state, job)
            elif job["job_id"]:
                try:
                    status = self.backend.status(job["job_id"])
                    if status.state in {"completed", "failed", "cancelled"}:
                        self._charge(job, status)
                        self._update(state, job, status="fetch_pending" if status.state == "completed" else status.state, error=status.error)
                        if status.state == "completed":
                            self._fetch(state, job)
                    elif status.state in {"queued", "running", "unknown"}:
                        changes = {"error": status.error}
                        if job["status"] != "cancel_requested":
                            changes["status"] = status.state
                        if status.state == "running" and job["started_at"] is None:
                            changes["started_at"] = self.clock()
                        self._update(state, job, **changes)
                        spec = job["packet"]["spec"]
                        queue_expired = status.state == "queued" and self.clock() - job["submitted_at"] >= spec["queue_timeout_seconds"]
                        runtime_expired = status.state == "running" and status.runtime_seconds is not None and status.runtime_seconds > spec["runtime_seconds"]
                        if job["cancel_intent"] or queue_expired or runtime_expired:
                            self._cancel(state, job)
                    else:
                        raise ValueError("invalid scheduler status")
                except Exception as exc:
                    self._update(state, job, error=f"scheduler state unavailable: {exc}")
            return RemoteJob.from_dict(job)

    def _cancel(self, state, job):
        self._update(state, job, status="cancel_requested", cancel_intent=True, error="cancellation awaiting scheduler confirmation")
        try:
            self.backend.cancel(job["job_id"])
        except Exception as exc:
            self._update(state, job, error=f"cancellation unconfirmed: {exc}")

    def cancel(self, key):
        with self._locked():
            state = self._load()
            job = state["jobs"][key]
            if job["status"] not in TERMINAL:
                self._update(state, job, cancel_intent=True)
                if job["status"] in {"reserved", "staged"}:
                    self._update(state, job, status="cancelled", accounting_final=True)
                    return RemoteJob.from_dict(job)
                if not job["job_id"]:
                    self._reconcile(state, job)
                if job["job_id"]:
                    self._cancel(state, job)
            return RemoteJob.from_dict(job)
