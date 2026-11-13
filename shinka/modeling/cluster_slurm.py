"""Optional cluster-slurm backend using an already commissioned managed profile.

No raw SSH, account guessing, environment bootstrap, package repair, GPU fallback,
or automatic submission replay occurs here. Keep this adapter and its records
outside author-writeable paths. A saved doctor receipt is evidence of a prior
operator check, not a fresh health check or cryptographic authentication.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import fnmatch
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import uuid

from .remote import JobStatus, Reconciliation


class ClusterSlurmError(RuntimeError):
    pass


class AmbiguousSubmission(ClusterSlurmError):
    pass


# This uses the installed skill's CLI, profile loader, renderer and transport.
# Its retry predicate is disabled because replaying sbatch after a lost response
# may create a second charged job. Header repair would invalidate our reserved
# resources and immutable script, so it also requires a new reviewed submission.
_CLI_LOADER = r'''
import importlib.util, sys
path, arguments = sys.argv[1], sys.argv[2:]
spec = importlib.util.spec_from_file_location("shinka_managed_cluster_cli", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
required = ("main", "is_transient_transport_failure", "run_bounded_sbatch_test_only", "validate_submit_args", "ClusterError")
if not all(hasattr(module, name) for name in required):
    raise RuntimeError("Unsupported cluster-slurm CLI; recommission the adapter")
module.is_transient_transport_failure = lambda *args, **kwargs: False
original_validation = module.run_bounded_sbatch_test_only
def guarded_validation(profile, header, test_once, apply_header_fix, **kwargs):
    def disallow_repair(*args, **kwargs):
        raise module.ClusterError("Automatic header repair requires a new resource reservation; submission stopped")
    return original_validation(profile, header, test_once, disallow_repair, **kwargs)
module.run_bounded_sbatch_test_only = guarded_validation
def bounded_submit_args(values):
    values = [str(value).strip() for value in values if str(value).strip()]
    if any(value != "--parsable" for value in values):
        raise module.ClusterError("Profile submit_args may not override the reserved job; move resource settings into reviewed headers")
    return values
module.validate_submit_args = bounded_submit_args
raise SystemExit(module.main(arguments))
'''


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(value: str) -> str:
    if not isinstance(value, str):
        raise ClusterSlurmError("Artifact paths must be strings")
    path = PurePosixPath(value)
    if value in {"", "."} or path.is_absolute() or ".." in path.parts or "\\" in value or path.as_posix() != value:
        raise ClusterSlurmError("Artifact paths must be normalized relative POSIX paths")
    return value


def _elapsed(value) -> float | None:
    if value is None or str(value).strip() in {"", "Unknown", "UNKNOWN", "N/A"}:
        return None
    text = str(value).strip()
    days, clock = (text.split("-", 1) if "-" in text else ("0", text))
    fields = clock.split(":")
    if len(fields) not in {2, 3}:
        return None
    numbers = [float(item) for item in fields]
    seconds = (numbers[0] * 3600 + numbers[1] * 60 + numbers[2]) if len(numbers) == 3 else numbers[0] * 60 + numbers[1]
    result = int(days) * 86400 + seconds
    return result if math.isfinite(result) and result >= 0 else None


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class ClusterSlurmBackend:
    def __init__(self, root, *, profile, config_path, approved_download_root,
                 doctor_receipt, cli_path=None, python_command=sys.executable,
                 command_runner=None, transport_timeout=60, max_download_bytes=1_000_000_000,
                 approved_artifact_patterns=("calibration-result.json", "intermediate-result.json", "worker-resources.json")):
        self.root = Path(root).resolve()
        self.profile = str(profile)
        self.config_path = Path(config_path).expanduser().resolve()
        self.download_root = Path(approved_download_root).expanduser().resolve()
        self.cli_path = Path(cli_path or "~/.codex/skills/cluster-slurm/scripts/cluster_slurm.py").expanduser().resolve()
        self.python_command = str(python_command)
        self.command_runner = command_runner or subprocess.run
        self.transport_timeout = float(transport_timeout)
        self.max_download_bytes = int(max_download_bytes)
        self.approved_artifact_patterns = tuple(approved_artifact_patterns)
        if not self.approved_artifact_patterns or any(not isinstance(pattern, str) or not pattern for pattern in self.approved_artifact_patterns):
            raise ValueError("Explicit artifact retrieval patterns are required")
        self.doctor_receipt = Path(doctor_receipt).resolve()
        if not self.profile or self.transport_timeout <= 0 or self.max_download_bytes <= 0:
            raise ValueError("Explicit profile and positive bounds are required")
        for path in (self.cli_path, self.config_path, self.doctor_receipt):
            if not path.is_file():
                raise ClusterSlurmError(f"Required commissioned configuration/receipt is missing: {path}")
        receipt = json.loads(self.doctor_receipt.read_text())
        profiles = receipt.get("result", {}).get("profiles", [])
        if receipt.get("ok") is not True or not any(item.get("profile") == self.profile and item.get("ok") is True for item in profiles):
            raise ClusterSlurmError("Provide a successful cluster-slurm doctor receipt for this exact profile before execution")
        self.identity = {"profile": self.profile, "cli_sha256": _digest(self.cli_path),
                         "config_sha256": _digest(self.config_path), "doctor_receipt_sha256": _digest(self.doctor_receipt),
                         "adapter_loader_sha256": hashlib.sha256(_CLI_LOADER.encode()).hexdigest(),
                         "approved_artifact_patterns": list(self.approved_artifact_patterns),
                         "approved_download_root": str(self.download_root), "max_download_bytes": self.max_download_bytes}
        self.root.mkdir(parents=True, exist_ok=True)

    def _verify_identity(self, state=None):
        if (_digest(self.cli_path) != self.identity["cli_sha256"] or _digest(self.config_path) != self.identity["config_sha256"]
                or _digest(self.doctor_receipt) != self.identity["doctor_receipt_sha256"]):
            raise ClusterSlurmError("CLI or profile configuration changed; recommission before further remote work")
        if state is not None and state["identity"] != self.identity:
            raise ClusterSlurmError("This operation belongs to a different commissioned CLI/profile identity")

    def _call(self, *arguments, dry_run=False):
        self._verify_identity()
        argv = [self.python_command, "-c", _CLI_LOADER, str(self.cli_path), "--config", str(self.config_path)]
        if dry_run:
            argv.append("--dry-run")
        argv.extend(str(item) for item in arguments)
        completed = self.command_runner(argv, cwd=str(self.root), timeout=self.transport_timeout,
                                        capture_output=True, text=True, check=False)
        if isinstance(completed, dict):
            payload = completed
        else:
            raw = completed.stdout if completed.returncode == 0 else completed.stderr
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError) as error:
                raise ClusterSlurmError("cluster-slurm returned an unparseable response; inspect private command logs") from error
        if payload.get("ok") is not True or not isinstance(payload.get("result"), dict):
            code = payload.get("error_code", "UNKNOWN")
            raise ClusterSlurmError(f"cluster-slurm {code}: {payload.get('error', 'command failed')}")
        return payload["result"]

    def _state_path(self, key):
        if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ValueError("Operation key must be a lowercase SHA256 digest")
        return self.root / "operations" / f"{key}.json"

    @contextmanager
    def _lock(self, key):
        path = self._state_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield path

    def _load(self, key):
        state = json.loads(self._state_path(key).read_text())
        self._verify_identity(state)
        return state

    def _job_state(self, job_id):
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", str(job_id)):
            raise ValueError("Invalid scheduler job ID")
        matches = []
        for path in (self.root / "operations").glob("*.json"):
            state = json.loads(path.read_text())
            if state.get("job_id") == str(job_id):
                self._verify_identity(state)
                matches.append(state)
        if len(matches) != 1:
            raise ClusterSlurmError("Job ID is not uniquely owned by this adapter")
        return matches[0]

    def stage(self, packet_dir: Path, key: str) -> str:
        packet_dir = Path(packet_dir).resolve()
        manifest = json.loads((packet_dir / "packet.json").read_text())
        if manifest.get("work_key") != key or manifest.get("spec", {}).get("profile") != self.profile:
            raise ClusterSlurmError("Packet work/profile identity does not match the adapter")
        with self._lock(key) as path:
            if path.exists():
                state = self._load(key)
                if state["packet_sha256"] != _digest(packet_dir / "packet.json"):
                    raise ClusterSlurmError("An existing operation cannot replace its immutable packet")
                if state.get("staged"):
                    return state["run_id"]
            else:
                run_id = f"shinka-{key}"
                preview = self._call("init-run", "--profile", self.profile, "--run-id", run_id, dry_run=True)
                state = {"key": key, "run_id": run_id, "identity": self.identity,
                         "local_run_dir": preview["local_run_dir"], "packet_sha256": _digest(packet_dir / "packet.json"),
                         "packet_spec": manifest["spec"], "packet_command": manifest["command"],
                         "submit_attempted": False, "staged": False, "job_id": None}
                _write(path, state)
            local_run = Path(state["local_run_dir"]) / "run.json"
            if local_run.exists():
                run = json.loads(local_run.read_text())
                if run.get("run_id") != state["run_id"] or run.get("profile_id") != self.profile:
                    raise ClusterSlurmError("Existing skill run has a different identity")
                if run.get("job_id") or state.get("submit_attempted"):
                    raise AmbiguousSubmission("Cannot restage or reinitialize a submitted run")
            else:
                initialized = self._call("init-run", "--profile", self.profile, "--run-id", state["run_id"])
                run = initialized["run"]
            state["remote_run_dir"] = run["remote_run_dir"]
            packet = self.root / "packets" / key / "packet"
            if packet.exists():
                shutil.rmtree(packet)  # Only our own pre-submit allowlisted staging copy.
            packet.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(packet_dir / "packet.json", packet / "packet.json")
            for entry in manifest["files"]:
                relative = _relative(entry["path"])
                source = packet_dir / "files" / relative
                if source.is_symlink() or not source.resolve().is_relative_to(packet_dir / "files") or not source.is_file():
                    raise ClusterSlurmError("Packet contains an unsafe input path")
                if _digest(source) != entry["sha256"] or source.stat().st_size != entry["bytes"]:
                    raise ClusterSlurmError("Packet input hash or length changed")
                destination = packet / "files" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            # Upload explicit files, not a directory: scp -r to an existing
            # destination can nest packet/packet on a resumed partial upload.
            groups = {"packet": [packet / "packet.json"]}
            for entry in manifest["files"]:
                relative = PurePosixPath(entry["path"])
                remote_subdir = str(PurePosixPath("packet/files") / relative.parent)
                groups.setdefault(remote_subdir, []).append(packet / "files" / entry["path"])
            for remote_subdir, files in groups.items():
                arguments = ["upload", "--run-id", state["run_id"], "--remote-subdir", remote_subdir]
                for file in files:
                    arguments.extend(["--local-path", str(file)])
                self._call(*arguments)
            state["staged"] = True
            _write(path, state)
            return state["run_id"]

    @staticmethod
    def _check_header(header, spec):
        cpus = int(header.get("cpus_per_task", 1))
        nodes, tasks = int(header.get("nodes", 1)), int(header.get("ntasks", 1))
        per_node = int(header.get("ntasks_per_node", 1))
        gpu_values = [int(header.get(key) or 0) for key in ("gpus", "gpus_per_node")]
        gres = str(header.get("gres", ""))
        if gres:
            matches = re.findall(r"gpu(?::[^:,=]+)?:(\d+)", gres)
            if not matches:
                raise ClusterSlurmError("Unrecognized profile GPU allocation")
            gpu_values.extend(int(value) for value in matches)
        if nodes != 1 or tasks != 1 or per_node != 1 or cpus != spec.cpus or max(gpu_values, default=0) != spec.gpus:
            raise ClusterSlurmError("Rendered profile allocation differs from the reserved CPU/GPU resources")

    def submit(self, handle, key, spec, command) -> str:
        with self._lock(key) as path:
            state = self._load(key)
            if handle != state["run_id"] or spec.profile != self.profile or not state.get("staged"):
                raise ClusterSlurmError("Unstaged or mismatched remote operation")
            if state["packet_spec"] != spec.to_dict() or state["packet_command"] != list(command):
                raise ClusterSlurmError("Submission changed the immutable packet resource or command contract")
            if state.get("job_id"):
                return state["job_id"]
            if state.get("submit_attempted"):
                raise AmbiguousSubmission("A submit was attempted; reconcile it instead of submitting again")
            if not command or any(not isinstance(value, str) or "\x00" in value for value in command):
                raise ValueError("Worker command must be a nonempty argv sequence")
            if not float(spec.runtime_seconds).is_integer():
                raise ValueError("SLURM runtime_seconds must be integral to match the exact resource reservation")
            seconds = int(spec.runtime_seconds)
            header = {"job_name": f"shinka-{key[:24]}", "nodes": 1, "ntasks": 1, "ntasks_per_node": 1, "cpus_per_task": spec.cpus,
                      "mem": f"{spec.memory_mb}M", "time": f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"}
            if spec.gpus:
                header["gpus"] = spec.gpus
            arguments = ["render-job", "--run-id", handle, "--script-name", "job.slurm",
                         "--command", "cd packet && exec " + shlex.join(command)]
            for name, value in header.items():
                arguments += ["--header", f"{name}={value}"]
            rendered = self._call(*arguments)
            run = json.loads((Path(state["local_run_dir"]) / "run.json").read_text())
            self._check_header(run["header"], spec)
            script = Path(rendered["local_script"])
            state.update({"script_sha256": _digest(script), "spec": spec.to_dict(),
                          "command": list(command), "submit_attempted": True, "submission_state": "unknown"})
            _write(path, state)  # Must precede the one possible charged side effect.
            try:
                submitted = self._call("submit-job", "--run-id", handle, "--script-name", "job.slurm")
                job_id = str(submitted["job_id"])
                if re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id) is None:
                    raise ClusterSlurmError("Invalid job ID in submission response")
            except Exception as error:
                state["submission_error"] = str(error)
                _write(path, state)
                raise AmbiguousSubmission("Submission response is uncertain; no automatic replay is allowed") from error
            state.update({"job_id": job_id, "submission_state": "submitted"})
            _write(path, state)
            return job_id

    def reconcile(self, key):
        path = self._state_path(key)
        if not path.exists():
            return Reconciliation("absent")
        with self._lock(key):
            state = self._load(key)
            if state.get("job_id"):
                return Reconciliation("found", state["job_id"])
            local_run = Path(state["local_run_dir"]) / "run.json"
            if local_run.exists():
                run = json.loads(local_run.read_text())
                if run.get("run_id") != state["run_id"] or run.get("profile_id") != self.profile:
                    raise ClusterSlurmError("Cannot reconcile a foreign skill run")
                job_id = str(run.get("job_id") or "")
                if re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
                    state.update({"job_id": job_id, "submission_state": "submitted"})
                    _write(path, state)
                    return Reconciliation("found", job_id)
            # A missing ledger job ID after sbatch transport loss is NOT absence.
            return Reconciliation("unknown" if state.get("submit_attempted") else "absent")

    def status(self, job_id):
        state = self._job_state(job_id)
        response = self._call("status", "--run-id", state["run_id"], "--job-id", str(job_id))
        status = response.get("status", {})
        scheduler_state = str(status.get("state", "UNKNOWN")).split()[0].rstrip("+")
        mapping = {"PENDING": "queued", "REQUEUED": "queued", "RUNNING": "running", "CONFIGURING": "running",
                   "COMPLETING": "running", "SUSPENDED": "running", "COMPLETED": "completed", "CANCELLED": "cancelled",
                   "FAILED": "failed", "TIMEOUT": "failed", "OUT_OF_MEMORY": "failed", "NODE_FAIL": "failed",
                   "BOOT_FAIL": "failed", "PREEMPTED": "failed", "DEADLINE": "failed"}
        runtime = _elapsed(status.get("elapsed"))
        state["last_status"] = {"state": scheduler_state, "runtime_seconds": runtime,
                                "allocated_cpu_seconds": runtime * state["spec"]["cpus"] if runtime is not None else None,
                                "allocated_gpu_seconds": runtime * state["spec"]["gpus"] if runtime is not None else None,
                                "accounting_basis": "verified requested allocation × scheduler elapsed; actual scheduler allocation may differ"}
        _write(self._state_path(state["key"]), state)
        normalized = mapping.get(scheduler_state, "unknown")
        exit_code = status.get("exit_code")
        if normalized == "completed" and exit_code not in {None, "0:0", "0", 0}:
            normalized = "failed"
        error = status.get("reason") if normalized in {"failed", "unknown"} else None
        if normalized == "failed" and exit_code is not None:
            error = f"scheduler exit_code={exit_code}"
        return JobStatus(normalized, runtime, error)

    def cancel(self, job_id):
        state = self._job_state(job_id)
        self._call("cancel", "--run-id", state["run_id"], "--job-id", str(job_id))
        state["cancel_requested"] = True
        _write(self._state_path(state["key"]), state)
        # scancel acceptance is not terminal status; the controller keeps polling.

    def _download(self, run_id, remote_path, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._call("download", "--run-id", run_id, "--remote-path", remote_path, "--local-path", str(destination))
        if not destination.is_file() or destination.is_symlink():
            raise ClusterSlurmError("Download did not produce a regular artifact file")

    def fetch(self, handle, job_id, destination: Path):
        state = self._job_state(job_id)
        destination = Path(destination).resolve()
        if handle != state["run_id"] or not destination.is_relative_to(self.download_root):
            raise ClusterSlurmError("Artifact destination or run is outside the approved retrieval scope")
        destination.mkdir(parents=True, exist_ok=True)
        manifest_path = destination / "result-manifest.json"
        temporary_manifest = destination / f".result-manifest.{uuid.uuid4().hex}.download"
        self._download(handle, "packet/results/result-manifest.json", temporary_manifest)
        if temporary_manifest.stat().st_size > 1024 * 1024:
            temporary_manifest.unlink()
            raise ClusterSlurmError("Result manifest exceeds the 1 MiB metadata bound")
        manifest = json.loads(temporary_manifest.read_text())
        manifest_digest = _digest(temporary_manifest)
        if manifest.get("work_key") != state["key"]:
            raise ClusterSlurmError("Returned manifest belongs to a different operation")
        total = sum(int(entry["bytes"]) for entry in manifest["artifacts"])
        if total < 0 or total > self.max_download_bytes:
            raise ClusterSlurmError("Returned artifact volume exceeds the explicit download bound")
        # Check disclosure scope before downloading any listed result artifact.
        for entry in manifest["artifacts"]:
            relative = _relative(entry["path"])
            if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in self.approved_artifact_patterns):
                raise ClusterSlurmError(f"Result type is outside approved retrieval patterns: {relative}")
        if state.get("downloaded_manifest_sha256") and state["downloaded_manifest_sha256"] != manifest_digest:
            raise ClusterSlurmError("Completed result manifest changed after prior retrieval")
        seen = set()
        for entry in manifest["artifacts"]:
            relative = _relative(entry["path"])
            if relative in seen or relative == "result-manifest.json" or int(entry["bytes"]) < 0:
                raise ClusterSlurmError("Duplicate or invalid result artifact")
            seen.add(relative)
            target = destination / relative
            if not target.resolve().is_relative_to(destination):
                raise ClusterSlurmError("Result artifact leaves the approved directory")
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.download")
            self._download(handle, "packet/results/" + relative, temporary)
            if temporary.stat().st_size != entry["bytes"] or _digest(temporary) != entry["sha256"]:
                temporary.unlink(missing_ok=True)
                raise ClusterSlurmError("Returned artifact hash or length mismatch")
            temporary.replace(target)
        temporary_manifest.replace(manifest_path)
        state["downloaded_manifest_sha256"] = manifest_digest
        _write(self._state_path(state["key"]), state)
