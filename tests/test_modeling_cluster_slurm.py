"""Fake CLI responses exercise orchestration; no profile or cluster is contacted."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from shinka.modeling.cluster_slurm import (
    AmbiguousSubmission, ClusterSlurmBackend, ClusterSlurmError, _CLI_LOADER,
)
from shinka.modeling.remote import RemoteSpec

KEY = "a" * 64
COMMAND = ("python3", "-m", "shinka.modeling.remote_worker", "--packet", "packet.json", "--output", "results", "--calibrator", "trusted:evaluate")


class FakeCLI:
    def __init__(self, root):
        self.root = root
        self.calls = []
        self.submit_mode = "success"
        self.scheduler_state = "COMPLETED"
        self.elapsed = "00:01:05"
        self.upscale = False
        self.downloads = {}

    def __call__(self, argv, **kwargs):
        arguments = argv[argv.index("--config") + 2:]
        dry = arguments[0] == "--dry-run"
        if dry:
            arguments = arguments[1:]
        command, rest = arguments[0], arguments[1:]
        self.calls.append((command, dry, rest, kwargs))
        def get(flag):
            return rest[rest.index(flag) + 1]
        run_id = get("--run-id") if "--run-id" in rest else ""
        directory = self.root / "skill-runs" / run_id
        run_file = directory / "run.json"
        if command == "init-run":
            if dry:
                result = {"local_run_dir": str(directory), "run_id": run_id, "dry_run": True}
            else:
                directory.mkdir(parents=True, exist_ok=True)
                run = {"run_id": run_id, "profile_id": "test-cpu", "local_run_dir": str(directory),
                       "remote_run_dir": f"/scratch/test/codex-runs/{run_id}", "job_id": None}
                run_file.write_text(json.dumps(run))
                result = {"run": run}
        elif command == "upload":
            local_paths = [Path(rest[index + 1]) for index, token in enumerate(rest) if token == "--local-path"]
            assert all(path.is_file() for path in local_paths)
            subdir = get("--remote-subdir")
            relative_dir = Path(subdir).relative_to("packet")
            names = [str(relative_dir / path.name) for path in local_paths]
            self.uploaded_names = sorted(set(getattr(self, "uploaded_names", []) + names))
            result = {"run_id": run_id, "uploaded": [{"local": str(path), "remote": f"/fake/{subdir}/{path.name}"} for path in local_paths]}
        elif command == "render-job":
            header = dict(rest[index + 1].split("=", 1) for index, item in enumerate(rest) if item == "--header")
            if self.upscale:
                header["cpus_per_task"] = 64
            run = json.loads(run_file.read_text())
            run["header"] = header
            run_file.write_text(json.dumps(run))
            script = directory / "job.slurm"
            script.write_text("#!/bin/bash\nmodule load configured/environment\n" + get("--command") + "\n")
            result = {"run_id": run_id, "local_script": str(script), "remote_script": "/fake/job.slurm"}
        elif command == "submit-job":
            if self.submit_mode != "lost_without_receipt":
                run = json.loads(run_file.read_text())
                run["job_id"] = "12345"
                run_file.write_text(json.dumps(run))
            if self.submit_mode.startswith("lost"):
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            result = {"run_id": run_id, "job_id": "12345", "applied_fixes": [], "submit_validation": {"ok": True}}
        elif command == "status":
            result = {"job_id": "12345", "status": {"state": self.scheduler_state, "elapsed": self.elapsed, "source": "sacct"},
                      "terminal": self.scheduler_state in {"COMPLETED", "FAILED", "CANCELLED"}}
        elif command == "cancel":
            result = {"ok": True, "job_id": "12345"}
        elif command == "download":
            destination = Path(get("--local-path"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.downloads[get("--remote-path")])
            result = {"run_id": run_id, "remote_path": get("--remote-path"), "local_path": str(destination)}
        else:
            raise AssertionError(command)
        return {"ok": True, "result": result}


@pytest.fixture
def setup(tmp_path):
    cli = tmp_path / "cluster_slurm.py"
    cli.write_text("# Fake fixture only; never a real commissioned cluster CLI.\n")
    config = tmp_path / "config.json"
    config.write_text('{"config_version":2,"profiles":{"test-cpu":{}}}')
    doctor = tmp_path / "doctor.json"
    doctor.write_text(json.dumps({"ok": True, "result": {"ok": True, "profiles": [{"profile": "test-cpu", "ok": True}]}}))
    fake = FakeCLI(tmp_path)
    def make(**overrides):
        return ClusterSlurmBackend(tmp_path / "adapter", profile="test-cpu", config_path=config,
            approved_download_root=tmp_path / "results", doctor_receipt=doctor, cli_path=cli,
            command_runner=fake, **overrides)
    source = tmp_path / "packet" / "files" / "model.py"
    source.parent.mkdir(parents=True)
    source.write_text("MODEL = 1\n")
    packet = {"work_key": KEY, "spec": RemoteSpec("test-cpu", cpus=2, runtime_seconds=120).to_dict(), "command": list(COMMAND),
              "files": [{"path": "model.py", "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "bytes": source.stat().st_size}]}
    (source.parent.parent / "packet.json").write_text(json.dumps(packet))
    # An undeclared local file must never be uploaded with the packet.
    (source.parent.parent / "do-not-upload.txt").write_text("private extra")
    return make, fake, source.parent.parent, tmp_path


def submit(setup):
    make, fake, packet, root = setup
    backend = make()
    handle = backend.stage(packet, KEY)
    job = backend.submit(handle, KEY, RemoteSpec("test-cpu", cpus=2, runtime_seconds=120), COMMAND)
    return backend, fake, handle, job, root


def test_stages_exact_inputs_and_never_reinitializes_existing_run(setup):
    make, fake, packet, _ = setup
    backend = make()
    handle = backend.stage(packet, KEY)
    assert make().stage(packet, KEY) == handle
    assert fake.uploaded_names == ["files/model.py", "packet.json"]
    assert sum(command == "init-run" and not dry for command, dry, _, _ in fake.calls) == 1
    assert all(kwargs["timeout"] == 60 for _, _, _, kwargs in fake.calls)


def test_submit_status_accounting_and_cancel_are_nonblocking(setup):
    backend, fake, handle, job, root = submit(setup)
    assert job == "12345"
    assert backend.reconcile(KEY).state == "found"
    status = backend.status(job)
    assert status.state == "completed" and status.runtime_seconds == 65
    record = json.loads((root / "adapter" / "operations" / f"{KEY}.json").read_text())
    assert record["last_status"]["allocated_cpu_seconds"] == 130
    fake.scheduler_state = "RUNNING"
    backend.cancel(job)
    assert backend.status(job).state == "running"
    commands = [item[0] for item in fake.calls]
    assert "run-workload" not in commands and "setup" not in commands and "doctor" not in commands
    assert commands.count("submit-job") == 1
    assert backend.submit(handle, KEY, RemoteSpec("test-cpu", cpus=2, runtime_seconds=120), COMMAND) == job
    with pytest.raises(ClusterSlurmError, match="immutable packet"):
        backend.submit(handle, KEY, RemoteSpec("test-cpu", cpus=2, runtime_seconds=120), ("changed",))
    assert [item[0] for item in fake.calls].count("submit-job") == 1


@pytest.mark.parametrize("mode,expected", [("lost_without_receipt", "unknown"), ("lost_after_receipt", "found")])
def test_lost_submission_is_reconciled_without_resubmit(setup, mode, expected):
    make, fake, packet, _ = setup
    fake.submit_mode = mode
    backend = make()
    handle = backend.stage(packet, KEY)
    with pytest.raises(AmbiguousSubmission):
        backend.submit(handle, KEY, RemoteSpec("test-cpu", cpus=2, runtime_seconds=120), COMMAND)
    restarted = make()
    assert restarted.reconcile(KEY).state == expected
    if expected == "unknown":
        with pytest.raises(AmbiguousSubmission):
            restarted.submit(handle, KEY, RemoteSpec("test-cpu", cpus=2, runtime_seconds=120), COMMAND)
    assert sum(call[0] == "submit-job" for call in fake.calls) == 1


def test_profile_resource_upscale_stops_before_submission(setup):
    make, fake, packet, _ = setup
    fake.upscale = True
    backend = make()
    handle = backend.stage(packet, KEY)
    with pytest.raises(ClusterSlurmError, match="allocation differs"):
        backend.submit(handle, KEY, RemoteSpec("test-cpu", cpus=2, runtime_seconds=120), COMMAND)
    assert not any(call[0] == "submit-job" for call in fake.calls)


def result_downloads(fake, payload=b'{"score":1}', name="calibration-result.json"):
    manifest = {"work_key": KEY, "artifacts": [{"path": name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}]}
    fake.downloads = {"packet/results/result-manifest.json": json.dumps(manifest).encode(), "packet/results/" + name: payload}


def test_fetch_checks_allowlist_hash_and_approved_destination(setup):
    backend, fake, handle, job, root = submit(setup)
    result_downloads(fake)
    destination = root / "results" / KEY
    backend.fetch(handle, job, destination)
    assert (destination / "calibration-result.json").read_bytes() == b'{"score":1}'
    with pytest.raises(ClusterSlurmError, match="approved retrieval scope"):
        backend.fetch(handle, job, root / "outside")
    fake.downloads["packet/results/calibration-result.json"] = b"corruption"
    with pytest.raises(ClusterSlurmError, match="hash or length"):
        backend.fetch(handle, job, destination)


def test_fetch_denies_unapproved_types_before_result_download(setup):
    backend, fake, handle, job, root = submit(setup)
    result_downloads(fake, name="raw-observations.csv")
    with pytest.raises(ClusterSlurmError, match="outside approved retrieval patterns"):
        backend.fetch(handle, job, root / "results" / KEY)
    downloads = [call for call in fake.calls if call[0] == "download"]
    assert len(downloads) == 1  # Only the artifact manifest was retrieved.


def test_changed_cli_or_profile_fails_closed(setup):
    make, fake, packet, root = setup
    backend = make()
    backend.stage(packet, KEY)
    (root / "cluster_slurm.py").write_text("# Changed CLI\n")
    with pytest.raises(ClusterSlurmError, match="changed"):
        backend.reconcile(KEY)
    with pytest.raises(ClusterSlurmError, match="different commissioned"):
        make().stage(packet, KEY)


def test_cli_loader_blocks_transport_retries_repairs_and_resource_overrides(tmp_path):
    # Execute the actual thin loader against a tiny stand-in module. No transport.
    fake_cli = tmp_path / "fake.py"
    fake_cli.write_text('''
class ClusterError(Exception): pass
def is_transient_transport_failure(*args): return True
def run_bounded_sbatch_test_only(profile, header, test_once, apply_header_fix, **kwargs):
    apply_header_fix({"cpus_per_task": 64})
def validate_submit_args(values): return values
def main(arguments):
    assert not is_transient_transport_failure(255, "", "lost")
    for operation in (
        lambda: validate_submit_args(["--cpus-per-task=99"]),
        lambda: run_bounded_sbatch_test_only("p", {}, None, lambda value: None),
    ):
        try: operation()
        except ClusterError: pass
        else: raise AssertionError("Unsafe operation passed")
    assert validate_submit_args(["--parsable"]) == ["--parsable"]
    return 0
''')
    import sys
    result = subprocess.run([sys.executable, "-c", _CLI_LOADER, str(fake_cli)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
