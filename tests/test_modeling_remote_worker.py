import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from shinka.modeling import CandidateSpec, StageSpec, EvaluationRequest, CalibrationResult
from shinka.modeling.remote_worker import run, file_identity
from shinka.modeling.remote import RemoteSpec


def calibration(request, artifact_dir):
    path = Path(artifact_dir) / "draws.json"
    path.write_text('[1, 2, 3]\n')
    return CalibrationResult(score=-1.2, reliable=True, posterior_kind="posterior_samples",
                             artifacts={"draws": str(path)})


def probe(request, artifact_dir):
    return {"simulator_calls": 4, "value": 1.5}


def escaping_artifact(request, artifact_dir):
    return CalibrationResult(score=0, reliable=True, artifacts={"outside": request.candidate.source_path})


def slow_probe(request, artifact_dir):
    time.sleep(10)
    return {"completed": True}


def descendant_calibration(request, artifact_dir):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    return CalibrationResult(score=1, posterior_kind="posterior_samples", diagnostics={"child_pid": child.pid})


def packet(tmp_path, task_kind="calibration"):
    root = tmp_path / "packet"
    (root / "files").mkdir(parents=True)
    source = root / "files/candidate.json"
    source.write_text('{"hypothesis": "Synthetic model"}\n')
    req = EvaluationRequest(CandidateSpec("synthetic", hashlib.sha256(source.read_bytes()).hexdigest(), "/different-host/candidate.json"),
                            StageSpec("refine", "mcmc"), 3, "data-sha", "development", "log-score", "trusted-evaluator-sha")
    value = {"schema_version": 1, "work_key": "work-sha", "request": req.to_dict(),
             "request_key": req.cache_key, "protocol_key": req.protocol_key,
             "source_sha": req.candidate.source_sha, "task_kind": task_kind,
             "parent_evaluation_key": "parent-sha" if task_kind != "calibration" else None,
             "spec": RemoteSpec("synthetic-profile", runtime_seconds=90).to_dict(),
             "files": [file_identity(source, "candidate.json")], "candidate_source": "candidate.json"}
    path = root / "packet.json"
    path.write_text(json.dumps(value))
    return path


def test_remote_worker_verifies_packet_and_writes_relative_hashed_outputs(tmp_path):
    path = packet(tmp_path)
    result = run(path, tmp_path / "output", f"{__name__}:calibration")
    assert result["result_path"] == "calibration-result.json"
    fitted = json.loads((tmp_path / "output/calibration-result.json").read_text())
    assert fitted["artifacts"]["draws"] == "draws.json"
    for item in result["artifacts"]:
        assert file_identity(tmp_path / "output" / item["path"], item["path"]) == item


def test_modified_input_rejected_before_running(tmp_path):
    path = packet(tmp_path)
    (path.parent / "files/candidate.json").write_text("changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        run(path, tmp_path / "output", f"{__name__}:calibration")
    assert not (tmp_path / "output").exists()


def test_intermediate_probe_has_no_rankable_calibration_result(tmp_path):
    manifest = run(packet(tmp_path, "likelihood-probe"), tmp_path / "output", f"{__name__}:probe")
    assert manifest["result_path"] is None
    assert manifest["parent_evaluation_key"] == "parent-sha"
    assert (tmp_path / "output/intermediate-result.json").exists()


def test_worker_rejects_artifacts_outside_result_directory(tmp_path):
    with pytest.raises(ValueError, match="within this result directory"):
        run(packet(tmp_path), tmp_path / "output", f"{__name__}:escaping_artifact")


def test_worker_enforces_intermediate_timeout_and_does_not_publish_manifest(tmp_path):
    path = packet(tmp_path, "probe")
    value = json.loads(path.read_text())
    value["request"]["stage"]["max_seconds"] = .3
    request = EvaluationRequest.from_dict(value["request"])
    value.update(request_key=request.cache_key, protocol_key=request.protocol_key)
    path.write_text(json.dumps(value))
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="TimeoutError"):
        run(path, tmp_path / "output", f"{__name__}:slow_probe")
    assert time.monotonic() - started < 4
    assert not (tmp_path / "output/result-manifest.json").exists()


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups")
def test_remote_success_cleans_up_solver_descendants(tmp_path):
    import psutil

    run(packet(tmp_path), tmp_path / "output", f"{__name__}:descendant_calibration")
    result = json.loads((tmp_path / "output/calibration-result.json").read_text())
    identity = result["diagnostics"]["child_pid"]
    for _ in range(20):
        try:
            if psutil.Process(identity).status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(.025)
    pytest.fail("remote callback left a solver descendant running")


def test_worker_rejects_stage_larger_than_scheduler_allocation(tmp_path):
    path = packet(tmp_path)
    value = json.loads(path.read_text())
    value["spec"]["runtime_seconds"] = 5
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="exceeds the remote"):
        run(path, tmp_path / "output", f"{__name__}:calibration")
