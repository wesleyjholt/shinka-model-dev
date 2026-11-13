"""Remote scheduler contract tests; every backend is local and synthetic."""
import hashlib
import json
from pathlib import Path

import pytest

from shinka.modeling import CalibrationResult, CandidateSpec, EvaluationRequest, StageSpec, BudgetExhausted
from shinka.modeling.remote import RemoteController, RemoteSpec, Reconciliation, JobStatus


class FakeScheduler:
    def __init__(self):
        self.jobs = {}
        self.packets = {}
        self.submit_calls = 0
        self.reconcile_calls = 0
        self.cancel_calls = []
        self.lose_response = False
        self.unknown_reconciliation = False
        self.tamper = None
        self.fetch_failures = 0

    def stage(self, packet_dir, key):
        self.packets[key] = json.loads((packet_dir / "packet.json").read_text())
        return key

    def submit(self, handle, key, spec, command):
        self.submit_calls += 1
        identity = str(len(self.jobs) + 1)
        self.jobs[identity] = {"key": key, "status": JobStatus("queued")}
        if self.lose_response:
            raise ConnectionError("response lost after scheduler accepted job")
        return identity

    def reconcile(self, key):
        self.reconcile_calls += 1
        if self.unknown_reconciliation:
            return Reconciliation("unknown")
        for identity, job in self.jobs.items():
            if job["key"] == key:
                return Reconciliation("found", identity)
        return Reconciliation("absent")

    def status(self, job_id):
        return self.jobs[job_id]["status"]

    def cancel(self, job_id):
        self.cancel_calls.append(job_id)

    def fetch(self, handle, job_id, destination):
        if self.fetch_failures:
            self.fetch_failures -= 1
            raise ConnectionError("download interrupted")
        packet = self.packets[handle]
        request = packet["request"]
        name = "result.json" if packet["task_kind"] == "calibration" else "simulation.json"
        value = CalibrationResult(3.2, reliable=True, posterior_kind="point_estimate").to_dict() if packet["task_kind"] == "calibration" else {"state": [1, 2, 3]}
        raw = json.dumps(value).encode()
        (destination / name).write_bytes(raw)
        manifest = {"schema_version": 1,
                    **{key: packet[key] for key in ("work_key", "request_key", "protocol_key", "source_sha", "task_kind", "parent_evaluation_key")},
                    **{key: request[key] for key in ("stage", "dataset_id", "split_id", "objective_id", "evaluator_id")},
                    "artifacts": [{"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}],
                    "result_path": name if packet["task_kind"] == "calibration" else None}
        if self.tamper == "identity":
            manifest["dataset_id"] = "another-dataset"
        if self.tamper == "checksum":
            manifest["artifacts"][0]["sha256"] = "0" * 64
        if self.tamper == "traversal":
            manifest["artifacts"][0]["path"] = "../outside.json"
        (destination / "result-manifest.json").write_text(json.dumps(manifest))
        (destination / "stdout.txt").write_text("combined_score = 1000000000")


def setup(tmp_path, scheduler=None, **kwargs):
    scheduler = scheduler or FakeScheduler()
    source = tmp_path / "candidate.py"
    source.write_text("def model(x): return x\n")
    request = EvaluationRequest(CandidateSpec("linear", hashlib.sha256(source.read_bytes()).hexdigest(), str(source)),
                                StageSpec("map", "MAP", max_seconds=.1), 3, "data-v1", "validation-v1", "log-predictive:v1", "evaluator-v1")
    controller = RemoteController(tmp_path / "remote", scheduler, max_jobs=3, max_inflight=2,
                                  max_cpu_seconds=200, max_gpu_seconds=100, **kwargs)
    spec = RemoteSpec("gautschi-cpu", cpus=2, runtime_seconds=20, queue_timeout_seconds=30)
    return controller, scheduler, request, spec, {"candidate.py": source}


def test_uncertain_submission_reconciles_before_retry_and_restart(tmp_path):
    controller, backend, request, spec, files = setup(tmp_path)
    backend.lose_response = True
    first = controller.submit(request, spec, files, ["python", "worker.py"])
    assert first.status == "uncertain" and backend.submit_calls == 1
    backend.unknown_reconciliation = True
    restarted = RemoteController(controller.root, backend)
    assert restarted.submit(request, spec, files, ["python", "worker.py"]).status == "uncertain"
    assert backend.submit_calls == 1 and backend.reconcile_calls == 1
    assert restarted.usage()["reserved_cpu_seconds"] == 40
    backend.unknown_reconciliation = False
    recovered = restarted.poll(first.key)
    assert recovered.job_id == "1" and backend.submit_calls == 1


def test_queue_time_is_separate_from_compute_and_cancel_stays_reserved(tmp_path):
    now = [1000.0]
    controller, backend, request, spec, files = setup(tmp_path, clock=lambda: now[0])
    job = controller.submit(request, spec, files, ["worker"])
    now[0] += 31
    queued = controller.poll(job.key)
    assert queued.status == "cancel_requested" and backend.cancel_calls == [job.job_id]
    assert controller.usage()["reserved_cpu_seconds"] == 40
    backend.jobs[job.job_id]["status"] = JobStatus("cancelled", runtime_seconds=0)
    cancelled = controller.poll(job.key)
    assert cancelled.status == "cancelled"
    assert controller.usage()["charged_cpu_seconds"] == 0
    assert controller.usage()["job_count"] == 1


def test_completed_result_requires_manifest_and_uses_scheduler_actual_cost(tmp_path):
    controller, backend, request, spec, files = setup(tmp_path)
    job = controller.submit(request, spec, files, ["worker"])
    backend.jobs[job.job_id]["status"] = JobStatus("running", runtime_seconds=5)
    assert controller.poll(job.key).status == "running"  # local .1s timeout is irrelevant
    backend.jobs[job.job_id]["status"] = JobStatus("completed", runtime_seconds=8, allocated_cpus=4)
    result = controller.poll(job.key)
    assert result.result.score == 3.2 and result.status == "completed"
    assert result.charged_cpu_seconds == 32
    assert controller.poll(job.key) == result
    assert backend.submit_calls == 1


@pytest.mark.parametrize("tamper", ["identity", "checksum", "traversal"])
def test_invalid_results_never_become_scores_but_keep_compute_charge(tmp_path, tamper):
    controller, backend, request, spec, files = setup(tmp_path)
    job = controller.submit(request, spec, files, ["worker"])
    backend.tamper = tamper
    backend.jobs[job.job_id]["status"] = JobStatus("completed", runtime_seconds=7)
    result = controller.poll(job.key)
    assert result.status == "invalid_result" and result.result is None
    assert result.charged_cpu_seconds == 14


def test_intermediate_work_has_lineage_artifacts_and_no_score(tmp_path):
    controller, backend, request, spec, files = setup(tmp_path)
    job = controller.submit(request, spec, files, ["worker"], task_kind="sensitivity",
                            parent_evaluation_key=request.cache_key)
    backend.jobs[job.job_id]["status"] = JobStatus("completed", runtime_seconds=2)
    result = controller.poll(job.key)
    assert result.status == "completed" and result.result is None
    assert "simulation.json" in result.output_paths
    assert result.packet["parent_evaluation_key"] == request.cache_key


def test_result_download_can_resume_without_new_submission(tmp_path):
    controller, backend, request, spec, files = setup(tmp_path)
    job = controller.submit(request, spec, files, ["worker"])
    backend.jobs[job.job_id]["status"] = JobStatus("completed", runtime_seconds=2)
    backend.fetch_failures = 1
    assert controller.poll(job.key).status == "fetch_pending"
    restarted = RemoteController(controller.root, backend)
    assert restarted.poll(job.key).status == "completed"
    assert backend.submit_calls == 1 and restarted.usage()["charged_cpu_seconds"] == 4


def test_failed_or_unknown_jobs_do_not_evade_resource_or_job_budgets(tmp_path):
    controller, backend, request, spec, files = setup(tmp_path)
    first = controller.submit(request, spec, files, ["worker"])
    second = controller.submit(request, spec, files, ["other-worker"])
    with pytest.raises(BudgetExhausted, match="in-flight"):
        controller.submit(request, spec, files, ["third-worker"])
    backend.jobs[first.job_id]["status"] = JobStatus("failed", runtime_seconds=None)
    assert controller.poll(first.key).charged_cpu_seconds == 40
    third = controller.submit(request, spec, files, ["third-worker"])
    assert controller.usage()["job_count"] == 3
    with pytest.raises(BudgetExhausted):
        controller.submit(request, spec, files, ["fourth-worker"])
    with pytest.raises(ValueError, match="cannot change"):
        RemoteController(controller.root, backend, max_jobs=100)


def test_unknown_cancellation_intent_survives_reconciliation(tmp_path):
    controller, backend, request, spec, files = setup(tmp_path)
    backend.lose_response = True
    backend.unknown_reconciliation = True
    job = controller.submit(request, spec, files, ["worker"])
    assert controller.cancel(job.key).cancel_intent
    backend.unknown_reconciliation = False
    assert controller.poll(job.key).status == "cancel_requested"
    assert backend.cancel_calls == ["1"]
