"""Shinka proposals remain pending until remote evidence is verified."""
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.launch import LocalJobConfig
from shinka.modeling.constraints import ConstraintPending
from shinka.modeling.contracts import CandidateSpec, StageSpec, EvaluationRequest, CalibrationResult
from shinka.modeling.pipeline import EvaluationInProgress
from shinka.modeling.remote import RemoteSpec
from shinka.modeling.remote_scheduler import RemoteCalibrationScheduler
from shinka.modeling.store import Ledger


class FakePipeline:
    """Deterministic stand-in; no cluster, transport, or author model is called."""

    def __init__(self, root):
        self.ledger = Ledger(root / "ledger.sqlite", max_evaluations=10, max_seconds=1000)
        self.controller = SimpleNamespace(root=root / "remote")
        self.jobs = {}
        self.submit_count = 0
        self.allowed = True
        self.next_state = "running"
        self.cancel_confirmed = False

    def job_for(self, key):
        return self.jobs.get(key)

    def submit(self, request, spec, files, command):
        if not self.allowed:
            raise ConstraintPending("Operator review pending")
        self.submit_count += 1
        record, claimed = self.ledger.reserve(request, str(self.controller.root / request.cache_key))
        if claimed:
            record = self.ledger.start(request.cache_key)
        self.jobs[request.cache_key] = SimpleNamespace(key="remote-" + request.cache_key,
                                                    status="queued", started_at=None)
        return record

    def poll(self, key):
        record = self.ledger.get(key)
        if record.status in {"completed", "failed"}:
            return record
        job = self.jobs[key]
        job.status = self.next_state
        if job.status == "completed":
            return self.ledger.complete(key, CalibrationResult(
                score=-2.5, reliable=True, posterior_kind="point_estimate",
                parameters={"PRIVATE_PARAMETER": 3.0}, notes=("PRIVATE_REPORT",)), 1.0)
        if job.status == "running":
            job.started_at = 1234.0
        return record

    def cancel(self, key):
        self.jobs[key].status = "cancelled" if self.cancel_confirmed else "cancel_requested"
        if self.cancel_confirmed:
            return self.ledger.fail(key, "Private cancellation detail", 1.0)
        return self.ledger.get(key)


def setup(tmp_path, pipeline=None):
    pipeline = pipeline or FakePipeline(tmp_path)
    source = tmp_path / "candidate.py"
    if not source.exists():
        source.write_text("model = 1\n")

    def request(path):
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return EvaluationRequest(CandidateSpec(sha[:12], sha, path), StageSpec("screen", "map"),
                                 1, "data", "development", "prediction-score", "trusted-evaluator")

    scheduler = RemoteCalibrationScheduler(
        pipeline, request, lambda req: {"candidate.py": req.candidate.source_path},
        RemoteSpec("fake-test-profile", runtime_seconds=60), ("python", "trusted_worker.py"),
        tmp_path / "scheduler", poll_interval=0.001,
    )
    return scheduler, pipeline, source


def test_pending_remote_fit_is_not_exported_as_failed_candidate(tmp_path):
    async def run():
        scheduler, pipeline, source = setup(tmp_path)
        results = tmp_path / "gen1/results"
        key = await scheduler.submit_async_nonblocking(source, results)
        job = SimpleNamespace(job_id=key, evaluation_started_at=None)
        assert await scheduler.batch_check_status_async([job]) == [True]
        assert job.evaluation_started_at == 1234.0
        with pytest.raises(EvaluationInProgress):
            await scheduler.get_job_results_async(key, results)
        pipeline.next_state = "completed"
        assert await scheduler.batch_check_status_async([job]) == [False]
        payload = await scheduler.get_job_results_async(key, results)
        assert payload["correct"] == {"correct": True, "error": None}
        assert payload["metrics"]["combined_score"] == -2.5
        assert "PRIVATE" not in json.dumps(payload)
        assert json.loads((results / "metrics.json").read_text()) == payload["metrics"]
    asyncio.run(run())


def test_unknown_and_unconfirmed_cancellation_remain_live(tmp_path):
    async def run():
        scheduler, pipeline, source = setup(tmp_path)
        key = await scheduler.submit_async_nonblocking(source, tmp_path / "results")
        pipeline.next_state = "unknown"
        job = SimpleNamespace(job_id=key)
        assert await scheduler.batch_check_status_async([job]) == [True]
        assert await scheduler.cancel_job_async(key) is False
        assert await scheduler.batch_check_status_async([job]) == [True]
        pipeline.cancel_confirmed = True
        assert await scheduler.cancel_job_async(key) is True
        assert await scheduler.batch_check_status_async([job]) == [False]
    asyncio.run(run())


def test_restart_reuses_binding_and_shutdown_preserves_cancel_intent(tmp_path):
    scheduler, pipeline, source = setup(tmp_path)
    key = scheduler.submit_async(source, tmp_path / "results")
    restarted, _, _ = setup(tmp_path, pipeline)
    assert restarted.submit_async(source, tmp_path / "results") == key
    assert pipeline.submit_count == 1
    restarted.shutdown()
    state = json.loads((restarted.root / "scheduler-state.json").read_text())
    assert state["jobs"][key]["cancel_requested"]
    assert not state["jobs"][key]["terminal"]
    observer, _, _ = setup(tmp_path, pipeline)
    pipeline.cancel_confirmed = True
    assert observer.check_job_status(SimpleNamespace(job_id=key)) is False
    assert pipeline.submit_count == 1


def test_pending_constraint_review_does_not_submit_or_raise_into_engine(tmp_path):
    scheduler, pipeline, source = setup(tmp_path)
    pipeline.allowed = False
    key = scheduler.submit_async(source, tmp_path / "results")
    assert pipeline.submit_count == 0
    assert scheduler.check_job_status(SimpleNamespace(job_id=key)) is True
    assert scheduler.pipeline.ledger.get(key) is None
    pipeline.allowed = True
    assert scheduler.check_job_status(SimpleNamespace(job_id=key)) is True
    assert pipeline.submit_count == 1


def test_generation_zero_has_normal_shinka_result_shape(tmp_path):
    scheduler, pipeline, source = setup(tmp_path)
    pipeline.next_state = "completed"
    results, elapsed = scheduler.run(source, tmp_path / "gen0/results")
    assert results["correct"]["correct"]
    assert results["metrics"]["combined_score"] == -2.5
    assert elapsed >= 0


def test_archive_protocol_and_confirmation_boundaries(tmp_path):
    scheduler, pipeline, source = setup(tmp_path)
    scheduler.submit_async(source, tmp_path / "results")
    from dataclasses import replace
    original = scheduler.build_request
    scheduler.build_request = lambda path: replace(original(path), stage=StageSpec("confirm", "smc", confirmation=True))
    with pytest.raises(ValueError, match="Confirmation"):
        scheduler.submit_async(source, tmp_path / "other-results")
    scheduler.build_request = lambda path: replace(original(path), stage=StageSpec("refine", "mcmc"))
    with pytest.raises(ValueError, match="protocol changed"):
        scheduler.submit_async(source, tmp_path / "other-results")
    assert pipeline.submit_count == 1


def test_remote_runner_injection_rejects_queue_blind_local_timeout_before_clients():
    with pytest.raises(ValueError, match="time=None"):
        ShinkaEvolveRunner(None, LocalJobConfig(time="00:01:00"), None,
                          scheduler=SimpleNamespace(job_type="remote_calibration"))


def test_result_directory_cannot_be_rebound_to_different_source(tmp_path):
    scheduler, pipeline, source = setup(tmp_path)
    scheduler.submit_async(source, tmp_path / "results")
    source.write_text("model = 2\n")
    with pytest.raises(ValueError, match="already bound"):
        scheduler.submit_async(source, tmp_path / "results")
    assert pipeline.submit_count == 1


def test_cached_result_read_does_not_wait_for_other_jobs_transport_lock(tmp_path):
    async def run():
        scheduler, pipeline, source = setup(tmp_path)
        pipeline.next_state = "completed"
        key = scheduler.submit_async(source, tmp_path / "results")
        scheduler.check_job_status(SimpleNamespace(job_id=key))
        with scheduler._locked():
            result = await asyncio.wait_for(scheduler.get_job_results_async(key, tmp_path / "results"), 0.5)
        assert result["correct"]["correct"]
    asyncio.run(run())


def test_shutdown_race_cannot_start_an_untracked_job(tmp_path):
    scheduler, pipeline, source = setup(tmp_path)
    original = scheduler.build_request

    def closing_request(path):
        request = original(path)
        scheduler.shutdown()
        return request

    scheduler.build_request = closing_request
    with pytest.raises(RuntimeError, match="began shutdown"):
        scheduler.submit_async(source, tmp_path / "results")
    assert pipeline.submit_count == 0
