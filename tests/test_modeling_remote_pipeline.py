from pathlib import Path

import pytest

from test_modeling_remote import setup
from shinka.modeling import Ledger
from shinka.modeling.constraints import ConstraintBook, ConstraintPending
from shinka.modeling.remote import RemoteController, JobStatus
from shinka.modeling.remote_pipeline import RemoteCalibrationPipeline


def test_remote_fit_joins_scientific_ledger_after_verified_completion(tmp_path):
    controller, backend, request, spec, files = setup(tmp_path)
    ledger = Ledger(tmp_path / "fits.sqlite", max_evaluations=3, max_seconds=200)
    pipeline = RemoteCalibrationPipeline(ledger, controller)
    assert pipeline.submit(request, spec, files, ["worker"]).status == "running"
    job = pipeline.job_for(request.cache_key)
    assert ledger.usage()["evaluation_count"] == 1
    backend.jobs[job.job_id]["status"] = JobStatus("completed", runtime_seconds=3)
    record = pipeline.poll(request.cache_key)
    assert record.status == "completed" and record.result.score == 3.2
    assert record.elapsed_seconds == 3
    assert pipeline.submit(request, spec, files, ["worker"]) == record
    assert backend.submit_calls == 1


def test_remote_restart_recovers_missing_pointer_without_duplicate_submission(tmp_path):
    controller, backend, request, spec, files = setup(tmp_path)
    ledger = Ledger(tmp_path / "fits.sqlite", max_evaluations=3, max_seconds=200)
    pipeline = RemoteCalibrationPipeline(ledger, controller)
    pipeline.submit(request, spec, files, ["worker"])
    pipeline._reference_path(request.cache_key).unlink()
    restarted = RemoteCalibrationPipeline(Ledger(ledger.path), RemoteController(controller.root, backend))
    assert restarted.submit(request, spec, files, ["worker"]).status == "running"
    assert backend.submit_calls == 1
    Path(files["candidate.py"]).write_text("changed input")
    with pytest.raises(ValueError, match="inputs changed"):
        restarted.submit(request, spec, files, ["worker"])


def test_remote_constraints_cover_fits_and_intermediate_probes_before_cost(tmp_path):
    controller, backend, request, spec, files = setup(tmp_path)
    book = ConstraintBook(tmp_path / "constraints")
    book.initialize("Follow the declared observation units.")
    ledger = Ledger(tmp_path / "fits.sqlite", max_evaluations=3, max_seconds=200)
    pipeline = RemoteCalibrationPipeline(ledger, controller, constraints=book)
    with pytest.raises(ConstraintPending):
        pipeline.submit(request, spec, files, ["worker"])
    with pytest.raises(ConstraintPending):
        pipeline.submit_probe(request, spec, files, ["worker"], task_kind="sensitivity", parent_evaluation_key=request.cache_key)
    assert controller.usage()["job_count"] == 0 and ledger.usage()["evaluation_count"] == 0
    book.review_compliance(request.candidate.source_sha, True, "Units checked", "trusted evaluator")
    probe = pipeline.submit_probe(request, spec, files, ["worker"], task_kind="sensitivity", parent_evaluation_key=request.cache_key)
    assert probe.packet["task_kind"] == "sensitivity"
    assert controller.usage()["reserved_cpu_seconds"] == 40
    assert ledger.usage()["evaluation_count"] == 0  # no model score for a probe


def test_remote_cancel_does_not_release_scientific_record_until_confirmed(tmp_path):
    controller, backend, request, spec, files = setup(tmp_path)
    ledger = Ledger(tmp_path / "fits.sqlite", max_evaluations=3, max_seconds=200)
    pipeline = RemoteCalibrationPipeline(ledger, controller)
    pipeline.submit(request, spec, files, ["worker"])
    assert pipeline.cancel(request.cache_key).status == "running"
    job = pipeline.job_for(request.cache_key)
    backend.jobs[job.job_id]["status"] = JobStatus("cancelled", runtime_seconds=2)
    assert pipeline.poll(request.cache_key).status == "failed"
    assert ledger.usage()["charged_seconds"] == 2
