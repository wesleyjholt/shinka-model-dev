import hashlib
import json
from dataclasses import replace

import pytest

from shinka.modeling import CalibrationPipeline, CandidateSpec, EvaluationRequest, Ledger, StageSpec, CalibrationResult
from shinka.modeling.campaign import promote
from shinka.modeling.cli import main
from shinka.modeling.contracts import EvaluationRecord
from shinka.modeling.reports import write_report
from shinka.modeling.shinka_adapter import author_feedback, evaluate_for_shinka


def calibrate(request, artifact_dir):
    # A tempting screen leader is worse under posterior prediction.
    values = {"a": (9., 1.), "b": (8., 3.), "novel": (-2., 7.)}
    final = request.stage.confirmation
    return CalibrationResult(score=values[request.candidate.candidate_id][int(final)],
                             reliable=True, posterior_kind="posterior_samples" if final else "point_estimate",
                             diagnostics={"private_observation": "never disclose to author"})


def request(tmp_path, name="a", stage=None):
    source = tmp_path / f"{name}.json"
    source.write_text(json.dumps({"structure": name}))
    candidate = CandidateSpec(name, hashlib.sha256(source.read_bytes()).hexdigest(), str(source),
                              ("different" if name == "novel" else "similar",))
    return EvaluationRequest(candidate, stage or StageSpec("screen", "map", max_seconds=5),
                             1, "data-v1", "development-v1", "log-score-v1", "evaluator-v1")


def test_shinka_adapter_freezes_archive_protocol_and_excludes_private_evidence(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite", max_evaluations=4, max_seconds=30)
    pipeline = CalibrationPipeline(ledger, tmp_path / "fits")
    req = request(tmp_path)
    kwargs = {"archive_contract_path": tmp_path / "archive-protocol.json"}
    record = evaluate_for_shinka(pipeline, req, calibrate, tmp_path / "shinka", **kwargs)
    assert record.status == "completed", record.error
    assert json.loads((tmp_path / "shinka/correct.json").read_text())["correct"]
    metrics = json.loads((tmp_path / "shinka/metrics.json").read_text())
    assert metrics["combined_score"] == 9
    assert "private_observation" not in json.dumps(metrics)
    with pytest.raises(ValueError, match="protocol changed"):
        evaluate_for_shinka(pipeline, replace(req, split_id="different"), calibrate, tmp_path / "other", **kwargs)
    with pytest.raises(ValueError, match="promotion ledger"):
        evaluate_for_shinka(pipeline, replace(req, stage=StageSpec("confirm", "smc", confirmation=True)),
                            calibrate, tmp_path / "other", **kwargs)
    assert ledger.usage()["evaluation_count"] == 1


def test_promotions_keep_diversity_and_never_rank_cross_fidelity(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite", max_evaluations=5, max_seconds=30)
    pipeline = CalibrationPipeline(ledger, tmp_path / "fits")
    screen = StageSpec("screen", "map", max_seconds=5)
    final = StageSpec("confirm", "smc", {"particles": 100}, max_seconds=5, confirmation=True)
    for name in ("a", "b", "novel"):
        assert pipeline.evaluate(request(tmp_path, name, screen), calibrate).status == "completed"
    promoted = promote(pipeline, calibrate, [screen, final], lock_path=tmp_path / "promote.lock",
                       exploration_fraction=0.5, limit_per_stage=2)
    assert {r.request.candidate.candidate_id for r in promoted} == {"a", "novel"}
    path = write_report(ledger.records(), tmp_path / "report", usage=ledger.usage())
    data = json.loads(path.with_suffix(".json").read_text())
    winners = {leader["confirmation"]: leader["candidate_id"] for leader in data["stage_leaders"]}
    assert winners == {False: "a", True: "novel"}
    assert ledger.usage()["evaluation_count"] == 5


def test_report_does_not_confirm_map_or_unscored_results(tmp_path):
    req = request(tmp_path, stage=StageSpec("confirm", "map", confirmation=True))
    result = CalibrationResult(score=5., reliable=True, posterior_kind="point_estimate")
    record = EvaluationRecord(req.cache_key, req, "completed", "unused", result=result)
    write_report([record], tmp_path / "report")
    assert json.loads((tmp_path / "report/report.json").read_text())["stage_leaders"] == []
    unscored = replace(record, request=replace(req, stage=StageSpec("screen", "map")),
                       result=CalibrationResult(score=None, reliable=False))
    write_report([unscored], tmp_path / "unscored")
    assert author_feedback(unscored)["public"]["calibration_reliable"] is False


def test_cli_reopens_persisted_budget_and_reports(tmp_path, capsys):
    path = tmp_path / "ledger.sqlite"
    Ledger(path, max_evaluations=8, max_seconds=120)
    assert main(["status", "--ledger", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["max_evaluations"] == 8
    assert main(["report", "--ledger", str(path), "--output", str(tmp_path / "report")]) == 0
    assert (tmp_path / "report/report.md").exists()


def test_full_intermediate_quota_still_promotes_existing_fit_downstream(tmp_path):
    screen = StageSpec("screen", "map", max_seconds=3)
    refine = StageSpec("refine", "mcmc", max_seconds=3)
    final = StageSpec("confirm", "smc", max_seconds=3, confirmation=True)
    ledger = Ledger(tmp_path / "ledger.sqlite", stage_limits={
        "screen": {"max_evaluations": 2, "max_seconds": 20},
        "refine": {"max_evaluations": 1, "max_seconds": 20},
        "confirm": {"max_evaluations": 1, "max_seconds": 20},
    })
    pipeline = CalibrationPipeline(ledger, tmp_path / "fits")
    for name in ("a", "b"):
        pipeline.evaluate(request(tmp_path, name, screen), calibrate)
    pipeline.evaluate(request(tmp_path, "a", refine), calibrate)
    result = promote(pipeline, calibrate, [screen, refine, final], lock_path=tmp_path / "lock")
    assert len(result) == 1
    assert result[0].request.stage.confirmation


def test_mixed_comparison_contract_rejected_before_promotion(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite", max_evaluations=8, max_seconds=60)
    pipeline = CalibrationPipeline(ledger, tmp_path / "fits")
    req = request(tmp_path)
    pipeline.evaluate(req, calibrate)
    pipeline.evaluate(replace(req, split_id="another-split"), calibrate)
    with pytest.raises(ValueError, match="one dataset"):
        promote(pipeline, calibrate, [req.stage, StageSpec("confirm", "smc", confirmation=True)],
                lock_path=tmp_path / "lock")
    assert ledger.usage()["evaluation_count"] == 2


def test_required_constraints_survive_restart_and_block_only_one_candidate(tmp_path):
    from shinka.modeling.constraints import ConstraintBook, ConstraintPending
    book = ConstraintBook(tmp_path / "constraints")
    book.initialize("Keep the defined units consistent.")
    ledger = Ledger(tmp_path / "ledger.sqlite", max_evaluations=4, max_seconds=30)
    pipeline = CalibrationPipeline(ledger, tmp_path / "fits", constraints=book)
    req = request(tmp_path)
    assert evaluate_for_shinka(pipeline, req, calibrate, tmp_path / "waiting",
                               archive_contract_path=tmp_path / "protocol.json") is None
    assert ledger.usage()["evaluation_count"] == 0
    assert json.loads((tmp_path / "waiting/metrics.json").read_text())["public"]["calibration_status"] == "pending_constraints"
    other = request(tmp_path, "b")
    book.review_compliance(other.candidate.source_sha, True, "Units checked by the trusted model validator", "validator")
    assert pipeline.evaluate(other, calibrate).status == "completed"
    reopened = CalibrationPipeline(Ledger(ledger.path), tmp_path / "fits")
    with pytest.raises(ConstraintPending):
        reopened.evaluate(req, calibrate)
    permission = book.request_exception(req.candidate.source_sha, "Scientist requested a unitless comparison model")
    book.approve(permission["id"], actor="scientist")
    assert reopened.evaluate(req, calibrate).status == "completed"
    assert ledger.usage()["evaluation_count"] == 2
    change = book.request_change("Keep the defined units and conserve mass.", "Add the scientist's new requirement")
    book.approve(change["id"], actor="scientist")
    with pytest.raises(ConstraintPending):
        reopened.evaluate(other, calibrate)
    import shutil
    shutil.rmtree(book.root)
    with pytest.raises(ValueError, match="missing or uninitialized"):
        reopened.evaluate(other, calibrate)
