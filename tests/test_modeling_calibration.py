"""Scientific orchestration invariants; no real data or expensive calibration."""

import hashlib
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from shinka.modeling import (
    BudgetExhausted,
    CalibrationPipeline,
    CalibrationResult,
    CandidateSpec,
    EvaluationInProgress,
    EvaluationRecord,
    EvaluationRequest,
    Ledger,
    PromotionPolicy,
    StageSpec,
    confirmed_records,
)


def successful_calibration(request, artifact_dir):
    artifact = Path(artifact_dir) / "executions.txt"
    # A second execution with the same key would fail, not silently pass.
    with artifact.open("x") as stream:
        stream.write("one execution")
    return CalibrationResult(
        score=2.5, parameters={"a": 1.2}, priors={"a": "normal(0, 1)"},
        fixed_parameters={"scale": {"value": 1, "reason": "known by construction"}},
        reliable=True, posterior_kind="point_estimate", artifacts={"log": str(artifact)},
    )


def failing_calibration(request, artifact_dir):
    time.sleep(0.05)
    raise RuntimeError("solver failed after doing work")


def slow_calibration(request, artifact_dir):
    time.sleep(10)
    return CalibrationResult(score=1)


def mislabeled_map(request, artifact_dir):
    return CalibrationResult(score=1, reliable=True, posterior_kind="posterior_samples")


def leave_descendant(request, artifact_dir):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    return CalibrationResult(score=1, posterior_kind="point_estimate", diagnostics={"child_pid": child.pid})


def request_for(name="a", stage=None, descriptor=("linear",)):
    return EvaluationRequest(
        CandidateSpec(name, hashlib.sha256(name.encode()).hexdigest(), descriptor=descriptor),
        stage or StageSpec("screen", "MAP", {"iterations": 10}, max_seconds=10),
        seed=17, dataset_id="synthetic-v1", split_id="train-validation-v1",
        objective_id="validation-log-predictive-density:v1:higher-better", evaluator_id="synthetic-evaluator-v1",
    )


def record_for(name, score, stage=None, descriptor=("linear",), reliable=True):
    request = request_for(name, stage, descriptor)
    result = CalibrationResult(
        score=score, reliable=reliable,
        posterior_kind="point_estimate" if request.stage.method.lower() == "map" else "posterior_samples",
    )
    return EvaluationRecord(request.cache_key, request, "completed", "/unused", result=result)


def test_rank_reversal_never_compares_different_fidelities():
    map_stage = StageSpec("screen", "MAP")
    smc_stage = StageSpec("confirm", "SMC", confirmation=True)
    records = [record_for("a", 100, map_stage), record_for("b", 90, map_stage),
               record_for("a", -10, smc_stage), record_for("b", 0, smc_stage)]
    policy = PromotionPolicy(exploration_fraction=0)
    assert policy.select(records, "screen", "mcmc", 1)[0].request.candidate.candidate_id == "a"
    assert policy.select(records, "confirm", "replicate", 1)[0].request.candidate.candidate_id == "b"
    assert {r.request.stage.name for r in confirmed_records(records)} == {"confirm"}


def test_novel_structure_gets_reserved_slot_and_uncertain_fit_can_be_rescued():
    records = [record_for("a", 10), record_for("b", 9),
               record_for("novel", -20, descriptor=("nonlinear", "delay"), reliable=False)]
    selected = PromotionPolicy(.5).select(records, "screen", "mcmc", 2)
    assert [r.request.candidate.candidate_id for r in selected] == ["a", "novel"]


def test_single_slot_uses_seeded_mixture_of_exploration_and_exploitation():
    records = [record_for("best", 10), record_for("other", -20, descriptor=("nonlinear",))]
    policy = PromotionPolicy(.25)
    choices = [policy.select(records, "screen", "mcmc", 1, seed=seed)[0].request.candidate.candidate_id
               for seed in range(400)]
    # With no archive, an exploration choice is uniform over the two models:
    # approximately half of the 25% exploration draws pick the weaker model.
    assert 25 <= choices.count("other") <= 80
    assert choices.count("best") > choices.count("other")
    assert policy.select(records, "screen", "mcmc", 1, seed=17) == policy.select(
        records, "screen", "mcmc", 1, seed=17
    )


def test_same_stage_name_different_budget_or_split_cannot_be_ranked():
    a = record_for("a", 1)
    b = record_for("b", 2, replace(a.request.stage, budget={"iterations": 100}))
    with pytest.raises(ValueError, match="comparable"):
        PromotionPolicy().select([a, b], "screen", "next", 1)
    b = replace(a, request=replace(a.request, split_id="another-split"))
    with pytest.raises(ValueError, match="comparable"):
        PromotionPolicy().select([a, b], "screen", "next", 1)


def test_only_reliable_posterior_confirmations_are_eligible():
    stage = StageSpec("confirm", "SMC", confirmation=True)
    good = record_for("good", 1, stage)
    bad = record_for("bad", 99, stage, reliable=False)
    point = replace(good, result=replace(good.result, posterior_kind="point_estimate"))
    failed = replace(good, status="failed")
    assert confirmed_records([good, bad, point, failed, record_for("screen", 999)]) == [good]


def test_metadata_is_immutable_snapshot_and_protocol_identity_is_complete():
    metadata = {"solver": {"steps": [1, 2]}}
    request = replace(request_for(), context=metadata)
    key = request.cache_key
    metadata["solver"]["steps"].append(3)
    assert request.context["solver"]["steps"] == (1, 2)
    with pytest.raises(TypeError):
        request.context["solver"]["new"] = 1
    restored = EvaluationRequest.from_dict(request.to_dict())
    assert restored == request and restored.cache_key == key
    assert replace(request, seed=18).cache_key != key
    assert replace(request, seed=18).protocol_key == request.protocol_key
    assert replace(request, evaluator_id="changed-code").protocol_key != request.protocol_key
    assert replace(request, objective_id="different-objective").cache_key != key


def test_replay_is_idempotent_and_ledger_survives_restart(tmp_path):
    path = tmp_path / "ledger.sqlite"
    ledger = Ledger(path, max_evaluations=1, max_seconds=20)
    request = request_for()
    pipeline = CalibrationPipeline(ledger, tmp_path / "artifacts")
    first = pipeline.evaluate(request, successful_calibration)
    assert first.status == "completed", first.error
    second = CalibrationPipeline(Ledger(path), tmp_path / "artifacts").evaluate(request, successful_calibration)
    assert second == first
    assert ledger.usage()["evaluation_count"] == 1
    assert ledger.usage()["charged_seconds"] > 0
    assert ledger.usage()["reserved_seconds"] == 0
    assert Ledger(path).usage()["max_evaluations"] == 1
    with pytest.raises(BudgetExhausted):
        pipeline.evaluate(request_for("b"), successful_calibration)
    with pytest.raises(ValueError, match="persisted"):
        Ledger(path, max_evaluations=2)


def test_failed_work_is_charged_cached_and_cannot_evade_count_budget(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite", max_evaluations=1)
    pipeline = CalibrationPipeline(ledger, tmp_path / "artifacts")
    result = pipeline.evaluate(request_for(), failing_calibration)
    assert result.status == "failed" and "solver failed" in result.error
    assert result.charge_seconds >= .05
    assert pipeline.evaluate(request_for(), successful_calibration) == result
    assert ledger.usage()["status_counts"] == {"failed": 1}
    with pytest.raises(BudgetExhausted):
        pipeline.evaluate(request_for("b"), successful_calibration)


def test_timeout_stops_worker_and_charges_attempt(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    pipeline = CalibrationPipeline(ledger, tmp_path / "artifacts")
    request = request_for(stage=StageSpec("bounded", "MCMC", max_seconds=.3))
    start = time.monotonic()
    result = pipeline.evaluate(request, slow_calibration)
    assert time.monotonic() - start < 4
    assert result.status == "failed" and "TimeoutError" in result.error
    assert result.charge_seconds >= .3
    assert ledger.usage()["reserved_seconds"] == 0


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups require POSIX")
def test_successful_worker_cannot_leave_running_descendants(tmp_path):
    import psutil

    result = CalibrationPipeline(Ledger(tmp_path / "ledger.sqlite"), tmp_path / "artifacts").evaluate(
        request_for(), leave_descendant
    )
    assert result.status == "completed", result.error
    child_pid = result.result.diagnostics["child_pid"]
    for _ in range(20):
        try:
            child = psutil.Process(child_pid)
            if child.status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(.025)
    pytest.fail("worker descendant survived stage cleanup")


def test_map_cannot_claim_posterior_samples(tmp_path):
    result = CalibrationPipeline(Ledger(tmp_path / "ledger.sqlite"), tmp_path / "artifacts").evaluate(
        request_for(), mislabeled_map
    )
    assert result.status == "failed"
    assert "MAP is a point estimate" in result.error


def test_source_sha_is_checked_even_for_cache_replays(tmp_path):
    source = tmp_path / "candidate.py"
    source.write_text("x = 1\n")
    candidate = CandidateSpec("a", hashlib.sha256(source.read_bytes()).hexdigest(), str(source))
    request = replace(request_for(), candidate=candidate)
    ledger = Ledger(tmp_path / "ledger.sqlite")
    pipeline = CalibrationPipeline(ledger, tmp_path / "artifacts")
    assert pipeline.evaluate(request, successful_calibration).status == "completed"
    source.write_text("x = 2\n")
    with pytest.raises(ValueError, match="SHA"):
        pipeline.evaluate(request, successful_calibration)
    assert ledger.usage()["evaluation_count"] == 1


def test_atomic_reservation_and_abandoned_work_are_not_free(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite", max_evaluations=1, max_seconds=10)
    request = request_for()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: ledger.reserve(request, str(tmp_path / "artifacts")), range(4)))
    assert sum(claimed for _, claimed in results) == 1
    assert ledger.usage()["reserved_seconds"] == 10
    with pytest.raises(EvaluationInProgress):
        CalibrationPipeline(ledger, tmp_path / "artifacts").evaluate(request, successful_calibration)
    abandoned = ledger.fail_abandoned(request.cache_key, "old worker confirmed stopped")
    assert abandoned.charge_seconds == 10
    assert ledger.usage()["charged_seconds"] == 10


def test_reservation_accounts_for_all_inflight_work(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite", max_seconds=15)
    ledger.reserve(request_for("a"), str(tmp_path / "a"))
    with pytest.raises(BudgetExhausted, match="wall-time"):
        ledger.reserve(request_for("b"), str(tmp_path / "b"))


def test_stage_quotas_preserve_expensive_confirmation_allocation(tmp_path):
    limits = {"screen": {"max_evaluations": 2, "max_seconds": 20},
              "confirm": {"max_evaluations": 1, "max_seconds": 30}}
    ledger = Ledger(tmp_path / "ledger.sqlite", stage_limits=limits)
    for name in ("a", "b"):
        request = request_for(name)
        ledger.reserve(request, str(tmp_path / name))
        ledger.start(request.cache_key)
        ledger.fail(request.cache_key, "used full stage allocation", 10)
    with pytest.raises(BudgetExhausted, match="screen stage evaluation"):
        ledger.reserve(request_for("c"), str(tmp_path / "c"))
    confirmation = request_for("a", StageSpec("confirm", "SMC", max_seconds=30, confirmation=True))
    assert ledger.reserve(confirmation, str(tmp_path / "confirm"))[1]
    usage = Ledger(tmp_path / "ledger.sqlite").usage()
    assert usage["max_evaluations"] == 3 and usage["max_seconds"] == 50
    assert usage["stage_limits"] == limits
    assert usage["stage_usage"]["screen"]["charged_seconds"] == 20
    assert usage["stage_usage"]["confirm"]["reserved_seconds"] == 30
    changed = {**limits, "screen": {"max_evaluations": 3, "max_seconds": 20}}
    with pytest.raises(ValueError, match="persisted"):
        Ledger(tmp_path / "ledger.sqlite", stage_limits=changed)


def test_stage_quota_requires_enough_global_budget_and_rejects_unlisted_stage(tmp_path):
    limits = {"screen": {"max_evaluations": 1, "max_seconds": 20}}
    with pytest.raises(ValueError, match="cover all reserved"):
        Ledger(tmp_path / "ledger.sqlite", max_seconds=10, stage_limits=limits)
    ledger = Ledger(tmp_path / "ledger.sqlite", stage_limits=limits)
    request = request_for(stage=StageSpec("unplanned", "SMC", max_seconds=10))
    with pytest.raises(BudgetExhausted, match="not in"):
        ledger.reserve(request, str(tmp_path / "unplanned"))


def test_omitted_global_budget_on_restart_adopts_persisted_total_with_same_quotas(tmp_path):
    limits = {"screen": {"max_evaluations": 1, "max_seconds": 20}}
    path = tmp_path / "ledger.sqlite"
    Ledger(path, max_evaluations=2, max_seconds=30, stage_limits=limits)
    usage = Ledger(path, stage_limits=limits).usage()
    assert usage["max_evaluations"] == 2 and usage["max_seconds"] == 30
