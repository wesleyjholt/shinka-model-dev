"""The demonstration must calibrate the known posterior and keep test data out."""
import numpy as np

from examples.model_calibration.calibrate import (
    exact_posterior, mcmc, predictive_score, reference_diagnostics, smc,
    systematic_resample,
)


def test_short_mcmc_matches_known_posterior():
    samples, diagnostic = mcmc(2, {"chains": 4, "warmup": 300, "draws": 700}, np.random.default_rng(812))
    comparison = reference_diagnostics(samples, 2)
    assert diagnostic["split_rhat_max"] < 1.1
    assert diagnostic["autocorrelation_ess_min"] >= 80
    assert comparison["analytic_mean_error_in_posterior_sd_max"] < 0.3
    assert comparison["analytic_covariance_relative_error"] < 0.25


def test_smc_matches_known_posterior():
    samples, diagnostic = smc(3, {"particles": 1024, "temperatures": 32, "rejuvenation_steps": 4}, np.random.default_rng(812))
    comparison = reference_diagnostics(samples, 3)
    assert diagnostic["resampling_events"] > 1
    assert 0 < diagnostic["rejuvenation_acceptance"] < 1
    assert comparison["analytic_mean_error_in_posterior_sd_max"] < 0.2
    assert comparison["analytic_covariance_relative_error"] < 0.2


def test_predictive_score_integrates_parameter_uncertainty():
    mean, covariance = exact_posterior(2)
    draws = np.random.default_rng(0).multivariate_normal(mean, covariance, size=4000)
    score = predictive_score(draws, 2)
    assert np.isfinite(score)
    assert abs(score - predictive_score(mean[None, :], 2)) > 1e-5


def test_resampling_handles_extreme_weights():
    indices = systematic_resample(np.array([0.0, 0.0, 1.0]), np.random.default_rng(1))
    np.testing.assert_array_equal(indices, np.array([2, 2, 2]))


def test_staged_campaign_replays_without_extra_evaluations(tmp_path, monkeypatch):
    from examples.model_calibration import run as demo

    original_score = demo.predictive_score
    final_reads = []

    def check_frozen_selection(samples, degree, split):
        assert split == "final_test"
        assert (tmp_path / "frozen_selection.json").exists()
        final_reads.append(split)
        return original_score(samples, degree, split)

    monkeypatch.setattr(demo, "predictive_score", check_frozen_selection)
    summary = demo.run(tmp_path, make_plot=False)
    assert final_reads == ["final_test"]
    assert summary["records_per_stage"] == {"screen": 4, "refine": 3, "confirm": 2}
    assert summary["evaluation_count"] == 9
    assert summary["selected"]["candidate_id"] in {"quadratic", "cubic"}
    assert summary["selected"]["selection_split"] == "development"
    assert summary["final_test"]["split_id"] == "final_test"
    assert (tmp_path / "frozen_selection.json").exists()
    assert (tmp_path / "report.md").exists()
    before = (tmp_path / "final_test.json").read_bytes()
    # A replay must use cached fit artifacts, and must not look at final data again.
    monkeypatch.setattr(demo, "predictive_score", lambda *a, **k: (_ for _ in ()).throw(AssertionError("test data read twice")))
    replay = demo.run(tmp_path, make_plot=False)
    assert replay == summary
    assert (tmp_path / "final_test.json").read_bytes() == before


def test_shinka_evaluator_never_executes_candidate(tmp_path):
    import json
    from examples.model_calibration.evaluate import evaluate

    marker = tmp_path / "must-not-exist"
    candidate = tmp_path / "candidate.py"
    candidate.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nMODEL_SPEC = {{'degree': 2, 'hypothesis': 'Curvature may help.'}}\n")
    result = evaluate(candidate, tmp_path / "metrics", tmp_path / "ledger")
    assert not marker.exists()
    assert result.status == "completed"
    assert result.result.posterior_kind == "point_estimate"
    assert json.loads((tmp_path / "metrics" / "correct.json").read_text())["correct"]
    metrics = json.loads((tmp_path / "metrics" / "metrics.json").read_text())
    assert metrics["public"]["method"] == "map"
    assert "samples" not in str(metrics)


def test_model_spec_rejects_executable_values(tmp_path):
    import pytest
    from examples.model_calibration.model_spec import read_model_spec

    candidate = tmp_path / "candidate.py"
    candidate.write_text("MODEL_SPEC = dict(degree=2, hypothesis='Computed expression')\n")
    with pytest.raises(ValueError):
        read_model_spec(candidate)


def test_live_promotion_waits_for_batch_and_keeps_final_test_closed(tmp_path):
    import pytest
    from examples.model_calibration.evaluate import evaluate
    from examples.model_calibration.promote import run as promote

    root = tmp_path / "ledger"
    for degree in (1, 2):
        candidate = tmp_path / f"candidate-{degree}.py"
        candidate.write_text(f"MODEL_SPEC = {{'degree': {degree}, 'hypothesis': 'Test this structure.'}}\n")
        evaluate(candidate, tmp_path / f"metrics-{degree}", root)
    early = promote(root, make_plot=False)
    assert early["status"] == "waiting_for_screening_batch"
    # An explicit smaller batch supports a small test or intentionally shorter run.
    result = promote(root, minimum_screens=2, make_plot=False)
    assert result["status"] == "promotion_complete"
    assert not (root / "final_test.json").exists()
    finished = promote(root, minimum_screens=2, finalize=True, make_plot=False)
    assert finished["records_per_stage"] == {"screen": 2, "refine": 2, "confirm": 2}
    assert (root / "final_test.json").exists()
    with pytest.raises(RuntimeError, match="finalized"):
        evaluate(candidate, tmp_path / "later-metrics", root)


def test_pending_constraint_review_does_not_stop_other_promotions(tmp_path):
    from examples.model_calibration.evaluate import evaluate, open_ledger
    from examples.model_calibration.promote import run as promote
    from shinka.modeling import CalibrationPipeline
    from shinka.modeling.constraints import ConstraintBook

    root = tmp_path / "campaign"
    for degree in (1, 2):
        candidate = tmp_path / f"candidate-{degree}.py"
        candidate.write_text(f"MODEL_SPEC = {{'degree': {degree}, 'hypothesis': 'Synthetic check'}}\n")
        evaluate(candidate, tmp_path / f"metrics-{degree}", root)
    ledger = open_ledger(root)
    records = ledger.records()
    book = ConstraintBook(tmp_path / "constraints")
    book.initialize("Use the scientist-reviewed observation units.")
    permitted = records[1].request.candidate.source_sha
    book.review_compliance(permitted, True, "Observation units checked", "trusted scientist")
    CalibrationPipeline(ledger, root / "artifacts", constraints=book)
    result = promote(root, minimum_screens=2, make_plot=False)
    assert result["status"] == "promotion_complete"
    promoted = [r for r in ledger.records() if r.request.stage.name != "screen"]
    assert len(promoted) == 2
    assert all(r.request.candidate.source_sha == permitted for r in promoted)
    assert any(event["kind"] == "review_requested" for event in book.events())
