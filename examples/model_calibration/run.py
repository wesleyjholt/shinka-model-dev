"""Run a small staged-calibration campaign without any language-model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from shinka.modeling import (
    CalibrationPipeline, CandidateSpec, EvaluationRequest, Ledger,
    PromotionPolicy, StageSpec, confirmed_records,
)
from shinka.modeling.reports import write_report

from .calibrate import (
    DATASET_ID, EVALUATOR_ID, OBJECTIVE_ID, calibrate, observations,
    predictive_score,
)


def candidates() -> list[CandidateSpec]:
    directory = Path(__file__).parent / "candidates"
    result = []
    for name in ("constant", "linear", "quadratic", "cubic"):
        source = directory / f"{name}.json"
        result.append(CandidateSpec(
            candidate_id=name,
            source_sha=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_path=str(source.resolve()),
            descriptor=(f"polynomial_degree_{json.loads(source.read_text())['degree']}",),
        ))
    return result


def stages():
    return (
        StageSpec("screen", "map", {}, max_seconds=15),
        StageSpec("refine", "mcmc", {"chains": 4, "warmup": 300, "draws": 700}, max_seconds=30),
        StageSpec("confirm", "smc", {"particles": 1024, "temperatures": 32, "rejuvenation_steps": 4}, max_seconds=30, confirmation=True),
    )


def _plot_final(record, output: Path):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prediction = np.genfromtxt(record.result.artifacts["predictions"], delimiter=",", names=True)
    fig, axis = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    x = prediction["x"]
    axis.fill_between(x, prediction["predictive_q05"], prediction["predictive_q95"], color="#92b8d0", alpha=0.45, label="90% predictive interval")
    axis.fill_between(x, prediction["predictive_q25"], prediction["predictive_q75"], color="#347aa4", alpha=0.55, label="50% predictive interval")
    train_x, train_y = observations("train")
    dev_x, dev_y = observations("development")
    axis.scatter(train_x, train_y, s=16, color="#343434", label="Training")
    axis.scatter(dev_x, dev_y, s=20, facecolors="none", edgecolors="#be783c", label="Development")
    axis.set(xlabel="Synthetic input x (dimensionless, linear scale)", ylabel="Synthetic response y (dimensionless, linear scale)", title=f"Selected {record.request.candidate.candidate_id} model — SMC")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(output / "selected_predictions.png", dpi=160)
    (output / "selected_predictions.caption.md").write_text(
        "Synthetic Gaussian polynomial data: 48 training observations and 40 development observations. "
        "Both axes use linear scales and dimensionless units. Bands contain the central 50% and 90% "
        "of posterior predictive draws, including known observation noise (SD 0.25). "
        "The structure was selected using the development set; final-test observations are not plotted. "
        "Source: examples/model_calibration/calibrate.py, dataset synthetic-quadratic-v1-seed-2026.\n"
    )
    plt.close(fig)


def run(output: str | Path, make_plot: bool = True) -> dict:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(output / "calibration.sqlite", max_evaluations=9, max_seconds=240, stage_limits={
        "screen": {"max_evaluations": 4, "max_seconds": 60},
        "refine": {"max_evaluations": 3, "max_seconds": 90},
        "confirm": {"max_evaluations": 2, "max_seconds": 60},
    })
    pipeline = CalibrationPipeline(ledger, output / "artifacts")
    policy = PromotionPolicy(exploration_fraction=0.34)
    screen, refine, confirm = stages()
    candidate_list = candidates()

    def evaluate(candidate, stage):
        return pipeline.evaluate(EvaluationRequest(
            candidate=candidate, stage=stage, seed=812,
            dataset_id=DATASET_ID, split_id="development",
            objective_id=OBJECTIVE_ID, evaluator_id=EVALUATOR_ID,
            context={"training_split": "train", "author_data": "synthetic public example"},
        ), calibrate)

    for candidate in candidate_list:
        evaluate(candidate, screen)
    existing = ledger.records()
    remaining_refinements = max(0, 3 - sum(record.request.stage.name == refine.name for record in existing))
    for record in policy.select(existing, source_stage=screen.name, target_stage=refine.name, limit=remaining_refinements, seed=81):
        evaluate(record.request.candidate, refine)
    existing = ledger.records()
    remaining_confirmations = max(0, 2 - sum(record.request.stage.name == confirm.name for record in existing))
    for record in policy.select(existing, source_stage=refine.name, target_stage=confirm.name, limit=remaining_confirmations, seed=82):
        evaluate(record.request.candidate, confirm)
    records = ledger.records()
    return finalize(records, output, make_plot=make_plot, usage=ledger.usage())


def finalize(records, output: Path, *, make_plot: bool = True, usage: dict | None = None):
    """Freeze a completed search, then evaluate its selected model once."""
    confirmed = confirmed_records(records)
    if not confirmed:
        write_report(records, output, usage=usage)
        raise RuntimeError("No reliable confirmation. Inspect report.md before increasing the inference budget.")
    if len({record.request.protocol_key for record in confirmed}) != 1:
        raise RuntimeError("Confirmation results have different protocols and cannot be ranked together")
    # The final test is never part of parent selection, promotion, or choosing
    # the winner. Only identical confirmation protocols can be ranked here.
    selected = max(confirmed, key=lambda record: record.result.score)
    selection = {
        "candidate_id": selected.request.candidate.candidate_id,
        "source_sha": selected.request.candidate.source_sha,
        "selection_split": "development",
        "confirmation_stage": selected.request.stage.name,
        "confirmation_method": selected.request.stage.method,
        "confirmation_budget": dict(selected.request.stage.budget),
        "development_log_predictive_density": selected.result.score,
    }
    selection_path = output / "frozen_selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise RuntimeError("The frozen selection changed. Use a fresh output directory for a new campaign.")
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    final_path = output / "final_test.json"
    if final_path.exists():
        final = json.loads(final_path.read_text())
        if final["candidate_id"] != selection["candidate_id"] or final["source_sha"] != selection["source_sha"]:
            raise RuntimeError("Final-test artifact does not match the frozen selection")
    else:
        fitted = np.load(selected.result.artifacts["samples"])
        final = {
            "candidate_id": selection["candidate_id"],
            "source_sha": selection["source_sha"],
            "split_id": "final_test",
            "mean_log_predictive_density": predictive_score(fitted["coefficients"], int(fitted["degree"]), "final_test"),
            "use": "One report after structure selection. Never used for ranking or promotion. No refit.",
        }
        final_path.write_text(json.dumps(final, indent=2) + "\n")
    report_path = write_report(records, output, usage=usage)
    if make_plot:
        _plot_final(selected, output)
    summary = {
        "selected": selection,
        "final_test": final,
        "evaluation_count": len(records),
        "records_per_stage": {stage.name: sum(record.request.stage.name == stage.name for record in records) for stage in stages()},
        "report": str(report_path),
        "paid_api_calls": 0,
        "scope": "Synthetic demonstration. This is not a resumed brain-model research campaign.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Persistent output directory; reruns reuse exact cached requests")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, make_plot=not args.no_plot), indent=2))


if __name__ == "__main__":
    main()
