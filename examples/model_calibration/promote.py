"""Promote models from a live Shinka ledger after its bounded screening batch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from shinka.modeling import CalibrationPipeline, EvaluationRequest, PromotionPolicy
from shinka.modeling.reports import write_report
from .calibrate import calibrate
from .evaluate import open_ledger
from .run import finalize as finalize_campaign, stages


def run(ledger_root: str | Path, *, minimum_screens: int = 10,
        finalize: bool = False, make_plot: bool = True) -> dict:
    """Default to a full screening batch; an early poll never blocks research.

    ``finalize=True`` must be called only after the Shinka search stops. Without
    it this command writes development evidence and never opens final-test data.
    """
    if not 1 <= minimum_screens <= 10:
        raise ValueError("minimum_screens must be between one and ten")
    root = Path(ledger_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    ledger = open_ledger(root)
    if (root / "frozen_selection.json").exists():
        if finalize:
            return finalize_campaign(ledger.records(), root, make_plot=make_plot, usage=ledger.usage())
        return {"status": "finalized", "report": str(root / "report.md")}
    existing = ledger.records()
    screens = [record for record in existing if record.request.stage.name == "screen"
               and record.status in {"completed", "failed"}]
    if len(screens) < minimum_screens:
        report = write_report(existing, root, usage=ledger.usage())
        return {"status": "waiting_for_screening_batch", "screened": len(screens),
                "minimum_screens": minimum_screens, "report": str(report)}
    pipeline = CalibrationPipeline(ledger, root / "artifacts")
    policy = PromotionPolicy(exploration_fraction=0.25)
    screen, refine, confirm = stages()
    for source, target, capacity, seed in ((screen, refine, 4, 81), (refine, confirm, 2, 82)):
        existing = ledger.records()
        remaining = max(0, capacity - sum(record.request.stage.name == target.name for record in existing))
        for record in policy.select(existing, source_stage=source.name, target_stage=target.name, limit=remaining, seed=seed):
            base = record.request
            request = EvaluationRequest(
                candidate=base.candidate, stage=target, seed=base.seed,
                dataset_id=base.dataset_id, split_id=base.split_id,
                objective_id=base.objective_id, evaluator_id=base.evaluator_id,
                context=base.context,
            )
            pipeline.evaluate(request, calibrate)
    records = ledger.records()
    if finalize:
        return finalize_campaign(records, root, make_plot=make_plot, usage=ledger.usage())
    report = write_report(records, root, usage=ledger.usage())
    return {"status": "promotion_complete", "report": str(report), "usage": ledger.usage(),
            "final_test_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", default=os.environ.get("MODEL_CALIBRATION_RUN"))
    parser.add_argument("--minimum-screens", type=int, default=10)
    parser.add_argument("--finalize", action="store_true", help="After stopping model search, freeze selection and use the final test once")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if not args.ledger_root:
        parser.error("Set MODEL_CALIBRATION_RUN or pass --ledger-root")
    print(json.dumps(run(args.ledger_root, minimum_screens=args.minimum_screens,
                         finalize=args.finalize, make_plot=not args.no_plot), indent=2))


if __name__ == "__main__":
    main()
