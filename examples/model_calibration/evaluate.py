"""Trusted Shinka evaluator for the synthetic polynomial example.

Accepts standard --program_path and --results_dir arguments. The candidate is
parsed as data, not imported. MODEL_CALIBRATION_RUN identifies the shared ledger
used across Shinka generations; it must be outside author-editable workspaces.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

# Support the upstream convention of launching the evaluator as a script.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shinka.modeling import CalibrationPipeline, CandidateSpec, EvaluationRequest, Ledger, StageSpec
from shinka.modeling.shinka_adapter import evaluate_for_shinka
from shinka.modeling.constraints import ConstraintBook
from examples.model_calibration.calibrate import DATASET_ID, EVALUATOR_ID, OBJECTIVE_ID, calibrate
from examples.model_calibration.model_spec import read_model_spec


def open_ledger(root: Path):
    return Ledger(root / "calibration.sqlite", max_evaluations=16, max_seconds=330, stage_limits={
        "screen": {"max_evaluations": 10, "max_seconds": 150},
        "refine": {"max_evaluations": 4, "max_seconds": 120},
        "confirm": {"max_evaluations": 2, "max_seconds": 60},
    })


def evaluate(program_path: str | Path, results_dir: str | Path, ledger_root: str | Path):
    root = Path(ledger_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "frozen_selection.json").exists():
        raise RuntimeError("This campaign was finalized. Start a new ledger before more model search.")
    payload = Path(program_path).read_bytes()
    if len(payload) > 256_000:
        raise ValueError("Candidate specification exceeds 256 KB")
    digest = hashlib.sha256(payload).hexdigest()
    source_directory = root / "candidate_sources"
    source_directory.mkdir(exist_ok=True)
    source = source_directory / f"{digest}.py"
    try:
        with source.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if source.read_bytes() != payload:
            raise ValueError("Content-addressed source snapshot was changed")
    spec = read_model_spec(source)
    candidate = CandidateSpec(
        candidate_id=f"polynomial-{digest[:12]}", source_sha=digest,
        source_path=str(source), descriptor=("polynomial", f"degree={spec['degree']}"),
    )
    request = EvaluationRequest(
        candidate=candidate,
        stage=StageSpec("screen", "map", {}, max_seconds=15),
        seed=812, dataset_id=DATASET_ID, split_id="development",
        objective_id=OBJECTIVE_ID, evaluator_id=EVALUATOR_ID,
        context={"training_split": "train", "author_data": "synthetic public example"},
    )
    constraint_root = os.environ.get("MODEL_CALIBRATION_CONSTRAINTS")
    pipeline = CalibrationPipeline(open_ledger(root), root / "artifacts",
        constraints=ConstraintBook(constraint_root) if constraint_root else None)
    return evaluate_for_shinka(
        pipeline, request, calibrate, results_dir,
        archive_contract_path=root / "shinka_archive_protocol.json",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--ledger_root", default=os.environ.get("MODEL_CALIBRATION_RUN"))
    args = parser.parse_args()
    if not args.ledger_root:
        parser.error("Set MODEL_CALIBRATION_RUN or pass --ledger_root for a shared bounded ledger")
    evaluate(args.program_path, args.results_dir, args.ledger_root)


if __name__ == "__main__":
    main()
