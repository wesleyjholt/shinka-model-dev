"""Expose one fixed development protocol to Shinka's evolutionary archive."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from .contracts import EvaluationRequest, EvaluationRecord
from .pipeline import CalibrationPipeline
from .constraints import ConstraintPending


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _bind_archive_protocol(path: Path, request: EvaluationRequest) -> None:
    """An archive must never silently mix calibration stages or objectives."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["protocol_key"] != request.protocol_key:
                raise ValueError("Shinka archive protocol changed; use a separate archive for this calibration protocol")
        else:
            _write_json(path, {"protocol_key": request.protocol_key,
                               "stage": request.stage.to_dict(),
                               "score_direction": "maximize"})


def author_feedback(record: EvaluationRecord) -> dict:
    """Small allowlisted development feedback; no samples, data, or raw errors.

    The campaign owner authorizes disclosure of the development score by using
    this adapter. Final test outcomes and confirmation artifacts stay separate.
    """
    if record.request.stage.confirmation:
        raise ValueError("Confirmation results must not enter the development archive")
    result = record.result
    usable = bool(record.status == "completed" and result and result.reliable
                  and result.score is not None and math.isfinite(result.score))
    return {
        "combined_score": result.score if usable else 0.0,
        "public": {"stage": record.request.stage.name, "method": record.request.stage.method,
                   "calibration_status": record.status, "calibration_reliable": usable,
                   "posterior_kind": result.posterior_kind if result else "unavailable",
                   "evidence_key": record.cache_key},
    }


def evaluate_for_shinka(
    pipeline: CalibrationPipeline, request: EvaluationRequest, calibrate,
    results_dir: str | Path, *, archive_contract_path: str | Path,
) -> EvaluationRecord | None:
    """Write standard metrics.json/correct.json from a trusted calibrator.

    This bridge is not a security sandbox. Run author code with the isolation
    required by the project and keep the evaluator and data out of its workspace.
    """
    if request.stage.confirmation:
        raise ValueError("Use the promotion ledger for confirmation results")
    _bind_archive_protocol(Path(archive_contract_path), request)
    try:
        record = pipeline.evaluate(request, calibrate)
    except ConstraintPending:
        # No calibration cost is incurred. This candidate waits while compliant
        # candidates can proceed through the search.
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        _write_json(results_dir / "correct.json", {"correct": False, "error": "Pending user-constraint review or permission"})
        _write_json(results_dir / "metrics.json", {
            "combined_score": 0.0, "public": {"calibration_status": "pending_constraints",
                                               "source_sha": request.candidate.source_sha},
        })
        return None
    metrics = author_feedback(record)
    correct = metrics["public"]["calibration_reliable"]
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(results_dir / "correct.json", {"correct": correct,
                "error": None if correct else "Calibration failed or is unreliable; inspect the private evaluator ledger."})
    _write_json(results_dir / "metrics.json", metrics)
    return record
