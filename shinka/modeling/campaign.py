"""Bounded promotion coordinator, independent of the fast Shinka search loop."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

from .contracts import EvaluationRecord, StageSpec
from .pipeline import CalibrationPipeline
from .policy import PromotionPolicy
from .store import BudgetExhausted
from .constraints import ConstraintPending


def promote(
    pipeline: CalibrationPipeline,
    calibrate: Callable,
    stages: Sequence[StageSpec],
    *,
    lock_path: str | Path,
    limit_per_stage: int = 2,
    exploration_fraction: float = 0.25,
    seed: int = 0,
) -> list[EvaluationRecord]:
    """Run one bounded promotion pass; another active coordinator returns []

    Each transition has its own ranking. Work already attempted at the target
    protocol is not silently repeated. Change the seed explicitly for replication
    through the lower-level evaluation API. The durable ledger enforces total cost.
    """
    import fcntl
    if limit_per_stage < 1:
        raise ValueError("limit_per_stage must be positive")
    if len(stages) < 2 or len({s.name for s in stages}) != len(stages):
        raise ValueError("Provide at least two uniquely named calibration stages")
    if any(s.confirmation for s in stages[:-1]) or not stages[-1].confirmation:
        raise ValueError("Only the last stage must be the confirmation stage")
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    completed = []
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return []
        protocols = {replace(r.request, stage=stages[0]).protocol_key
                     for r in pipeline.ledger.records() if r.request.stage in stages}
        if len(protocols) > 1:
            raise ValueError("Promotion requires one dataset/split/objective/evaluator/context contract per ledger; separate mixed campaigns before promoting")
        policy = PromotionPolicy(exploration_fraction=exploration_fraction)
        for source, target in zip(stages, stages[1:]):
            records = pipeline.ledger.records()
            comparable = [r for r in records if r.request.stage in (source, target)]
            selected = policy.select(comparable, source_stage=source.name,
                                     target_stage=target.name, limit=limit_per_stage, seed=seed)
            for record in selected:
                try:
                    completed.append(pipeline.evaluate(replace(record.request, stage=target), calibrate))
                except BudgetExhausted:
                    # A full intermediate tier must not prevent spending the
                    # separately reserved confirmation budget on existing fits.
                    break
                except ConstraintPending:
                    # Permission is local to this candidate; continue work that
                    # complies with the user's active guidelines.
                    continue
    return completed
