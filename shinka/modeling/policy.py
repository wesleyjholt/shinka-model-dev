"""Promotion uses comparable evidence plus a reserved structural diversity quota."""

from __future__ import annotations

import math
import random
from collections import defaultdict


def _identity(record):
    candidate = record.request.candidate
    return candidate.candidate_id, candidate.source_sha


def _distance(left, right):
    left, right = set(left), set(right)
    return len(left ^ right) / len(left | right) if left or right else 0.0


def confirmed_records(records):
    """Return eligible confirmations; callers still group by protocol before ranking."""
    return [record for record in records if (
        record.status == "completed" and record.request.stage.confirmation
        and record.result is not None and record.result.reliable
        and record.result.score is not None
        and record.result.posterior_kind not in {"none", "point_estimate"}
    )]


class PromotionPolicy:
    def __init__(self, exploration_fraction=0.25):
        if not 0 <= exploration_fraction <= 1:
            raise ValueError("exploration_fraction must be between zero and one")
        self.exploration_fraction = exploration_fraction

    def select(self, records, source_stage, target_stage, limit, seed=0):
        """Select records to calibrate at the next fidelity, never mix fidelities.

        Scores are averaged over available seeds within each candidate source.
        Distinct candidates reserve ceil(limit * exploration_fraction) slots,
        based on distance from the promoted archive and exploitation winners.
        A single available slot uses a seeded Bernoulli choice so repeated small
        batches do not turn into exploration-only search.
        Equal descriptors fall back to seeded random choice. Failed/uncertain
        fits do not claim success, but finite uncertain fits may be promoted to
        resolve their uncertainty. A target-stage attempt counts even if failed.
        """
        if limit <= 0:
            return []
        records = list(records)
        eligible = [r for r in records if r.status == "completed"
                    and r.request.stage.name == source_stage
                    and r.result is not None and r.result.score is not None]
        if len({r.request.protocol_key for r in eligible}) > 1:
            raise ValueError("promotion requires one comparable source protocol; filter dataset/split/objective/stage configuration first")
        if not eligible:
            return []
        reference = eligible[0].request
        # Other datasets/evaluators must not suppress fresh work in this campaign.
        target_records = [r for r in records if r.request.stage.name == target_stage
                          and r.request.dataset_id == reference.dataset_id
                          and r.request.split_id == reference.split_id
                          and r.request.objective_id == reference.objective_id
                          and r.request.evaluator_id == reference.evaluator_id
                          and r.request.context == reference.context]
        attempted = {_identity(r) for r in target_records}
        groups = defaultdict(list)
        for record in eligible:
            if _identity(record) not in attempted:
                groups[_identity(record)].append(record)
        candidates = [(sum(r.result.score for r in group) / len(group), group[0]) for group in groups.values()]
        rng = random.Random(seed)
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: item[0], reverse=True)
        count = min(limit, len(candidates))
        explore = (int(rng.random() < self.exploration_fraction) if count == 1
                   else min(count, math.ceil(count * self.exploration_fraction)))
        winners = [r for _, r in candidates[:count - explore]]
        remaining = [r for _, r in candidates[count - explore:]]
        archive = [r.request.candidate.descriptor for r in target_records] + [r.request.candidate.descriptor for r in winners]
        rng.shuffle(remaining)
        for _ in range(explore):
            # With no prior archive the first exploration seed is random.
            chosen = max(remaining, key=lambda r: min(
                (_distance(r.request.candidate.descriptor, descriptor) for descriptor in archive), default=0.0
            ))
            winners.append(chosen)
            archive.append(chosen.request.candidate.descriptor)
            remaining.remove(chosen)
        return winners
