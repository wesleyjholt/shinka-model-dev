"""Serializable contracts for scientific calibration, independent of model family."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


class FrozenDict(dict):
    """A picklable JSON mapping whose nested values are also immutable."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("calibration metadata is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable
    __ior__ = _immutable

    def __reduce__(self):
        return FrozenDict, (dict(self),)

    def __deepcopy__(self, memo):
        return self


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON metadata keys must be strings")
        return FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("metadata must contain finite JSON values")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class _Serializable:
    def to_dict(self) -> dict[str, Any]:
        # Return independent, ordinary JSON objects to callers and SQLite.
        return json.loads(json.dumps(asdict(self), allow_nan=False))


@dataclass(frozen=True)
class StageSpec(_Serializable):
    name: str
    method: str
    budget: dict[str, Any] = field(default_factory=dict)
    max_seconds: float = 60.0
    confirmation: bool = False

    def __post_init__(self):
        if not self.name or not self.method:
            raise ValueError("stage name and calibration method are required")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive and finite")
        object.__setattr__(self, "max_seconds", float(self.max_seconds))
        object.__setattr__(self, "budget", _freeze(self.budget))

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class CandidateSpec(_Serializable):
    candidate_id: str
    source_sha: str
    source_path: str = ""
    descriptor: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.candidate_id or not self.source_sha:
            raise ValueError("candidate identity and immutable source SHA are required")
        if any(not isinstance(item, str) for item in self.descriptor):
            raise TypeError("descriptor must contain structural or behavioral tags")
        object.__setattr__(self, "descriptor", tuple(self.descriptor))

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class EvaluationRequest(_Serializable):
    candidate: CandidateSpec
    stage: StageSpec
    seed: int
    dataset_id: str
    split_id: str
    objective_id: str
    evaluator_id: str
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if any(not value for value in (
            self.dataset_id, self.split_id, self.objective_id, self.evaluator_id
        )):
            raise ValueError("dataset, split, objective and evaluator identities are required")
        object.__setattr__(self, "context", _freeze(self.context))

    @property
    def protocol_key(self) -> str:
        """Only records with this same key have directly comparable scores."""
        value = self.to_dict()
        value.pop("candidate")
        value.pop("seed")
        return _digest({"schema": 1, **value})

    @property
    def cache_key(self) -> str:
        value = self.to_dict()
        # Location and descriptive labels do not change an immutable source.
        value["candidate"] = {
            "candidate_id": self.candidate.candidate_id,
            "source_sha": self.candidate.source_sha,
        }
        return _digest({"schema": 1, **value})

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["candidate"] = CandidateSpec.from_dict(value["candidate"])
        value["stage"] = StageSpec.from_dict(value["stage"])
        return cls(**value)


@dataclass(frozen=True)
class CalibrationResult(_Serializable):
    """The evaluator supplies a score on a declared higher-is-better objective.

    Diagnostic reliability is determined by the scientific evaluator. The
    orchestration layer does not pretend one ESS/R-hat rule fits all methods.
    Large samples and predictions belong in artifacts, not in the ledger.
    """

    score: float | None
    parameters: dict[str, Any] = field(default_factory=dict)
    priors: dict[str, Any] = field(default_factory=dict)
    fixed_parameters: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    reliable: bool = False
    reliability_reasons: tuple[str, ...] = ()
    posterior_kind: str = "none"
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("scores must be finite; use None for an unscored result")
        if self.posterior_kind not in {
            "none", "point_estimate", "approximate_posterior", "posterior_samples"
        }:
            raise ValueError("unknown posterior_kind")
        for name in ("parameters", "priors", "fixed_parameters", "diagnostics", "artifacts"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        object.__setattr__(self, "reliability_reasons", tuple(self.reliability_reasons))
        object.__setattr__(self, "notes", tuple(self.notes))

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class EvaluationRecord(_Serializable):
    cache_key: str
    request: EvaluationRequest
    status: str
    artifact_dir: str
    result: CalibrationResult | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0
    charge_seconds: float = 0.0
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["request"] = EvaluationRequest.from_dict(value["request"])
        if value.get("result") is not None:
            value["result"] = CalibrationResult.from_dict(value["result"])
        return cls(**value)
