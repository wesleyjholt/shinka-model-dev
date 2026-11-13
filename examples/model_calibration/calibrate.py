"""Small reference inference routines, not a general-purpose inference library.

The example uses a conjugate model to check inference against a known answer.
MAP returns a point estimate. MCMC and SMC return posterior draws. All methods
fit training observations only; the score uses a distinct development split.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from shinka.modeling import CalibrationResult
from .model_spec import read_model_spec

NOISE_SD = 0.25
PRIOR_SD = 2.0
DATASET_ID = "synthetic-quadratic-v1-seed-2026"
OBJECTIVE_ID = "mean-log-predictive-density-v1-higher-is-better"
EVALUATOR_ID = "gaussian-polynomial-calibration-v1"


def observations(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Separate seeds keep final-test generation out of calibration."""
    seeds = {"train": 2026, "development": 2027, "final_test": 2028}
    if split not in seeds:
        raise ValueError(f"Unknown split: {split}")
    rng = np.random.default_rng(seeds[split])
    x = np.linspace(-1.0, 1.0, 48) if split == "train" else rng.uniform(-1, 1, 40)
    y = 0.4 + 0.75 * x - 0.9 * x**2 + rng.normal(0, NOISE_SD, x.size)
    return x, y


def design(x: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(x, N=degree + 1, increasing=True)


def exact_posterior(degree: int, temperature: float = 1.0):
    x, y = observations("train")
    matrix = design(x, degree)
    precision = np.eye(degree + 1) / PRIOR_SD**2 + temperature * matrix.T @ matrix / NOISE_SD**2
    covariance = np.linalg.inv(precision)
    mean = covariance @ (temperature * matrix.T @ y / NOISE_SD**2)
    return mean, covariance


def log_likelihood(samples: np.ndarray, degree: int) -> np.ndarray:
    x, y = observations("train")
    residual = y[:, None] - design(x, degree) @ samples.T
    return -0.5 * np.sum((residual / NOISE_SD)**2, axis=0)


def log_target(samples: np.ndarray, degree: int, temperature: float = 1.0) -> np.ndarray:
    return temperature * log_likelihood(samples, degree) - 0.5 * np.sum((samples / PRIOR_SD)**2, axis=1)


def predictive_score(samples: np.ndarray, degree: int, split: str = "development") -> float:
    x, y = observations(split)
    residual = y[:, None] - design(x, degree) @ samples.T
    density = -0.5 * (residual / NOISE_SD)**2 - np.log(NOISE_SD * np.sqrt(2 * np.pi))
    return float(np.mean(logsumexp(density, axis=1) - np.log(len(samples))))


def _rhat(chains: np.ndarray) -> np.ndarray:
    # Split-chain variance diagnostic. It is not the rank-normalized modern R-hat.
    _, count, _ = chains.shape
    half = count // 2
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    within = np.mean(np.var(split, axis=1, ddof=1), axis=0)
    between = half * np.var(np.mean(split, axis=1), axis=0, ddof=1)
    return np.sqrt(((half - 1) / half * within + between / half) / within)


def _ess(chains: np.ndarray) -> np.ndarray:
    """Conservative initial-positive-pair autocorrelation estimate for this demo."""
    n_chains, count, dimension = chains.shape
    result = []
    for coordinate in range(dimension):
        centered = chains[:, :, coordinate] - chains[:, :, coordinate].mean(axis=1, keepdims=True)
        variance = np.mean(np.sum(centered**2, axis=1) / count)
        correlations = []
        for lag in range(1, min(count // 2, 250)):
            correlations.append(float(np.mean(np.sum(centered[:, :-lag] * centered[:, lag:], axis=1) / (count - lag)) / variance))
        paired = [sum(correlations[i:i + 2]) for i in range(0, len(correlations) - 1, 2)]
        positive = []
        for pair in paired:
            if pair <= 0:
                break
            positive.append(pair)
        result.append(min(n_chains * count, n_chains * count / max(1.0, 1 + 2 * sum(positive))))
    return np.asarray(result)


def mcmc(degree: int, budget: dict, rng: np.random.Generator):
    chains = int(budget.get("chains", 4))
    warmup = int(budget.get("warmup", 300))
    draws = int(budget.get("draws", 500))
    if chains < 2 or warmup < 0 or draws < 20:
        raise ValueError("MCMC needs at least two chains and 20 retained draws")
    mean, covariance = exact_posterior(degree)
    # Gaussian curvature is available cheaply here, just as a MAP Hessian can be.
    proposal = np.linalg.cholesky(covariance) * (2.38 / np.sqrt(degree + 1))
    current = mean + rng.normal(size=(chains, degree + 1)) @ np.linalg.cholesky(covariance).T * 4
    current_target = log_target(current, degree)
    samples = np.empty((chains, draws, degree + 1))
    accepted = 0
    for step in range(warmup + draws):
        proposed = current + rng.normal(size=current.shape) @ proposal.T
        proposed_target = log_target(proposed, degree)
        accept = np.log(rng.uniform(size=chains)) < proposed_target - current_target
        current[accept] = proposed[accept]
        current_target[accept] = proposed_target[accept]
        accepted += np.sum(accept)
        if step >= warmup:
            samples[:, step - warmup] = current
    rhat, ess = _rhat(samples), _ess(samples)
    diagnostics = {
        "split_rhat_max": float(max(rhat)),
        "autocorrelation_ess_min": float(min(ess)),
        "acceptance_rate": float(accepted / (chains * (warmup + draws))),
        "chains": chains, "retained_draws_per_chain": draws,
        "diagnostic_scope": "illustrative split R-hat and autocorrelation ESS; not a production diagnostic suite",
    }
    return samples.reshape(-1, degree + 1), diagnostics


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positions = (rng.uniform() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def smc(degree: int, budget: dict, rng: np.random.Generator):
    n = int(budget.get("particles", 1024))
    temperatures = int(budget.get("temperatures", 32))
    moves = int(budget.get("rejuvenation_steps", 4))
    if n < 32 or temperatures < 2 or moves < 1:
        raise ValueError("SMC needs at least 32 particles, two temperatures and one move")
    samples = rng.normal(0, PRIOR_SD, size=(n, degree + 1))
    log_weights = np.full(n, -np.log(n))
    ancestors = np.arange(n)
    previous_temperature = 0.0
    minimum_ess = float(n)
    accepted, attempted, resamples = 0, 0, 0
    # A quadratic schedule resolves the sharp transition from diffuse prior.
    schedule = np.linspace(0, 1, temperatures + 1)[1:]**2
    for temperature in schedule:
        log_weights += (temperature - previous_temperature) * log_likelihood(samples, degree)
        log_weights -= logsumexp(log_weights)
        weights = np.exp(log_weights)
        ess = float(1 / np.sum(weights**2))
        minimum_ess = min(minimum_ess, ess)
        # At the last temperature, return equally weighted posterior samples.
        if ess < n * 0.7 or temperature == 1.0:
            indices = systematic_resample(weights, rng)
            samples, ancestors = samples[indices], ancestors[indices]
            log_weights = np.full(n, -np.log(n))
            resamples += 1
        _, covariance = exact_posterior(degree, temperature)
        proposal = np.linalg.cholesky(covariance) * (2.38 / np.sqrt(degree + 1))
        current_target = log_target(samples, degree, temperature)
        for _ in range(moves):
            proposed = samples + rng.normal(size=samples.shape) @ proposal.T
            proposed_target = log_target(proposed, degree, temperature)
            accept = np.log(rng.uniform(size=n)) < proposed_target - current_target
            samples[accept] = proposed[accept]
            current_target[accept] = proposed_target[accept]
            accepted += int(np.sum(accept))
            attempted += n
        previous_temperature = float(temperature)
    return samples, {
        "particles": n, "temperature_steps": temperatures,
        "minimum_weight_ess": minimum_ess,
        "unique_initial_ancestors": int(len(np.unique(ancestors))),
        "rejuvenation_acceptance": accepted / attempted,
        "resampling_events": resamples,
        "diagnostic_scope": "weight ESS and genealogy do not establish posterior convergence",
    }


def reference_diagnostics(samples: np.ndarray, degree: int) -> dict:
    exact_mean, exact_covariance = exact_posterior(degree)
    relative_mean_error = np.abs(samples.mean(axis=0) - exact_mean) / np.sqrt(np.diag(exact_covariance))
    covariance_error = np.linalg.norm(np.cov(samples, rowvar=False) - exact_covariance) / np.linalg.norm(exact_covariance)
    return {
        "analytic_mean_error_in_posterior_sd_max": float(max(relative_mean_error)),
        "analytic_covariance_relative_error": float(covariance_error),
        "reference_check": "Conjugate synthetic example only. Real models require their own inference checks.",
    }


def save_predictions(samples: np.ndarray, degree: int, path: Path, seed: int):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, 161)
    latent = design(x, degree) @ samples.T
    predicted = latent + rng.normal(0, NOISE_SD, size=latent.shape)
    quantiles = np.quantile(predicted, [0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x", "predictive_q05", "predictive_q25", "predictive_q50", "predictive_q75", "predictive_q95"])
        writer.writerows(np.column_stack([x, quantiles]))


def calibrate(request, artifact_dir: str) -> CalibrationResult:
    if request.split_id != "development":
        raise ValueError("Search calibration can score development observations only")
    candidate = read_model_spec(request.candidate.source_path)
    degree = candidate["degree"]
    if not isinstance(degree, int) or not 0 <= degree <= 8:
        raise ValueError("The polynomial demo expects a degree from zero to eight")
    rng = np.random.default_rng(request.seed)
    method = request.stage.method
    if method == "map":
        mean, _ = exact_posterior(degree)
        samples = mean[None, :]
        diagnostics = {"optimizer": "closed-form strictly concave Gaussian posterior", "uncertainty": "point estimate only"}
        kind = "point_estimate"
        reliable, reasons = True, ()
    elif method == "mcmc":
        samples, diagnostics = mcmc(degree, dict(request.stage.budget), rng)
        diagnostics.update(reference_diagnostics(samples, degree))
        reliable = diagnostics["split_rhat_max"] < 1.1 and diagnostics["autocorrelation_ess_min"] >= 80 and diagnostics["analytic_mean_error_in_posterior_sd_max"] < 0.3
        kind = "posterior_samples"
        reasons = () if reliable else ("Short-chain diagnostic or analytical reference check failed; allocate inference work before judging the model.",)
    elif method == "smc":
        samples, diagnostics = smc(degree, dict(request.stage.budget), rng)
        diagnostics.update(reference_diagnostics(samples, degree))
        reliable = diagnostics["analytic_mean_error_in_posterior_sd_max"] < 0.2 and diagnostics["analytic_covariance_relative_error"] < 0.2
        kind = "posterior_samples"
        reasons = () if reliable else ("Synthetic analytical reference check failed; this is an inference failure, not evidence against the structure.",)
    else:
        raise ValueError(f"Unsupported demonstration method: {method}")
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    sample_path = directory / "calibration_samples.npz"
    np.savez_compressed(sample_path, coefficients=samples, degree=degree)
    artifacts = {"samples": str(sample_path)}
    if kind == "posterior_samples":
        prediction_path = directory / "predictions.csv"
        save_predictions(samples, degree, prediction_path, request.seed + 1)
        artifacts["predictions"] = str(prediction_path)
    parameters = {}
    for index in range(degree + 1):
        entry = {"mean": float(samples[:, index].mean())}
        if kind == "posterior_samples":
            entry.update({"sd": float(samples[:, index].std(ddof=1)), "q05": float(np.quantile(samples[:, index], 0.05)), "q95": float(np.quantile(samples[:, index], 0.95))})
        parameters[f"beta_{index}"] = entry
    return CalibrationResult(
        score=predictive_score(samples, degree), parameters=parameters,
        priors={f"beta_{index}": {"distribution": "Normal", "mean": 0, "sd": PRIOR_SD} for index in range(degree + 1)},
        fixed_parameters={"observation_sd": {"value": NOISE_SD, "reason": "Known synthetic data-generation noise; an assumption specific to this example."}},
        diagnostics=diagnostics, reliable=bool(reliable), reliability_reasons=reasons,
        posterior_kind=kind, artifacts=artifacts,
        notes=(candidate["hypothesis"], "Calibration uses training observations. Score uses repeatedly searched development observations, not the final test."),
    )
