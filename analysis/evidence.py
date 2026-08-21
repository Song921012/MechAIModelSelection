"""Calibrate the local evidence criterion by Laplace-proposal importance sampling."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechai_experiments.benchmark import _truth_data
from mechai_experiments.config import SCENARIOS
from mechai_experiments.dynamics import CANDIDATES
from mechai_experiments.fitting import gaussian_deviance


torch.set_default_dtype(torch.float64)
SCENARIO_LOOKUP = {scenario.name: scenario for scenario in SCENARIOS}
SCENARIOS_FOR_CALIBRATION = (
    "regular_sir_full",
    "early_seir_infected_only",
    "missing_time_varying_transmission",
)
CANDIDATES_FOR_CALIBRATION = ("sir", "seir", "tv_sir")


def _normal_log_density(
    theta: torch.Tensor,
    mean: torch.Tensor,
    precision: torch.Tensor,
) -> torch.Tensor:
    delta = theta - mean
    sign, logdet = torch.linalg.slogdet(precision)
    if sign <= 0:
        raise ValueError("precision must be positive definite")
    quadratic = delta @ precision @ delta
    dimension = theta.numel()
    return 0.5 * logdet - 0.5 * dimension * math.log(2.0 * math.pi) - 0.5 * quadratic


def _proposal_draws(
    mean: torch.Tensor,
    precision: torch.Tensor,
    draws: int,
    seed: int,
) -> torch.Tensor:
    dimension = mean.numel()
    chol = torch.linalg.cholesky(precision)
    engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
    uniforms = engine.draw(draws).to(dtype=torch.float64)
    uniforms = torch.clamp(uniforms, 1e-10, 1.0 - 1e-10)
    standard = math.sqrt(2.0) * torch.erfinv(2.0 * uniforms - 1.0)
    perturbations = torch.linalg.solve_triangular(
        chol.mT, standard.mT, upper=True
    ).mT
    return mean.unsqueeze(0) + perturbations


def _logmeanexp(values: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(values, dim=0) - math.log(values.numel())


def evaluate_cell(path_string: str, draws: int) -> dict:
    torch.set_num_threads(1)
    path = Path(path_string)
    record = json.loads(path.read_text(encoding="utf-8"))
    scenario = SCENARIO_LOOKUP[record["scenario"]]
    candidate = CANDIDATES[record["candidate"]]
    theta_hat = torch.tensor(record["theta"], dtype=torch.float64)
    information = torch.tensor(record["information"], dtype=torch.float64)
    prior_precision = candidate.prior_precision.to(dtype=torch.float64)
    posterior_precision = 0.5 * (information + information.mT) + prior_precision
    jitter = 1e-9 * max(1.0, float(torch.trace(posterior_precision)) / candidate.dimension)
    posterior_precision = posterior_precision + jitter * torch.eye(candidate.dimension)
    samples = _proposal_draws(
        theta_hat,
        posterior_precision,
        draws,
        seed=31_337 + 1009 * int(record["seed"]) + 17 * candidate.dimension,
    )
    times, _, noisy = _truth_data(scenario, int(record["seed"]))
    log_weights = []
    invalid = 0
    for theta in samples:
        try:
            prediction = candidate.observe(theta, times, scenario.observation)
            if not torch.isfinite(prediction).all():
                invalid += 1
                continue
            log_likelihood = -0.5 * gaussian_deviance(
                prediction, noisy, scenario.noise
            )
            log_prior = _normal_log_density(
                theta, candidate.prior_mean, prior_precision
            )
            log_proposal = _normal_log_density(
                theta, theta_hat, posterior_precision
            )
            value = log_likelihood + log_prior - log_proposal
            if torch.isfinite(value):
                log_weights.append(value)
            else:
                invalid += 1
        except (RuntimeError, ValueError, OverflowError):
            invalid += 1
    if len(log_weights) < max(32, draws // 2):
        return {
            "status": "insufficient_finite_draws",
            "scenario": scenario.name,
            "truth": record["truth"],
            "candidate": record["candidate"],
            "seed": int(record["seed"]),
            "draws": draws,
            "finite_draws": len(log_weights),
            "invalid_draws": invalid,
        }
    weights_tensor = torch.stack(log_weights)
    log_evidence = _logmeanexp(weights_tensor)
    normalized = torch.softmax(weights_tensor, dim=0)
    ess = 1.0 / torch.sum(normalized * normalized)
    local_score = float(record.get("gic_evid", record["ogic_e"]))
    exact_score = float(-2.0 * log_evidence)
    return {
        "status": "ok",
        "scenario": scenario.name,
        "truth": record["truth"],
        "candidate": record["candidate"],
        "seed": int(record["seed"]),
        "dimension": candidate.dimension,
        "draws": draws,
        "finite_draws": len(log_weights),
        "invalid_draws": invalid,
        "importance_ess": float(ess),
        "relative_ess": float(ess / len(log_weights)),
        "maximum_normalized_weight": float(torch.max(normalized)),
        "minus2_log_evidence_is": exact_score,
        "gic_evid": local_score,
        "bic": float(record["bic"]),
        "gic_evid_error": local_score - exact_score,
    }


def _selection_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    criteria = ("minus2_log_evidence_is", "gic_evid", "bic")
    rows = []
    for (scenario, seed), group in frame.groupby(["scenario", "seed"], sort=True):
        for criterion in criteria:
            winner = group.loc[group[criterion].astype(float).idxmin()]
            rows.append({
                "scenario": scenario,
                "seed": int(seed),
                "criterion": criterion,
                "truth": winner["truth"],
                "selected": winner["candidate"],
                "correct": int(winner["candidate"] == winner["truth"]),
            })
    selections = pd.DataFrame(rows)
    summary = (
        selections.groupby(["scenario", "criterion"], sort=True)
        .agg(recovery_rate=("correct", "mean"), n=("correct", "size"))
        .reset_index()
    )
    return selections, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=512)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.draws < 64:
        raise ValueError("--draws must be at least 64")
    raw = ROOT / "results" / "records" / "submission" / "core" / "raw"
    paths = []
    for path in sorted(raw.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("status") == "ok"
            and record.get("scenario") in SCENARIOS_FOR_CALIBRATION
            and record.get("candidate") in CANDIDATES_FOR_CALIBRATION
            and int(record.get("seed", -1)) < args.seeds
        ):
            paths.append(path)
    started = time.perf_counter()
    results = []
    if args.workers == 1:
        for index, path in enumerate(paths, start=1):
            results.append(evaluate_cell(str(path), args.draws))
            elapsed = time.perf_counter() - started
            print(
                f"evidence {index}/{len(paths)} "
                f"({100.0 * index / len(paths):.1f}%); elapsed={elapsed:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(evaluate_cell, str(path), args.draws): path
                for path in paths
            }
            for index, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                elapsed = time.perf_counter() - started
                rate = index / max(elapsed, 1e-9)
                eta = (len(paths) - index) / max(rate, 1e-9)
                print(
                    f"evidence {index}/{len(paths)} "
                    f"({100.0 * index / len(paths):.1f}%); "
                    f"elapsed={elapsed:.1f}s; eta={eta:.1f}s",
                    flush=True,
                )
    output = ROOT / "results" / "summary" / "first_principles"
    output.mkdir(parents=True, exist_ok=True)
    all_rows = pd.DataFrame(results)
    all_rows.to_csv(output / "evidence_calibration_all.csv", index=False)
    valid = all_rows.loc[all_rows["status"] == "ok"].copy()
    valid.to_csv(output / "evidence_calibration.csv", index=False)
    if len(valid):
        selections, summary = _selection_summary(valid)
        selections.to_csv(output / "evidence_calibration_selections.csv", index=False)
        summary.to_csv(output / "evidence_calibration_selection_summary.csv", index=False)
        diagnostics = (
            valid.groupby(["scenario", "candidate"], sort=True)
            .agg(
                n=("seed", "size"),
                median_relative_ess=("relative_ess", "median"),
                minimum_relative_ess=("relative_ess", "min"),
                median_absolute_gic_error=(
                    "gic_evid_error", lambda x: float(np.median(np.abs(x)))
                ),
                mean_gic_error=("gic_evid_error", "mean"),
            )
            .reset_index()
        )
        diagnostics.to_csv(
            output / "evidence_calibration_summary.csv", index=False
        )
    print(
        f"wrote {len(valid)}/{len(all_rows)} finite calibration cells; "
        f"elapsed={time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
