"""Rolling-origin validation and predictive model averaging comparisons."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

from mechai_model_selection import (  # noqa: E402
    criterion_weights, model_average, rolling_origin_splits, stacking_weights,
)
from mechai_experiments.config import SCENARIOS, TRUE_PARAMETERS  # noqa: E402
from mechai_experiments.dynamics import CANDIDATES  # noqa: E402
from mechai_experiments.records import load_compatible, protocol_hash, result_path, write_record  # noqa: E402
from mechai_experiments.fitting import fit_map, information_matrix  # noqa: E402
from mechai_experiments.benchmark import _truth_data  # noqa: E402


SCENARIO_NAMES = ("early_seir_infected_only", "missing_neural_feedback")
CANDIDATE_KEYS = ("sir", "tv_sir", "ude_sir_h2", "neural_ode_h2")


def _bootstrap_mean_interval(values, *, seed: int, draws: int = 5000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a nonempty finite vector")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(draws, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


@dataclass(frozen=True)
class PrefixCandidate:
    base: object
    full_times: torch.Tensor

    @property
    def name(self): return self.base.name
    @property
    def dimension(self): return self.base.dimension
    @property
    def prior_mean(self): return self.base.prior_mean
    @property
    def prior_precision(self): return self.base.prior_precision

    def observe(self, theta, times, observation):
        return self.base.observe(theta, self.full_times, observation)[:len(times)]


def run_rolling_cell(scenario_name: str, candidate_key: str, seed: int, profile: str, overwrite: bool, resume: bool = False) -> dict:
    torch.set_num_threads(1)
    settings = {
        "scenario": scenario_name, "candidate": candidate_key,
        "fractions": (0.5, 0.6, 0.7), "validation_fraction": 0.1,
        "starts": 3 if profile == "submission" else 1,
        "adam_steps": 100 if profile == "submission" else 15,
        "lbfgs_steps": 18 if profile == "submission" else 3,
        "refine_steps": 120 if profile == "submission" else 5,
    }
    digest = protocol_hash(settings)
    path = result_path(ROOT, "predictive", f"{scenario_name}__{candidate_key}__seed{seed:03d}")
    previous = load_compatible(path, digest, overwrite, replace_incompatible=resume)
    if previous is not None:
        return previous
    scenario = next(item for item in SCENARIOS if item.name == scenario_name)
    candidate = CANDIDATES[candidate_key]
    times, _clean, noisy = _truth_data(scenario, seed)
    wrapper = PrefixCandidate(candidate, times)
    folds = []
    try:
        for fold_index, (train_slice, valid_slice) in enumerate(rolling_origin_splits(len(times))):
            train_times = times[train_slice]
            fit = fit_map(
                wrapper, train_times, noisy[train_slice], scenario.observation,
                scenario.noise, seed=10000 * fold_index + seed,
                starts=settings["starts"], adam_steps=settings["adam_steps"],
                lbfgs_steps=settings["lbfgs_steps"],
                refine_steps=settings["refine_steps"],
            )
            prediction = candidate.observe(fit.theta, times, scenario.observation)[valid_slice]
            residual = (prediction - noisy[valid_slice]) / scenario.noise
            pointwise = -0.5 * residual**2 - math.log(scenario.noise * math.sqrt(2.0 * math.pi))
            folds.append({
                "fold": fold_index, "train_end": train_slice.stop,
                "validation_start": valid_slice.start, "validation_end": valid_slice.stop,
                "pointwise_log_density": pointwise.reshape(-1).tolist(),
                "objective": fit.objective, "gradient_norm": fit.gradient_norm,
                "start_diagnostics": fit.start_diagnostics,
            })
        record = {
            "status": "ok", "scenario": scenario_name, "candidate": candidate_key,
            "truth": scenario.truth, "seed": seed, "folds": folds,
        }
    except Exception as exc:
        record = {
            "status": "failed", "scenario": scenario_name, "candidate": candidate_key,
            "truth": scenario.truth, "seed": seed, "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    write_record(path, digest, record)
    return record


def _delta_prediction(candidate, theta, fit_times, forecast_times, observation, noise):
    information, _ = information_matrix(candidate, theta, fit_times, observation, noise)
    precision = information + candidate.prior_precision
    covariance = torch.linalg.inv(
        precision + 1e-8 * torch.eye(candidate.dimension, dtype=torch.float64)
    )
    local = theta.detach().clone().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: candidate.observe(value, forecast_times, observation).reshape(-1),
        local, vectorize=True,
    ).reshape(-1, candidate.dimension)
    mean = candidate.observe(theta, forecast_times, observation).reshape(-1)
    variance = torch.sum((jacobian @ covariance) * jacobian, dim=1) + noise**2
    return mean, torch.clamp(variance, min=noise**2)


def aggregate(profile: str, max_seeds: int | None = None) -> None:
    core_table = pd.read_csv(ROOT / "results" / "summary" / "numerical" / "core_scores.csv")
    predictive_raw = ROOT / "results" / "records" / "submission" / "predictive" / "raw"
    output = ROOT / "results" / "summary" / "numerical"
    rows = []
    methods = ("equal", "aic", "bic", "gic_eff", "gic_vol_050", "stacking", "hard_gic")
    scenario_lookup = {item.name: item for item in SCENARIOS}
    seeds = max_seeds if max_seeds is not None else (50 if profile == "submission" else 1)
    for scenario_name in SCENARIO_NAMES:
        scenario = scenario_lookup[scenario_name]
        fit_times = torch.linspace(0.0, scenario.horizon, scenario.n_times)
        forecast_times = torch.linspace(0.0, 1.35 * scenario.horizon, int(1.35 * scenario.n_times))
        truth = CANDIDATES[scenario.truth].observe(
            TRUE_PARAMETERS[scenario.truth], forecast_times, scenario.observation
        ).reshape(-1)
        test_slice = slice(scenario.n_times - 1, None)
        for seed in range(seeds):
            score_group = core_table[
                (core_table["scenario"] == scenario_name)
                & (core_table["seed"] == seed)
                & (core_table["candidate"].isin(CANDIDATE_KEYS))
            ].set_index("candidate").loc[list(CANDIDATE_KEYS)]
            means, variances, lpd = [], [], []
            for candidate_key in CANDIDATE_KEYS:
                core_path = ROOT / "results" / "records" / "submission" / "core" / "raw" / f"{scenario_name}__{candidate_key}__seed{seed:03d}.json"
                core_record = json.loads(core_path.read_text(encoding="utf-8"))
                theta = torch.tensor(core_record["theta"], dtype=torch.float64)
                mean, variance = _delta_prediction(
                    CANDIDATES[candidate_key], theta, fit_times, forecast_times,
                    scenario.observation, scenario.noise,
                )
                means.append(mean); variances.append(variance)
                rolling_path = predictive_raw / f"{scenario_name}__{candidate_key}__seed{seed:03d}.json"
                rolling = json.loads(rolling_path.read_text(encoding="utf-8"))
                lpd.append(torch.tensor([
                    value for fold in rolling["folds"] for value in fold["pointwise_log_density"]
                ], dtype=torch.float64))
            mean_tensor, variance_tensor = torch.stack(means), torch.stack(variances)
            weight_map = {"equal": torch.full((len(CANDIDATE_KEYS),), 1.0 / len(CANDIDATE_KEYS))}
            for criterion in ("aic", "bic", "gic_eff", "gic_vol_050"):
                weight_map[criterion] = criterion_weights(
                    torch.tensor(score_group[criterion].to_numpy(dtype=float))
                )
            weight_map["stacking"] = stacking_weights(torch.stack(lpd))
            hard = torch.zeros(len(CANDIDATE_KEYS), dtype=torch.float64)
            hard[int(torch.argmin(torch.tensor(score_group["gic_eff"].to_numpy(dtype=float))))] = 1.0
            weight_map["hard_gic"] = hard
            for method in methods:
                averaged = model_average(mean_tensor, variance_tensor, weight_map[method], confidence=0.90)
                target = truth[test_slice]
                mean = averaged["mean"][test_slice]
                variance = averaged["variance"][test_slice]
                lower, upper = averaged["lower"][test_slice], averaged["upper"][test_slice]
                rows.append({
                    "scenario": scenario_name, "seed": seed, "method": method,
                    "mse": float(torch.mean((mean - target)**2)),
                    "log_score": float(torch.mean(-0.5 * ((target - mean)**2 / variance + torch.log(2.0 * math.pi * variance)))),
                    "coverage90": float(torch.mean(((target >= lower) & (target <= upper)).double())),
                    "mean_width90": float(torch.mean(upper - lower)),
                    "within_variance": float(torch.mean(averaged["within"][test_slice])),
                    "between_variance": float(torch.mean(averaged["between"][test_slice])),
                    **{f"weight_{key}": float(value) for key, value in zip(CANDIDATE_KEYS, weight_map[method])},
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "predictive_model_averaging.csv", index=False)
    summary_rows = []
    for scenario_index, scenario_name in enumerate(SCENARIO_NAMES):
        scenario_frame = frame[frame["scenario"] == scenario_name]
        hard = scenario_frame[scenario_frame["method"] == "hard_gic"].set_index("seed")["mse"]
        for method_index, method in enumerate(methods):
            group = scenario_frame[scenario_frame["method"] == method].sort_values("seed")
            paired_hard = hard.loc[group["seed"]].to_numpy()
            mse_difference = group["mse"].to_numpy() - paired_hard
            base_seed = 310_000 + 100 * scenario_index + method_index
            mse_low, mse_high = _bootstrap_mean_interval(group["mse"], seed=base_seed)
            log_low, log_high = _bootstrap_mean_interval(group["log_score"], seed=base_seed + 10_000)
            coverage_low, coverage_high = _bootstrap_mean_interval(group["coverage90"], seed=base_seed + 20_000)
            difference_low, difference_high = _bootstrap_mean_interval(mse_difference, seed=base_seed + 30_000)
            summary_rows.append({
                "scenario": scenario_name, "method": method,
                "mse_mean": group["mse"].mean(), "mse_median": group["mse"].median(),
                "mse_ci_low": mse_low, "mse_ci_high": mse_high,
                "log_score_mean": group["log_score"].mean(),
                "log_score_ci_low": log_low, "log_score_ci_high": log_high,
                "coverage90": group["coverage90"].mean(),
                "coverage90_ci_low": coverage_low, "coverage90_ci_high": coverage_high,
                "width90": group["mean_width90"].mean(),
                "within": group["within_variance"].mean(),
                "between": group["between_variance"].mean(),
                "mse_difference_vs_hard": mse_difference.mean(),
                "mse_difference_ci_low": difference_low,
                "mse_difference_ci_high": difference_high,
                "n": len(group),
            })
    pd.DataFrame(summary_rows).to_csv(
        output / "predictive_model_averaging_summary.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "submission"), default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=None,
                        help="Queue the first N seeds without changing the per-fit protocol hash.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    seeds = args.seeds if args.seeds is not None else (50 if args.profile == "submission" else 1)
    if seeds < 1 or seeds > (50 if args.profile == "submission" else 1):
        parser.error("--seeds must be between 1 and the selected profile maximum")
    tasks = [(scenario, candidate, seed, args.profile, args.overwrite, args.resume)
             for scenario in SCENARIO_NAMES for seed in range(seeds) for candidate in CANDIDATE_KEYS]
    if args.workers == 1:
        for task in tasks:
            result = run_rolling_cell(*task)
            print(task[:3], result["status"], flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_rolling_cell, *task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                print(result["scenario"], result["candidate"], result["seed"], result["status"], flush=True)
    aggregate(args.profile, seeds)


if __name__ == "__main__":
    main()


