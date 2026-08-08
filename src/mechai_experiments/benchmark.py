"""Formal, resumable PyTorch benchmark for observable-geometry selection."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import torch

from .config import DEFAULT_CANDIDATES, RESOLUTION_GRID, SCENARIOS, TRUE_PARAMETERS
from .dynamics import CANDIDATES
from .fitting import fit_map, gaussian_deviance, information_matrix, laplace_loglik_draws
from .scoring import score_fit


CRITERIA = (
    "aic", "aicc", "bic", "blocked_validation", "waic_laplace",
    "ogic_p", "gic_eff", "gic_vol_025", "gic_vol_050", "gic_vol_100", "ogic_e",
)


@dataclass(frozen=True)
class _ValidationCandidate:
    """Keep the declared experiment horizon fixed while fitting a time prefix."""

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
        full = self.base.observe(theta, self.full_times, observation)
        return full[: len(times)]


def _truth_data(scenario, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    times = torch.linspace(0.0, scenario.horizon, scenario.n_times, dtype=torch.float64)
    truth = CANDIDATES[scenario.truth]
    clean = truth.observe(TRUE_PARAMETERS[scenario.truth], times, scenario.observation)
    generator = torch.Generator().manual_seed(10_007 * seed + 37 * scenario.n_times)
    noisy = clean + scenario.noise * torch.randn(clean.shape, generator=generator, dtype=torch.float64)
    return times, clean, noisy


def _forecast_error(candidate, theta, scenario, seed: int) -> float:
    forecast_times = torch.linspace(0.0, 1.35 * scenario.horizon, int(1.35 * scenario.n_times), dtype=torch.float64)
    truth = CANDIDATES[scenario.truth]
    clean = truth.observe(TRUE_PARAMETERS[scenario.truth], forecast_times, scenario.observation)
    prediction = candidate.observe(theta, forecast_times, scenario.observation)
    start = max(1, scenario.n_times - 1)
    return float(torch.mean((prediction[start:] - clean[start:]) ** 2))


def _blocked_validation(candidate, scenario, times, noisy, seed: int, args) -> float:
    cutoff = max(4, int(round(0.7 * len(times))))
    fitting_candidate = _ValidationCandidate(candidate, times)
    fit = fit_map(
        fitting_candidate, times[:cutoff], noisy[:cutoff], scenario.observation,
        scenario.noise, seed=50_000 + seed, starts=1,
        adam_steps=max(50, args.adam_steps // 2),
        lbfgs_steps=max(8, args.lbfgs_steps // 2),
    )
    prediction = candidate.observe(fit.theta, times, scenario.observation)
    return float(gaussian_deviance(prediction[cutoff:], noisy[cutoff:], scenario.noise))


def run_cell(root: Path, scenario, candidate_key: str, seed: int, args) -> dict:
    torch.set_num_threads(1)
    raw_dir = root / "results" / "formal" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{scenario.name}__{candidate_key}__seed{seed:03d}.json"
    if path.exists() and not args.overwrite:
        return json.loads(path.read_text(encoding="utf-8"))
    candidate = CANDIDATES[candidate_key]
    times, clean, noisy = _truth_data(scenario, seed)
    try:
        fit = fit_map(
            candidate, times, noisy, scenario.observation, scenario.noise,
            seed=seed, starts=args.starts, adam_steps=args.adam_steps,
            lbfgs_steps=args.lbfgs_steps,
        )
        information, jacobian = information_matrix(candidate, fit.theta, times, scenario.observation, scenario.noise)
        draws = laplace_loglik_draws(
            candidate, fit, information, times, noisy, scenario.observation,
            scenario.noise, draws=args.posterior_draws, seed=seed,
        )
        scores, profile = score_fit(
            candidate, fit, information, draws, noisy.numel(), RESOLUTION_GRID,
        )
        eigenvalues = torch.linalg.eigvalsh(information).clamp_min(0)
        record = {
            "status": "ok",
            "scenario": scenario.name,
            "truth": scenario.truth,
            "candidate": candidate_key,
            "candidate_label": candidate.name,
            "seed": seed,
            "observation": scenario.observation,
            "noise": scenario.noise,
            "n_observations": noisy.numel(),
            "dimension": candidate.dimension,
            "objective": fit.objective,
            "deviance": fit.deviance,
            "converged": fit.converged,
            "iterations": fit.iterations,
            "wall_seconds": fit.wall_seconds,
            "forecast_mse": _forecast_error(candidate, fit.theta, scenario, seed),
            "blocked_validation": (
                float("nan") if args.skip_validation
                else _blocked_validation(candidate, scenario, times, noisy, seed, args)
            ),
            "condition_information": float(torch.linalg.cond(information + 1e-10 * torch.eye(candidate.dimension))),
            "jacobian_rank": int(torch.linalg.matrix_rank(jacobian, rtol=1e-7)),
            "theta": fit.theta.tolist(),
            "information_eigenvalues": eigenvalues.tolist(),
            "generalized_eigenvalues": profile["eigenvalues"].tolist(),
            "profile": {
                "resolution": profile["resolution"].tolist(),
                "dimension": profile["dimension"].tolist(),
                "complexity": profile["complexity"].tolist(),
            },
            **scores,
        }
    except Exception as exc:
        record = {
            "status": "failed",
            "scenario": scenario.name,
            "truth": scenario.truth,
            "candidate": candidate_key,
            "candidate_label": candidate.name,
            "seed": seed,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def aggregate(root: Path) -> None:
    raw_dir = root / "results" / "formal" / "raw"
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(raw_dir.glob("*.json"))]
    ok = [record for record in records if record.get("status") == "ok"]
    scalar_exclusions = {"theta", "information_eigenvalues", "generalized_eigenvalues", "profile", "traceback"}
    rows = [{key: value for key, value in record.items() if key not in scalar_exclusions} for record in ok]
    frame = pd.DataFrame(rows)
    log_n = np.log(frame["n_observations"].astype(float))
    frame["gic_eff"] = frame["deviance"] + log_n * frame["d_obs"]
    for suffix, weight in (("025", 0.25), ("050", 0.50), ("100", 1.00)):
        frame[f"gic_vol_{suffix}"] = frame["gic_eff"] + weight * frame["c_obs"]
    frame.to_csv(table_dir / "formal_model_scores.csv", index=False)

    profile_rows = []
    for record in ok:
        profile = record["profile"]
        for tau, dimension, complexity in zip(profile["resolution"], profile["dimension"], profile["complexity"]):
            profile_rows.append({
                "scenario": record["scenario"], "candidate": record["candidate"],
                "seed": record["seed"], "resolution": tau,
                "d_obs": dimension, "c_obs": complexity,
            })
    pd.DataFrame(profile_rows).to_csv(table_dir / "formal_resolution_profiles.csv", index=False)

    selections = []
    weights = []
    for (scenario, seed), group in frame.groupby(["scenario", "seed"]):
        for criterion in CRITERIA:
            finite = group[np.isfinite(group[criterion].astype(float))].copy()
            if finite.empty:
                continue
            values = finite[criterion].astype(float).to_numpy()
            shifted = values - np.min(values)
            raw_weights = np.exp(-0.5 * np.clip(shifted, 0, 1400))
            raw_weights /= raw_weights.sum()
            winner_index = int(np.argmin(values))
            winner = finite.iloc[winner_index]
            selections.append({
                "scenario": scenario, "seed": seed, "criterion": criterion,
                "selected": winner["candidate"], "truth": winner["truth"],
                "correct": int(winner["candidate"] == winner["truth"]),
            })
            for (_, row), weight in zip(finite.iterrows(), raw_weights):
                weights.append({
                    "scenario": scenario, "seed": seed, "criterion": criterion,
                    "candidate": row["candidate"], "weight": float(weight),
                })
    selection_frame = pd.DataFrame(selections)
    weight_frame = pd.DataFrame(weights)
    selection_frame.to_csv(table_dir / "formal_selections.csv", index=False)
    weight_frame.to_csv(table_dir / "formal_model_weights.csv", index=False)
    if not selection_frame.empty:
        summary = selection_frame.groupby(["scenario", "criterion"]).agg(
            recovery_rate=("correct", "mean"),
            n=("correct", "size"),
            modal_selection=("selected", lambda x: x.value_counts().index[0]),
            modal_frequency=("selected", lambda x: x.value_counts().iloc[0] / len(x)),
        ).reset_index()
        z = 1.959963984540054
        n = summary["n"].astype(float)
        p = summary["recovery_rate"].astype(float)
        denominator = 1.0 + z**2 / n
        center = (p + z**2 / (2.0 * n)) / denominator
        half_width = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denominator
        summary["recovery_ci_low"] = np.maximum(0.0, center - half_width)
        summary["recovery_ci_high"] = np.minimum(1.0, center + half_width)
        summary.to_csv(table_dir / "formal_selection_summary.csv", index=False)
    metadata = {
        "cells_total": len(records),
        "cells_ok": len(ok),
        "cells_failed": len(records) - len(ok),
        "criteria": list(CRITERIA),
        "diagnostic_nonfinite": {
            "waic_laplace": int((~np.isfinite(pd.to_numeric(frame["waic_laplace"], errors="coerce"))).sum()),
        },
    }
    (root / "results" / "formal" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--adam-steps", type=int, default=140)
    parser.add_argument("--lbfgs-steps", type=int, default=25)
    parser.add_argument("--posterior-draws", type=int, default=64)
    parser.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--scenarios", nargs="+", default=[scenario.name for scenario in SCENARIOS])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scenario_lookup = {scenario.name: scenario for scenario in SCENARIOS}
    tasks = [
        (scenario_lookup[scenario_name], candidate_key, seed)
        for scenario_name in args.scenarios
        for seed in range(args.seeds)
        for candidate_key in args.candidates
    ]
    if args.workers == 1:
        for scenario, candidate_key, seed in tasks:
            result = run_cell(root, scenario, candidate_key, seed, args)
            print(scenario.name, seed, candidate_key, result["status"], result.get("ogic_e", result.get("error", "")), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_cell, root, scenario, candidate_key, seed, args): (scenario, candidate_key, seed)
                for scenario, candidate_key, seed in tasks
            }
            for future in as_completed(futures):
                scenario, candidate_key, seed = futures[future]
                result = future.result()
                print(scenario.name, seed, candidate_key, result["status"], result.get("ogic_e", result.get("error", "")), flush=True)
    aggregate(root)


if __name__ == "__main__":
    main()

