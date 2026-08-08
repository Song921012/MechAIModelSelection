"""Neural-width stress test for UDE observable complexity."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import traceback
import pandas as pd
import torch

from mechai_experiments.config import RESOLUTION_GRID, SCENARIOS, TRUE_PARAMETERS
from mechai_experiments.dynamics import CANDIDATES
from mechai_experiments.fitting import fit_map, information_matrix, laplace_loglik_draws
from scoring import score_fit


WIDTHS = ("ude_sir_h2", "ude_sir_h4", "ude_sir_h8")


def data_for(seed: int):
    scenario = next(item for item in SCENARIOS if item.name == "missing_neural_feedback")
    times = torch.linspace(0.0, scenario.horizon, scenario.n_times, dtype=torch.float64)
    truth = CANDIDATES[scenario.truth]
    clean = truth.observe(TRUE_PARAMETERS[scenario.truth], times, scenario.observation)
    generator = torch.Generator().manual_seed(90_001 + seed)
    noisy = clean + scenario.noise * torch.randn(clean.shape, generator=generator, dtype=torch.float64)
    return scenario, times, clean, noisy


def run_cell(root: Path, candidate_key: str, seed: int, args) -> dict:
    torch.set_num_threads(1)
    output = root / "results" / "capacity" / "raw"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{candidate_key}__seed{seed:03d}.json"
    if path.exists() and not args.overwrite:
        return json.loads(path.read_text(encoding="utf-8"))
    scenario, times, clean, noisy = data_for(seed)
    candidate = CANDIDATES[candidate_key]
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
        scores, profile = score_fit(candidate, fit, information, draws, noisy.numel(), RESOLUTION_GRID)
        prediction = candidate.observe(fit.theta, times, scenario.observation)
        record = {
            "status": "ok", "candidate": candidate_key, "seed": seed,
            "dimension": candidate.dimension, "deviance": fit.deviance,
            "trajectory_mse": float(torch.mean((prediction - clean) ** 2)),
            "jacobian_rank": int(torch.linalg.matrix_rank(jacobian, rtol=1e-7)),
            "wall_seconds": fit.wall_seconds,
            "theta": fit.theta.tolist(),
            "generalized_eigenvalues": profile["eigenvalues"].tolist(),
            "profile": {
                "resolution": profile["resolution"].tolist(),
                "dimension": profile["dimension"].tolist(),
                "complexity": profile["complexity"].tolist(),
            },
            **scores,
        }
    except Exception as exc:
        record = {"status": "failed", "candidate": candidate_key, "seed": seed, "error": repr(exc), "traceback": traceback.format_exc()}
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def aggregate(root: Path) -> None:
    raw = root / "results" / "capacity" / "raw"
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(raw.glob("*.json"))]
    ok = [record for record in records if record["status"] == "ok"]
    rows = [{key: value for key, value in record.items() if key not in {"theta", "generalized_eigenvalues", "profile", "traceback"}} for record in ok]
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "tables" / "capacity_study.csv", index=False)
    summary = frame.groupby("candidate").agg(
        dimension=("dimension", "first"), d_obs_mean=("d_obs", "mean"),
        d_obs_std=("d_obs", "std"), c_obs_mean=("c_obs", "mean"),
        trajectory_mse_median=("trajectory_mse", "median"),
        ogic_e_mean=("ogic_e", "mean"), n=("seed", "size"),
    ).reset_index()
    summary.to_csv(root / "tables" / "capacity_study_summary.csv", index=False)
    metadata = {"cells": len(records), "ok": len(ok), "failed": len(records) - len(ok)}
    (root / "results" / "capacity" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--adam-steps", type=int, default=160)
    parser.add_argument("--lbfgs-steps", type=int, default=30)
    parser.add_argument("--posterior-draws", type=int, default=64)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tasks = [(candidate, seed) for seed in range(args.seeds) for candidate in WIDTHS]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_cell, root, candidate, seed, args): (candidate, seed) for candidate, seed in tasks}
        for future in as_completed(futures):
            candidate, seed = futures[future]
            result = future.result()
            print(candidate, seed, result["status"], result.get("ogic_e", result.get("error", "")), flush=True)
    aggregate(root)


if __name__ == "__main__":
    main()


