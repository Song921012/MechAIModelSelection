"""High-repetition epidemic benchmark with optimization diagnostics."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import traceback

import pandas as pd
import numpy as np
import torch

from mechai_experiments.config import DEFAULT_CANDIDATES, SCENARIOS, TRUE_PARAMETERS
from mechai_experiments.criteria import primary_scores
from mechai_experiments.dynamics import CANDIDATES
from mechai_experiments.records import finite_or_none, load_compatible, objective_gap, protocol_hash, result_path, write_record
from mechai_experiments.fitting import fit_map, information_matrix
from mechai_experiments.benchmark import _forecast_error, _truth_data


ROOT = Path(__file__).resolve().parents[1]


def run_cell(scenario_name: str, candidate_key: str, seed: int, profile: str, overwrite: bool, resume: bool = False) -> dict:
    torch.set_num_threads(1)
    scenario = next(item for item in SCENARIOS if item.name == scenario_name)
    settings = {
        "study": "core", "scenario": asdict(scenario), "candidate": candidate_key,
        "seeds": 100 if profile == "submission" else 1,
        "starts": 5 if profile == "submission" else 1,
        "adam_steps": 180 if profile == "submission" else 20,
        "lbfgs_steps": 35 if profile == "submission" else 4,
        "refine_steps": 250 if profile == "submission" else 8,
    }
    digest = protocol_hash(settings)
    path = result_path(ROOT, "core", f"{scenario_name}__{candidate_key}__seed{seed:03d}")
    previous = load_compatible(path, digest, overwrite, replace_incompatible=resume)
    if previous is not None:
        return previous
    candidate = CANDIDATES[candidate_key]
    times, _clean, noisy = _truth_data(scenario, seed)
    try:
        fit = fit_map(
            candidate, times, noisy, scenario.observation, scenario.noise,
            seed=seed, starts=settings["starts"], adam_steps=settings["adam_steps"],
            lbfgs_steps=settings["lbfgs_steps"],
            refine_steps=settings["refine_steps"],
        )
        information, jacobian = information_matrix(
            candidate, fit.theta, times, scenario.observation, scenario.noise
        )
        scores = primary_scores(candidate, fit, information, noisy.numel())
        record = {
            "status": "ok", "scenario": scenario.name, "truth": scenario.truth,
            "candidate": candidate_key, "seed": seed, "observation": scenario.observation,
            "noise": scenario.noise, "n_observations": noisy.numel(),
            "dimension": candidate.dimension, "theta": fit.theta.tolist(),
            "objective": fit.objective, "deviance": fit.deviance,
            "forecast_mse": _forecast_error(candidate, fit.theta, scenario, seed),
            "gradient_norm": finite_or_none(fit.gradient_norm), "best_start": fit.best_start,
            "start_diagnostics": fit.start_diagnostics, "wall_seconds": fit.wall_seconds,
            "second_best_objective_gap": objective_gap(fit.start_diagnostics),
            "jacobian_rank": int(torch.linalg.matrix_rank(jacobian, rtol=1e-7)),
            "information": information.tolist(),
            **{key: finite_or_none(value) for key, value in scores.items()},
        }
    except Exception as exc:
        record = {
            "status": "failed", "scenario": scenario.name, "truth": scenario.truth,
            "candidate": candidate_key, "seed": seed, "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    write_record(path, digest, record)
    return record


def aggregate(max_seeds: int | None = None) -> None:
    raw = ROOT / "results" / "records" / "submission" / "core" / "raw"
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(raw.glob("*.json"))]
    if max_seeds is not None:
        records = [record for record in records if int(record.get("seed", max_seeds)) < max_seeds]
    excluded = {"theta", "information", "start_diagnostics", "traceback"}
    rows = [{key: value for key, value in record.items() if key not in excluded}
            for record in records if record.get("status") == "ok"]
    frame = pd.DataFrame(rows)
    output = ROOT / "results" / "summary" / "numerical"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "core_scores.csv", index=False)
    criteria = ("aic", "aicc", "bic", "gic_eff", "gic_vol_025", "gic_vol_050", "gic_vol_100", "ogic_e")
    selections = []
    for (scenario, seed), group in frame.groupby(["scenario", "seed"]):
        for criterion in criteria:
            finite = group[group[criterion].notna()]
            if finite.empty:
                continue
            winner = finite.loc[finite[criterion].idxmin()]
            selections.append({
                "scenario": scenario, "seed": seed, "criterion": criterion,
                "truth": winner["truth"], "selected": winner["candidate"],
                "correct": int(winner["truth"] == winner["candidate"]),
            })
    selection = pd.DataFrame(selections)
    selection.to_csv(output / "core_selections.csv", index=False)
    if not selection.empty:
        summary = selection.groupby(["scenario", "criterion"])["correct"].agg(["mean", "count"]).reset_index()
        z = 1.959963984540054
        denominator = 1.0 + z**2 / summary["count"]
        center = (summary["mean"] + z**2 / (2.0 * summary["count"])) / denominator
        half = z * np.sqrt(summary["mean"] * (1.0 - summary["mean"]) / summary["count"] + z**2 / (4.0 * summary["count"]**2)) / denominator
        summary["ci_low"] = center - half
        summary["ci_high"] = center + half
        summary.to_csv(output / "core_selection_summary.csv", index=False)
    frame.groupby(["scenario", "candidate"]).agg(
        n=("seed", "count"), median_gradient_norm=("gradient_norm", "median"),
        median_wall_seconds=("wall_seconds", "median"),
        median_objective=("objective", "median"),
    ).reset_index().to_csv(output / "core_fit_diagnostics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "submission"), default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=None,
                        help="Queue the first N seeds without changing the per-fit protocol hash.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scenarios", nargs="+", default=[item.name for item in SCENARIOS])
    parser.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else (100 if args.profile == "submission" else 1)
    if seeds < 1 or seeds > (100 if args.profile == "submission" else 1):
        parser.error("--seeds must be between 1 and the selected profile maximum")
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    tasks = [(scenario, candidate, seed, args.profile, args.overwrite, args.resume)
             for scenario in args.scenarios for seed in range(seeds) for candidate in args.candidates]
    if args.workers == 1:
        for task in tasks:
            result = run_cell(*task)
            print(task[0], task[1], task[2], result["status"], flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_cell, *task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                print(result["scenario"], result["candidate"], result["seed"], result["status"], flush=True)
    aggregate(seeds)


if __name__ == "__main__":
    main()


