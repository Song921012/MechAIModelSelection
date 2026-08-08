"""Exploratory local-Gaussian WAIC baseline on a prespecified seed subset."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import traceback

import pandas as pd
import torch

from mechai_experiments.config import SCENARIOS
from mechai_experiments.dynamics import CANDIDATES
from mechai_experiments.records import finite_or_none, load_compatible, protocol_hash, result_path, write_record
from mechai_experiments.fitting import FitResult, laplace_loglik_draws
from mechai_experiments.benchmark import _truth_data
from scoring import score_fit


ROOT = Path(__file__).resolve().parents[1]
RESOLUTIONS = torch.logspace(-2, 2, 25, dtype=torch.float64)


def run_cell(scenario_name: str, candidate_key: str, seed: int, draws: int, overwrite: bool, resume: bool = False) -> dict:
    torch.set_num_threads(1)
    settings = {
        "scenario": scenario_name, "candidate": candidate_key, "seed": seed,
        "draws": draws, "posterior": "local_gaussian", "seed_subset": 30,
    }
    digest = protocol_hash(settings)
    path = result_path(ROOT, "waic", f"{scenario_name}__{candidate_key}__seed{seed:03d}")
    previous = load_compatible(path, digest, overwrite, replace_incompatible=resume)
    if previous is not None:
        return previous
    scenario = next(item for item in SCENARIOS if item.name == scenario_name)
    core_path = ROOT / "results" / "records" / "submission" / "core" / "raw" / f"{scenario_name}__{candidate_key}__seed{seed:03d}.json"
    core = json.loads(core_path.read_text(encoding="utf-8"))
    candidate = CANDIDATES[candidate_key]
    times, _clean, noisy = _truth_data(scenario, seed)
    try:
        theta = torch.tensor(core["theta"], dtype=torch.float64)
        information = torch.tensor(core["information"], dtype=torch.float64)
        fit = FitResult(
            theta, core["objective"], core["deviance"], -0.5 * core["deviance"],
            True, 0, 0.0,
        )
        samples = laplace_loglik_draws(
            candidate, fit, information, times, noisy, scenario.observation,
            scenario.noise, draws=draws, seed=seed,
        )
        scores, _profile = score_fit(
            candidate, fit, information, samples, noisy.numel(), RESOLUTIONS
        )
        waic_value = finite_or_none(scores["waic_laplace"])
        p_waic_value = finite_or_none(scores["p_waic"])
        record = {
            "status": "ok" if waic_value is not None and p_waic_value is not None else "nonfinite_score",
            "scenario": scenario_name, "candidate": candidate_key,
            "truth": scenario.truth, "seed": seed, "draws": draws,
            "waic_laplace": waic_value,
            "p_waic": p_waic_value,
        }
    except Exception as exc:
        record = {
            "status": "failed", "scenario": scenario_name, "candidate": candidate_key,
            "truth": scenario.truth, "seed": seed, "draws": draws,
            "error": repr(exc), "traceback": traceback.format_exc(),
        }
    write_record(path, digest, record)
    return record


def aggregate() -> None:
    raw = ROOT / "results" / "records" / "submission" / "waic" / "raw"
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(raw.glob("*.json"))]
    frame = pd.DataFrame([{key: value for key, value in record.items() if key != "traceback"}
                          for record in records])
    output = ROOT / "results" / "summary" / "numerical"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "waic_local_gaussian.csv", index=False)
    selections = []
    ok = frame[(frame["status"] == "ok") & frame["waic_laplace"].notna()]
    for (scenario, seed), group in ok.groupby(["scenario", "seed"]):
        if group["candidate"].nunique() != 5:
            continue
        winner = group.loc[group["waic_laplace"].idxmin()]
        selections.append({
            "scenario": scenario, "seed": seed, "selected": winner["candidate"],
            "truth": winner["truth"], "correct": int(winner["candidate"] == winner["truth"]),
        })
    pd.DataFrame(selections).to_csv(output / "waic_local_gaussian_selections.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=512)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    tasks = [(scenario.name, candidate, seed, args.draws, args.overwrite, args.resume)
             for scenario in SCENARIOS for seed in range(args.seeds) for candidate in CANDIDATES
             if candidate in ("sir", "seir", "tv_sir", "ude_sir_h2", "neural_ode_h2")]
    if args.workers == 1:
        for task in tasks:
            result = run_cell(*task); print(task[:3], result["status"], flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_cell, *task) for task in tasks]
            for future in as_completed(futures):
                result = future.result(); print(result["scenario"], result["candidate"], result["seed"], result["status"], flush=True)
    aggregate()


if __name__ == "__main__":
    main()


