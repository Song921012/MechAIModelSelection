"""Calibrate predictive optimism without refitting any candidate model."""

from __future__ import annotations

import argparse
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


torch.set_default_dtype(torch.float64)
SCENARIO_LOOKUP = {scenario.name: scenario for scenario in SCENARIOS}


def _deviance_draws(
    prediction: torch.Tensor,
    clean: torch.Tensor,
    noise: float,
    draws: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(8_300_003 + seed)
    residual_noise = torch.randn(
        (draws, clean.numel()), generator=generator, dtype=torch.float64
    )
    targets = clean.reshape(1, -1) + noise * residual_noise
    standardized = (prediction.reshape(1, -1) - targets) / noise
    constant = clean.numel() * math.log(2.0 * math.pi * noise * noise)
    return torch.sum(standardized * standardized, dim=1) + constant


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (scenario, candidate), group in frame.groupby(
        ["scenario", "candidate"], sort=True
    ):
        empirical = group["empirical_optimism"].to_numpy(float)
        predicted = group["predicted_optimism"].to_numpy(float)
        residual = empirical - predicted
        n = len(group)
        rows.append({
            "scenario": scenario,
            "candidate": candidate,
            "n_fits": n,
            "new_response_draws_per_fit": int(group["new_response_draws"].iloc[0]),
            "mean_empirical_optimism": float(np.mean(empirical)),
            "se_empirical_optimism": float(np.std(empirical, ddof=1) / math.sqrt(n)),
            "mean_predicted_optimism": float(np.mean(predicted)),
            "mean_calibration_residual": float(np.mean(residual)),
            "residual_ci_low": float(
                np.mean(residual) - 1.959963984540054 * np.std(residual, ddof=1) / math.sqrt(n)
            ),
            "residual_ci_high": float(
                np.mean(residual) + 1.959963984540054 * np.std(residual, ddof=1) / math.sqrt(n)
            ),
            "mean_ratio_empirical_to_predicted": float(
                np.mean(empirical) / np.mean(predicted)
            ) if abs(np.mean(predicted)) > 1e-12 else math.nan,
        })
    return pd.DataFrame(rows)


def run(draws: int, seed_limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = ROOT / "results" / "records" / "submission" / "core" / "raw"
    paths = sorted(raw.glob("*.json"))
    rows: list[dict] = []
    started = time.perf_counter()
    eligible = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "ok" and int(record["seed"]) < seed_limit:
            eligible.append(record)
    for index, record in enumerate(eligible, start=1):
        scenario = SCENARIO_LOOKUP[record["scenario"]]
        candidate = CANDIDATES[record["candidate"]]
        theta = torch.tensor(record["theta"], dtype=torch.float64)
        times, clean, _ = _truth_data(scenario, int(record["seed"]))
        with torch.no_grad():
            prediction = candidate.observe(theta, times, scenario.observation)
            test_deviance = _deviance_draws(
                prediction,
                clean,
                float(scenario.noise),
                draws,
                seed=97_003 * int(record["seed"]) + candidate.dimension,
            )
        empirical = float(torch.mean(test_deviance)) - float(record["deviance"])
        predicted = 2.0 * float(
            record.get("effective_dimension", record.get("d_obs"))
        )
        rows.append({
            "scenario": scenario.name,
            "truth": record["truth"],
            "candidate": record["candidate"],
            "seed": int(record["seed"]),
            "training_deviance": float(record["deviance"]),
            "mean_new_deviance": float(torch.mean(test_deviance)),
            "mc_se_new_deviance": float(
                torch.std(test_deviance, correction=1) / math.sqrt(draws)
            ),
            "empirical_optimism": empirical,
            "predicted_optimism": predicted,
            "calibration_residual": empirical - predicted,
            "effective_dimension": predicted / 2.0,
            "new_response_draws": draws,
        })
        if index % 100 == 0 or index == len(eligible):
            elapsed = time.perf_counter() - started
            rate = index / max(elapsed, 1e-9)
            remaining = (len(eligible) - index) / max(rate, 1e-9)
            print(
                f"optimism {index}/{len(eligible)} "
                f"({100.0 * index / len(eligible):.1f}%); "
                f"elapsed={elapsed:.1f}s; eta={remaining:.1f}s",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    return frame, _summarize(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed-limit", type=int, default=80)
    args = parser.parse_args()
    if args.draws < 2:
        raise ValueError("--draws must be at least 2")
    frame, summary = run(args.draws, args.seed_limit)
    output = ROOT / "results" / "summary" / "first_principles"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "optimism_calibration.csv", index=False)
    summary.to_csv(output / "optimism_calibration_summary.csv", index=False)
    print(f"wrote {len(frame)} fit-level rows and {len(summary)} summaries")


if __name__ == "__main__":
    main()
