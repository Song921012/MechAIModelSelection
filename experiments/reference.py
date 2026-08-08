"""Reference-geometry and resolution sensitivity from fitted core models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

from mechai_model_selection import block_reference_metric, geometry_sensitivity_grid  # noqa: E402
from mechai_experiments.dynamics import CANDIDATES  # noqa: E402


RHO_GRID = (1.0, 2.0, 4.0, 8.0, 16.0)
LAMBDA_GRID = tuple(float(value) for value in torch.logspace(-2, 2, 9))
GAMMA_GRID = (0.0, 0.25, 0.5, 1.0)
MECHANISTIC_DIMENSION = {
    "sir": 2, "seir": 3, "tv_sir": 4, "ude_sir_h2": 2, "neural_ode_h2": 0,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("submission", "legacy"), default="submission")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()
    raw = ROOT / "results" / "records" / ("submission/core/raw" if args.source == "submission" else "formal/raw")
    rows = []
    for path in sorted(raw.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "ok":
            continue
        if args.seeds is not None and int(record.get("seed", -1)) >= args.seeds:
            continue
        candidate_key = record["candidate"]
        candidate = CANDIDATES[candidate_key]
        if "information" in record:
            information = torch.tensor(record["information"], dtype=torch.float64)
        else:
            continue
        mechanistic = MECHANISTIC_DIMENSION[candidate_key]
        references = []
        for rho in RHO_GRID:
            if mechanistic == 0:
                references.append(rho * torch.eye(candidate.dimension, dtype=torch.float64))
            elif mechanistic == candidate.dimension:
                references.append(torch.eye(candidate.dimension, dtype=torch.float64))
            else:
                references.append(block_reference_metric(
                    (mechanistic, candidate.dimension - mechanistic), (1.0, rho)
                ))
        grid = geometry_sensitivity_grid(
            information, references, torch.tensor(LAMBDA_GRID, dtype=torch.float64)
        )
        for rho_index, rho in enumerate(RHO_GRID):
            for lambda_index, resolution in enumerate(LAMBDA_GRID):
                d_obs = float(grid["dimension"][rho_index, lambda_index])
                c_obs = float(grid["complexity"][rho_index, lambda_index])
                for gamma in GAMMA_GRID:
                    rows.append({
                        "scenario": record["scenario"], "seed": record["seed"],
                        "candidate": candidate_key, "truth": record["truth"],
                        "rho_nn": rho, "resolution": resolution, "gamma": gamma,
                        "d_obs": d_obs, "c_obs": c_obs,
                        "score": record["deviance"] + math.log(record["n_observations"]) * d_obs + gamma * c_obs,
                    })
    frame = pd.DataFrame(rows)
    output = ROOT / "results" / "summary" / "numerical"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "reference_sensitivity.csv", index=False)
    selections = []
    for keys, group in frame.groupby(["scenario", "seed", "rho_nn", "resolution", "gamma"]):
        winner = group.loc[group["score"].idxmin()]
        selections.append({
            "scenario": keys[0], "seed": keys[1], "rho_nn": keys[2],
            "resolution": keys[3], "gamma": keys[4],
            "selected": winner["candidate"], "truth": winner["truth"],
            "correct": int(winner["candidate"] == winner["truth"]),
        })
    selection = pd.DataFrame(selections)
    selection.to_csv(output / "reference_sensitivity_selections.csv", index=False)
    selection.groupby(["scenario", "rho_nn", "resolution", "gamma"])["correct"].agg(["mean", "count"]).reset_index().to_csv(
        output / "reference_sensitivity_summary.csv", index=False
    )


if __name__ == "__main__":
    main()



