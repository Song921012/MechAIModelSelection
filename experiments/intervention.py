"""Multi-initial-condition study that separates state feedback from time forcing."""

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

from mechai_experiments.config import DEFAULT_CANDIDATES, RESOLUTION_GRID, TRUE_PARAMETERS
from mechai_experiments.dynamics import CANDIDATES, Candidate
from mechai_experiments.fitting import fit_map, gaussian_deviance, information_matrix, laplace_loglik_draws
from scoring import score_fit


DTYPE = torch.float64
TRAIN_SIR_STATES = (
    torch.tensor([0.995, 0.005, 0.0], dtype=DTYPE),
    torch.tensor([0.985, 0.015, 0.0], dtype=DTYPE),
    torch.tensor([0.960, 0.040, 0.0], dtype=DTYPE),
)
HELD_OUT_SIR_STATE = torch.tensor([0.975, 0.025, 0.0], dtype=DTYPE)
CRITERIA = (
    "aic", "aicc", "bic", "intervention_validation", "waic_laplace",
    "ogic_p", "gic_eff", "gic_vol_025", "gic_vol_050", "gic_vol_100", "ogic_e",
)


def _state_for(candidate: Candidate, sir_state: torch.Tensor) -> torch.Tensor:
    if candidate.state_kind == "sir":
        return sir_state
    susceptible, infected, removed = sir_state
    exposed = torch.tensor(0.005, dtype=DTYPE)
    return torch.stack((susceptible - exposed, exposed, infected, removed))


@dataclass(frozen=True)
class MultiTrajectoryCandidate:
    base: Candidate
    initial_states: tuple[torch.Tensor, ...]

    @property
    def name(self):
        return self.base.name

    @property
    def dimension(self):
        return self.base.dimension

    @property
    def prior_mean(self):
        return self.base.prior_mean

    @property
    def prior_precision(self):
        return self.base.prior_precision

    def observe(self, theta: torch.Tensor, times: torch.Tensor, observation: str) -> torch.Tensor:
        trajectories = [
            self.base.observe(theta, times, observation, _state_for(self.base, state))
            for state in self.initial_states
        ]
        return torch.cat(trajectories, dim=0)


def _data(seed: int, times: torch.Tensor, noise: float):
    truth = CANDIDATES["ude_sir_h2"]
    wrapped = MultiTrajectoryCandidate(truth, TRAIN_SIR_STATES)
    clean = wrapped.observe(TRUE_PARAMETERS["ude_sir_h2"], times, "infected")
    generator = torch.Generator().manual_seed(910_003 + seed)
    noisy = clean + noise * torch.randn(clean.shape, generator=generator, dtype=DTYPE)
    return clean, noisy


def _held_out_score(candidate: Candidate, theta: torch.Tensor, times: torch.Tensor, noise: float, seed: int):
    truth = CANDIDATES["ude_sir_h2"]
    truth_state = _state_for(truth, HELD_OUT_SIR_STATE)
    candidate_state = _state_for(candidate, HELD_OUT_SIR_STATE)
    clean = truth.observe(TRUE_PARAMETERS["ude_sir_h2"], times, "infected", truth_state)
    generator = torch.Generator().manual_seed(920_003 + seed)
    noisy = clean + noise * torch.randn(clean.shape, generator=generator, dtype=DTYPE)
    prediction = candidate.observe(theta, times, "infected", candidate_state)
    return float(gaussian_deviance(prediction, noisy, noise)), float(torch.mean((prediction - clean) ** 2))


def run_cell(root: Path, key: str, seed: int, args):
    torch.set_num_threads(1)
    raw_dir = root / "results" / "intervention" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{key}__seed{seed:03d}.json"
    if path.exists() and not args.overwrite:
        return json.loads(path.read_text(encoding="utf-8"))

    base = CANDIDATES[key]
    candidate = MultiTrajectoryCandidate(base, TRAIN_SIR_STATES)
    times = torch.linspace(0.0, 18.0, 37, dtype=DTYPE)
    noise = 0.015
    _, noisy = _data(seed, times, noise)
    try:
        fit = fit_map(
            candidate, times, noisy, "infected", noise, seed=seed,
            starts=args.starts, adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps,
        )
        information, jacobian = information_matrix(candidate, fit.theta, times, "infected", noise)
        draws = laplace_loglik_draws(
            candidate, fit, information, times, noisy, "infected", noise,
            draws=args.posterior_draws, seed=seed,
        )
        scores, profile = score_fit(candidate, fit, information, draws, noisy.numel(), RESOLUTION_GRID)
        validation, held_out_mse = _held_out_score(base, fit.theta, times, noise, seed)
        record = {
            "status": "ok", "candidate": key, "candidate_label": base.name,
            "truth": "ude_sir_h2", "seed": seed, "dimension": base.dimension,
            "deviance": fit.deviance, "intervention_validation": validation,
            "held_out_mse": held_out_mse, "jacobian_rank": int(torch.linalg.matrix_rank(jacobian, rtol=1e-7)),
            "wall_seconds": fit.wall_seconds, "theta": fit.theta.tolist(),
            "generalized_eigenvalues": profile["eigenvalues"].tolist(), **scores,
        }
    except Exception as exc:
        record = {
            "status": "failed", "candidate": key, "seed": seed,
            "error": repr(exc), "traceback": traceback.format_exc(),
        }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _wilson(successes: pd.Series):
    n = float(len(successes)); p = float(successes.mean()); z = 1.959963984540054
    denominator = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def aggregate(root: Path):
    paths = sorted((root / "results" / "intervention" / "raw").glob("*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    ok = [record for record in records if record["status"] == "ok"]
    frame = pd.DataFrame([
        {key: value for key, value in record.items() if key not in {"theta", "traceback", "generalized_eigenvalues"}}
        for record in ok
    ])
    log_n = np.log(3 * 37)
    frame["gic_eff"] = frame["deviance"] + log_n * frame["d_obs"]
    for suffix, weight in (("025", 0.25), ("050", 0.50), ("100", 1.00)):
        frame[f"gic_vol_{suffix}"] = frame["gic_eff"] + weight * frame["c_obs"]
    table_dir = root / "tables"; table_dir.mkdir(exist_ok=True)
    frame.to_csv(table_dir / "intervention_model_scores.csv", index=False)
    selections = []
    for seed, group in frame.groupby("seed"):
        for criterion in CRITERIA:
            finite = group[np.isfinite(pd.to_numeric(group[criterion], errors="coerce"))]
            if finite.empty:
                continue
            winner = finite.loc[pd.to_numeric(finite[criterion]).idxmin()]
            selections.append({
                "seed": seed, "criterion": criterion, "selected": winner.candidate,
                "correct": int(winner.candidate == "ude_sir_h2"),
            })
    selected = pd.DataFrame(selections)
    selected.to_csv(table_dir / "intervention_selections.csv", index=False)
    summary_rows = []
    for criterion, group in selected.groupby("criterion"):
        low, high = _wilson(group.correct)
        summary_rows.append({
            "criterion": criterion, "recovery_rate": group.correct.mean(), "n": len(group),
            "recovery_ci_low": low, "recovery_ci_high": high,
            "modal_selection": group.selected.value_counts().index[0],
            "modal_frequency": group.selected.value_counts().iloc[0] / len(group),
        })
    pd.DataFrame(summary_rows).to_csv(table_dir / "intervention_selection_summary.csv", index=False)
    metadata = {"cells": len(records), "ok": len(ok), "failed": len(records) - len(ok)}
    (root / "results" / "intervention" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--adam-steps", type=int, default=160)
    parser.add_argument("--lbfgs-steps", type=int, default=30)
    parser.add_argument("--posterior-draws", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tasks = [(key, seed) for seed in range(args.seeds) for key in DEFAULT_CANDIDATES]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_cell, root, key, seed, args): (key, seed) for key, seed in tasks}
        for future in as_completed(futures):
            key, seed = futures[future]
            result = future.result()
            print(key, seed, result["status"], result.get("ogic_e", result.get("error", "")), flush=True)
    aggregate(root)


if __name__ == "__main__":
    main()


