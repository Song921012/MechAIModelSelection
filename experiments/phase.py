"""Controlled information phase diagram for a UDE-SIR data-generating system."""

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

from mechai_experiments.config import TRUE_PARAMETERS
from mechai_experiments.criteria import primary_scores
from mechai_experiments.dynamics import CANDIDATES
from mechai_experiments.records import finite_or_none, load_compatible, objective_gap, protocol_hash, result_path, write_record
from mechai_experiments.fitting import fit_map, information_matrix


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_KEYS = ("sir", "tv_sir", "ude_sir_h2", "neural_ode_h2")
N_OBSERVATIONS = (10, 20, 40)
NOISE_LEVELS = (0.005, 0.015, 0.045)
TRAJECTORY_COUNTS = (1, 2, 4)
INITIAL_INFECTED = {
    1: (0.015,), 2: (0.0075, 0.03), 4: (0.005, 0.015, 0.03, 0.06),
}


@dataclass(frozen=True)
class MultiTrajectoryCandidate:
    base: object
    initial_states: tuple[torch.Tensor, ...]

    @property
    def name(self): return self.base.name
    @property
    def dimension(self): return self.base.dimension
    @property
    def prior_mean(self): return self.base.prior_mean
    @property
    def prior_precision(self): return self.base.prior_precision

    def observe(self, theta, times, observation):
        return torch.cat([
            self.base.observe(theta, times, observation, initial_state=initial)
            for initial in self.initial_states
        ], dim=0)


def _initial_states(count: int) -> tuple[torch.Tensor, ...]:
    return tuple(torch.tensor([1.0 - infected, infected, 0.0], dtype=torch.float64)
                 for infected in INITIAL_INFECTED[count])


def run_cell(n_times: int, noise: float, trajectories: int, candidate_key: str, seed: int, profile: str, overwrite: bool, resume: bool = False) -> dict:
    torch.set_num_threads(1)
    settings = {
        "n_times": n_times, "noise": noise, "trajectories": trajectories,
        "candidate": candidate_key, "horizon": 18.0,
        "starts": 3 if profile == "submission" else 1,
        "adam_steps": 140 if profile == "submission" else 20,
        "lbfgs_steps": 25 if profile == "submission" else 4,
        "refine_steps": 220 if profile == "submission" else 8,
        "truth_parameters": TRUE_PARAMETERS["ude_sir_h2"].tolist(),
    }
    digest = protocol_hash(settings)
    label = f"n{n_times}__noise{noise:.3f}__traj{trajectories}__{candidate_key}__seed{seed:03d}"
    path = result_path(ROOT, "phase_diagram", label)
    previous = load_compatible(path, digest, overwrite, replace_incompatible=resume)
    if previous is not None:
        return previous
    times = torch.linspace(0.0, 18.0, n_times, dtype=torch.float64)
    initials = _initial_states(trajectories)
    truth = MultiTrajectoryCandidate(CANDIDATES["ude_sir_h2"], initials)
    candidate = MultiTrajectoryCandidate(CANDIDATES[candidate_key], initials)
    clean = truth.observe(TRUE_PARAMETERS["ude_sir_h2"], times, "infected")
    generator = torch.Generator().manual_seed(65537 * seed + n_times + trajectories)
    noisy = clean + noise * torch.randn(clean.shape, generator=generator)
    try:
        fit = fit_map(
            candidate, times, noisy, "infected", noise, seed=seed,
            starts=settings["starts"], adam_steps=settings["adam_steps"],
            lbfgs_steps=settings["lbfgs_steps"],
            refine_steps=settings["refine_steps"],
        )
        information, jacobian = information_matrix(candidate, fit.theta, times, "infected", noise)
        scores = primary_scores(candidate, fit, information, noisy.numel())
        record = {
            "status": "ok", "n_times": n_times, "noise": noise,
            "trajectories": trajectories, "truth": "ude_sir_h2",
            "candidate": candidate_key, "seed": seed, "dimension": candidate.dimension,
            "n_observations": noisy.numel(), "theta": fit.theta.tolist(),
            "objective": fit.objective, "deviance": fit.deviance,
            "gradient_norm": finite_or_none(fit.gradient_norm), "best_start": fit.best_start,
            "start_diagnostics": fit.start_diagnostics, "wall_seconds": fit.wall_seconds,
            "second_best_objective_gap": objective_gap(fit.start_diagnostics),
            "jacobian_rank": int(torch.linalg.matrix_rank(jacobian, rtol=1e-7)),
            "information": information.tolist(),
            **{key: finite_or_none(value) for key, value in scores.items()},
        }
    except Exception as exc:
        record = {
            "status": "failed", "n_times": n_times, "noise": noise,
            "trajectories": trajectories, "truth": "ude_sir_h2",
            "candidate": candidate_key, "seed": seed, "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    write_record(path, digest, record)
    return record


def aggregate() -> None:
    raw = ROOT / "results" / "records" / "submission" / "phase_diagram" / "raw"
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(raw.glob("*.json"))]
    excluded = {"theta", "information", "start_diagnostics", "traceback"}
    frame = pd.DataFrame([{key: value for key, value in record.items() if key not in excluded}
                          for record in records if record.get("status") == "ok"])
    output = ROOT / "results" / "summary" / "numerical"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "phase_diagram_scores.csv", index=False)
    selections = []
    for keys, group in frame.groupby(["n_times", "noise", "trajectories", "seed"]):
        for criterion in ("aic", "bic", "gic_eff", "gic_vol_050", "ogic_e"):
            finite = group[group[criterion].notna()]
            winner = finite.loc[finite[criterion].idxmin()]
            selections.append({
                "n_times": keys[0], "noise": keys[1], "trajectories": keys[2],
                "seed": keys[3], "criterion": criterion, "selected": winner["candidate"],
                "correct": int(winner["candidate"] == "ude_sir_h2"),
            })
    selection = pd.DataFrame(selections)
    selection.to_csv(output / "phase_diagram_selections.csv", index=False)
    summary = selection.groupby(["n_times", "noise", "trajectories", "criterion"])["correct"].agg(["mean", "count"]).reset_index()
    z = 1.959963984540054
    denominator = 1.0 + z**2 / summary["count"]
    center = (summary["mean"] + z**2 / (2.0 * summary["count"])) / denominator
    half = z * np.sqrt(summary["mean"] * (1.0 - summary["mean"]) / summary["count"] + z**2 / (4.0 * summary["count"]**2)) / denominator
    summary["ci_low"] = center - half
    summary["ci_high"] = center + half
    summary.to_csv(output / "phase_diagram_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "submission"), default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=None,
                        help="Queue the first N seeds without changing the per-fit protocol hash.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else (30 if args.profile == "submission" else 1)
    if seeds < 1 or seeds > (30 if args.profile == "submission" else 1):
        parser.error("--seeds must be between 1 and the selected profile maximum")
    grid = ((20, 0.015, 2),) if args.profile == "smoke" else [
        (n_times, noise, trajectories) for n_times in N_OBSERVATIONS
        for noise in NOISE_LEVELS for trajectories in TRAJECTORY_COUNTS
    ]
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    tasks = [(n_times, noise, trajectories, candidate, seed, args.profile, args.overwrite, args.resume)
             for n_times, noise, trajectories in grid for seed in range(seeds)
             for candidate in CANDIDATE_KEYS]
    if args.workers == 1:
        for task in tasks:
            result = run_cell(*task)
            print(task[:5], result["status"], flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_cell, *task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                print(result.get("n_times"), result.get("candidate"), result.get("seed"), result["status"], flush=True)
    aggregate()


if __name__ == "__main__":
    main()


