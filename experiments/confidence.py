"""Repeated-sampling coverage for rank-aware geometric confidence regions."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
import torch

ROOT = Path(__file__).resolve().parents[1]

from mechai_model_selection import coverage_summary, generalized_spectrum, quotient_rank
from mechai_experiments.config import SCENARIOS, TRUE_PARAMETERS
from mechai_experiments.dynamics import CANDIDATES
from mechai_experiments.fitting import gaussian_deviance, information_matrix
from mechai_experiments.records import SCHEMA_VERSION, protocol_hash


torch.set_default_dtype(torch.float64)
SCENARIO_NAMES = ("regular_sir_full", "early_seir_infected_only")
LEVELS = (0.50, 0.80, 0.90, 0.95)


def _protocol_digest(scenario_name: str) -> str:
    scenario = next(item for item in SCENARIOS if item.name == scenario_name)
    return protocol_hash({
        "study": "confidence",
        "scenario": scenario_name,
        "truth": scenario.truth,
        "observation": scenario.observation,
        "horizon": scenario.horizon,
        "n_times": scenario.n_times,
        "noise": scenario.noise,
        "optimizer": {"starts": 1, "adam_steps": 90, "lbfgs_steps": 22},
        "rank_relative_tolerance": 1e-4,
        "rank_absolute_tolerance": 1e-8,
    })


def _with_record_metadata(record: dict) -> dict:
    expected = _protocol_digest(record["scenario"])
    existing_schema = record.get("schema_version")
    existing_hash = record.get("protocol_hash")
    if existing_schema not in (None, SCHEMA_VERSION):
        raise RuntimeError(f"confidence record schema mismatch: {existing_schema}")
    if existing_hash not in (None, expected):
        raise RuntimeError("confidence record protocol mismatch")
    return {"schema_version": SCHEMA_VERSION, "protocol_hash": expected, **record}


def _fit_likelihood(candidate, times, target, observation, noise, seed):
    generator = torch.Generator().manual_seed(91_003 + 7_919 * seed + candidate.dimension)
    best = None
    for scale in (0.20,):
        theta = (
            candidate.prior_mean
            + scale * torch.randn(candidate.dimension, generator=generator)
        ).clone().requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=0.025)
        for _ in range(90):
            optimizer.zero_grad()
            loss = 0.5 * gaussian_deviance(
                candidate.observe(theta, times, observation), target, noise
            )
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], 20.0)
            optimizer.step()
        optimizer2 = torch.optim.LBFGS(
            [theta],
            lr=0.5,
            max_iter=22,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer2.zero_grad()
            value = 0.5 * gaussian_deviance(
                candidate.observe(theta, times, observation), target, noise
            )
            value.backward()
            return value

        try:
            optimizer2.step(closure)
        except (RuntimeError, AssertionError):
            pass
        fitted = theta.detach()
        objective = float(
            0.5
            * gaussian_deviance(
                candidate.observe(fitted, times, observation), target, noise
            )
        )
        if best is None or objective < best[0]:
            best = (objective, fitted)
    return best[1]


def _cell(scenario_name: str, seed: int, calibration: bool) -> dict:
    torch.set_num_threads(1)
    scenario = next(item for item in SCENARIOS if item.name == scenario_name)
    candidate = CANDIDATES[scenario.truth]
    truth = TRUE_PARAMETERS[scenario.truth]
    times = torch.linspace(0.0, scenario.horizon, scenario.n_times)
    clean = candidate.observe(truth, times, scenario.observation)
    generator_seed = (5_000_003 if calibration else 1_000_003) + seed * 10_007
    generator = torch.Generator().manual_seed(generator_seed)
    target = clean + scenario.noise * torch.randn(
        clean.shape, generator=generator, dtype=torch.float64
    )
    begin = time.perf_counter()
    estimate = _fit_likelihood(
        candidate, times, target, scenario.observation, scenario.noise, generator_seed
    )
    information, _ = information_matrix(
        candidate, estimate, times, scenario.observation, scenario.noise
    )
    reference = candidate.prior_precision
    rank = quotient_rank(
        information,
        reference,
        relative_tolerance=1e-4,
        absolute_tolerance=1e-8,
    )
    generalized = generalized_spectrum(information, reference)
    positive = generalized[generalized > max(1e-8, 1e-4 * float(generalized.max()))]
    delta = estimate - truth
    statistic = float(delta @ information @ delta)
    return _with_record_metadata({
        "scenario": scenario_name,
        "seed": seed,
        "sample": "calibration" if calibration else "evaluation",
        "dimension": candidate.dimension,
        "rank": rank,
        "statistic": statistic,
        "log_information_volume": float(torch.sum(torch.log(positive))) if rank else math.nan,
        "objective": float(
            0.5
            * gaussian_deviance(
                candidate.observe(estimate, times, scenario.observation),
                target,
                scenario.noise,
            )
        ),
        "parameter_error": float(torch.linalg.vector_norm(delta)),
        "wall_seconds": time.perf_counter() - begin,
        "estimate": estimate.tolist(),
    })


def _write_outputs(records: list[dict]) -> None:
    records = [_with_record_metadata(record) for record in records]
    keys = [(row["scenario"], row["sample"], int(row["seed"])) for row in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate confidence scenario-sample-seed key")
    if any(not math.isfinite(float(row["statistic"])) for row in records):
        raise RuntimeError("nonfinite confidence statistic")
    scalar = [{key: value for key, value in row.items() if key != "estimate"} for row in records]
    frame = pd.DataFrame(scalar)
    frame.to_csv(ROOT / "results" / "summary" / "confidence_study.csv", index=False)
    rows = []
    for scenario_name in SCENARIO_NAMES:
        calibration = frame[
            (frame["scenario"] == scenario_name) & (frame["sample"] == "calibration")
        ]
        evaluation = frame[
            (frame["scenario"] == scenario_name) & (frame["sample"] == "evaluation")
        ]
        modal_rank = int(evaluation["rank"].mode().iloc[0])
        dimension = int(evaluation["dimension"].iloc[0])
        for level in LEVELS:
            thresholds = {
                "naive_wald": float(chi2.ppf(level, dimension)),
                "geometric_quotient": float(chi2.ppf(level, modal_rank)),
                "simulation_calibrated": float(calibration["statistic"].quantile(level)),
            }
            for method, threshold in thresholds.items():
                covered = torch.tensor(
                    (evaluation["statistic"].to_numpy() <= threshold).tolist()
                )
                if method == "naive_wald":
                    rank_for_volume = dimension
                else:
                    rank_for_volume = modal_rank
                if method == "naive_wald" and modal_rank < dimension:
                    # The raw-coordinate set is cylindrical along null
                    # directions, so a finite determinant width is undefined.
                    log_width = math.inf
                else:
                    log_width = (
                        0.5 * rank_for_volume * math.log(max(threshold, 1e-14))
                        - 0.5 * evaluation["log_information_volume"].mean()
                    )
                summary = coverage_summary(covered)
                rows.append(
                    {
                        "scenario": scenario_name,
                        "method": method,
                        "nominal": level,
                        "rank": rank_for_volume,
                        "threshold": threshold,
                        "coverage": summary["coverage"],
                        "coverage_ci_low": summary["wilson_lower"],
                        "coverage_ci_high": summary["wilson_upper"],
                        "mean_log_relative_width": log_width,
                        "n": summary["n"],
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "results" / "summary" / "confidence_coverage_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True)
    labels = {
        "naive_wald": "Wald (raw dimension)",
        "geometric_quotient": "Geometric quotient",
        "simulation_calibrated": "Simulation calibrated",
    }
    markers = {"naive_wald": "s", "geometric_quotient": "o", "simulation_calibrated": "^"}
    titles = {"regular_sir_full": "Regular full observation", "early_seir_infected_only": "Early partial observation"}
    for axis, scenario_name in zip(axes, SCENARIO_NAMES):
        subset = summary[summary["scenario"] == scenario_name]
        for method in labels:
            values = subset[subset["method"] == method].sort_values("nominal")
            axis.plot(
                values["nominal"],
                values["coverage"],
                marker=markers[method],
                label=labels[method],
            )
            axis.fill_between(
                values["nominal"],
                values["coverage_ci_low"],
                values["coverage_ci_high"],
                alpha=0.10,
            )
        axis.plot([0.45, 1.0], [0.45, 1.0], "k--", linewidth=1)
        axis.set_xlim(0.47, 0.98)
        axis.set_ylim(0.35, 1.01)
        axis.set_title(titles[scenario_name])
        axis.set_xlabel("Nominal level")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(ROOT / "figures" / "confidence_coverage.pdf", bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "evaluation_replicates_per_scenario": int(
            frame[frame["sample"] == "evaluation"].groupby("scenario").size().min()
        ),
        "calibration_replicates_per_scenario": int(
            frame[frame["sample"] == "calibration"].groupby("scenario").size().min()
        ),
        "rank_relative_tolerance": 1e-4,
        "levels": list(LEVELS),
    }
    (ROOT / "results" / "records" / "confidence").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "records" / "confidence" / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    tasks = [
        (scenario_name, seed, calibration)
        for scenario_name in SCENARIO_NAMES
        for calibration in (True, False)
        for seed in range(args.replicates)
    ]
    records = []
    pending = []
    raw_directory = ROOT / "results" / "records" / "confidence" / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    for scenario_name, seed, calibration in tasks:
        sample = "calibration" if calibration else "evaluation"
        raw_path = raw_directory / f"{scenario_name}__{sample}__seed_{seed:03d}.json"
        if raw_path.exists() and not args.overwrite:
            record = _with_record_metadata(json.loads(raw_path.read_text(encoding="utf-8")))
            raw_path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
            records.append(record)
        else:
            pending.append((scenario_name, seed, calibration))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_cell, scenario_name, seed, calibration): (
                scenario_name,
                seed,
                calibration,
            )
            for scenario_name, seed, calibration in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            raw_path = raw_directory / (
                f"{record['scenario']}__{record['sample']}__"
                f"seed_{int(record['seed']):03d}.json"
            )
            raw_path.write_text(
                json.dumps(record, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            if index % 20 == 0:
                print(f"completed {index}/{len(pending)} new records", flush=True)
    _write_outputs(records)


if __name__ == "__main__":
    main()


