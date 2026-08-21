"""Loss-metric matching experiment for positive pulse responses."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = ROOT.parent / "mechai-model-selection" / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from mechai_model_selection import generalized_optimism


torch.set_default_dtype(torch.float64)
GRID = torch.linspace(0.0, 1.0, 61)
PROBABILITIES = torch.linspace(0.05, 0.95, 41)
WIDTH = 0.12
PRIOR_MEAN = torch.tensor([0.0, 0.5])
REFERENCE = torch.diag(torch.tensor([1.0, 4.0]))
RAW_NOISE = 0.03
QUANTILE_NOISE = 0.006


def raw_response(theta: torch.Tensor) -> torch.Tensor:
    log_amplitude, location = theta
    return torch.exp(log_amplitude) * torch.exp(
        -0.5 * ((GRID - location) / WIDTH) ** 2
    )


def transport_quantiles(theta: torch.Tensor) -> torch.Tensor:
    _, location = theta
    standard_normal = math.sqrt(2.0) * torch.erfinv(2.0 * PROBABILITIES - 1.0)
    return location + WIDTH * standard_normal


FEATURES = {
    "fisher_trajectory": (raw_response, RAW_NOISE),
    "wasserstein_quantile": (transport_quantiles, QUANTILE_NOISE),
}
TRUTHS = {
    "amplitude_change": torch.tensor([math.log(1.35), 0.5]),
    "time_shift": torch.tensor([0.0, 0.62]),
}


def deviance(
    feature,
    theta: torch.Tensor,
    target: torch.Tensor,
    noise: float,
) -> torch.Tensor:
    residual = (feature(theta) - target) / noise
    return torch.sum(residual * residual) + target.numel() * math.log(
        2.0 * math.pi * noise * noise
    )


def fit_map(feature, target: torch.Tensor, noise: float) -> torch.Tensor:
    theta = PRIOR_MEAN.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [theta],
        lr=0.8,
        max_iter=40,
        history_size=20,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        delta = theta - PRIOR_MEAN
        value = 0.5 * (
            deviance(feature, theta, target, noise)
            + delta @ REFERENCE @ delta
        )
        value.backward()
        return value

    optimizer.step(closure)
    return theta.detach()


def pullback(feature, theta: torch.Tensor, noise: float) -> torch.Tensor:
    local = theta.clone().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: feature(value) / noise, local, vectorize=True
    )
    matrix = jacobian.mT @ jacobian
    return 0.5 * (matrix + matrix.mT)


def run(seeds: int, new_draws: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    geometry_rows = []
    started = time.perf_counter()
    total = len(TRUTHS) * len(FEATURES) * seeds
    completed = 0
    for truth_name, theta_true in TRUTHS.items():
        for loss_name, (feature, noise) in FEATURES.items():
            clean = feature(theta_true)
            for seed in range(seeds):
                generator = torch.Generator().manual_seed(
                    4_100_003 + 10_007 * seed
                    + (0 if truth_name == "amplitude_change" else 1_000_000)
                    + (0 if loss_name == "fisher_trajectory" else 2_000_000)
                )
                training = clean + noise * torch.randn(
                    clean.shape, generator=generator
                )
                theta_hat = fit_map(feature, training, noise)
                training_deviance = float(
                    deviance(feature, theta_hat, training, noise)
                )
                new_targets = clean.reshape(1, -1) + noise * torch.randn(
                    (new_draws, clean.numel()), generator=generator
                )
                predictions = feature(theta_hat).reshape(1, -1)
                residual = (predictions - new_targets) / noise
                constant = clean.numel() * math.log(
                    2.0 * math.pi * noise * noise
                )
                new_deviances = torch.sum(residual * residual, dim=1) + constant
                empirical = float(torch.mean(new_deviances)) - training_deviance

                fisher_g = pullback(
                    raw_response, theta_hat, RAW_NOISE
                )
                wasserstein_g = pullback(
                    transport_quantiles, theta_hat, QUANTILE_NOISE
                )
                metric_matrices = {
                    "fisher_trajectory": fisher_g,
                    "wasserstein_quantile": wasserstein_g,
                }
                matched_g = metric_matrices[loss_name]
                other_name = (
                    "wasserstein_quantile"
                    if loss_name == "fisher_trajectory"
                    else "fisher_trajectory"
                )
                other_g = metric_matrices[other_name]
                matched_df = generalized_optimism(
                    matched_g, matched_g, REFERENCE, 1.0
                )
                mismatched_df = generalized_optimism(
                    other_g, other_g, REFERENCE, 1.0
                )
                rows.append({
                    "truth": truth_name,
                    "loss": loss_name,
                    "seed": seed,
                    "theta_hat_log_amplitude": float(theta_hat[0]),
                    "theta_hat_location": float(theta_hat[1]),
                    "training_deviance": training_deviance,
                    "mean_new_deviance": float(torch.mean(new_deviances)),
                    "empirical_optimism": empirical,
                    "matched_metric": loss_name,
                    "mismatched_metric": other_name,
                    "matched_predicted_optimism": 2.0 * matched_df,
                    "mismatched_predicted_optimism": 2.0 * mismatched_df,
                    "matched_residual": empirical - 2.0 * matched_df,
                    "mismatched_residual": empirical - 2.0 * mismatched_df,
                    "new_response_draws": new_draws,
                })
                if seed == 0:
                    for metric_name, matrix in metric_matrices.items():
                        eigenvalues = torch.linalg.eigvalsh(matrix).clamp_min(0.0)
                        geometry_rows.append({
                            "truth": truth_name,
                            "loss": loss_name,
                            "metric": metric_name,
                            "amplitude_information": float(matrix[0, 0]),
                            "location_information": float(matrix[1, 1]),
                            "cross_information": float(matrix[0, 1]),
                            "minimum_eigenvalue": float(eigenvalues[0]),
                            "maximum_eigenvalue": float(eigenvalues[-1]),
                        })
                completed += 1
                if completed % 100 == 0 or completed == total:
                    elapsed = time.perf_counter() - started
                    print(
                        f"metric-loss {completed}/{total} "
                        f"({100.0 * completed / total:.1f}%); "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
    frame = pd.DataFrame(rows)
    summaries = []
    for (truth, loss), group in frame.groupby(["truth", "loss"], sort=True):
        n = len(group)
        for metric_type, residual_column, prediction_column in (
            ("matched", "matched_residual", "matched_predicted_optimism"),
            ("mismatched", "mismatched_residual", "mismatched_predicted_optimism"),
        ):
            residual = group[residual_column].to_numpy(float)
            summaries.append({
                "truth": truth,
                "loss": loss,
                "metric_relation": metric_type,
                "n_fits": n,
                "new_response_draws_per_fit": new_draws,
                "mean_empirical_optimism": float(group["empirical_optimism"].mean()),
                "mean_predicted_optimism": float(group[prediction_column].mean()),
                "mean_calibration_residual": float(np.mean(residual)),
                "residual_ci_low": float(
                    np.mean(residual)
                    - 1.959963984540054 * np.std(residual, ddof=1) / math.sqrt(n)
                ),
                "residual_ci_high": float(
                    np.mean(residual)
                    + 1.959963984540054 * np.std(residual, ddof=1) / math.sqrt(n)
                ),
                "mean_absolute_residual": float(np.mean(np.abs(residual))),
            })
    return frame, pd.DataFrame(summaries), pd.DataFrame(geometry_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--new-draws", type=int, default=200)
    args = parser.parse_args()
    frame, summary, geometry = run(args.seeds, args.new_draws)
    output = ROOT / "results" / "summary" / "first_principles"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "metric_loss_matching.csv", index=False)
    summary.to_csv(output / "metric_loss_matching_summary.csv", index=False)
    geometry.to_csv(output / "metric_loss_geometry.csv", index=False)
    print(f"wrote {len(frame)} fit-level rows and {len(summary)} summaries")


if __name__ == "__main__":
    main()
