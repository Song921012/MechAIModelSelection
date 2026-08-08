"""Controlled Fisher-versus-Wasserstein pullback boundary experiment."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

from mechai_model_selection import fisher_pullback, wasserstein_pullback_1d


torch.set_default_dtype(torch.float64)


def _positive_curve(theta: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    log_amplitude, shift = theta
    return torch.exp(log_amplitude) * torch.exp(-0.5 * ((grid - shift) / 0.13) ** 2)


def _normalized_quantile(theta: torch.Tensor, grid: torch.Tensor, probabilities: torch.Tensor):
    density = _positive_curve(theta, grid)
    density = density / torch.trapz(density, grid)
    cdf = torch.cumsum(density, dim=0)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    # Smooth inverse-CDF approximation using narrow soft assignments.
    distances = (cdf.unsqueeze(0) - probabilities.unsqueeze(1)) ** 2
    assignments = torch.softmax(-distances / 2e-5, dim=1)
    return assignments @ grid


def main() -> None:
    grid = torch.linspace(0.0, 1.0, 301)
    probabilities = torch.linspace(0.01, 0.99, 99)
    theta = torch.tensor([0.0, 0.5])
    fisher, _ = fisher_pullback(
        lambda value: _positive_curve(value, grid),
        theta,
        observation_precision=torch.eye(grid.numel()),
    )
    wasserstein, _ = wasserstein_pullback_1d(
        lambda value: _normalized_quantile(value, grid, probabilities),
        theta,
    )
    rows = []
    for metric_name, matrix in (("Fisher trajectory", fisher), ("Wasserstein shape", wasserstein)):
        eigenvalues = torch.linalg.eigvalsh(matrix).clamp_min(0)
        rows.append(
            {
                "metric": metric_name,
                "amplitude_information": float(matrix[0, 0]),
                "shift_information": float(matrix[1, 1]),
                "cross_information": float(matrix[0, 1]),
                "minimum_eigenvalue": float(eigenvalues[0]),
                "maximum_eigenvalue": float(eigenvalues[-1]),
            }
        )
    pd.DataFrame(rows).to_csv(ROOT / "results" / "summary" / "metric_boundary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    axes[0].plot(grid, _positive_curve(theta, grid), color="#1f77b4")
    axes[0].plot(grid, _positive_curve(theta + torch.tensor([0.25, 0.0]), grid), "--", label="Amplitude")
    axes[0].plot(grid, _positive_curve(theta + torch.tensor([0.0, 0.08]), grid), ":", label="Shift")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Positive response")
    axes[0].legend(frameon=False)
    values = pd.DataFrame(rows).set_index("metric")
    values[["amplitude_information", "shift_information"]].plot.bar(ax=axes[1], rot=0)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Pullback information")
    axes[1].legend(["Amplitude", "Shift"], frameon=False)
    figure.tight_layout()
    figure.savefig(ROOT / "figures" / "metric_boundary.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()


