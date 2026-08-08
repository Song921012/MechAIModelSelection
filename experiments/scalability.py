"""Cost of Jacobian construction and generalized spectral analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

from mechai_model_selection import generalized_spectrum, residual_jacobian  # noqa: E402


def benchmark_dimension(dimension: int, repeats: int, observations: int = 240) -> list[dict]:
    generator = torch.Generator().manual_seed(7919 + dimension)
    design = torch.randn(observations, dimension, generator=generator, dtype=torch.float64) / dimension**0.5
    theta = torch.randn(dimension, generator=generator, dtype=torch.float64) / dimension**0.5
    reference = torch.diag(torch.linspace(1.0, 4.0, dimension, dtype=torch.float64))

    def solution_map(value):
        linear = design @ value
        return linear + 0.1 * torch.sin(linear)

    rows = []
    for repeat in range(repeats + 3):
        begin = time.perf_counter()
        jacobian = residual_jacobian(solution_map, theta)
        jacobian_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        information = jacobian.mT @ jacobian
        gram_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        spectrum = generalized_spectrum(information, reference)
        spectrum_seconds = time.perf_counter() - begin
        if repeat >= 3:
            rows.append({
                "dimension": dimension, "observations": observations,
                "repeat": repeat - 3, "jacobian_seconds": jacobian_seconds,
                "gram_seconds": gram_seconds, "spectrum_seconds": spectrum_seconds,
                "tensor_megabytes": 8.0 * (jacobian.numel() + information.numel() + spectrum.numel()) / 2**20,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "submission"), default="smoke")
    args = parser.parse_args()
    dimensions = (10, 50) if args.profile == "smoke" else (10, 20, 50, 100, 200)
    repeats = 3 if args.profile == "smoke" else 20
    rows = []
    for dimension in dimensions:
        rows.extend(benchmark_dimension(dimension, repeats))
        print("dimension", dimension, "complete", flush=True)
    output = ROOT / "results" / "summary" / "numerical"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "scalability_timings.csv", index=False)
    frame.groupby("dimension").agg(
        jacobian_median=("jacobian_seconds", "median"),
        gram_median=("gram_seconds", "median"),
        spectrum_median=("spectrum_seconds", "median"),
        tensor_megabytes=("tensor_megabytes", "first"),
    ).reset_index().to_csv(output / "scalability_summary.csv", index=False)


if __name__ == "__main__":
    main()


