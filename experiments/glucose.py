"""External glucose minimal-model case using the public UDE data release."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

from mechai_model_selection import PullbackGeometry, criterion_weights, generalized_spectrum


torch.set_default_dtype(torch.float64)
DATA_ROOT = ROOT / "experiments" / "data" / "raw" / "ude-regularization" / "minimal-model" / "data"
MODEL_KEYS = ("minimal_fixed", "minimal_gamma", "ude_appearance", "neural_ode")


def _interpolate(times, values, query):
    indices = torch.searchsorted(times, query).clamp(1, len(times) - 1)
    left_t, right_t = times[indices - 1], times[indices]
    left_v, right_v = values[indices - 1], values[indices]
    fraction = (query - left_t) / (right_t - left_t)
    return left_v + fraction * (right_v - left_v)


def _rk4(rhs, x0, times, max_step=10.0):
    states = [x0]
    x = x0
    t = times[0]
    for target in times[1:]:
        steps = max(1, int(math.ceil(float(target - t) / max_step)))
        h = (target - t) / steps
        for _ in range(steps):
            k1 = rhs(t, x)
            k2 = rhs(t + h / 2, x + h * k1 / 2)
            k3 = rhs(t + h / 2, x + h * k2 / 2)
            k4 = rhs(t + h, x + h * k3)
            x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t = t + h
        states.append(x)
    return torch.stack(states)


def _mlp_time(theta, t):
    w1 = theta[:2]
    b1 = theta[2:4]
    w2 = theta[4:6]
    b2 = theta[6]
    hidden = torch.tanh(w1 * (t / 120.0 - 1.0) + b1)
    return torch.dot(w2, hidden) + b2


def _simulate(key, theta, times, insulin_times, insulin, basal_glucose, basal_insulin):
    if key == "neural_ode":
        def rhs(t, state):
            g = state[0]
            inputs = torch.stack(
                ((g - basal_glucose) / basal_glucose, (_interpolate(insulin_times, insulin, t) - basal_insulin) / 60.0, t / 120.0 - 1.0)
            )
            w1 = theta[:6].reshape(2, 3)
            b1 = theta[6:8]
            w2 = theta[8:10]
            b2 = theta[10]
            derivative = 0.04 * (torch.dot(w2, torch.tanh(w1 @ inputs + b1)) + b2)
            return derivative.reshape(1)
        return _rk4(rhs, torch.tensor([basal_glucose]), times)[:, 0]

    p1, p2, p3 = torch.exp(theta[:3])
    if key == "minimal_gamma":
        sigma, rate = torch.exp(theta[3:5])
    else:
        sigma = torch.tensor(1.4)
        rate = torch.tensor(0.014)
    conversion = 0.005551 / 18.57
    meal_mass = 85_500.0

    def rhs(t, state):
        glucose, remote = state
        insulin_value = _interpolate(insulin_times, insulin, t)
        if key == "ude_appearance":
            appearance = 0.12 * torch.nn.functional.softplus(_mlp_time(theta[3:], t))
        else:
            positive_t = torch.clamp(t, min=1e-8)
            log_gamma = torch.lgamma(sigma)
            appearance = conversion * meal_mass * torch.exp(
                sigma * torch.log(rate) + (sigma - 1.0) * torch.log(positive_t)
                - rate * positive_t - log_gamma
            )
            appearance = torch.where(t > 0, appearance, torch.zeros_like(appearance))
        return torch.stack(
            (
                -glucose * remote - p3 * (glucose - basal_glucose) + appearance,
                -p1 * remote + p2 * (insulin_value - basal_insulin),
            )
        )
    initial = torch.tensor([basal_glucose, 0.0])
    return _rk4(rhs, initial, times)[:, 0]


def _specification(key):
    if key == "minimal_fixed":
        mean = torch.log(torch.tensor([0.019, 2.65e-4, 0.026]))
        precision = torch.eye(3)
    elif key == "minimal_gamma":
        mean = torch.cat((torch.log(torch.tensor([0.019, 2.65e-4, 0.026])), torch.log(torch.tensor([1.4, 0.014]))))
        precision = torch.eye(5)
    elif key == "ude_appearance":
        mean = torch.cat((torch.log(torch.tensor([0.019, 2.65e-4, 0.026])), torch.zeros(7)))
        precision = torch.diag(torch.cat((torch.ones(3), torch.full((7,), 4.0))))
    elif key == "neural_ode":
        mean = torch.zeros(11)
        precision = 4.0 * torch.eye(11)
    else:
        raise KeyError(key)
    return mean, precision


def _fit(key, seed, observed, glucose_times, glucose_std, insulin_times, insulin):
    mean, precision = _specification(key)
    train_count = 5
    generator = torch.Generator().manual_seed(730_001 + 9_973 * seed + len(mean))
    best = None
    for start in range(1):
        theta = (mean + 0.16 * torch.randn(len(mean), generator=generator)).clone().requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=0.018 if len(mean) > 5 else 0.025)
        for _ in range(90):
            optimizer.zero_grad()
            prediction = _simulate(
                key, theta, glucose_times, insulin_times, insulin, observed[0], insulin[0]
            )
            residual = (prediction[:train_count] - observed[:train_count]) / glucose_std[:train_count]
            prior = theta - mean
            loss = 0.5 * (torch.sum(residual**2) + 0.05 * prior @ precision @ prior)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], 20.0)
            optimizer.step()
        fitted = theta.detach()
        with torch.no_grad():
            prediction = _simulate(
                key, fitted, glucose_times, insulin_times, insulin, observed[0], insulin[0]
            )
            train_residual = (prediction[:train_count] - observed[:train_count]) / glucose_std[:train_count]
            objective = float(torch.sum(train_residual**2))
        if best is None or objective < best[0]:
            best = (objective, fitted, prediction)
    objective, theta, prediction = best
    local = theta.clone().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: (
            _simulate(key, value, glucose_times, insulin_times, insulin, observed[0], insulin[0])[:train_count]
            / glucose_std[:train_count]
        ),
        local,
        vectorize=True,
    ).reshape(train_count, len(theta))
    information = jacobian.mT @ jacobian
    eigenvalues = generalized_spectrum(information, precision)
    geometry = PullbackGeometry(eigenvalues, resolution=1.0)
    deviance = objective + float(torch.sum(torch.log(2.0 * math.pi * glucose_std[:train_count] ** 2)))
    n = train_count
    scores = {
        "aic": deviance + 2 * len(theta),
        "bic": deviance + math.log(n) * len(theta),
        "gic_eff": deviance + math.log(n) * geometry.effective_dimension,
        "gic_vol": deviance + math.log(n) * geometry.effective_dimension + 0.5 * geometry.complexity,
    }
    holdout_residual = (prediction[train_count:] - observed[train_count:]) / glucose_std[train_count:]
    return {
        "candidate": key,
        "seed": seed,
        "theta": theta.tolist(),
        "train_deviance": deviance,
        "blocked_standardized_mse": float(torch.mean(holdout_residual**2)),
        "observable_dimension": geometry.effective_dimension,
        "observable_complexity": geometry.complexity,
        "prediction": prediction.tolist(),
        **scores,
    }


def _bootstrap_cell(seed):
    glucose = pd.read_csv(DATA_ROOT / "mean_glucose.csv")
    insulin_frame = pd.read_csv(DATA_ROOT / "mean_insulin.csv")
    glucose_times = torch.tensor(glucose["time"].to_numpy())
    mean_glucose = torch.tensor(glucose["glucose"].to_numpy())
    glucose_std = torch.tensor(glucose["std"].to_numpy())
    insulin_times = torch.tensor(insulin_frame["time"].to_numpy())
    insulin = torch.tensor(insulin_frame["insulin"].to_numpy())
    generator = torch.Generator().manual_seed(600_007 + seed)
    observed = mean_glucose + glucose_std * torch.randn(mean_glucose.shape, generator=generator)
    observed[0] = mean_glucose[0]
    return [
        _fit(key, seed, observed, glucose_times, glucose_std, insulin_times, insulin)
        for key in MODEL_KEYS
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_bootstrap_cell, seed) for seed in range(args.bootstrap)]
        for index, future in enumerate(as_completed(futures), start=1):
            completed = future.result()
            records.extend(completed)
            raw_directory = ROOT / "results" / "records" / "glucose" / "raw"
            raw_directory.mkdir(parents=True, exist_ok=True)
            for record in completed:
                raw_path = raw_directory / (
                    f"{record['candidate']}__seed_{int(record['seed']):03d}.json"
                )
                raw_path.write_text(
                    json.dumps(record, indent=2, allow_nan=False),
                    encoding="utf-8",
                )
            print(f"completed {index}/{args.bootstrap}", flush=True)
    exclusions = {"theta", "prediction"}
    frame = pd.DataFrame(
        [{key: value for key, value in row.items() if key not in exclusions} for row in records]
    )
    frame.to_csv(ROOT / "results" / "summary" / "glucose_case_scores.csv", index=False)
    selections = []
    for seed, group in frame.groupby("seed"):
        ordered = group.set_index("candidate").loc[list(MODEL_KEYS)]
        for criterion in ("aic", "bic", "gic_eff", "gic_vol"):
            values = torch.tensor(ordered[criterion].to_numpy())
            weights = criterion_weights(values)
            for candidate, weight in zip(MODEL_KEYS, weights):
                selections.append(
                    {
                        "seed": seed,
                        "criterion": criterion,
                        "candidate": candidate,
                        "weight": float(weight),
                        "selected": int(candidate == MODEL_KEYS[int(torch.argmin(values))]),
                    }
                )
    selection_frame = pd.DataFrame(selections)
    selection_frame.to_csv(ROOT / "results" / "summary" / "glucose_case_weights.csv", index=False)
    summary = (
        frame.groupby("candidate")
        .agg(
            n=("seed", "size"),
            blocked_mse=("blocked_standardized_mse", "mean"),
            blocked_mse_sd=("blocked_standardized_mse", "std"),
            observable_dimension=("observable_dimension", "mean"),
            observable_dimension_sd=("observable_dimension", "std"),
        )
        .reset_index()
    )
    summary.to_csv(ROOT / "results" / "summary" / "glucose_case_summary.csv", index=False)

    glucose = pd.read_csv(DATA_ROOT / "mean_glucose.csv")
    figure, axes = plt.subplots(1, 2, figsize=(8.3, 3.3))
    axes[0].errorbar(
        glucose["time"], glucose["glucose"], yerr=glucose["std"],
        fmt="o", color="black", capsize=2, label="Mean data",
    )
    for key in MODEL_KEYS:
        predictions = np.array(
            [row["prediction"] for row in records if row["candidate"] == key]
        )
        axes[0].plot(glucose["time"], predictions.mean(axis=0), label=key.replace("_", " "))
    axes[0].axvline(120, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Minutes after meal")
    axes[0].set_ylabel("Glucose")
    axes[0].legend(frameon=False, fontsize=7)
    weight_summary = (
        selection_frame[selection_frame["criterion"].isin(["bic", "gic_eff"])]
        .groupby(["criterion", "candidate"])["weight"].mean()
        .unstack(0)
        .loc[list(MODEL_KEYS)]
    )
    weight_summary.plot.bar(ax=axes[1], rot=20)
    axes[1].set_ylabel("Mean support weight")
    axes[1].legend(["BIC", "geometric BIC approximation"], frameon=False)
    figure.tight_layout()
    figure.savefig(ROOT / "figures" / "glucose_case.pdf", bbox_inches="tight")
    plt.close(figure)

    source = DATA_ROOT / "mean_glucose.csv"
    commit_file = ROOT / "experiments" / "data" / "raw" / "ude-regularization" / ".git" / "refs" / "heads" / "main"
    metadata = {
        "repository": "https://github.com/Computational-Biology-TUe/ude-regularization",
        "commit": commit_file.read_text(encoding="utf-8").strip() if commit_file.exists() else "packed-ref",
        "license": "MIT",
        "accessed": date.today().isoformat(),
        "glucose_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "insulin_sha256": hashlib.sha256(
            (DATA_ROOT / "mean_insulin.csv").read_bytes()
        ).hexdigest(),
        "bootstrap_replicates": args.bootstrap,
        "training_times_minutes": [0, 15, 30, 60, 120],
        "held_out_times_minutes": [180, 240],
    }
    (ROOT / "results" / "records" / "glucose").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "records" / "glucose" / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()


