"""Create the five main figures for the first-principles manuscript."""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
FP = REPO / "results" / "summary" / "first_principles"
NUM = REPO / "results" / "summary" / "numerical"
TABLES = ROOT / "results" / "summary"
OUT = ROOT / "figures" / "submission"
OUT.mkdir(parents=True, exist_ok=True)
TIFF_OUT = os.environ.get("MECHAI_TIFF_OUT")

MM = 1 / 25.4
width_mm = 183  # JUQ full-text width
WIDTH = 7.2047
INK = "#252A34"
GRAY = "#7A8088"
LIGHT = "#D9DEE5"
BLUE = "#275D8C"
BLUE2 = "#6F9CC3"
TEAL = "#2F8F92"
PURPLE = "#755C91"
RED = "#B5514F"
GOLD = "#B38A3E"
CMAP = LinearSegmentedColormap.from_list(
    "geometry_blue", ["#F6F8FA", "#C5D8E7", BLUE]
)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.6,
    "axes.titlesize": 7.1,
    "axes.labelsize": 6.6,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 5.9,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})

CRITERIA = ["aic", "aicc", "bic", "gic_pred", "gic_eff_logn", "gic_evid"]
CLABEL = {
    "aic": "AIC",
    "aicc": "AICc",
    "bic": "BIC",
    "gic_pred": "GIC-pred",
    "gic_eff_logn": "Geometric BIC",
    "gic_evid": "GIC-evid",
}
CCOLOR = {
    "aic": LIGHT,
    "aicc": "#B8BDC4",
    "bic": GRAY,
    "gic_pred": BLUE,
    "gic_eff_logn": TEAL,
    "gic_evid": PURPLE,
}
SCENARIOS = [
    "regular_sir_full",
    "early_seir_infected_only",
    "missing_time_varying_transmission",
    "missing_neural_feedback",
    "noisy_sir_overfit_risk",
]
SLABEL = {
    "regular_sir_full": "SIR, full",
    "early_seir_infected_only": "SEIR, early",
    "missing_time_varying_transmission": "Time-varying",
    "missing_neural_feedback": "Neural feedback",
    "noisy_sir_overfit_risk": "SIR, noisy",
}
STUDIES = [
    "biochemical_haldane",
    "biochemical_ude",
    "ecology_rm",
    "fhn_standard",
    "fhn_ude",
]
STLABEL = {
    "biochemical_haldane": "Haldane",
    "biochemical_ude": "Biochemical UDE",
    "ecology_rm": "Predator-prey",
    "fhn_standard": "FitzHugh-Nagumo",
    "fhn_ude": "FHN-UDE",
}
PMETHODS = ["equal", "aic", "bic", "gic_evid", "stacking", "hard_gic_pred"]
PLABEL = {
    "equal": "Equal",
    "aic": "AIC",
    "bic": "BIC",
    "gic_evid": "GIC-evid",
    "stacking": "Stacking",
    "hard_gic_pred": "Hard GIC-pred",
}
PCOLOR = {
    "equal": LIGHT,
    "aic": "#B8BDC4",
    "bic": GRAY,
    "gic_evid": PURPLE,
    "stacking": TEAL,
    "hard_gic_pred": BLUE,
}
MODEL_COLORS = {
    "sir": LIGHT,
    "tv_sir": GRAY,
    "ude_sir_h2": BLUE,
    "neural_ode_h2": TEAL,
}
MODEL_LABEL = {
    "sir": "SIR",
    "seir": "SEIR",
    "tv_sir": "Time-varying SIR",
    "ude_sir_h2": "UDE-SIR",
    "neural_ode_h2": "Neural ODE",
}


def read(path: Path, *required: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    return frame


def panel(ax, label: str) -> None:
    ax.text(
        -0.12, 1.07, label, transform=ax.transAxes,
        fontsize=8, fontweight="bold", ha="left", va="top", clip_on=False,
    )


def clean(ax) -> None:
    ax.grid(axis="y", color="#E7EBEF", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)


def heat(ax, matrix: pd.DataFrame, *, vmin=0.0, vmax=1.0, annotate=True):
    image = ax.imshow(
        matrix.to_numpy(float), cmap=CMAP, vmin=vmin, vmax=vmax, aspect="auto"
    )
    if annotate:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iloc[row, column]
                if pd.notna(value):
                    color = "white" if value > vmin + 0.62 * (vmax - vmin) else INK
                    ax.text(
                        column, row, f"{value:.2f}",
                        ha="center", va="center", fontsize=5.5, color=color,
                    )
    return image


def export(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
    svg_path = OUT / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.025)
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.025)
    if TIFF_OUT:
        target = Path(TIFF_OUT)
        target.mkdir(parents=True, exist_ok=True)
        fig.savefig(target / f"{stem}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def selection_entropy(values: pd.Series) -> float:
    probabilities = values.value_counts(normalize=True).to_numpy(float)
    if len(probabilities) < 2:
        return 0.0
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(len(probabilities)))


def runner_up_margin(group: pd.DataFrame, score: str) -> float:
    numeric = pd.to_numeric(group[score], errors="coerce")
    values = np.sort(numeric[numeric.notna()])
    return float(values[1] - values[0]) if len(values) > 1 else math.nan


def figure1() -> None:
    summary = read(FP / "core_selection_summary.csv", "scenario", "criterion", "recovery_rate")
    scores = read(
        FP / "core_scores.csv", "scenario", "candidate", "seed", "dimension",
        "effective_dimension", "relative_log_volume", "deviance",
    )
    selections = read(FP / "core_selections.csv", "scenario", "criterion", "selected")
    matrix = (
        summary.pivot(index="scenario", columns="criterion", values="recovery_rate")
        .reindex(SCENARIOS)[CRITERIA]
    )

    fig = plt.figure(figsize=(WIDTH, 4.25), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.22, 1.22, 1.0, 1.0])
    ax_heat = fig.add_subplot(grid[:, :2])
    ax_change = fig.add_subplot(grid[0, 2])
    ax_dimension = fig.add_subplot(grid[0, 3])
    ax_volume = fig.add_subplot(grid[1, 2])
    ax_entropy = fig.add_subplot(grid[1, 3])

    image = heat(ax_heat, matrix)
    ax_heat.set_xticks(range(len(CRITERIA)), [CLABEL[item] for item in CRITERIA], rotation=32, ha="right")
    ax_heat.set_yticks(range(len(SCENARIOS)), [SLABEL[item] for item in SCENARIOS])
    ax_heat.set_title("Truth recovery under competing complexity penalties")
    fig.colorbar(image, ax=ax_heat, fraction=0.035, pad=0.02, label="Recovery rate")

    wide = matrix
    y = np.arange(len(SCENARIOS))
    ax_change.barh(y - 0.17, wide["gic_pred"] - wide["bic"], 0.32, color=BLUE, label="GIC-pred")
    ax_change.barh(y + 0.17, wide["gic_evid"] - wide["bic"], 0.32, color=PURPLE, label="GIC-evid")
    ax_change.axvline(0.0, color=INK, linewidth=0.7)
    ax_change.set_yticks(y, [SLABEL[item] for item in SCENARIOS])
    ax_change.set_xlabel("Recovery difference from BIC")
    ax_change.set_title("Recovery relative to BIC")
    ax_change.legend(loc="lower right")
    clean(ax_change)

    dimensions = (
        scores.groupby("candidate")
        .agg(formal=("dimension", "first"), effective=("effective_dimension", "mean"))
        .sort_values("formal")
    )
    candidate_style = {
        "sir": (GRAY, "o"),
        "seir": (BLUE2, "s"),
        "tv_sir": (TEAL, "D"),
        "ude_sir_h2": (PURPLE, "^"),
        "neural_ode_h2": (GOLD, "P"),
    }
    for name, row in dimensions.iterrows():
        color, marker = candidate_style[name]
        ax_dimension.scatter(
            row["formal"], row["effective"], color=color, marker=marker,
            s=22, label=MODEL_LABEL[name], zorder=3,
        )
    limit = float(max(dimensions["formal"].max(), dimensions["effective"].max()))
    ax_dimension.plot([0, limit], [0, limit], "--", color=GRAY, linewidth=0.7)
    ax_dimension.legend(loc="upper right", handletextpad=0.35, labelspacing=0.25)
    ax_dimension.set_xlabel("Formal dimension")
    ax_dimension.set_ylabel("Mean effective dimension")
    ax_dimension.set_title("Formal versus effective dimension")
    clean(ax_dimension)

    volumes = (
        scores.groupby("candidate")["relative_log_volume"]
        .agg(["median", "std"]).sort_values("median")
    )
    ax_volume.barh(
        range(len(volumes)), volumes["median"], xerr=volumes["std"],
        color=TEAL, capsize=1.5,
    )
    ax_volume.set_yticks(range(len(volumes)), [MODEL_LABEL.get(name, name.replace("_", " ")) for name in volumes.index])
    ax_volume.set_xlabel("Relative log-volume")
    ax_volume.set_title("Local evidence complexity")
    clean(ax_volume)

    entropy = (
        selections.groupby(["scenario", "criterion"])["selected"]
        .apply(selection_entropy).unstack().reindex(SCENARIOS)
    )
    for criterion in ("bic", "gic_pred", "gic_eff_logn", "gic_evid"):
        ax_entropy.plot(
            range(len(SCENARIOS)), entropy[criterion],
            marker="o", markersize=2.8, linewidth=1.0,
            color=CCOLOR[criterion], label=CLABEL[criterion],
        )
    ax_entropy.set_xticks(range(len(SCENARIOS)), [SLABEL[item] for item in SCENARIOS], rotation=30, ha="right")
    ax_entropy.set_ylim(-0.03, 1.03)
    ax_entropy.set_ylabel("Normalized selection entropy")
    ax_entropy.set_title("Selection ambiguity")
    ax_entropy.legend(ncol=2, loc="upper center")
    clean(ax_entropy)

    for label, axis in zip("abcde", [ax_heat, ax_change, ax_dimension, ax_volume, ax_entropy]):
        if label in {"c", "d"}:
            x_position = 0.01 if label == "c" else -0.22
            axis.text(
                x_position, 1.07, label, transform=axis.transAxes,
                fontsize=8, fontweight="bold", ha="left", va="top",
                clip_on=False,
            )
        else:
            panel(axis, label)
    export(fig, "fig1_selection_complexity")


def figure2() -> None:
    summary = read(
        FP / "phase_diagram_selection_summary.csv",
        "n_times", "noise", "trajectories", "criterion", "recovery_rate",
    )
    scores = read(
        FP / "phase_diagram_scores.csv",
        "n_times", "noise", "trajectories", "candidate", "effective_dimension",
    )
    ude = (
        scores[scores["candidate"] == "ude_sir_h2"]
        .groupby(["n_times", "noise", "trajectories"])["effective_dimension"]
        .mean().reset_index()
    )
    fig, axes = plt.subplots(
        3, 3, figsize=(WIDTH, 4.8), sharex=True, sharey=True,
        constrained_layout=True,
    )
    specifications = [
        ("gic_pred", "GIC-pred recovery", 0.0, 1.0),
        ("bic", "BIC recovery", 0.0, 1.0),
        (None, "UDE effective dimension", float(ude.effective_dimension.min()), float(ude.effective_dimension.max())),
    ]
    for row, (criterion, label, vmin, vmax) in enumerate(specifications):
        for column, trajectories in enumerate((1, 2, 4)):
            source = (
                ude[ude["trajectories"] == trajectories]
                if criterion is None
                else summary[
                    (summary["criterion"] == criterion)
                    & (summary["trajectories"] == trajectories)
                ]
            )
            value = "effective_dimension" if criterion is None else "recovery_rate"
            matrix = source.pivot(index="noise", columns="n_times", values=value).sort_index(ascending=False)
            image = heat(axes[row, column], matrix, vmin=vmin, vmax=vmax)
            axes[row, column].set_xticks(range(len(matrix.columns)), matrix.columns)
            axes[row, column].set_yticks(range(len(matrix.index)), [f"{item:.3f}" for item in matrix.index])
            if row == 0:
                axes[row, column].set_title(
                    f"{trajectories} independent trajectory"
                    + ("ies" if trajectories > 1 else "")
                )
            if column == 0:
                axes[row, column].set_ylabel(label + "\nNoise s.d.")
            if row == 2:
                axes[row, column].set_xlabel("Observation times")
            panel(axes[row, column], chr(97 + row * 3 + column))
    fig.colorbar(axes[0, 0].images[0], ax=axes[:2, :], fraction=0.014, pad=0.015, label="Recovery rate")
    fig.colorbar(axes[2, 0].images[0], ax=axes[2, :], fraction=0.014, pad=0.015, label="Effective dimension")
    export(fig, "fig2_information_phase")


def figure3() -> None:
    sensitivity = read(
        NUM / "reference_sensitivity_summary.csv",
        "scenario", "rho_nn", "resolution", "gamma", "mean",
    )
    optimism = read(
        FP / "optimism_calibration_summary.csv",
        "scenario", "candidate", "mean_empirical_optimism",
        "se_empirical_optimism", "mean_predicted_optimism",
    )
    evidence = read(
        FP / "evidence_calibration.csv",
        "minus2_log_evidence_is", "gic_evid", "relative_ess", "gic_evid_error",
    )
    metric = read(
        FP / "metric_loss_matching_summary.csv",
        "truth", "loss", "metric_relation", "mean_calibration_residual",
    )
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.4), constrained_layout=True)

    for axis, scenario, title in (
        (axes[0, 0], "early_seir_infected_only", "Early SEIR"),
        (axes[0, 1], "missing_neural_feedback", "Missing neural feedback"),
    ):
        subset = sensitivity[
            (sensitivity["scenario"] == scenario)
            & np.isclose(sensitivity["gamma"], 0.0)
        ]
        matrix = subset.pivot(index="rho_nn", columns="resolution", values="mean").sort_index(ascending=False)
        image = heat(axis, matrix, annotate=False)
        axis.set_xticks(range(len(matrix.columns)), [f"{math.log10(item):.1f}" for item in matrix.columns], rotation=30, ha="right")
        axis.set_yticks(range(len(matrix.index)), [f"{item:g}" for item in matrix.index])
        axis.set_xlabel(r"$\log_{10}\lambda$")
        axis.set_ylabel(r"Neural-block precision $\rho$")
        axis.set_title(title)
    fig.colorbar(image, ax=axes[0, :2], fraction=0.025, pad=0.015, label="Geometric BIC recovery")

    scenario_colors = dict(zip(SCENARIOS, [GRAY, GOLD, TEAL, PURPLE, BLUE]))
    for scenario, group in optimism.groupby("scenario"):
        axes[0, 2].errorbar(
            group["mean_predicted_optimism"], group["mean_empirical_optimism"],
            yerr=1.96 * group["se_empirical_optimism"],
            fmt="o", markersize=3.0, linewidth=0.7, capsize=1.4,
            color=scenario_colors[scenario], alpha=0.85, label=SLABEL[scenario],
        )
    limits = [
        min(optimism["mean_predicted_optimism"].min(), optimism["mean_empirical_optimism"].min()),
        max(optimism["mean_predicted_optimism"].max(), optimism["mean_empirical_optimism"].max()),
    ]
    axes[0, 2].plot(limits, limits, "--", color=INK, linewidth=0.8)
    axes[0, 2].set_xlabel(r"Predicted optimism $2d_{\rm eff}$")
    axes[0, 2].set_ylabel("Independent-response optimism")
    axes[0, 2].set_title("Predictive-risk calibration")
    axes[0, 2].legend(ncol=2, loc="upper left")
    clean(axes[0, 2])

    scatter = axes[1, 0].scatter(
        evidence["minus2_log_evidence_is"], evidence["gic_evid"],
        c=evidence["relative_ess"], cmap="viridis", s=13, alpha=0.72,
        edgecolors="none",
    )
    low = min(evidence["minus2_log_evidence_is"].min(), evidence["gic_evid"].min())
    high = max(evidence["minus2_log_evidence_is"].max(), evidence["gic_evid"].max())
    axes[1, 0].plot([low, high], [low, high], "--", color=INK, linewidth=0.8)
    axes[1, 0].set_xlabel(r"$-2\log p(y\mid M)$, importance sampling")
    axes[1, 0].set_ylabel("GIC-evid")
    axes[1, 0].set_title("Local evidence approximation")
    fig.colorbar(scatter, ax=axes[1, 0], fraction=0.045, pad=0.02, label="Relative ESS")
    clean(axes[1, 0])

    axes[1, 1].scatter(
        evidence["relative_ess"].clip(lower=1e-4),
        np.abs(evidence["gic_evid_error"]),
        color=PURPLE, alpha=0.62, s=13,
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Relative importance-sampling ESS")
    axes[1, 1].set_ylabel("Absolute local-evidence error")
    axes[1, 1].set_title("Error near weak modes")
    clean(axes[1, 1])

    combinations = [
        ("amplitude_change", "fisher_trajectory"),
        ("amplitude_change", "wasserstein_quantile"),
        ("time_shift", "fisher_trajectory"),
        ("time_shift", "wasserstein_quantile"),
    ]
    labels = ["Amplitude / trajectory", "Amplitude / transport", "Shift / trajectory", "Shift / transport"]
    matched = []
    mismatched = []
    for truth, loss in combinations:
        subset = metric[(metric["truth"] == truth) & (metric["loss"] == loss)].set_index("metric_relation")
        matched.append(abs(float(subset.loc["matched", "mean_calibration_residual"])))
        mismatched.append(abs(float(subset.loc["mismatched", "mean_calibration_residual"])))
    x = np.arange(len(combinations))
    axes[1, 2].bar(x - 0.18, matched, 0.36, color=BLUE, label="Matched")
    axes[1, 2].bar(x + 0.18, mismatched, 0.36, color=LIGHT, edgecolor=GRAY, linewidth=0.5, label="Mismatched")
    axes[1, 2].set_xticks(x, labels, rotation=38, ha="right", rotation_mode="anchor")
    axes[1, 2].set_ylabel("Absolute mean calibration bias")
    axes[1, 2].set_title("Metric and risk must match")
    axes[1, 2].legend()
    clean(axes[1, 2])

    for label, axis in zip("abcdef", axes.flat):
        panel(axis, label)
    export(fig, "fig3_derived_criteria")


def figure4() -> None:
    scores = read(
        FP / "biological_systems_scores.csv",
        "study", "domain", "candidate", "seed", "dimension",
        "effective_dimension", "deviance", "gic_pred", "gradient_norm", "wall_seconds",
    )
    summary = read(
        FP / "biological_systems_selection_summary.csv",
        "study", "criterion", "recovery_rate",
    )
    matrix = (
        summary.pivot(index="study", columns="criterion", values="recovery_rate")
        .reindex(STUDIES)[CRITERIA]
    )
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.25), constrained_layout=True)
    image = heat(axes[0, 0], matrix)
    axes[0, 0].set_xticks(range(len(CRITERIA)), [CLABEL[item] for item in CRITERIA], rotation=32, ha="right")
    axes[0, 0].set_yticks(range(len(STUDIES)), [STLABEL[item] for item in STUDIES])
    axes[0, 0].set_title("Recovery by biological system")
    fig.colorbar(image, ax=axes[0, 0], fraction=0.04, pad=0.02)

    dimensions = (
        scores.groupby(["domain", "candidate"])
        .agg(formal=("dimension", "first"), effective=("effective_dimension", "mean"))
        .reset_index()
    )
    styles = [("biochemical", "Biochemical", BLUE, "o"), ("ecology", "Ecological", TEAL, "s"), ("fhn", "Electrophysiological", PURPLE, "^")]
    for domain, domain_label, color, marker in styles:
        subset = dimensions[dimensions["domain"] == domain]
        axes[0, 1].scatter(subset["formal"], subset["effective"], color=color, marker=marker, s=24, label=domain_label)
    limit = max(dimensions["formal"].max(), dimensions["effective"].max())
    axes[0, 1].plot([0, limit], [0, limit], "--", color=GRAY, linewidth=0.7)
    axes[0, 1].set_xlabel("Formal dimension")
    axes[0, 1].set_ylabel("Mean effective dimension")
    axes[0, 1].set_title("Geometric compression")
    axes[0, 1].legend()
    clean(axes[0, 1])

    decomposition = scores.copy()
    decomposition["fit_gap"] = decomposition["deviance"] - decomposition.groupby(["study", "seed"])["deviance"].transform("min")
    decomposition = (
        decomposition.groupby(["domain", "candidate"])
        .agg(fit=("fit_gap", "mean"), penalty=("effective_dimension", lambda values: 2.0 * values.mean()))
        .reset_index()
    )
    for domain, domain_label, color, marker in styles:
        subset = decomposition[decomposition["domain"] == domain]
        axes[0, 2].scatter(np.log10(1.0 + subset["fit"].clip(lower=0.0)), subset["penalty"], color=color, marker=marker, s=22)
    axes[0, 2].set_xlabel(r"$\log_{10}(1+\mathrm{mean\ deviance\ gap})$")
    axes[0, 2].set_ylabel(r"$2d_{\rm eff}$")
    axes[0, 2].set_title("Predictive fit-complexity balance")
    clean(axes[0, 2])

    margin = (
        scores.groupby(["study", "seed"])
        .apply(lambda block: runner_up_margin(block, "gic_pred"), include_groups=False)
        .rename("margin").reset_index()
    )
    boxes = axes[1, 0].boxplot(
        [margin.loc[margin["study"] == item, "margin"].clip(lower=1e-12) for item in STUDIES],
        showfliers=False, patch_artist=True, medianprops={"color": INK},
    )
    for patch, color in zip(boxes["boxes"], [BLUE, BLUE, TEAL, PURPLE, PURPLE]):
        patch.set_facecolor(color)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(range(1, 6), [STLABEL[item] for item in STUDIES], rotation=28, ha="right")
    axes[1, 0].set_ylabel("GIC-pred runner-up margin")
    axes[1, 0].set_title("Selection separation")
    clean(axes[1, 0])

    for axis, column, title, ylabel in (
        (axes[1, 1], "gradient_norm", "Optimization residual", "Gradient norm"),
        (axes[1, 2], "wall_seconds", "Computational cost", "Wall time (s)"),
    ):
        groups = ["biochemical", "ecology", "fhn"]
        boxes = axis.boxplot(
            [scores.loc[scores["domain"] == group, column].clip(lower=1e-12) for group in groups],
            showfliers=False, patch_artist=True, medianprops={"color": INK},
        )
        for patch, color in zip(boxes["boxes"], [BLUE, TEAL, PURPLE]):
            patch.set_facecolor(color)
        axis.set_yscale("log")
        axis.set_xticks(range(1, 4), ["Biochemical", "Ecological", "Electrophysiological"], rotation=22, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        clean(axis)
    for label, axis in zip("abcdef", axes.flat):
        panel(axis, label)
    export(fig, "fig4_biological_systems")


def figure5() -> None:
    coverage = read(
        TABLES / "confidence_coverage_summary.csv",
        "scenario", "method", "nominal", "coverage", "coverage_ci_low", "coverage_ci_high",
    )
    prediction = read(
        NUM / "predictive_model_averaging_summary.csv",
        "scenario", "method", "mse_mean", "mse_ci_low", "mse_ci_high", "within", "between",
    )
    weights = read(
        NUM / "predictive_model_averaging.csv",
        "scenario", "method", "weight_sir", "weight_tv_sir",
        "weight_ude_sir_h2", "weight_neural_ode_h2",
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(WIDTH, 4.55),
        gridspec_kw={"width_ratios": [1.0, 1.08, 1.18]},
        constrained_layout=True,
    )
    styles = [
        ("naive_wald", GRAY, "s", "Raw Wald"),
        ("geometric_quotient", BLUE, "o", "Geometric quotient"),
        ("simulation_calibrated", TEAL, "^", "Calibrated"),
    ]
    for axis, scenario, title in (
        (axes[0, 0], "regular_sir_full", "Regular SIR"),
        (axes[0, 1], "early_seir_infected_only", "Early partial SEIR"),
    ):
        for method, color, marker, label in styles:
            subset = coverage[
                (coverage["scenario"] == scenario)
                & (coverage["method"] == method)
            ].sort_values("nominal")
            axis.errorbar(
                subset["nominal"], subset["coverage"],
                yerr=[
                    subset["coverage"] - subset["coverage_ci_low"],
                    subset["coverage_ci_high"] - subset["coverage"],
                ],
                color=color, marker=marker, markersize=3, linewidth=1,
                capsize=1.5, label=label,
            )
        axis.plot([0.48, 0.97], [0.48, 0.97], "--", color=INK, linewidth=0.7)
        axis.set_xlim(0.48, 0.97)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("Nominal coverage")
        axis.set_ylabel("Empirical coverage")
        axis.set_title(title)
        clean(axis)
    axes[0, 0].legend(loc="lower right")

    for axis, scenario, title in (
        (axes[0, 2], "early_seir_infected_only", "Early partial SEIR"),
        (axes[1, 0], "missing_neural_feedback", "Missing neural feedback"),
    ):
        subset = (
            prediction[prediction["scenario"] == scenario]
            .set_index("method").reindex(PMETHODS)
        )
        subset = subset[subset["mse_mean"].notna()].sort_values("mse_mean")
        y = np.arange(len(subset))
        axis.barh(y, subset["mse_mean"], color=[PCOLOR[item] for item in subset.index])
        axis.errorbar(
            subset["mse_mean"], y,
            xerr=[
                subset["mse_mean"] - subset["mse_ci_low"],
                subset["mse_ci_high"] - subset["mse_mean"],
            ],
            fmt="none", ecolor=INK, linewidth=0.7, capsize=1.3,
        )
        axis.set_yticks(y, [PLABEL[item] for item in subset.index])
        axis.invert_yaxis()
        axis.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
        axis.set_xlabel("Held-out MSE")
        axis.set_title(title)
        clean(axis)

    methods = ["aic", "bic", "gic_evid", "stacking"]
    x = np.arange(len(methods))
    width = 0.34
    for index, (scenario, hatch) in enumerate((
        ("early_seir_infected_only", ""),
        ("missing_neural_feedback", "///"),
    )):
        subset = prediction[prediction["scenario"] == scenario].set_index("method").reindex(methods)
        position = x + (index - 0.5) * width
        axes[1, 1].bar(
            position, subset["within"], width, color=BLUE2,
            edgecolor=INK, linewidth=0.35, hatch=hatch,
            label="Within model" if index == 0 else None,
        )
        axes[1, 1].bar(
            position, subset["between"], width, bottom=subset["within"],
            color=TEAL, edgecolor=INK, linewidth=0.35, hatch=hatch,
            label="Between models" if index == 0 else None,
        )
    axes[1, 1].set_xticks(x, [PLABEL[item] for item in methods], rotation=22, ha="right")
    axes[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[1, 1].set_ylabel("Mean predictive variance")
    axes[1, 1].set_title("Within- and between-model uncertainty")
    axes[1, 1].set_ylim(
        0.0, 1.18 * float((prediction["within"] + prediction["between"]).max())
    )
    axes[1, 1].legend(ncol=2, loc="upper center")
    clean(axes[1, 1])

    focus = (
        weights[weights["method"].isin(["gic_evid", "stacking"])]
        .groupby(["scenario", "method"])[
            ["weight_sir", "weight_tv_sir", "weight_ude_sir_h2", "weight_neural_ode_h2"]
        ].mean()
    )
    order = [
        ("early_seir_infected_only", "gic_evid"),
        ("early_seir_infected_only", "stacking"),
        ("missing_neural_feedback", "gic_evid"),
        ("missing_neural_feedback", "stacking"),
    ]
    focus = focus.reindex(pd.MultiIndex.from_tuples(order))
    left = np.zeros(len(order))
    for column in focus.columns:
        model = column.replace("weight_", "")
        values = focus[column].to_numpy()
        axes[1, 2].barh(
            range(len(order)), values, left=left,
            color=MODEL_COLORS[model], label=MODEL_LABEL.get(model, model.replace("_", " ")),
        )
        left += values
    axes[1, 2].set_yticks(
        range(len(order)),
        [
            f"{'SEIR' if scenario.startswith('early') else 'UDE'}: {PLABEL[method]}"
            for scenario, method in order
        ],
    )
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_xlim(0.0, 1.0)
    axes[1, 2].set_xlabel("Mean model weight")
    axes[1, 2].set_title("Evidence weights and stacking")
    axes[1, 2].legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    clean(axes[1, 2])

    for label, axis in zip("abcdef", axes.flat):
        panel(axis, label)
    export(fig, "fig5_uncertainty_validation")


def main() -> None:
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    print(f"Wrote five main figures to {OUT}")


if __name__ == "__main__":
    main()
