"""Cross-domain biochemical and electrophysiological benchmarks."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import json
import numpy as np
from pathlib import Path
import traceback

import torch

from mechai_experiments.models.biochemical import CANDIDATES as BIO_CANDIDATES, TRUE_PARAMETERS as BIO_TRUE
from mechai_experiments.criteria import primary_scores
from mechai_experiments.records import finite_or_none, load_compatible, objective_gap, protocol_hash, result_path, write_record
from mechai_experiments.models.electrophysiology import CANDIDATES as FHN_CANDIDATES, TRUE_PARAMETERS as FHN_TRUE
from mechai_experiments.models.ecology import ECOLOGY_CANDIDATES, TRUE_RM
from mechai_experiments.fitting import fit_map, information_matrix
from mechai_model_selection import ObservableGeometry, resolution_profile


ROOT = Path(__file__).resolve().parents[1]
RESOLUTIONS = torch.logspace(-2, 2, 25, dtype=torch.float64)


@dataclass(frozen=True)
class DomainProtocol:
    domain: str
    truth: str
    candidates: tuple[str, ...]
    observation: str
    horizon: float
    n_times: int
    noise: float
    seeds: int
    starts: int
    adam_steps: int
    lbfgs_steps: int
    refine_steps: int


PROTOCOLS = {
    "ecology_rm": (ECOLOGY_CANDIDATES, {"rm": TRUE_RM}, DomainProtocol(
        "ecology", "rm", tuple(ECOLOGY_CANDIDATES), "full", 12.0, 31, 0.025,
        50, 5, 160, 30, 250,
    )),
    "biochemical_haldane": (BIO_CANDIDATES, BIO_TRUE, DomainProtocol(
        "biochemical", "haldane", tuple(BIO_CANDIDATES), "substrate", 4.0, 25, 0.01,
        50, 5, 180, 35, 250,
    )),
    "biochemical_ude": (BIO_CANDIDATES, BIO_TRUE, DomainProtocol(
        "biochemical", "ude_mm", tuple(BIO_CANDIDATES), "substrate", 4.0, 25, 0.01,
        50, 5, 180, 35, 250,
    )),
    "fhn_standard": (FHN_CANDIDATES, FHN_TRUE, DomainProtocol(
        "fhn", "fhn", tuple(FHN_CANDIDATES), "voltage", 30.0, 121, 0.03,
        50, 5, 180, 35, 250,
    )),
    "fhn_ude": (FHN_CANDIDATES, FHN_TRUE, DomainProtocol(
        "fhn", "ude_fhn", tuple(FHN_CANDIDATES), "voltage", 30.0, 121, 0.03,
        50, 5, 180, 35, 250,
    )),
}


def _record_key(study: str, candidate: str, seed: int) -> str:
    return f"{study}__{candidate}__seed{seed:03d}"


def run_cell(study: str, candidate_key: str, seed: int, protocol: DomainProtocol, overwrite: bool, resume: bool = False) -> dict:
    torch.set_num_threads(1)
    candidates, truths, _ = PROTOCOLS[study]
    digest = protocol_hash({
        "protocol": asdict(protocol),
        "truth_parameters": truths[protocol.truth].tolist(),
        "score_protocol": "primary-without-posterior-sampling",
    })
    path = result_path(ROOT, "crossdomain", _record_key(study, candidate_key, seed))
    previous = load_compatible(path, digest, overwrite, replace_incompatible=resume)
    if previous is not None:
        return previous
    candidate = candidates[candidate_key]
    times = torch.linspace(0.0, protocol.horizon, protocol.n_times, dtype=torch.float64)
    truth = candidates[protocol.truth]
    clean = truth.observe(truths[protocol.truth], times, protocol.observation)
    generator = torch.Generator().manual_seed(104729 * seed + protocol.n_times)
    noisy = clean + protocol.noise * torch.randn(clean.shape, generator=generator)
    try:
        fit = fit_map(
            candidate, times, noisy, protocol.observation, protocol.noise,
            seed=seed, starts=protocol.starts, adam_steps=protocol.adam_steps,
            lbfgs_steps=protocol.lbfgs_steps,
            refine_steps=protocol.refine_steps,
        )
        information, jacobian = information_matrix(
            candidate, fit.theta, times, protocol.observation, protocol.noise
        )
        scores = primary_scores(candidate, fit, information, noisy.numel())
        geometry = ObservableGeometry.from_matrices(
            information, candidate.prior_precision, resolution=1.0
        )
        profile = resolution_profile(geometry.eigenvalues, RESOLUTIONS)
        record = {
            "status": "ok", "study": study, "domain": protocol.domain,
            "truth": protocol.truth, "candidate": candidate_key, "seed": seed,
            "n_observations": noisy.numel(), "dimension": candidate.dimension,
            "theta": fit.theta.tolist(), "objective": fit.objective,
            "deviance": fit.deviance, "gradient_norm": finite_or_none(fit.gradient_norm),
            "best_start": fit.best_start, "start_diagnostics": fit.start_diagnostics,
            "second_best_objective_gap": objective_gap(fit.start_diagnostics),
            "wall_seconds": fit.wall_seconds,
            "jacobian_rank": int(torch.linalg.matrix_rank(jacobian, rtol=1e-7)),
            "information": information.tolist(),
            "generalized_eigenvalues": geometry.eigenvalues.tolist(),
            **{key: finite_or_none(value) for key, value in scores.items()},
        }
    except Exception as exc:
        record = {
            "status": "failed", "study": study, "domain": protocol.domain,
            "truth": protocol.truth, "candidate": candidate_key, "seed": seed,
            "error": repr(exc), "traceback": traceback.format_exc(),
        }
    write_record(path, digest, record)
    return record


def aggregate() -> None:
    import pandas as pd
    raw = ROOT / "results" / "records" / "submission" / "crossdomain" / "raw"
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(raw.glob("*.json"))]
    rows = []
    excluded = {"theta", "information", "generalized_eigenvalues", "start_diagnostics", "traceback"}
    for record in records:
        if record.get("status") == "ok":
            rows.append({key: value for key, value in record.items() if key not in excluded})
    frame = pd.DataFrame(rows)
    output = ROOT / "results" / "summary" / "numerical"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "crossdomain_scores.csv", index=False)
    selections = []
    criteria = ("aic", "aicc", "bic", "gic_eff", "gic_vol_050", "ogic_e")
    for (study, seed), group in frame.groupby(["study", "seed"]):
        for criterion in criteria:
            finite = group[group[criterion].notna()]
            if finite.empty:
                continue
            winner = finite.loc[finite[criterion].idxmin()]
            selections.append({
                "study": study, "seed": seed, "criterion": criterion,
                "truth": winner["truth"], "selected": winner["candidate"],
                "correct": int(winner["truth"] == winner["candidate"]),
            })
    selection = pd.DataFrame(selections)
    selection.to_csv(output / "crossdomain_selections.csv", index=False)
    if not selection.empty:
        summary = selection.groupby(["study", "criterion"])["correct"].agg(["mean", "count"]).reset_index()
        z = 1.959963984540054
        denominator = 1.0 + z**2 / summary["count"]
        center = (summary["mean"] + z**2 / (2.0 * summary["count"])) / denominator
        half = z * np.sqrt(summary["mean"] * (1.0 - summary["mean"]) / summary["count"] + z**2 / (4.0 * summary["count"]**2)) / denominator
        summary["ci_low"] = center - half
        summary["ci_high"] = center + half
        summary.to_csv(output / "crossdomain_summary.csv", index=False)
    frame.groupby(["study", "candidate"]).agg(
        n=("seed", "count"), median_gradient_norm=("gradient_norm", "median"),
        median_wall_seconds=("wall_seconds", "median"),
    ).reset_index().to_csv(output / "crossdomain_fit_diagnostics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "submission"), default="smoke")
    parser.add_argument("--studies", nargs="+", default=list(PROTOCOLS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=None,
                        help="Queue the first N seeds without changing the per-fit protocol hash.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    tasks = []
    for study in args.studies:
        candidates, _truths, base = PROTOCOLS[study]
        protocol = base if args.profile == "submission" else DomainProtocol(
            **{**asdict(base), "seeds": 1, "starts": 1, "adam_steps": 20, "lbfgs_steps": 4, "refine_steps": 8}
        )
        seeds = args.seeds if args.seeds is not None else protocol.seeds
        if seeds < 1 or seeds > protocol.seeds:
            parser.error("--seeds must be between 1 and the selected profile maximum")
        for seed in range(seeds):
            for candidate in candidates:
                tasks.append((study, candidate, seed, protocol, args.overwrite, args.resume))
    if args.workers == 1:
        for task in tasks:
            result = run_cell(*task)
            print(task[0], task[1], task[2], result["status"], flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_cell, *task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                print(result["study"], result["candidate"], result["seed"], result["status"], flush=True)
    aggregate()


if __name__ == "__main__":
    main()


