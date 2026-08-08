"""Single-source scoring through the public Python package."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import torch

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "package" / "python" / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from mechai_model_selection import (  # noqa: E402
    ObservableGeometry,
    aic,
    aicc,
    bic,
    gic_effective,
    gic_volume,
    ogic_evidence,
    ogic_predictive,
    resolution_profile,
    waic,
)

from .dynamics import Candidate  # noqa: E402
from .fitting import FitResult, prior_energy  # noqa: E402


def score_fit(
    candidate: Candidate,
    fit: FitResult,
    information: torch.Tensor,
    pointwise_loglik_draws: torch.Tensor,
    n_observations: int,
    resolutions: torch.Tensor,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    geometry = ObservableGeometry.from_matrices(information, candidate.prior_precision, resolution=1.0)
    posterior = waic(pointwise_loglik_draws)
    energy = float(prior_energy(candidate, fit.theta))
    scores = {
        "aic": aic(fit.log_likelihood, candidate.dimension),
        "aicc": aicc(fit.log_likelihood, candidate.dimension, n_observations),
        "bic": bic(fit.log_likelihood, candidate.dimension, n_observations),
        "waic_laplace": posterior["waic"],
        "p_waic": posterior["p_waic"],
        "ogic_p": ogic_predictive(fit.deviance, geometry),
        "ogic_e": ogic_evidence(fit.deviance, geometry, prior_energy=energy),
        "gic_eff": gic_effective(
            fit.deviance, geometry, penalty_factor=math.log(n_observations),
        ),
        "gic_vol_025": gic_volume(
            fit.deviance, geometry, penalty_factor=math.log(n_observations), volume_weight=0.25,
        ),
        "gic_vol_050": gic_volume(
            fit.deviance, geometry, penalty_factor=math.log(n_observations), volume_weight=0.50,
        ),
        "gic_vol_100": gic_volume(
            fit.deviance, geometry, penalty_factor=math.log(n_observations), volume_weight=1.00,
        ),
        "d_obs": geometry.effective_dimension,
        "c_obs": geometry.complexity,
        "prior_energy": energy,
    }
    profile = resolution_profile(geometry.eigenvalues, resolutions)
    profile["eigenvalues"] = geometry.eigenvalues
    return scores, profile

