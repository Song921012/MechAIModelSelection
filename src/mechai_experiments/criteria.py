"""Shared criterion evaluation for the versioned numerical studies."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "package" / "python" / "src"))

from mechai_model_selection import (  # noqa: E402
    ObservableGeometry, aic, aicc, bic, gic_effective, gic_volume,
    ogic_evidence, ogic_predictive,
)

from .fitting import prior_energy  # noqa: E402


def primary_scores(candidate, fit, information: torch.Tensor, n_observations: int) -> dict[str, float]:
    geometry = ObservableGeometry.from_matrices(
        information, candidate.prior_precision, resolution=1.0
    )
    energy = float(prior_energy(candidate, fit.theta))
    factor = math.log(n_observations)
    return {
        "aic": aic(fit.log_likelihood, candidate.dimension),
        "aicc": aicc(fit.log_likelihood, candidate.dimension, n_observations),
        "bic": bic(fit.log_likelihood, candidate.dimension, n_observations),
        "gic_eff": gic_effective(fit.deviance, geometry, penalty_factor=factor),
        "gic_vol_025": gic_volume(fit.deviance, geometry, penalty_factor=factor, volume_weight=0.25),
        "gic_vol_050": gic_volume(fit.deviance, geometry, penalty_factor=factor, volume_weight=0.50),
        "gic_vol_100": gic_volume(fit.deviance, geometry, penalty_factor=factor, volume_weight=1.00),
        "ogic_p": ogic_predictive(fit.deviance, geometry),
        "ogic_e": ogic_evidence(fit.deviance, geometry, prior_energy=energy),
        "d_obs": geometry.effective_dimension,
        "c_obs": geometry.complexity,
        "prior_energy": energy,
    }

