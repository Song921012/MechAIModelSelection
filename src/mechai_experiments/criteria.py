"""Shared criterion evaluation for the versioned numerical studies."""

from __future__ import annotations

import math

import torch
from mechai_model_selection import (
    PullbackGeometry,
    aic,
    aicc,
    bic,
    gic_effective,
    gic_evidence,
    gic_predictive,
    gic_volume,
)

from .fitting import prior_energy


def primary_scores(
    candidate, fit, information: torch.Tensor, n_observations: int
) -> dict[str, float]:
    geometry = PullbackGeometry.from_matrices(
        information, candidate.prior_precision, resolution=1.0
    )
    energy = float(prior_energy(candidate, fit.theta))
    factor = math.log(n_observations)
    return {
        "aic": aic(fit.log_likelihood, candidate.dimension),
        "aicc": aicc(fit.log_likelihood, candidate.dimension, n_observations),
        "bic": bic(fit.log_likelihood, candidate.dimension, n_observations),
        "gic_eff_logn": gic_effective(fit.deviance, geometry, penalty_factor=factor),
        "gic_eff": gic_effective(
            fit.deviance, geometry, penalty_factor=math.log(n_observations)
        ),
        "gic_vol_025": gic_volume(
            fit.deviance, geometry, penalty_factor=factor, volume_weight=0.25
        ),
        "gic_vol_050": gic_volume(
            fit.deviance, geometry, penalty_factor=factor, volume_weight=0.50
        ),
        "gic_vol_100": gic_volume(
            fit.deviance, geometry, penalty_factor=factor, volume_weight=1.00
        ),
        "gic_pred": gic_predictive(fit.deviance, geometry),
        "ogic_p": gic_predictive(fit.deviance, geometry),
        "gic_evid": gic_evidence(fit.deviance, geometry, prior_energy=energy),
        "ogic_e": gic_evidence(fit.deviance, geometry, prior_energy=energy),
        "effective_dimension": geometry.effective_dimension,
        "d_obs": geometry.effective_dimension,
        "relative_log_volume": geometry.relative_log_volume,
        "c_obs": geometry.complexity,
        "prior_energy": energy,
    }
