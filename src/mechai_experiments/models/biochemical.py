"""Mechanistic and learned rate laws for a biochemical conversion system."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import torch

from ..dynamics import rk4


DTYPE = torch.float64
X0 = torch.tensor([1.0, 0.0], dtype=DTYPE)


def _scalar_mlp(inputs: torch.Tensor, flat: torch.Tensor, hidden: int) -> torch.Tensor:
    offset = 0
    w1 = flat[offset:offset + 2 * hidden].reshape(hidden, 2); offset += 2 * hidden
    b1 = flat[offset:offset + hidden]; offset += hidden
    w2 = flat[offset:offset + hidden]; offset += hidden
    b2 = flat[offset]
    return torch.dot(w2, torch.tanh(w1 @ inputs + b1)) + b2


@dataclass(frozen=True)
class BiochemicalCandidate:
    name: str
    dimension: int
    prior_mean: torch.Tensor
    prior_precision: torch.Tensor
    simulate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def observe(self, theta: torch.Tensor, times: torch.Tensor, observation: str) -> torch.Tensor:
        trajectory = self.simulate(theta, times)
        if observation == "full":
            return trajectory
        if observation == "substrate":
            return trajectory[:, :1]
        raise ValueError(observation)


def _integrate(rate_law, theta: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    def rhs(t, x):
        substrate = torch.clamp(x[0], min=1e-8)
        rate = torch.clamp(rate_law(theta, t, substrate, times[-1]), min=0.0, max=4.0)
        return torch.stack((-rate, rate))
    return rk4(rhs, X0.to(theta), times, max_step=0.10)


def _mm(theta, times):
    def rate(z, _t, substrate, _horizon):
        vmax, km = torch.exp(z)
        return vmax * substrate / (km + substrate)
    return _integrate(rate, theta, times)


def _haldane(theta, times):
    def rate(z, _t, substrate, _horizon):
        vmax, km, ki = torch.exp(z)
        return vmax * substrate / (km + substrate + substrate**2 / ki)
    return _integrate(rate, theta, times)


def _ude_factory(hidden: int):
    def simulate(theta, times):
        def rate(z, t, substrate, horizon):
            vmax, km = torch.exp(z[:2])
            inputs = torch.stack((2.0 * substrate - 1.0, 2.0 * t / horizon - 1.0))
            correction = 0.5 * _scalar_mlp(inputs, z[2:], hidden)
            return vmax * substrate / (km + substrate) * torch.exp(torch.clamp(correction, -2.0, 2.0))
        return _integrate(rate, theta, times)
    return simulate


def _neural_factory(hidden: int):
    def simulate(theta, times):
        def rate(z, t, substrate, horizon):
            inputs = torch.stack((2.0 * substrate - 1.0, 2.0 * t / horizon - 1.0))
            return torch.nn.functional.softplus(_scalar_mlp(inputs, z, hidden))
        return _integrate(rate, theta, times)
    return simulate


def build_candidates() -> dict[str, BiochemicalCandidate]:
    hidden = 2
    neural_dim = 4 * hidden + 1
    ude_dim = 2 + neural_dim
    return {
        "mm": BiochemicalCandidate("Michaelis--Menten", 2, torch.log(torch.tensor([0.7, 0.2])), torch.eye(2), _mm),
        "haldane": BiochemicalCandidate("Substrate inhibition", 3, torch.log(torch.tensor([0.8, 0.15, 0.45])), torch.eye(3), _haldane),
        "ude_mm": BiochemicalCandidate("UDE rate law", ude_dim, torch.cat((torch.log(torch.tensor([0.7, 0.2])), torch.zeros(neural_dim))), torch.diag(torch.cat((torch.ones(2), 4.0 * torch.ones(neural_dim)))), _ude_factory(hidden)),
        "neural_rate": BiochemicalCandidate("Neural rate law", neural_dim, torch.zeros(neural_dim), 4.0 * torch.eye(neural_dim), _neural_factory(hidden)),
    }


CANDIDATES = build_candidates()
TRUE_PARAMETERS = {
    "haldane": torch.log(torch.tensor([0.8, 0.15, 0.45], dtype=DTYPE)),
    "ude_mm": torch.tensor([
        -0.3566749439, -1.6094379124,
        2.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.7, 0.0, 0.0,
    ], dtype=DTYPE),
}

