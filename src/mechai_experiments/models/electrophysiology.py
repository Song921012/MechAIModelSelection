"""FitzHugh--Nagumo candidates with mechanistic and learned current terms."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import torch

from ..dynamics import rk4


DTYPE = torch.float64
X0 = torch.tensor([-1.0, 1.0], dtype=DTYPE)


def _scalar_mlp(inputs: torch.Tensor, flat: torch.Tensor, hidden: int) -> torch.Tensor:
    offset = 0
    w1 = flat[offset:offset + 2 * hidden].reshape(hidden, 2); offset += 2 * hidden
    b1 = flat[offset:offset + hidden]; offset += hidden
    w2 = flat[offset:offset + hidden]; offset += hidden
    b2 = flat[offset]
    return torch.dot(w2, torch.tanh(w1 @ inputs + b1)) + b2


def _two_head_mlp(inputs: torch.Tensor, flat: torch.Tensor, hidden: int) -> torch.Tensor:
    offset = 0
    w1 = flat[offset:offset + 3 * hidden].reshape(hidden, 3); offset += 3 * hidden
    b1 = flat[offset:offset + hidden]; offset += hidden
    w2 = flat[offset:offset + 2 * hidden].reshape(2, hidden); offset += 2 * hidden
    b2 = flat[offset:offset + 2]
    return w2 @ torch.tanh(w1 @ inputs + b1) + b2


@dataclass(frozen=True)
class FHNCandidate:
    name: str
    dimension: int
    prior_mean: torch.Tensor
    prior_precision: torch.Tensor
    simulate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def observe(self, theta: torch.Tensor, times: torch.Tensor, observation: str) -> torch.Tensor:
        trajectory = self.simulate(theta, times)
        if observation == "voltage":
            return trajectory[:, :1]
        if observation == "full":
            return trajectory
        raise ValueError(observation)


def _mechanistic_rhs(theta, correction):
    a, b, log_eps, current = theta[:4]
    eps = torch.exp(log_eps)
    def rhs(t, x):
        v, w = x
        extra = correction(theta[4:], t, v, w)
        return torch.stack((v - v**3 / 3.0 - w + current + extra, eps * (v + a - b * w)))
    return rhs


def _fhn(theta, times):
    return rk4(_mechanistic_rhs(theta, lambda *_: torch.tensor(0.0, dtype=theta.dtype)), X0.to(theta), times, max_step=0.10)


def _tv_fhn(theta, times):
    horizon = times[-1]
    return rk4(_mechanistic_rhs(theta, lambda z, t, _v, _w: z[0] * torch.sin(torch.pi * t / horizon) + z[1] * torch.cos(torch.pi * t / horizon)), X0.to(theta), times, max_step=0.10)


def _ude_factory(hidden):
    def simulate(theta, times):
        horizon = times[-1]
        correction = lambda z, t, v, _w: 0.3 * _scalar_mlp(torch.stack((v / 2.0, 2.0 * t / horizon - 1.0)), z, hidden)
        return rk4(_mechanistic_rhs(theta, correction), X0.to(theta), times, max_step=0.10)
    return simulate


def _node_factory(hidden):
    def simulate(theta, times):
        horizon = times[-1]
        def rhs(t, x):
            inputs = torch.stack((x[0] / 2.0, x[1] / 2.0, 2.0 * t / horizon - 1.0))
            return 2.0 * torch.tanh(_two_head_mlp(inputs, theta, hidden))
        return rk4(rhs, X0.to(theta), times, max_step=0.10)
    return simulate


def build_candidates() -> dict[str, FHNCandidate]:
    hidden = 2
    base = torch.tensor([0.7, 0.8, -2.5257286443, 0.5], dtype=DTYPE)
    ude_neural = 4 * hidden + 1
    node_dim = 6 * hidden + 2
    return {
        "fhn": FHNCandidate("FitzHugh--Nagumo", 4, base, torch.eye(4, dtype=DTYPE), _fhn),
        "tv_fhn": FHNCandidate("Time-varying input FHN", 6, torch.cat((base, torch.zeros(2, dtype=DTYPE))), torch.eye(6, dtype=DTYPE), _tv_fhn),
        "ude_fhn": FHNCandidate("UDE current FHN", 4 + ude_neural, torch.cat((base, torch.zeros(ude_neural, dtype=DTYPE))), torch.diag(torch.cat((torch.ones(4, dtype=DTYPE), 4.0 * torch.ones(ude_neural, dtype=DTYPE)))), _ude_factory(hidden)),
        "neural_ode": FHNCandidate("Neural ODE", node_dim, torch.zeros(node_dim, dtype=DTYPE), 4.0 * torch.eye(node_dim, dtype=DTYPE), _node_factory(hidden)),
    }


CANDIDATES = build_candidates()
TRUE_PARAMETERS = {
    "fhn": torch.tensor([0.7, 0.8, -2.5257286443, 0.5], dtype=DTYPE),
    "ude_fhn": torch.tensor([
        0.7, 0.8, -2.5257286443, 0.5,
        3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0,
    ], dtype=DTYPE),
}

