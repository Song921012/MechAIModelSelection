"""Ecological candidate family for cross-domain model-selection transfer."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import math
import torch

from ..dynamics import rk4


DTYPE = torch.float64
Z0 = torch.log(torch.tensor([0.75, 0.35], dtype=DTYPE))


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
class EcologyCandidate:
    name: str
    dimension: int
    prior_mean: torch.Tensor
    prior_precision: torch.Tensor
    simulate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def observe(self, theta: torch.Tensor, times: torch.Tensor, observation: str) -> torch.Tensor:
        trajectory = self.simulate(theta, times)
        if observation == "full":
            return trajectory
        if observation == "prey":
            return trajectory[:, :1]
        raise ValueError(observation)


def _lv(theta: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    alpha, beta, delta, gamma = torch.exp(theta)
    def rhs(_t, z):
        prey, predator = torch.exp(z)
        return torch.stack((alpha - beta * predator, delta * prey - gamma))
    return torch.exp(rk4(rhs, Z0.to(theta), times, max_step=0.12))


def _rm(theta: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    growth, capacity, attack, handling, efficiency, mortality = torch.exp(theta)
    def rhs(_t, z):
        prey, predator = torch.exp(z)
        response = attack * prey / (1.0 + attack * handling * prey)
        return torch.stack((growth * (1.0 - prey / capacity) - response * predator / prey, efficiency * response - mortality))
    return torch.exp(rk4(rhs, Z0.to(theta), times, max_step=0.10))


def _ude_factory(hidden: int):
    def simulate(theta: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        alpha, beta, delta, gamma = torch.exp(theta[:4])
        neural = theta[4:]
        def rhs(_t, z):
            prey, predator = torch.exp(z)
            inputs = torch.stack((prey - 0.7, predator - 0.4))
            modifier = torch.exp(torch.clamp(0.35 * _scalar_mlp(inputs, neural, hidden), -1.5, 1.5))
            interaction = beta * modifier * predator
            return torch.stack((alpha - interaction, delta * prey - gamma))
        return torch.exp(rk4(rhs, Z0.to(theta), times, max_step=0.10))
    return simulate


def _node_factory(hidden: int):
    def simulate(theta: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        horizon = torch.clamp(times[-1], min=1.0)
        def rhs(t, z):
            populations = torch.exp(z)
            inputs = torch.stack((populations[0] - 0.7, populations[1] - 0.4, 2 * t / horizon - 1))
            return 1.5 * torch.tanh(_two_head_mlp(inputs, theta, hidden))
        return torch.exp(rk4(rhs, Z0.to(theta), times, max_step=0.10))
    return simulate


def candidates() -> dict[str, EcologyCandidate]:
    hidden = 2
    ude_dim = 4 + 4 * hidden + 1
    node_dim = 6 * hidden + 2
    ude_mean = torch.zeros(ude_dim, dtype=DTYPE)
    ude_mean[:4] = torch.log(torch.tensor([0.9, 1.0, 0.8, 0.6], dtype=DTYPE))
    node_mean = torch.zeros(node_dim, dtype=DTYPE)
    ude_precision = torch.diag(torch.cat((torch.ones(4), torch.full((ude_dim - 4,), 4.0))))
    return {
        "lv": EcologyCandidate("Lotka--Volterra", 4, torch.log(torch.tensor([0.9, 1.0, 0.8, 0.6])), torch.eye(4), _lv),
        "rm": EcologyCandidate("Rosenzweig--MacArthur", 6, torch.log(torch.tensor([1.0, 1.5, 1.1, 0.7, 0.75, 0.55])), torch.eye(6), _rm),
        "ude_lv": EcologyCandidate("UDE predator--prey", ude_dim, ude_mean, ude_precision, _ude_factory(hidden)),
        "neural_ode": EcologyCandidate("Neural ODE", node_dim, node_mean, 4.0 * torch.eye(node_dim), _node_factory(hidden)),
    }


ECOLOGY_CANDIDATES = candidates()
TRUE_RM = torch.log(torch.tensor([1.0, 1.55, 1.15, 0.65, 0.78, 0.56], dtype=DTYPE))

