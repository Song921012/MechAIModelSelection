"""Differentiable epidemic mechanism--AI candidate models."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import math
import torch


DTYPE = torch.float64
X0_SIR = torch.tensor([0.985, 0.015, 0.0], dtype=DTYPE)
X0_SEIR = torch.tensor([0.975, 0.010, 0.015, 0.0], dtype=DTYPE)


def rk4(rhs: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x0: torch.Tensor, times: torch.Tensor, max_step: float = 0.25) -> torch.Tensor:
    states = [x0]
    x = x0
    t = times[0]
    for target in times[1:]:
        interval = float(target - t)
        n_steps = max(1, int(math.ceil(abs(interval) / max_step)))
        h = (target - t) / n_steps
        for _ in range(n_steps):
            k1 = rhs(t, x)
            k2 = rhs(t + h / 2, x + h * k1 / 2)
            k3 = rhs(t + h / 2, x + h * k2 / 2)
            k4 = rhs(t + h, x + h * k3)
            x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t = t + h
        states.append(x)
    return torch.stack(states)


def _mlp_scalar(inputs: torch.Tensor, flat: torch.Tensor, hidden: int) -> torch.Tensor:
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
class Candidate:
    name: str
    dimension: int
    prior_mean: torch.Tensor
    prior_precision: torch.Tensor
    simulate: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]
    state_kind: str

    def observe(
        self,
        theta: torch.Tensor,
        times: torch.Tensor,
        observation: str,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        trajectory = self.simulate(theta, times, initial_state)
        if self.state_kind == "seir":
            sir_view = trajectory[:, (0, 2, 3)]
        else:
            sir_view = trajectory
        if observation == "infected":
            return sir_view[:, 1:2]
        if observation == "full":
            return sir_view
        raise ValueError(f"unknown observation operator: {observation}")


def _sir(theta: torch.Tensor, times: torch.Tensor, initial_state: torch.Tensor | None = None) -> torch.Tensor:
    beta, gamma = torch.exp(theta)
    def rhs(_t, x):
        s, i, r = x
        infection = beta * s * i
        return torch.stack((-infection, infection - gamma * i, gamma * i))
    x0 = X0_SIR if initial_state is None else initial_state
    return rk4(rhs, x0.to(theta), times)


def _seir(theta: torch.Tensor, times: torch.Tensor, initial_state: torch.Tensor | None = None) -> torch.Tensor:
    beta, sigma, gamma = torch.exp(theta)
    def rhs(_t, x):
        s, e, i, r = x
        infection = beta * s * i
        return torch.stack((-infection, infection - sigma * e, sigma * e - gamma * i, gamma * i))
    x0 = X0_SEIR if initial_state is None else initial_state
    return rk4(rhs, x0.to(theta), times)


def _tv_sir(theta: torch.Tensor, times: torch.Tensor, initial_state: torch.Tensor | None = None) -> torch.Tensor:
    log_beta, sine, cosine, log_gamma = theta
    gamma = torch.exp(log_gamma)
    horizon = torch.clamp(times[-1], min=1.0)
    def rhs(t, x):
        s, i, r = x
        phase = math.pi * t / horizon
        beta = torch.exp(torch.clamp(log_beta + sine * torch.sin(phase) + cosine * torch.cos(phase), -4.0, 2.0))
        infection = beta * s * i
        return torch.stack((-infection, infection - gamma * i, gamma * i))
    x0 = X0_SIR if initial_state is None else initial_state
    return rk4(rhs, x0.to(theta), times)


def _ude_factory(hidden: int):
    def simulate(theta: torch.Tensor, times: torch.Tensor, initial_state: torch.Tensor | None = None) -> torch.Tensor:
        log_beta, log_gamma = theta[:2]
        neural = theta[2:]
        gamma = torch.exp(log_gamma)
        horizon = torch.clamp(times[-1], min=1.0)
        def rhs(t, x):
            s, i, r = x
            inputs = torch.stack((2 * t / horizon - 1, 20 * i - 0.5))
            correction = 0.5 * _mlp_scalar(inputs, neural, hidden)
            beta = torch.exp(torch.clamp(log_beta + correction, -4.0, 2.0))
            infection = beta * s * i
            return torch.stack((-infection, infection - gamma * i, gamma * i))
        x0 = X0_SIR if initial_state is None else initial_state
        return rk4(rhs, x0.to(theta), times)
    return simulate


def _neural_ode_factory(hidden: int):
    def simulate(theta: torch.Tensor, times: torch.Tensor, initial_state: torch.Tensor | None = None) -> torch.Tensor:
        horizon = torch.clamp(times[-1], min=1.0)
        def rhs(t, x):
            s, i, r = x
            inputs = torch.stack((s, i, 2 * t / horizon - 1))
            rates = torch.nn.functional.softplus(_two_head_mlp(inputs, theta, hidden))
            flow_si = rates[0] * s
            flow_ir = rates[1] * i
            return torch.stack((-flow_si, flow_si - flow_ir, flow_ir))
        x0 = X0_SIR if initial_state is None else initial_state
        return rk4(rhs, x0.to(theta), times)
    return simulate


def _diagonal_precision(dimension: int, mechanistic: int, neural_sd: float = 0.5) -> torch.Tensor:
    diagonal = torch.full((dimension,), 1.0 / neural_sd**2, dtype=DTYPE)
    diagonal[:mechanistic] = 1.0
    return torch.diag(diagonal)


def build_candidates() -> dict[str, Candidate]:
    candidates = {
        "sir": Candidate("SIR", 2, torch.tensor([0.0, -1.0], dtype=DTYPE), torch.eye(2, dtype=DTYPE), _sir, "sir"),
        "seir": Candidate("SEIR", 3, torch.tensor([0.0, -0.5, -1.0], dtype=DTYPE), torch.eye(3, dtype=DTYPE), _seir, "seir"),
        "tv_sir": Candidate("TV-SIR", 4, torch.tensor([0.0, 0.0, 0.0, -1.0], dtype=DTYPE), torch.eye(4, dtype=DTYPE), _tv_sir, "sir"),
    }
    for hidden in (2, 4, 8):
        ude_dim = 2 + 4 * hidden + 1
        ude_mean = torch.zeros(ude_dim, dtype=DTYPE); ude_mean[1] = -1.0
        candidates[f"ude_sir_h{hidden}"] = Candidate(
            f"UDE-SIR({hidden})", ude_dim, ude_mean,
            _diagonal_precision(ude_dim, 2), _ude_factory(hidden), "sir",
        )
        node_dim = 6 * hidden + 2
        node_mean = torch.zeros(node_dim, dtype=DTYPE)
        candidates[f"neural_ode_h{hidden}"] = Candidate(
            f"NeuralODE({hidden})", node_dim, node_mean,
            _diagonal_precision(node_dim, 0), _neural_ode_factory(hidden), "sir",
        )
    return candidates


CANDIDATES = build_candidates()

