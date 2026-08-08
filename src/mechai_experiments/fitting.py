"""MAP fitting, geometric diagnostics, and Laplace predictive sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
import torch

from .dynamics import Candidate


@dataclass
class FitResult:
    theta: torch.Tensor
    objective: float
    deviance: float
    log_likelihood: float
    converged: bool
    iterations: int
    wall_seconds: float
    gradient_norm: float = math.nan
    best_start: int = -1
    start_diagnostics: list[dict[str, float | int | bool | str]] = field(default_factory=list)


def gaussian_deviance(prediction: torch.Tensor, target: torch.Tensor, noise: float) -> torch.Tensor:
    residual = (prediction - target) / noise
    n = residual.numel()
    return torch.sum(residual**2) + n * math.log(2.0 * math.pi * noise**2)


def prior_energy(candidate: Candidate, theta: torch.Tensor) -> torch.Tensor:
    delta = theta - candidate.prior_mean.to(theta)
    return delta @ candidate.prior_precision.to(theta) @ delta


def map_objective(candidate: Candidate, theta: torch.Tensor, times: torch.Tensor, target: torch.Tensor, observation: str, noise: float) -> torch.Tensor:
    prediction = candidate.observe(theta, times, observation)
    return 0.5 * (gaussian_deviance(prediction, target, noise) + prior_energy(candidate, theta))


def _initial_theta(candidate: Candidate, seed: int, start: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(100_003 * seed + 997 * start + candidate.dimension)
    scale = 0.20 if start == 0 else 0.45
    precision_diag = torch.diag(candidate.prior_precision)
    perturbation = torch.randn(candidate.dimension, generator=generator, dtype=torch.float64) * scale / torch.sqrt(precision_diag)
    return candidate.prior_mean + perturbation


def fit_map(
    candidate: Candidate,
    times: torch.Tensor,
    target: torch.Tensor,
    observation: str,
    noise: float,
    *,
    seed: int,
    starts: int = 2,
    adam_steps: int = 180,
    lbfgs_steps: int = 35,
    refine_steps: int = 0,
) -> FitResult:
    best: FitResult | None = None
    begin = time.perf_counter()
    diagnostics: list[dict[str, float | int | bool | str]] = []
    for start in range(starts):
        start_begin = time.perf_counter()
        theta = _initial_theta(candidate, seed, start).clone().requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=0.025 if candidate.dimension < 10 else 0.015)
        last = math.inf
        iterations = 0
        for step in range(adam_steps):
            optimizer.zero_grad()
            loss = map_objective(candidate, theta, times, target, observation, noise)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], 20.0)
            optimizer.step()
            iterations += 1
            last = float(loss.detach())
        if math.isfinite(last):
            optimizer2 = torch.optim.LBFGS([theta], lr=0.5, max_iter=lbfgs_steps, history_size=20, line_search_fn="strong_wolfe")
            calls = 0
            def closure():
                nonlocal calls
                optimizer2.zero_grad()
                value = map_objective(candidate, theta, times, target, observation, noise)
                value.backward()
                calls += 1
                return value
            try:
                optimizer2.step(closure)
                iterations += calls
            except (RuntimeError, AssertionError):
                pass
        fitted = theta.detach()
        with torch.no_grad():
            objective = float(map_objective(candidate, fitted, times, target, observation, noise))
            deviance = float(gaussian_deviance(candidate.observe(fitted, times, observation), target, noise))
        gradient_theta = fitted.clone().requires_grad_(True)
        gradient_value = map_objective(candidate, gradient_theta, times, target, observation, noise)
        if torch.isfinite(gradient_value):
            gradient_value.backward()
            gradient_norm = float(torch.linalg.vector_norm(gradient_theta.grad.detach()))
        else:
            gradient_norm = math.inf
        diagnostic = {
            "start": start,
            "objective": objective,
            "deviance": deviance,
            "gradient_norm": gradient_norm,
            "iterations": iterations,
            "finite": math.isfinite(objective) and math.isfinite(gradient_norm),
            "wall_seconds": time.perf_counter() - start_begin,
        }
        diagnostics.append(diagnostic)
        result = FitResult(
            fitted,
            objective,
            deviance,
            -0.5 * deviance,
            math.isfinite(objective),
            iterations,
            diagnostic["wall_seconds"],
            gradient_norm,
            start,
            [],
        )
        if best is None or result.objective < best.objective:
            best = result
    assert best is not None
    if refine_steps > 0 and best.converged:
        refinement_begin = time.perf_counter()
        theta = best.theta.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [theta], lr=0.25, max_iter=refine_steps, history_size=50,
            tolerance_grad=1e-9, tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
        calls = 0

        def refinement_closure():
            nonlocal calls
            optimizer.zero_grad()
            value = map_objective(candidate, theta, times, target, observation, noise)
            value.backward()
            calls += 1
            return value

        try:
            optimizer.step(refinement_closure)
        except (RuntimeError, AssertionError):
            pass
        refined = theta.detach()
        gradient_theta = refined.clone().requires_grad_(True)
        gradient_value = map_objective(
            candidate, gradient_theta, times, target, observation, noise
        )
        if torch.isfinite(gradient_value):
            gradient_value.backward()
            gradient_norm = float(torch.linalg.vector_norm(gradient_theta.grad.detach()))
            objective = float(gradient_value.detach())
            with torch.no_grad():
                deviance = float(gaussian_deviance(
                    candidate.observe(refined, times, observation), target, noise
                ))
            if objective <= best.objective + 1e-8:
                best.theta = refined
                best.objective = objective
                best.deviance = deviance
                best.log_likelihood = -0.5 * deviance
                best.gradient_norm = gradient_norm
                best.iterations += calls
        diagnostics.append({
            "stage": "refinement", "start": best.best_start,
            "objective": best.objective, "deviance": best.deviance,
            "gradient_norm": best.gradient_norm, "iterations": calls,
            "finite": best.converged,
            "wall_seconds": time.perf_counter() - refinement_begin,
        })
    best.wall_seconds = time.perf_counter() - begin
    best.start_diagnostics = diagnostics
    return best


def information_matrix(candidate: Candidate, theta: torch.Tensor, times: torch.Tensor, observation: str, noise: float) -> tuple[torch.Tensor, torch.Tensor]:
    local = theta.detach().clone().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(
        lambda z: candidate.observe(z, times, observation).reshape(-1) / noise,
        local,
        vectorize=True,
    ).reshape(-1, candidate.dimension)
    information = jacobian.mT @ jacobian
    return 0.5 * (information + information.mT), jacobian


def laplace_loglik_draws(
    candidate: Candidate,
    fit: FitResult,
    information: torch.Tensor,
    times: torch.Tensor,
    target: torch.Tensor,
    observation: str,
    noise: float,
    *,
    draws: int = 128,
    seed: int = 0,
) -> torch.Tensor:
    precision = information + candidate.prior_precision
    covariance = torch.linalg.inv(precision + 1e-8 * torch.eye(candidate.dimension, dtype=torch.float64))
    chol = torch.linalg.cholesky(covariance)
    generator = torch.Generator().manual_seed(700_001 + seed)
    standard = torch.randn(draws, candidate.dimension, generator=generator, dtype=torch.float64)
    parameters = fit.theta.unsqueeze(0) + standard @ chol.mT
    values = []
    constant = -0.5 * math.log(2.0 * math.pi * noise**2)
    for theta in parameters:
        prediction = candidate.observe(theta, times, observation)
        pointwise = constant - 0.5 * ((prediction - target) / noise) ** 2
        values.append(pointwise.reshape(-1))
    return torch.stack(values)

