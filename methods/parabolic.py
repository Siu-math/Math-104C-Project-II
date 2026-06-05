from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np


HeatInitialFunction = Callable[[np.ndarray], np.ndarray]


def solve_forward_difference(
    h: float,
    k: float,
    t_end: float,
    initial_condition: HeatInitialFunction | None = None,
) -> dict:
    x, t = _build_grid(h, k, t_end)
    lambda_value = k / h**2
    w = np.zeros((len(t), len(x)), dtype=float)
    w[0, :] = _initial_values(x, initial_condition)
    w[0, 0] = 0.0
    w[0, -1] = 0.0

    for n in range(len(t) - 1):
        w[n + 1, 1:-1] = (1.0 - 2.0 * lambda_value) * w[n, 1:-1] + lambda_value * (w[n, :-2] + w[n, 2:])

    return _make_result("Forward Difference", h, k, t_end, x, t, w, [], [], [])


def solve_backward_difference_gauss_seidel(
    h: float,
    k: float,
    t_end: float,
    tolerance: float = 1e-12,
    max_iterations: int = 100000,
    initial_condition: HeatInitialFunction | None = None,
) -> dict:
    x, t = _build_grid(h, k, t_end)
    lambda_value = k / h**2
    w = np.zeros((len(t), len(x)), dtype=float)
    w[0, :] = _initial_values(x, initial_condition)
    w[0, 0] = 0.0
    w[0, -1] = 0.0
    iterations = []
    max_updates = []
    converged_steps = []

    for n in range(len(t) - 1):
        rhs = w[n, 1:-1].copy()
        initial = w[n, 1:-1].copy()
        solution, step_iterations, max_update, converged = _solve_tridiagonal_gauss_seidel(
            diagonal=1.0 + 2.0 * lambda_value,
            off_diagonal=-lambda_value,
            rhs=rhs,
            initial=initial,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        w[n + 1, 1:-1] = solution
        iterations.append(step_iterations)
        max_updates.append(max_update)
        converged_steps.append(converged)
        if not converged:
            raise RuntimeError("Backward Difference Gauss-Seidel did not converge.")

    return _make_result("Backward Difference", h, k, t_end, x, t, w, iterations, max_updates, converged_steps)


def solve_crank_nicolson_gauss_seidel(
    h: float,
    k: float,
    t_end: float,
    tolerance: float = 1e-12,
    max_iterations: int = 100000,
    initial_condition: HeatInitialFunction | None = None,
) -> dict:
    x, t = _build_grid(h, k, t_end)
    lambda_value = k / h**2
    w = np.zeros((len(t), len(x)), dtype=float)
    w[0, :] = _initial_values(x, initial_condition)
    w[0, 0] = 0.0
    w[0, -1] = 0.0
    iterations = []
    max_updates = []
    converged_steps = []

    for n in range(len(t) - 1):
        rhs = (
            0.5 * lambda_value * w[n, :-2]
            + (1.0 - lambda_value) * w[n, 1:-1]
            + 0.5 * lambda_value * w[n, 2:]
        )
        initial = w[n, 1:-1].copy()
        solution, step_iterations, max_update, converged = _solve_tridiagonal_gauss_seidel(
            diagonal=1.0 + lambda_value,
            off_diagonal=-0.5 * lambda_value,
            rhs=rhs,
            initial=initial,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        w[n + 1, 1:-1] = solution
        iterations.append(step_iterations)
        max_updates.append(max_update)
        converged_steps.append(converged)
        if not converged:
            raise RuntimeError("Crank-Nicolson Gauss-Seidel did not converge.")

    return _make_result("Crank-Nicolson", h, k, t_end, x, t, w, iterations, max_updates, converged_steps)


def _build_grid(h: float, k: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    if h <= 0.0 or k <= 0.0 or t_end <= 0.0:
        raise ValueError("h, k, and t_end must be positive.")
    raw_m = 1.0 / h
    raw_n = t_end / k
    m = int(round(raw_m))
    n = int(round(raw_n))
    if m < 2 or not np.isclose(raw_m, m, rtol=1e-12, atol=1e-12):
        raise ValueError(f"h={h} must divide the unit interval.")
    if n < 1 or not np.isclose(raw_n, n, rtol=1e-12, atol=1e-12):
        raise ValueError(f"k={k} must divide the time interval.")
    x = np.linspace(0.0, 1.0, m + 1)
    t = k * np.arange(n + 1, dtype=float)
    t[-1] = t_end
    return x, t


def _initial_values(x: np.ndarray, initial_condition: HeatInitialFunction | None) -> np.ndarray:
    if initial_condition is None:
        return np.sin(math.pi * x)
    return np.asarray(initial_condition(x), dtype=float)


def _solve_tridiagonal_gauss_seidel(
    diagonal: float,
    off_diagonal: float,
    rhs: np.ndarray,
    initial: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, float, bool]:
    current = initial.copy()
    size = len(current)
    if size == 0:
        return current, 0, 0.0, True

    max_update = float("inf")
    for iteration in range(1, max_iterations + 1):
        max_update = 0.0
        for i in range(size):
            left = off_diagonal * current[i - 1] if i > 0 else 0.0
            right = off_diagonal * current[i + 1] if i < size - 1 else 0.0
            value = (rhs[i] - left - right) / diagonal
            update = abs(value - current[i])
            current[i] = value
            if update > max_update:
                max_update = update
        if max_update <= tolerance:
            return current, iteration, float(max_update), True

    return current, max_iterations, float(max_update), False


def _make_result(
    method: str,
    h: float,
    k: float,
    t_end: float,
    x: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    iterations: list[int],
    max_updates: list[float],
    converged_steps: list[bool],
) -> dict:
    lambda_value = k / h**2
    return {
        "method": method,
        "h": h,
        "k": k,
        "t_end": t_end,
        "lambda": lambda_value,
        "stability_label": "stable" if lambda_value <= 0.5 else "unstable",
        "x": x,
        "t": t,
        "numerical": w,
        "iterations_per_step": np.array(iterations, dtype=int),
        "max_updates_per_step": np.array(max_updates, dtype=float),
        "converged_per_step": np.array(converged_steps, dtype=bool),
    }
