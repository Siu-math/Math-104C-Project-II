from __future__ import annotations

import numpy as np


def solve_laplace_gauss_seidel(
    h: float,
    tolerance: float = 1e-12,
    max_iterations: int = 100000,
) -> dict:
    m = _mesh_intervals(h)
    x = np.linspace(0.0, 1.0, m + 1)
    y = np.linspace(0.0, 1.0, m + 1)
    w = np.zeros((m + 1, m + 1), dtype=float)
    w[-1, :] = 100.0 * x
    w[:, -1] = 100.0 * y

    iterations = 0
    max_update = float("inf")
    converged = False

    for iteration in range(1, max_iterations + 1):
        max_update = 0.0
        for j in range(1, m):
            for i in range(1, m):
                previous = w[j, i]
                current = 0.25 * (w[j, i - 1] + w[j, i + 1] + w[j - 1, i] + w[j + 1, i])
                w[j, i] = current
                update = abs(current - previous)
                if update > max_update:
                    max_update = update

        iterations = iteration
        if max_update <= tolerance:
            converged = True
            break

    residual = _residual(w)
    max_residual = float(np.max(np.abs(residual))) if residual.size else 0.0

    return {
        "h": h,
        "x": x,
        "y": y,
        "numerical": w,
        "iterations": iterations,
        "max_update": float(max_update),
        "converged": converged,
        "residual": residual,
        "max_residual": max_residual,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
    }


def _mesh_intervals(h: float) -> int:
    if h <= 0.0:
        raise ValueError("h must be positive.")
    raw = 1.0 / h
    m = int(round(raw))
    if m < 2 or not np.isclose(raw, m, rtol=1e-12, atol=1e-12):
        raise ValueError(f"h={h} must divide the unit interval.")
    return m


def _residual(w: np.ndarray) -> np.ndarray:
    if w.shape[0] <= 2:
        return np.empty((0, 0), dtype=float)
    return 4.0 * w[1:-1, 1:-1] - w[1:-1, :-2] - w[1:-1, 2:] - w[:-2, 1:-1] - w[2:, 1:-1]
