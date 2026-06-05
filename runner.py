from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from definitions import (
    ELLIPTIC_H_VALUES,
    MAX_ITERATIONS,
    TOLERANCE,
    HeatCase,
    ProblemCHeatCase,
    get_heat_cases,
    get_problem_c_cases,
)
from methods import ELLIPTIC_METHODS, HEAT_METHODS


@dataclass(frozen=True)
class ExperimentResult:
    problem: str
    method: str
    case_name: str
    raw: dict
    detail_table: pd.DataFrame | None
    point_table: pd.DataFrame | None
    summary_row: dict


def run_all_experiments() -> dict[str, list[ExperimentResult]]:
    return {
        "Problem A": run_problem_a(),
        "Problem B": run_problem_b(),
        "Problem C": run_problem_c(),
    }


def run_problem_a() -> list[ExperimentResult]:
    results = []
    for h in ELLIPTIC_H_VALUES:
        for method_name, solve in ELLIPTIC_METHODS:
            raw = solve(h, TOLERANCE, MAX_ITERATIONS)
            exact = _elliptic_exact(raw["x"], raw["y"])
            absolute_error = np.abs(exact - raw["numerical"])
            raw = {
                **raw,
                "exact": exact,
                "absolute_error": absolute_error,
            }
            detail_table = _elliptic_detail_table(raw)
            summary_row = _elliptic_summary_row(method_name, raw)
            results.append(
                ExperimentResult(
                    problem="Problem A",
                    method=method_name,
                    case_name=f"h_{_value_key(h)}",
                    raw=raw,
                    detail_table=detail_table,
                    point_table=None,
                    summary_row=summary_row,
                )
            )
    return results


def run_problem_b() -> list[ExperimentResult]:
    results = []
    for case in get_heat_cases():
        for method_name, solve in HEAT_METHODS:
            raw = _solve_heat_case(solve, case)
            exact = _heat_exact(raw["x"], raw["t"])
            absolute_error = np.abs(exact - raw["numerical"])
            raw = {
                **raw,
                "case": case,
                "exact": exact,
                "absolute_error": absolute_error,
                "max_error_over_time": np.max(absolute_error, axis=1),
                "max_abs_over_time": np.max(np.abs(raw["numerical"]), axis=1),
            }
            point_table = _heat_point_table(raw, 0.5, [0.2, 0.5, 0.8]) if case.name == "stable_h_0p1_k_0p005" and method_name == "Crank-Nicolson" else None
            summary_row = _heat_summary_row(method_name, case, raw)
            results.append(
                ExperimentResult(
                    problem="Problem B",
                    method=method_name,
                    case_name=case.name,
                    raw=raw,
                    detail_table=None,
                    point_table=point_table,
                    summary_row=summary_row,
                )
            )
    return results


def run_problem_c() -> list[ExperimentResult]:
    results = []
    for case in get_problem_c_cases():
        initial_condition = _problem_c_initial_condition(case.initial_condition)
        for method_name, solve in HEAT_METHODS:
            raw = _solve_heat_case(solve, case, initial_condition)
            exact = _problem_c_exact(case.initial_condition, raw["x"], raw["t"])
            absolute_error = np.abs(exact - raw["numerical"])
            raw = {
                **raw,
                "case": case,
                "experiment": case.experiment,
                "initial_condition": case.initial_condition,
                "exact": exact,
                "absolute_error": absolute_error,
                "max_error_over_time": np.max(absolute_error, axis=1),
                "max_abs_over_time": np.max(np.abs(raw["numerical"]), axis=1),
            }
            point_table = _heat_point_table(raw, 0.05, [0.25, 0.5, 0.75]) if case.name == "different_initial_mixed_modes" and method_name == "Crank-Nicolson" else None
            summary_row = _problem_c_summary_row(method_name, case, raw)
            results.append(
                ExperimentResult(
                    problem="Problem C",
                    method=method_name,
                    case_name=case.name,
                    raw=raw,
                    detail_table=None,
                    point_table=point_table,
                    summary_row=summary_row,
                )
            )
    return results


def _solve_heat_case(solve, case: HeatCase | ProblemCHeatCase, initial_condition=None) -> dict:
    if solve.__name__ == "solve_forward_difference":
        return solve(case.h, case.k, case.t_end, initial_condition)
    return solve(case.h, case.k, case.t_end, TOLERANCE, MAX_ITERATIONS, initial_condition)


def _elliptic_exact(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 100.0 * np.outer(y, x)


def _heat_exact(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.exp(-(np.pi**2) * t)[:, None] * np.sin(np.pi * x)[None, :]


def _problem_c_initial_condition(name: str):
    if name == "sin_2pi":
        return lambda x: np.sin(2.0 * np.pi * x)
    if name == "mixed_modes":
        return lambda x: np.sin(np.pi * x) + 0.5 * np.sin(3.0 * np.pi * x)
    raise ValueError(f"Unknown Problem C initial condition: {name}")


def _problem_c_exact(name: str, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    if name == "sin_2pi":
        return np.exp(-((2.0 * np.pi) ** 2) * t)[:, None] * np.sin(2.0 * np.pi * x)[None, :]
    if name == "mixed_modes":
        first_mode = np.exp(-(np.pi**2) * t)[:, None] * np.sin(np.pi * x)[None, :]
        third_mode = 0.5 * np.exp(-((3.0 * np.pi) ** 2) * t)[:, None] * np.sin(3.0 * np.pi * x)[None, :]
        return first_mode + third_mode
    raise ValueError(f"Unknown Problem C exact solution: {name}")


def _elliptic_detail_table(raw: dict) -> pd.DataFrame:
    rows = []
    for j, y_value in enumerate(raw["y"]):
        for i, x_value in enumerate(raw["x"]):
            rows.append(
                {
                    "x": x_value,
                    "y": y_value,
                    "exact solution": raw["exact"][j, i],
                    "numerical approximation": raw["numerical"][j, i],
                    "absolute error": raw["absolute_error"][j, i],
                }
            )
    return pd.DataFrame(rows)


def _heat_point_table(raw: dict, time_value: float, x_values: list[float]) -> pd.DataFrame:
    rows = []
    time_index = int(round(time_value / raw["k"]))
    for x_value in x_values:
        space_index = int(np.argmin(np.abs(raw["x"] - x_value)))
        rows.append(
            {
                "x": raw["x"][space_index],
                "t": raw["t"][time_index],
                "exact solution": raw["exact"][time_index, space_index],
                "numerical approximation": raw["numerical"][time_index, space_index],
                "absolute error": raw["absolute_error"][time_index, space_index],
            }
        )
    return pd.DataFrame(rows)


def _elliptic_summary_row(method_name: str, raw: dict) -> dict:
    absolute_error = raw["absolute_error"]
    return {
        "problem": "Problem A",
        "method": method_name,
        "h": raw["h"],
        "mesh intervals": int(round(1.0 / raw["h"])),
        "interior points": int((round(1.0 / raw["h"]) - 1) ** 2),
        "iterations": int(raw["iterations"]),
        "converged": bool(raw["converged"]),
        "max update": float(raw["max_update"]),
        "max residual": float(raw["max_residual"]),
        "max error": float(np.max(absolute_error)),
        "mean error": float(np.mean(absolute_error)),
    }


def _heat_summary_row(method_name: str, case: HeatCase, raw: dict) -> dict:
    iterations = raw["iterations_per_step"]
    absolute_error = raw["absolute_error"]
    if len(iterations):
        total_iterations = int(np.sum(iterations))
        max_step_iterations = int(np.max(iterations))
        converged = bool(np.all(raw["converged_per_step"]))
    else:
        total_iterations = 0
        max_step_iterations = 0
        converged = True
    return {
        "problem": "Problem B",
        "method": method_name,
        "case": case.name,
        "h": case.h,
        "k": case.k,
        "lambda": case.lambda_value,
        "stability": case.stability_label,
        "time steps": len(raw["t"]) - 1,
        "space intervals": len(raw["x"]) - 1,
        "total iterations": total_iterations,
        "max step iterations": max_step_iterations,
        "converged": converged,
        "final max error": float(np.max(absolute_error[-1, :])),
        "max error": float(np.max(absolute_error)),
        "mean error": float(np.mean(absolute_error)),
    }


def _problem_c_summary_row(method_name: str, case: ProblemCHeatCase, raw: dict) -> dict:
    iterations = raw["iterations_per_step"]
    absolute_error = raw["absolute_error"]
    if len(iterations):
        total_iterations = int(np.sum(iterations))
        max_step_iterations = int(np.max(iterations))
        converged = bool(np.all(raw["converged_per_step"]))
    else:
        total_iterations = 0
        max_step_iterations = 0
        converged = True
    return {
        "problem": "Problem C",
        "experiment": case.experiment,
        "initial condition": case.initial_condition,
        "method": method_name,
        "case": case.name,
        "h": case.h,
        "k": case.k,
        "lambda": case.lambda_value,
        "stability": case.stability_label,
        "time steps": len(raw["t"]) - 1,
        "space intervals": len(raw["x"]) - 1,
        "total iterations": total_iterations,
        "max step iterations": max_step_iterations,
        "converged": converged,
        "final max error": float(np.max(absolute_error[-1, :])),
        "max error": float(np.max(absolute_error)),
        "mean error": float(np.mean(absolute_error)),
    }


def _value_key(value: float) -> str:
    return f"{value:g}".replace(".", "p")
