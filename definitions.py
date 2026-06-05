from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeatCase:
    name: str
    h: float
    k: float
    t_end: float
    lambda_value: float
    stability_label: str


@dataclass(frozen=True)
class ProblemCHeatCase:
    name: str
    experiment: str
    initial_condition: str
    h: float
    k: float
    t_end: float
    lambda_value: float
    stability_label: str


ELLIPTIC_H_VALUES = [0.25, 0.125, 0.0625]
HEAT_T_END = 2.0
PROBLEM_C_T_END = 0.1
TOLERANCE = 1e-12
MAX_ITERATIONS = 100000
SNAPSHOT_TIMES = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0]
PROBLEM_C_SNAPSHOT_TIMES = [0.0, 0.02, 0.05, 0.1]


def get_heat_cases() -> list[HeatCase]:
    raw_cases = [
        ("stable_h_0p1_k_0p0005", 0.1, 0.0005),
        ("stable_h_0p1_k_0p005", 0.1, 0.005),
        ("stable_h_0p05_k_0p001", 0.05, 0.001),
        ("stable_h_0p025_k_0p00025", 0.025, 0.00025),
        ("unstable_h_0p1_k_0p01", 0.1, 0.01),
    ]
    cases = []
    for name, h, k in raw_cases:
        lambda_value = k / h**2
        stability_label = "stable" if lambda_value <= 0.5 else "unstable"
        cases.append(HeatCase(name, h, k, HEAT_T_END, lambda_value, stability_label))
    return cases


def get_problem_c_cases() -> list[ProblemCHeatCase]:
    raw_cases = [
        ("different_initial_sin_2pi", "different_initial_conditions", "sin_2pi", 0.05, 0.001),
        ("different_initial_mixed_modes", "different_initial_conditions", "mixed_modes", 0.05, 0.001),
        ("mesh_refinement_h_0p1", "mesh_refinement", "mixed_modes", 0.1, 0.004),
        ("mesh_refinement_h_0p05", "mesh_refinement", "mixed_modes", 0.05, 0.001),
        ("mesh_refinement_h_0p025", "mesh_refinement", "mixed_modes", 0.025, 0.00025),
        ("mesh_refinement_h_0p0125", "mesh_refinement", "mixed_modes", 0.0125, 0.0000625),
    ]
    cases = []
    for name, experiment, initial_condition, h, k in raw_cases:
        lambda_value = k / h**2
        stability_label = "stable" if lambda_value <= 0.5 else "unstable"
        cases.append(
            ProblemCHeatCase(
                name,
                experiment,
                initial_condition,
                h,
                k,
                PROBLEM_C_T_END,
                lambda_value,
                stability_label,
            )
        )
    return cases
