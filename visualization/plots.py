from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from definitions import PROBLEM_C_SNAPSHOT_TIMES
from .tables import safe_name


def save_all_plots(results: dict, output_dir: Path) -> None:
    save_problem_a_plots(results["Problem A"], output_dir)
    save_problem_b_plots(results["Problem B"], output_dir)
    save_problem_c_plots(results["Problem C"], output_dir)


def save_problem_a_plots(results: list, output_dir: Path) -> None:
    figure_dir = output_dir / "figures" / "problem_a"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        _save_elliptic_heatmaps(result, figure_dir)
    _save_elliptic_convergence(results, figure_dir)
    _save_elliptic_iterations(results, figure_dir)


def save_problem_b_plots(results: list, output_dir: Path) -> None:
    figure_dir = output_dir / "figures" / "problem_b"
    figure_dir.mkdir(parents=True, exist_ok=True)
    cases = list(dict.fromkeys(result.case_name for result in results if result.raw["stability_label"] == "unstable"))
    for case_name in cases:
        case_results = [result for result in results if result.case_name == case_name]
        _save_heat_max_error(case_results, figure_dir)
    _save_forward_stability_comparison(results, figure_dir)
    _save_heat_refinement_summary(results, figure_dir)


def save_problem_c_plots(results: list, output_dir: Path) -> None:
    figure_dir = output_dir / "figures" / "problem_c"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _save_problem_c_initial_condition_snapshots(results, figure_dir)
    _save_problem_c_mesh_refinement(results, figure_dir)


def _save_elliptic_heatmaps(result, figure_dir: Path) -> None:
    raw = result.raw
    x = raw["x"]
    y = raw["y"]
    h_key = _value_key(raw["h"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("Exact Solution", raw["exact"]),
        ("Numerical Approximation", raw["numerical"]),
        ("Absolute Error", raw["absolute_error"]),
    ]
    for ax, (title, values) in zip(axes, panels):
        image = ax.imshow(values, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Problem A: h={raw['h']:g}")
    fig.tight_layout()
    fig.savefig(figure_dir / f"laplace_heatmaps_h_{h_key}.png", dpi=200)
    plt.close(fig)


def _save_elliptic_convergence(results: list, figure_dir: Path) -> None:
    ordered = sorted(results, key=lambda result: result.summary_row["h"], reverse=True)
    labels = [f"h={result.summary_row['h']:g}" for result in ordered]
    errors = np.array([result.summary_row["max error"] for result in ordered], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, errors, color="steelblue")
    ax.set_title("Problem A: Max Error vs Mesh Size")
    ax.set_xlabel("h")
    ax.set_ylabel("max absolute error")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(figure_dir / "laplace_max_error_vs_h.png", dpi=200)
    plt.close(fig)


def _save_elliptic_iterations(results: list, figure_dir: Path) -> None:
    ordered = sorted(results, key=lambda result: result.summary_row["h"], reverse=True)
    labels = [f"h={result.summary_row['h']:g}" for result in ordered]
    iterations = [result.summary_row["iterations"] for result in ordered]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, iterations, color="seagreen")
    ax.set_title("Problem A: Gauss-Seidel Iterations")
    ax.set_xlabel("h")
    ax.set_ylabel("iterations")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(figure_dir / "laplace_gauss_seidel_iterations.png", dpi=200)
    plt.close(fig)


def _save_heat_max_error(results: list, figure_dir: Path) -> None:
    case_name = results[0].case_name
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = _colors(results)
    for result in results:
        error = np.where(result.raw["max_error_over_time"] > 0.0, result.raw["max_error_over_time"], np.nan)
        ax.plot(result.raw["t"], error, linewidth=1.5, color=colors[result.method], label=result.method)
    ax.set_title(f"Problem B Max Error Over Time: {case_name}")
    ax.set_xlabel("t")
    ax.set_ylabel("max absolute error")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / f"heat_max_error_{safe_name(case_name)}.png", dpi=200)
    plt.close(fig)


def _save_forward_stability_comparison(results: list, figure_dir: Path) -> None:
    forward_results = [result for result in results if result.method == "Forward Difference"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("viridis")
    for index, result in enumerate(forward_results):
        label = f"h={result.raw['h']:g}, k={result.raw['k']:g}, lambda={result.raw['lambda']:g}"
        values = np.where(result.raw["max_abs_over_time"] > 0.0, result.raw["max_abs_over_time"], np.nan)
        ax.plot(result.raw["t"], values, linewidth=1.4, color=cmap(index / max(1, len(forward_results) - 1)), label=label)
    ax.set_title("Problem B: Forward Difference Stability Comparison")
    ax.set_xlabel("t")
    ax.set_ylabel("max absolute numerical value")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "forward_stability_comparison.png", dpi=200)
    plt.close(fig)


def _save_heat_refinement_summary(results: list, figure_dir: Path) -> None:
    stable_results = [result for result in results if result.raw["stability_label"] == "stable"]
    methods = list(dict.fromkeys(result.method for result in stable_results))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = _colors(stable_results)
    for method in methods:
        method_results = sorted([result for result in stable_results if result.method == method], key=lambda item: (item.raw["h"], item.raw["k"]))
        x_labels = [f"h={result.raw['h']:g}\nk={result.raw['k']:g}" for result in method_results]
        y_values = [result.summary_row["final max error"] for result in method_results]
        ax.plot(range(len(method_results)), y_values, marker="o", linewidth=1.5, color=colors[method], label=method)
    ax.set_title("Problem B Stable Cases: Final Max Error")
    ax.set_xlabel("mesh and time-step case")
    ax.set_ylabel("final max absolute error")
    ax.set_yscale("log")
    ax.set_xticks(range(max(len([result for result in stable_results if result.method == method]) for method in methods)))
    ax.set_xticklabels([f"case {index + 1}" for index in range(max(len([result for result in stable_results if result.method == method]) for method in methods))])
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "heat_stable_final_error_summary.png", dpi=200)
    plt.close(fig)


def _save_problem_c_initial_condition_snapshots(results: list, figure_dir: Path) -> None:
    selected = [
        result
        for result in results
        if result.raw["experiment"] == "different_initial_conditions" and result.method == "Crank-Nicolson"
    ]
    selected = sorted(selected, key=lambda result: result.raw["initial_condition"])
    fig, axes = plt.subplots(1, len(selected), figsize=(12, 5), sharey=True)
    if len(selected) == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis")
    for ax, result in zip(axes, selected):
        raw = result.raw
        for index, snapshot_time in enumerate(PROBLEM_C_SNAPSHOT_TIMES):
            time_index = int(round(snapshot_time / raw["k"]))
            color = cmap(index / max(1, len(PROBLEM_C_SNAPSHOT_TIMES) - 1))
            ax.plot(raw["x"], raw["numerical"][time_index, :], linewidth=1.8, color=color, label=f"t={snapshot_time:g}")
        ax.set_title(_initial_condition_title(raw["initial_condition"]))
        ax.set_xlabel("x")
        ax.grid(True, linestyle=":", linewidth=0.6)
    axes[0].set_ylabel("u(x,t)")
    axes[-1].legend()
    fig.suptitle("Problem C: Different Initial Conditions")
    fig.tight_layout()
    fig.savefig(figure_dir / "different_initial_conditions_snapshots.png", dpi=200)
    plt.close(fig)


def _save_problem_c_mesh_refinement(results: list, figure_dir: Path) -> None:
    selected = [result for result in results if result.raw["experiment"] == "mesh_refinement"]
    methods = list(dict.fromkeys(result.method for result in selected))
    colors = _colors(selected)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in methods:
        method_results = sorted([result for result in selected if result.method == method], key=lambda result: result.raw["h"], reverse=True)
        h_values = np.array([result.raw["h"] for result in method_results], dtype=float)
        errors = np.array([result.summary_row["max error"] for result in method_results], dtype=float)
        ax.loglog(h_values, errors, marker="o", linewidth=1.8, color=colors[method], label=method)
    ax.invert_xaxis()
    ax.set_title("Problem C: Mesh Refinement Error")
    ax.set_xlabel("h")
    ax.set_ylabel("max absolute error")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "mesh_refinement_error_vs_h.png", dpi=200)
    plt.close(fig)


def _initial_condition_title(value: str) -> str:
    titles = {
        "mixed_modes": "sin(pi x) + 0.5 sin(3 pi x)",
        "sin_2pi": "sin(2 pi x)",
    }
    return titles[value]


def _colors(results: list) -> dict[str, tuple[float, float, float, float]]:
    methods = list(dict.fromkeys(result.method for result in results))
    cmap = plt.get_cmap("tab10")
    return {method: cmap(index % 10) for index, method in enumerate(methods)}


def _value_key(value: float) -> str:
    return f"{value:g}".replace(".", "p")
