from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_all_tables(results: dict, output_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "Problem A": save_problem_a_tables(results["Problem A"], output_dir),
        "Problem B": save_problem_b_tables(results["Problem B"], output_dir),
        "Problem C": save_problem_c_tables(results["Problem C"], output_dir),
    }


def save_problem_a_tables(results: list, output_dir: Path) -> pd.DataFrame:
    table_dir = output_dir / "tables" / "problem_a"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([result.summary_row for result in results]).sort_values(["h"])
    save_dataframe(summary, table_dir / "summary")
    save_dataframe(summary[["h", "iterations", "max update", "max residual", "max error"]], table_dir / "iteration_counts")
    for result in results:
        path = table_dir / f"{safe_name(result.method)}_{safe_name(result.case_name)}_grid"
        save_dataframe(result.detail_table, path)
    return summary


def save_problem_b_tables(results: list, output_dir: Path) -> pd.DataFrame:
    table_dir = output_dir / "tables" / "problem_b"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([result.summary_row for result in results]).sort_values(["case", "method"])
    save_dataframe(summary, table_dir / "summary")
    iteration_columns = [
        "method",
        "case",
        "h",
        "k",
        "lambda",
        "total iterations",
        "max step iterations",
        "converged",
    ]
    save_dataframe(summary[iteration_columns], table_dir / "iteration_counts")
    for result in results:
        if result.point_table is not None:
            save_dataframe(result.point_table, table_dir / "representative_crank_nicolson_t_0p5")
    return summary


def save_problem_c_tables(results: list, output_dir: Path) -> pd.DataFrame:
    table_dir = output_dir / "tables" / "problem_c"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([result.summary_row for result in results]).sort_values(["experiment", "case", "method"])
    save_dataframe(summary, table_dir / "summary")
    save_dataframe(_mesh_refinement_orders(summary), table_dir / "mesh_refinement_orders")
    for result in results:
        if result.point_table is not None:
            save_dataframe(result.point_table, table_dir / "representative_mixed_modes_crank_nicolson_t_0p05")
    return summary


def _mesh_refinement_orders(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mesh_summary = summary[summary["experiment"] == "mesh_refinement"]
    for method, method_df in mesh_summary.groupby("method", sort=True):
        ordered = method_df.sort_values("h", ascending=False)
        previous_h = None
        previous_max_error = None
        previous_final_error = None
        for row in ordered.to_dict("records"):
            max_order = np.nan
            final_order = np.nan
            if previous_h is not None:
                max_order = np.log(previous_max_error / row["max error"]) / np.log(previous_h / row["h"])
                final_order = np.log(previous_final_error / row["final max error"]) / np.log(previous_h / row["h"])
            rows.append(
                {
                    "method": method,
                    "h": row["h"],
                    "k": row["k"],
                    "lambda": row["lambda"],
                    "max error": row["max error"],
                    "empirical order by max error": max_order,
                    "final max error": row["final max error"],
                    "empirical order by final max error": final_order,
                }
            )
            previous_h = row["h"]
            previous_max_error = row["max error"]
            previous_final_error = row["final max error"]
    return pd.DataFrame(rows)


def save_dataframe(df: pd.DataFrame, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_without_suffix.with_suffix(".csv"), index=False)


def safe_name(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("=", "")
        .replace(".", "p")
        .replace("__", "_")
    )
