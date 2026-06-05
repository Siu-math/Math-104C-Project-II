from __future__ import annotations

from pathlib import Path

from runner import run_all_experiments
from visualization import save_all_plots, save_all_tables


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def main() -> None:
    results = run_all_experiments()
    summaries = save_all_tables(results, OUTPUT_DIR)
    save_all_plots(results, OUTPUT_DIR)

    for problem_name, summary in summaries.items():
        print(problem_name)
        print(summary.to_string(index=False))
        print()

    print(f"Outputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
