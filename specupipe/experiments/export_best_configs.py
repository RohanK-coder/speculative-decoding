import pandas as pd
from pathlib import Path


def main():
    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = pd.read_csv(out_dir / "grid_summary.csv")
    category = pd.read_csv(out_dir / "category_summary.csv")

    best_overall = grid.sort_values("speedup", ascending=False).head(10)
    worst_overall = grid.sort_values("speedup", ascending=True).head(10)

    best_by_length = (
        grid.sort_values("speedup", ascending=False)
        .groupby("max_new_tokens", as_index=False)
        .head(3)
    )

    best_by_category = (
        category.sort_values("speedup", ascending=False)
        .groupby(["category", "max_new_tokens"], as_index=False)
        .head(3)
    )

    correctness_summary = {
        "avg_output_match": float(grid["output_match"].mean()),
        "avg_speedup": float(grid["speedup"].mean()),
        "best_speedup": float(grid["speedup"].max()),
        "worst_speedup": float(grid["speedup"].min()),
    }

    best_overall.to_csv(out_dir / "best_overall_configs.csv", index=False)
    worst_overall.to_csv(out_dir / "worst_overall_configs.csv", index=False)
    best_by_length.to_csv(out_dir / "best_by_output_length.csv", index=False)
    best_by_category.to_csv(out_dir / "best_by_category.csv", index=False)

    print("\nBest overall configs:")
    print(best_overall[[
        "mode", "k", "max_new_tokens", "speedup",
        "acceptance_rate", "pipeline_utilization",
        "peak_estimated_kv_overhead_units", "output_match"
    ]].to_string(index=False))

    print("\nBest by output length:")
    print(best_by_length[[
        "mode", "k", "max_new_tokens", "speedup",
        "acceptance_rate", "pipeline_utilization",
        "peak_estimated_kv_overhead_units", "output_match"
    ]].to_string(index=False))

    print("\nBest by category:")
    print(best_by_category[[
        "category", "mode", "k", "max_new_tokens", "speedup",
        "acceptance_rate", "pipeline_utilization",
        "peak_estimated_kv_overhead_units", "output_match"
    ]].to_string(index=False))

    print("\nCorrectness / result summary:")
    print(correctness_summary)

    print("\nSaved:")
    print(" - outputs/results/best_overall_configs.csv")
    print(" - outputs/results/worst_overall_configs.csv")
    print(" - outputs/results/best_by_output_length.csv")
    print(" - outputs/results/best_by_category.csv")


if __name__ == "__main__":
    main()
