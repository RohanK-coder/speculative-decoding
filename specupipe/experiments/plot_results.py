from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(df, metric, ylabel, title, filename, token_len=None, ratio_case=None):
    if metric not in df.columns:
        print(f"Skipping missing column: {metric}")
        return

    plt.figure(figsize=(7, 5))
    base_df = df
    if token_len is not None and "max_new_tokens" in df.columns:
        base_df = base_df[base_df["max_new_tokens"] == token_len]
    if ratio_case is not None and "speed_ratio_case" in df.columns:
        base_df = base_df[base_df["speed_ratio_case"] == ratio_case]

    for mode in sorted(base_df["mode"].unique()):
        subset = base_df[base_df["mode"] == mode]
        plt.plot(subset["k"], subset[metric], marker="o", label=mode)

    suffix = ""
    if token_len is not None:
        suffix += f" ({token_len} tokens)"
    if ratio_case is not None:
        suffix += f" [{ratio_case}]"

    plt.xlabel("Speculation Depth (k)")
    plt.ylabel(ylabel)
    plt.title(title + suffix)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def main():
    out_dir = Path("outputs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("outputs/results/grid_summary.csv")
    print("Columns in grid_summary.csv:")
    print(list(df.columns))

    for token_len in sorted(df["max_new_tokens"].unique()):
        for ratio_case in sorted(df["speed_ratio_case"].unique()):
            plot_metric(df, "speedup", "Average Speedup vs Baseline", "Speedup vs Speculation Depth", out_dir / f"speedup_vs_k_{token_len}_{ratio_case}.png", token_len, ratio_case)
            plot_metric(df, "acceptance_rate", "Average Acceptance Rate", "Acceptance Rate vs Speculation Depth", out_dir / f"acceptance_vs_k_{token_len}_{ratio_case}.png", token_len, ratio_case)
            plot_metric(df, "verify_bottleneck_ratio", "Verify Bottleneck Ratio", "Verify Bottleneck vs Speculation Depth", out_dir / f"verify_bottleneck_vs_k_{token_len}_{ratio_case}.png", token_len, ratio_case)
            plot_metric(df, "stall_rounds", "Average Stall Rounds", "Stall Rounds vs Speculation Depth", out_dir / f"stall_rounds_vs_k_{token_len}_{ratio_case}.png", token_len, ratio_case)
            plot_metric(df, "backpressure_events", "Backpressure Events", "Backpressure vs Speculation Depth", out_dir / f"backpressure_vs_k_{token_len}_{ratio_case}.png", token_len, ratio_case)
            plot_metric(df, "energy_per_token_proxy", "Energy Proxy per Token", "Energy per Token Proxy vs Speculation Depth", out_dir / f"energy_per_token_vs_k_{token_len}_{ratio_case}.png", token_len, ratio_case)


if __name__ == "__main__":
    main()
