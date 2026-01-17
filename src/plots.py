from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_arm_sizes(arm_sizes: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arm_sizes.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.bar(arm_sizes["arm"], arm_sizes["count"])
    plt.title("Arm Sizes")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_outcome_means_ci(outcome_means: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if outcome_means.empty:
        return
    for outcome in outcome_means["outcome"].unique():
        subset = outcome_means[outcome_means["outcome"] == outcome]
        plt.figure(figsize=(6, 4))
        y = subset["mean"].to_numpy()
        lower = subset["ci_low"].to_numpy()
        upper = subset["ci_high"].to_numpy()
        yerr = [y - lower, upper - y]
        plt.errorbar(subset["arm"], y, yerr=yerr, fmt="o", capsize=4)
        plt.title(f"{outcome} Mean by Arm")
        plt.ylabel(outcome)
        plt.tight_layout()
        plt.savefig(output_dir / f"mean_ci_{outcome}.png", dpi=150)
        plt.close()


def plot_uplift_ci(uplift: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if uplift.empty:
        return
    for outcome in uplift["outcome"].unique():
        subset = uplift[uplift["outcome"] == outcome]
        plt.figure(figsize=(6, 4))
        y = subset["estimate"].to_numpy()
        lower = subset["ci_low"].to_numpy()
        upper = subset["ci_high"].to_numpy()
        yerr = [y - lower, upper - y]
        plt.errorbar(subset["arm"], y, yerr=yerr, fmt="o", capsize=4)
        plt.axhline(0.0, color="black", linewidth=1)
        plt.title(f"{outcome} Uplift vs Control")
        plt.ylabel("Difference")
        plt.tight_layout()
        plt.savefig(output_dir / f"uplift_ci_{outcome}.png", dpi=150)
        plt.close()
