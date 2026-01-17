from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_tables(output_dir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False)


def write_figure_profit_ci(output_dir: Path, main_results: pd.DataFrame) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if main_results.empty:
        return

    arms = main_results["arm"].tolist()
    tau = main_results["tau_hat"].to_numpy()
    lower = main_results["ci_lower"].to_numpy()
    upper = main_results["ci_upper"].to_numpy()
    yerr = [tau - lower, upper - tau]

    plt.figure(figsize=(6, 4))
    plt.errorbar(arms, tau, yerr=yerr, fmt="o", capsize=4)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Profit ATE vs Control")
    plt.ylabel("Profit per customer")
    plt.tight_layout()
    plt.savefig(fig_dir / "profit_ate_ci.png", dpi=150)
    plt.close()


def write_metadata(output_dir: Path, metadata: Dict[str, Any]) -> None:
    path = output_dir / "run_metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
