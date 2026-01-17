from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from data import build_processed, clean_data, normalize_columns, validate_schema, validate_values
from data_summary import write_summary
from estimation import bootstrap_ci, estimate_diff_in_means, holm_adjust
from experiment_checks import balance_table
from plots import plot_arm_sizes, plot_outcome_means_ci, plot_uplift_ci
from hillstrom_emails.cleaning import normalize_arm
from hillstrom_emails.config import load_config
from hillstrom_emails.checks import srm_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hillstrom experiment analysis runner")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=n, replace=True).mean()
    lower = float(np.percentile(means, 100 * (alpha / 2)))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lower, upper


def _canonicalize_arms(df: pd.DataFrame, arm_col: str, arm_map: Dict[str, List[str]]) -> pd.DataFrame:
    df = df.copy()
    df["arm"] = df[arm_col].apply(lambda v: normalize_arm(v, arm_map))
    return df[df["arm"].notna()].copy()


def _arm_sizes(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["arm"].value_counts(dropna=False)
    return pd.DataFrame({"arm": counts.index.astype(str), "count": counts.values})


def _outcome_means_ci(
    df: pd.DataFrame,
    outcomes: Iterable[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        for arm, group in df.groupby("arm"):
            values = pd.to_numeric(group[outcome], errors="coerce").dropna().to_numpy()
            mean = float(np.mean(values)) if len(values) else float("nan")
            ci_low, ci_high = _bootstrap_mean_ci(values, n_boot, seed)
            rows.append(
                {
                    "outcome": outcome,
                    "arm": arm,
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def _uplift_ci(
    df: pd.DataFrame,
    outcomes: Iterable[str],
    control_arm: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    treat_arms = sorted(a for a in df["arm"].unique() if a != control_arm)
    for outcome in outcomes:
        for arm in treat_arms:
            estimate, se, ci_low, ci_high, p_value = estimate_diff_in_means(df, outcome, "arm", arm, control_arm)
            t = pd.to_numeric(df[df["arm"] == arm][outcome], errors="coerce").dropna().to_numpy()
            c = pd.to_numeric(df[df["arm"] == control_arm][outcome], errors="coerce").dropna().to_numpy()
            boot_low, boot_high = bootstrap_ci(t, c, n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "outcome": outcome,
                    "arm": arm,
                    "estimate": estimate,
                    "se": se,
                    "ci_low": boot_low,
                    "ci_high": boot_high,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = data_cfg.get("raw_dir")
    processed_path = data_cfg.get("processed_path", "data/processed/hillstrom_clean.csv")
    if raw_dir:
        processed = build_processed(Path(raw_dir), Path(processed_path))
        df = pd.read_csv(processed)
    else:
        df = pd.read_csv(data_cfg["input_csv"])
        df = normalize_columns(df)
        validate_schema(df)
        validate_values(df)
        df = clean_data(df)
        Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)

    write_summary(df, output_dir)

    df = _canonicalize_arms(df, data_cfg["arm_col"], data_cfg["arm_map"])

    margin = cfg["profit"]["margin"]
    email_cost = cfg["profit"]["email_cost"]
    df["emailed"] = df["arm"].isin(["mens", "womens"]).astype(int)
    df["profit"] = margin * df[data_cfg["spend_col"]] - email_cost * df["emailed"]

    outcomes = ["profit", data_cfg["visit_col"], data_cfg["conversion_col"], data_cfg["spend_col"]]

    n_boot = int(cfg["bootstrap"]["iterations"])
    seed = int(cfg["bootstrap"]["seed"])

    arm_sizes = _arm_sizes(df)
    arm_sizes.to_csv(tables_dir / "arm_sizes.csv", index=False)
    plot_arm_sizes(arm_sizes, figures_dir / "arm_sizes.png")

    outcome_means = _outcome_means_ci(df, outcomes, n_boot, seed)
    outcome_means.to_csv(tables_dir / "outcome_means_ci.csv", index=False)
    plot_outcome_means_ci(outcome_means, figures_dir)

    uplift = _uplift_ci(df, outcomes, "control", n_boot, seed)
    if not uplift.empty:
        profit_mask = uplift["outcome"] == "profit"
        if profit_mask.any():
            uplift.loc[profit_mask, "p_value_holm"] = holm_adjust(uplift.loc[profit_mask, "p_value"])
        else:
            uplift["p_value_holm"] = np.nan
    uplift.to_csv(tables_dir / "uplift_ci.csv", index=False)
    plot_uplift_ci(uplift, figures_dir)

    guardrails = []
    for metric, delta_pp in [
        (data_cfg["visit_col"], cfg["guardrails"]["visit_delta_pp"]),
        (data_cfg["conversion_col"], cfg["guardrails"]["conversion_delta_pp"]),
    ]:
        sub = uplift[uplift["outcome"] == metric].copy()
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            ci_low_pp = row["ci_low"] * 100.0
            ci_high_pp = row["ci_high"] * 100.0
            guardrails.append(
                {
                    "arm": row["arm"],
                    "metric": metric,
                    "delta_hat_pp": row["estimate"] * 100.0,
                    "ci_low_pp": ci_low_pp,
                    "ci_high_pp": ci_high_pp,
                    "threshold_pp": delta_pp,
                    "pass": ci_low_pp >= float(delta_pp),
                }
            )
    guardrails_df = pd.DataFrame(guardrails)
    guardrails_df.to_csv(tables_dir / "guardrails.csv", index=False)

    srm = srm_check(df, cfg)
    srm["table"].to_csv(tables_dir / "srm.csv", index=False)
    balance = balance_table(df, data_cfg.get("balance_covariates", []), "arm", "control")
    balance.to_csv(tables_dir / "balance.csv", index=False)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": data_cfg.get("input_csv"),
        "rows_clean": int(len(df)),
        "health": {
            "srm_flagged": bool(srm["flagged"]),
        },
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
