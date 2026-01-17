from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from data import build_processed, clean_data, normalize_columns, validate_schema, validate_values, validate_no_missing
from data_summary import write_summary
from estimation import bootstrap_ci, estimate_diff_in_means, holm_adjust
from experiment_checks import balance_table
from plots import plot_arm_sizes, plot_outcome_means_ci, plot_uplift_ci, plot_influence_top_share, plot_tail_sensitivity
from hillstrom_emails.cleaning import normalize_arm
from hillstrom_emails.config import load_config
from hillstrom_emails.checks import srm_check
from robustness_suite import compute_robustness
from two_part import summarize_two_part
from comparisons import mens_vs_womens
from influence import build_influence_table


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


def _bootstrap_p_one_sided(treat: np.ndarray, control: np.ndarray, n_boot: int, seed: int) -> float:
    diffs = np.empty(n_boot, dtype=float)
    rng = np.random.default_rng(seed)
    for i in range(n_boot):
        t = rng.choice(treat, size=len(treat), replace=True)
        c = rng.choice(control, size=len(control), replace=True)
        diffs[i] = t.mean() - c.mean()
    return float((np.sum(diffs <= 0.0) + 1.0) / (len(diffs) + 1.0))


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
        validate_no_missing(df)
        df = clean_data(df)
        Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)

    write_summary(df, output_dir)

    df = _canonicalize_arms(df, data_cfg["arm_col"], data_cfg["arm_map"])
    id_col = data_cfg.get("id_col")
    if not id_col or id_col not in df.columns:
        df = df.reset_index(drop=True)
        df["__row_id__"] = df.index.astype(str)
        id_col = "__row_id__"

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

    srm = srm_check(df, cfg)
    srm["table"].to_csv(tables_dir / "srm.csv", index=False)
    balance = balance_table(df, data_cfg.get("balance_covariates", []), "arm", "control")
    balance.to_csv(tables_dir / "balance.csv", index=False)

    health_pass = not bool(srm["flagged"])

    uplift = pd.DataFrame()
    guardrails_df = pd.DataFrame()
    if health_pass:
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

    # Main results: profit-only, one-sided inference, Holm, MES, guardrails, eligibility
    profit_uplift = uplift[uplift["outcome"] == "profit"].copy() if not uplift.empty else pd.DataFrame()
    if health_pass and not profit_uplift.empty:
        p_vals = []
        for _, row in profit_uplift.iterrows():
            arm = row["arm"]
            t = pd.to_numeric(df[df["arm"] == arm]["profit"], errors="coerce").dropna().to_numpy()
            c = pd.to_numeric(df[df["arm"] == "control"]["profit"], errors="coerce").dropna().to_numpy()
            p_vals.append(_bootstrap_p_one_sided(t, c, n_boot, seed))
        profit_uplift["p_value_one_sided"] = p_vals
        profit_uplift["p_value_holm"] = holm_adjust(profit_uplift["p_value_one_sided"].to_numpy())
        profit_uplift["reject_holm"] = profit_uplift["p_value_holm"] < 0.05
        mes = float(cfg["mes"]["min_effect"])
        profit_uplift["mes_pass"] = profit_uplift["estimate"] >= mes
        guardrail_pass = guardrails_df.pivot_table(index="arm", values="pass", aggfunc="all").reset_index()
        guardrail_pass = guardrail_pass.rename(columns={"pass": "guardrails_pass"})
        main_results = profit_uplift.merge(guardrail_pass, on="arm", how="left")
        main_results["n"] = main_results["arm"].apply(lambda a: int((df["arm"] == a).sum()))
        main_results["mean_profit"] = main_results["arm"].apply(
            lambda a: float(pd.to_numeric(df[df["arm"] == a]["profit"], errors="coerce").mean())
        )
        visit_pass = guardrails_df[guardrails_df["metric"] == data_cfg["visit_col"]][["arm", "pass"]]
        conv_pass = guardrails_df[guardrails_df["metric"] == data_cfg["conversion_col"]][["arm", "pass"]]
        main_results = main_results.merge(
            visit_pass.rename(columns={"pass": "guardrail_visit_pass"}), on="arm", how="left"
        )
        main_results = main_results.merge(
            conv_pass.rename(columns={"pass": "guardrail_conversion_pass"}), on="arm", how="left"
        )
        main_results["eligible"] = (
            main_results["reject_holm"] & main_results["mes_pass"] & main_results["guardrails_pass"]
        )
        selected_arm = None
        eligible = main_results[main_results["eligible"]]
        if not eligible.empty:
            selected_arm = eligible.sort_values("estimate", ascending=False).iloc[0]["arm"]
        main_results["selected"] = main_results["arm"] == selected_arm
        main_results["note"] = "primary_one_sided;mens_vs_womens_exploratory"
        main_results = main_results.rename(
            columns={
                "estimate": "tau_hat",
                "ci_low": "ci_lower",
                "ci_high": "ci_upper",
            }
        )
        main_results = main_results[
            [
                "arm",
                "n",
                "mean_profit",
                "tau_hat",
                "ci_lower",
                "ci_upper",
                "p_value_one_sided",
                "p_value_holm",
                "reject_holm",
                "mes_pass",
                "guardrail_visit_pass",
                "guardrail_conversion_pass",
                "guardrails_pass",
                "eligible",
                "selected",
                "note",
            ]
        ]
    else:
        main_results = pd.DataFrame()
    main_results.to_csv(tables_dir / "main_results.csv", index=False)

    if health_pass:
        # Robustness suite (sensitivity estimands)
        robustness = compute_robustness(
            df,
            spend_col=data_cfg["spend_col"],
            id_col=id_col,
            margin=cfg["profit"]["margin"],
            email_cost=cfg["profit"]["email_cost"],
            n_boot=n_boot,
            seed=seed,
        )
        robustness.to_csv(tables_dir / "robustness.csv", index=False)

        # Two-part decomposition
        two_part = summarize_two_part(
            df,
            arm_col="arm",
            conversion_col=data_cfg["conversion_col"],
            spend_col=data_cfg["spend_col"],
            control_arm="control",
            n_boot=n_boot,
            seed=seed,
        )
        two_part.to_csv(tables_dir / "two_part.csv", index=False)

        # Mens vs Womens direct comparison
        rows = []
        for outcome in ["profit", data_cfg["spend_col"]]:
            est, ci_low, ci_high, p_two, p_gt_0, B, seed_used = mens_vs_womens(df, outcome, "arm", n_boot, seed)
            rows.append(
                {
                    "metric": outcome,
                    "delta_hat": est,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "p_two_sided": p_two,
                    "p_delta_gt_0": p_gt_0,
                    "B": B,
                    "seed": seed_used,
                    "notes": "exploratory_two_sided_bootstrap",
                }
            )
        mens_womens = pd.DataFrame(rows)
        mens_womens.to_csv(tables_dir / "mens_vs_womens.csv", index=False)

        # Influence diagnostics
        influence = build_influence_table(
            df,
            data_cfg["spend_col"],
            "arm",
            "control",
            id_col=id_col,
            margin=cfg["profit"]["margin"],
            email_cost=cfg["profit"]["email_cost"],
        )
        influence.to_csv(tables_dir / "influence.csv", index=False)
        plot_influence_top_share(influence, figures_dir / "influence_shares.png")

        # Tail sensitivity curve for profit
        tail_rows = []
        for x in [0.0, 0.001, 0.005, 0.01, 0.02]:
            for arm in ["mens", "womens"]:
                t = df[df["arm"] == arm].copy()
                c = df[df["arm"] == "control"].copy()
                if x > 0:
                    t_spend = pd.to_numeric(t[data_cfg["spend_col"]], errors="coerce").fillna(0.0).to_numpy()
                    c_spend = pd.to_numeric(c[data_cfg["spend_col"]], errors="coerce").fillna(0.0).to_numpy()
                    t_ids = t[id_col].astype(str).to_numpy()
                    c_ids = c[id_col].astype(str).to_numpy()
                    t_k = int(np.ceil(x * len(t_spend)))
                    c_k = int(np.ceil(x * len(c_spend)))
                    t_idx = np.lexsort((t_ids, -t_spend))[:t_k]
                    c_idx = np.lexsort((c_ids, -c_spend))[:c_k]
                    t = t.drop(t.index[t_idx])
                    c = c.drop(c.index[c_idx])
                t_profit = cfg["profit"]["margin"] * t[data_cfg["spend_col"]] - cfg["profit"]["email_cost"]
                c_profit = cfg["profit"]["margin"] * c[data_cfg["spend_col"]]
                tau = float(t_profit.mean() - c_profit.mean())
                tail_rows.append({"arm": arm, "x_removed": x, "tau_hat": tau})
        tail_df = pd.DataFrame(tail_rows)
        tail_df.to_csv(tables_dir / "tail_sensitivity_profit.csv", index=False)
        plot_tail_sensitivity(tail_df, figures_dir / "tail_sensitivity_profit.png")
    else:
        pd.DataFrame().to_csv(tables_dir / "robustness.csv", index=False)
        pd.DataFrame().to_csv(tables_dir / "two_part.csv", index=False)
        pd.DataFrame().to_csv(tables_dir / "mens_vs_womens.csv", index=False)
        pd.DataFrame().to_csv(tables_dir / "influence.csv", index=False)
        pd.DataFrame().to_csv(tables_dir / "tail_sensitivity_profit.csv", index=False)

    git_hash = None
    try:
        import subprocess

        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_hash = None

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": data_cfg.get("input_csv"),
        "rows_clean": int(len(df)),
        "health": {
            "srm_flagged": bool(srm["flagged"]),
            "health_pass": bool(health_pass),
        },
        "inference": {
            "primary": "one_sided_bootstrap",
            "exploratory": "two_sided_bootstrap",
        },
        "srm": {
            "expected_allocation": cfg["srm"].get("expected_allocation"),
        },
        "bootstrap": {
            "B": n_boot,
            "seed": seed,
        },
        "git_commit": git_hash,
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
