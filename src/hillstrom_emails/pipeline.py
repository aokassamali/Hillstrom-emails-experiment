from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from .config import load_config
from .io import load_csv
from .cleaning import prepare_data
from .checks import data_sanity, srm_check, covariate_balance
from .estimators import estimate_primary, estimate_guardrails, ols_cross_check, adjusted_ate
from .robustness import run_robustness
from .reporting import write_tables, write_figure_profit_ci, write_metadata
from .stats import holm_adjust
from data import build_processed, normalize_columns, validate_schema, validate_values, clean_data
from data_summary import write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hillstrom email experiment pipeline")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    raw_dir = data_cfg.get("raw_dir")
    processed_path = data_cfg.get("processed_path", "data/processed/hillstrom_clean.csv")

    if raw_dir:
        processed = build_processed(Path(raw_dir), Path(processed_path))
        df_raw = load_csv(processed)
    else:
        df_raw = load_csv(data_cfg["input_csv"])
        df_raw = normalize_columns(df_raw)
        validate_schema(df_raw)
        validate_values(df_raw)
        df_raw = clean_data(df_raw)
        Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(processed_path, index=False)

    write_summary(df_raw, output_dir)
    df, cleaning_df = prepare_data(df_raw, cfg)

    issues = data_sanity(df, cfg)
    srm = srm_check(df, cfg)
    balance_df, balance_flag = covariate_balance(df, cfg)

    health_pass = (not srm["flagged"]) and (len(issues) == 0)

    main_results = pd.DataFrame()
    guardrails_df = pd.DataFrame()
    robustness_df = pd.DataFrame()
    ols_df = pd.DataFrame()
    adjusted_df = pd.DataFrame()

    if health_pass:
        main_results, df_with_profit = estimate_primary(df, cfg)
        ols_df = ols_cross_check(df_with_profit)
        guardrails_df = estimate_guardrails(df_with_profit, cfg)
        robustness_df = run_robustness(df_with_profit, cfg)
        adjusted_df = adjusted_ate(df_with_profit, cfg)

        pvals = main_results["p_value_raw"].to_numpy()
        holm = holm_adjust(pvals)
        main_results["p_value_holm"] = holm
        main_results["reject_holm"] = main_results["p_value_holm"] < 0.05

        mes = float(cfg["mes"]["min_effect"])
        main_results["mes_pass"] = main_results["tau_hat"] >= mes

        guardrail_pass = guardrails_df.pivot_table(index="arm", values="pass", aggfunc="all").reset_index()
        guardrail_pass = guardrail_pass.rename(columns={"pass": "guardrails_pass"})
        main_results = main_results.merge(guardrail_pass, on="arm", how="left")

        visit_pass = guardrails_df[guardrails_df["metric"] == "visit"]["pass"].values
        conv_pass = guardrails_df[guardrails_df["metric"] == "conversion"]["pass"].values
        if len(visit_pass) == 2:
            main_results["guardrail_visit_pass"] = visit_pass
        if len(conv_pass) == 2:
            main_results["guardrail_conversion_pass"] = conv_pass

        main_results["eligible"] = (
            main_results["reject_holm"] &
            main_results["mes_pass"] &
            main_results["guardrails_pass"]
        )

        selected_arm = None
        eligible = main_results[main_results["eligible"]]
        if not eligible.empty:
            selected_arm = eligible.sort_values("tau_hat", ascending=False).iloc[0]["arm"]
        main_results["selected"] = main_results["arm"].apply(lambda a: a == selected_arm)
    else:
        selected_arm = None

    tables = {
        "main_results": main_results,
        "guardrails": guardrails_df,
        "robustness": robustness_df,
        "balance": balance_df,
        "srm": srm["table"],
        "cleaning": cleaning_df,
        "ols_cross_check": ols_df,
        "adjusted_ate": adjusted_df,
    }

    write_tables(output_dir, tables)
    write_figure_profit_ci(output_dir, main_results)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": cfg["data"]["input_csv"],
        "rows_raw": int(len(df_raw)),
        "rows_clean": int(len(df)),
        "health": {
            "srm_flagged": bool(srm["flagged"]),
            "data_sanity_issues": issues,
            "balance_flag": bool(balance_flag),
        },
        "assumptions": {
            "srm_expected_default": bool(srm["expected_is_default"]),
            "srm_practical_delta_pp": cfg["srm"].get("practical_delta_pp"),
        },
        "decision": {
            "analysis_blocked": not health_pass,
            "selected_arm": selected_arm,
        },
    }
    write_metadata(output_dir, metadata)


if __name__ == "__main__":
    main()
