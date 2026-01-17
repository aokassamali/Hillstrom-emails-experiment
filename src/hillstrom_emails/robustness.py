from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd
from scipy.stats import trim_mean


def winsorize_series(series: pd.Series, quantile: float) -> pd.Series:
    cap = series.quantile(quantile)
    return series.clip(upper=cap)


def run_robustness(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    spend_col = cfg["data"]["spend_col"]
    margin = cfg["profit"]["margin"]
    email_cost = cfg["profit"]["email_cost"]

    def profit_from_spend(spend: pd.Series, emailed: pd.Series) -> pd.Series:
        return margin * spend - email_cost * emailed

    rows = []
    control = df[df["arm"] == "control"]

    # 1) Winsorize spend at 99th percentile
    spend_w = winsorize_series(df[spend_col], 0.99)
    df_w = df.copy()
    df_w["profit"] = profit_from_spend(spend_w, df_w["emailed"])
    for arm in ["mens", "womens"]:
        tau_hat = df_w[df_w["arm"] == arm]["profit"].mean() - df_w[df_w["arm"] == "control"]["profit"].mean()
        rows.append({"method": "winsorize_p99", "arm": arm, "tau_hat": float(tau_hat)})

    # 2) 1% trimmed mean on profit
    df_p = df.copy()
    df_p["profit"] = profit_from_spend(df_p[spend_col], df_p["emailed"])
    for arm in ["mens", "womens"]:
        t_vals = df_p[df_p["arm"] == arm]["profit"].to_numpy()
        c_vals = df_p[df_p["arm"] == "control"]["profit"].to_numpy()
        tau_hat = trim_mean(t_vals, 0.01) - trim_mean(c_vals, 0.01)
        rows.append({"method": "trim_mean_1pct", "arm": arm, "tau_hat": float(tau_hat)})

    # 3) log(1+spend) scale
    df_log = df.copy()
    df_log["log_spend"] = np.log1p(df_log[spend_col])
    for arm in ["mens", "womens"]:
        tau_hat = df_log[df_log["arm"] == arm]["log_spend"].mean() - df_log[df_log["arm"] == "control"]["log_spend"].mean()
        rows.append({"method": "log1p_spend", "arm": arm, "tau_hat": float(tau_hat)})

    # 4) Two-part decomposition
    conv_col = cfg["data"]["conversion_col"]
    for arm in ["mens", "womens"]:
        t = df[df["arm"] == arm]
        c = control
        conv_diff = t[conv_col].mean() - c[conv_col].mean()
        rows.append({"method": "two_part_conversion", "arm": arm, "tau_hat": float(conv_diff)})

        t_spend = t.loc[t[conv_col] == 1, spend_col]
        c_spend = c.loc[c[conv_col] == 1, spend_col]
        spend_diff = t_spend.mean() - c_spend.mean()
        rows.append({"method": "two_part_spend_among_converters", "arm": arm, "tau_hat": float(spend_diff)})

    return pd.DataFrame(rows)
