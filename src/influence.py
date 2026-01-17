from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def top_share(df: pd.DataFrame, spend_col: str, arm_col: str, pct: float) -> pd.DataFrame:
    rows = []
    for arm, group in df.groupby(arm_col):
        values = pd.to_numeric(group[spend_col], errors="coerce").dropna().to_numpy()
        if len(values) == 0:
            rows.append({"arm": arm, "pct": pct, "top_share": np.nan})
            continue
        cutoff = np.quantile(values, 1 - pct)
        top_sum = values[values >= cutoff].sum()
        total = values.sum()
        share = float(top_sum / total) if total > 0 else np.nan
        rows.append({"arm": arm, "pct": pct, "top_share": share})
    return pd.DataFrame(rows)


def uplift_after_removal(
    df: pd.DataFrame,
    spend_col: str,
    arm_col: str,
    control_arm: str,
    remove_pct: float,
) -> pd.DataFrame:
    rows = []
    arms = sorted(a for a in df[arm_col].unique() if a != control_arm)
    pooled = pd.to_numeric(df[spend_col], errors="coerce").dropna().to_numpy()
    if len(pooled) == 0:
        return pd.DataFrame(columns=["arm", "remove_pct", "tau_hat", "tau_hat_removed", "fraction_attributable"])
    cutoff = np.quantile(pooled, 1 - remove_pct)
    df_trim = df.copy()
    df_trim = df_trim[df_trim[spend_col] < cutoff].copy()

    for arm in arms:
        t = pd.to_numeric(df[df[arm_col] == arm][spend_col], errors="coerce").dropna().to_numpy()
        c = pd.to_numeric(df[df[arm_col] == control_arm][spend_col], errors="coerce").dropna().to_numpy()
        t_trim = pd.to_numeric(df_trim[df_trim[arm_col] == arm][spend_col], errors="coerce").dropna().to_numpy()
        c_trim = pd.to_numeric(df_trim[df_trim[arm_col] == control_arm][spend_col], errors="coerce").dropna().to_numpy()

        tau_hat = float(t.mean() - c.mean()) if len(t) and len(c) else np.nan
        tau_hat_trim = float(t_trim.mean() - c_trim.mean()) if len(t_trim) and len(c_trim) else np.nan
        frac = np.nan
        if tau_hat not in [0.0, np.nan] and tau_hat_trim == tau_hat_trim:
            if tau_hat != 0:
                frac = float((tau_hat - tau_hat_trim) / tau_hat)
        rows.append(
            {
                "arm": arm,
                "remove_pct": remove_pct,
                "tau_hat": tau_hat,
                "tau_hat_removed": tau_hat_trim,
                "fraction_attributable": frac,
            }
        )
    return pd.DataFrame(rows)


def build_influence_table(
    df: pd.DataFrame,
    spend_col: str,
    arm_col: str,
    control_arm: str,
    pct_list: List[float] | None = None,
    remove_pct_list: List[float] | None = None,
) -> pd.DataFrame:
    if pct_list is None:
        pct_list = [0.001, 0.01]
    if remove_pct_list is None:
        remove_pct_list = [0.001, 0.01]

    frames = []
    for pct in pct_list:
        share = top_share(df, spend_col, arm_col, pct)
        share["section"] = "top_share"
        frames.append(share)

    for pct in remove_pct_list:
        uplift = uplift_after_removal(df, spend_col, arm_col, control_arm, pct)
        uplift["section"] = "uplift_after_removal"
        frames.append(uplift)

    return pd.concat(frames, ignore_index=True, sort=False)
