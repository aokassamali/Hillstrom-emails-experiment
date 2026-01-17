from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _top_k_indices(spend: np.ndarray, ids: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=int)
    order = np.lexsort((ids, -spend))
    return order[:k]


def top_share(df: pd.DataFrame, spend_col: str, arm_col: str, id_col: str, pct: float) -> pd.DataFrame:
    rows = []
    for arm, group in df.groupby(arm_col):
        spend = pd.to_numeric(group[spend_col], errors="coerce").fillna(0.0).to_numpy()
        ids = group[id_col].astype(str).to_numpy()
        n = len(spend)
        if n == 0:
            rows.append({"arm": arm, "x_removed": pct, "top_share": 0.0})
            continue
        k = int(np.ceil(pct * n))
        idx = _top_k_indices(spend, ids, k)
        top_sum = spend[idx].sum() if len(idx) else 0.0
        total = spend.sum()
        share = float(top_sum / total) if total > 0 else 0.0
        rows.append({"arm": arm, "x_removed": pct, "top_share": share})
    return pd.DataFrame(rows)


def uplift_after_removal(
    df: pd.DataFrame,
    spend_col: str,
    arm_col: str,
    control_arm: str,
    id_col: str,
    remove_pct: float,
    margin: float,
    email_cost: float,
) -> pd.DataFrame:
    rows = []
    arms = sorted(a for a in df[arm_col].unique() if a != control_arm)
    for arm in arms:
        t = df[df[arm_col] == arm]
        c = df[df[arm_col] == control_arm]

        t_spend = pd.to_numeric(t[spend_col], errors="coerce").fillna(0.0).to_numpy()
        c_spend = pd.to_numeric(c[spend_col], errors="coerce").fillna(0.0).to_numpy()
        t_ids = t[id_col].astype(str).to_numpy()
        c_ids = c[id_col].astype(str).to_numpy()

        t_k = int(np.ceil(remove_pct * len(t_spend)))
        c_k = int(np.ceil(remove_pct * len(c_spend)))
        t_idx = _top_k_indices(t_spend, t_ids, t_k)
        c_idx = _top_k_indices(c_spend, c_ids, c_k)

        t_keep = np.ones(len(t_spend), dtype=bool)
        c_keep = np.ones(len(c_spend), dtype=bool)
        t_keep[t_idx] = False
        c_keep[c_idx] = False

        t_trim = t_spend[t_keep]
        c_trim = c_spend[c_keep]

        t_profit = margin * t_spend - email_cost
        c_profit = margin * c_spend
        t_profit_trim = margin * t_trim - email_cost
        c_profit_trim = margin * c_trim

        tau_profit_trim = float(t_profit_trim.mean() - c_profit_trim.mean()) if len(t_profit_trim) and len(c_profit_trim) else np.nan
        tau_spend_trim = float(t_trim.mean() - c_trim.mean()) if len(t_trim) and len(c_trim) else np.nan

        rows.append(
            {
                "arm": arm,
                "x_removed": remove_pct,
                "top_share": np.nan,
                "uplift_after_removal_profit": tau_profit_trim,
                "uplift_after_removal_spend": tau_spend_trim,
                "notes": "rank_remove_within_arm",
            }
        )
    return pd.DataFrame(rows)


def build_influence_table(
    df: pd.DataFrame,
    spend_col: str,
    arm_col: str,
    control_arm: str,
    id_col: str,
    margin: float,
    email_cost: float,
    pct_list: List[float] | None = None,
    remove_pct_list: List[float] | None = None,
) -> pd.DataFrame:
    if pct_list is None:
        pct_list = [0.001, 0.005, 0.01]
    if remove_pct_list is None:
        remove_pct_list = [0.001, 0.005, 0.01, 0.02]

    frames = []
    for pct in pct_list:
        share = top_share(df, spend_col, arm_col, id_col, pct)
        share["notes"] = "rank_top_share"
        frames.append(share)

    for pct in remove_pct_list:
        uplift = uplift_after_removal(df, spend_col, arm_col, control_arm, id_col, pct, margin, email_cost)
        frames.append(uplift)

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.fillna({"top_share": 0.0})
    return out
