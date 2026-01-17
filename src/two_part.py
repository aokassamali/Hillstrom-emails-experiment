from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _bootstrap_diff(
    treat: np.ndarray,
    control: np.ndarray,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_t = len(treat)
    n_c = len(control)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        t = rng.choice(treat, size=n_t, replace=True)
        c = rng.choice(control, size=n_c, replace=True)
        diffs[i] = t.mean() - c.mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def summarize_two_part(
    df: pd.DataFrame,
    arm_col: str,
    conversion_col: str,
    spend_col: str,
    control_arm: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    arms = sorted(a for a in df[arm_col].unique() if a != control_arm)
    all_arms = [control_arm] + arms

    # Descriptive converter stats by arm
    for arm in all_arms:
        group = df[df[arm_col] == arm]
        spend = pd.to_numeric(group.loc[group[conversion_col] == 1, spend_col], errors="coerce").dropna().to_numpy()
        if len(spend) == 0:
            median, iqr, p90, p99 = (np.nan, np.nan, np.nan, np.nan)
        else:
            q1, q3 = np.quantile(spend, [0.25, 0.75])
            median = float(np.median(spend))
            iqr = float(q3 - q1)
            p90 = float(np.quantile(spend, 0.90))
            p99 = float(np.quantile(spend, 0.99))
        rows.append(
            {
                "section": "converter_distribution",
                "arm": arm,
                "estimate": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "n_converters": int(len(spend)),
                "median": median,
                "iqr": iqr,
                "p90": p90,
                "p99": p99,
                "notes": "descriptive_only",
            }
        )

    for arm in arms:
        treat = df[df[arm_col] == arm]
        control = df[df[arm_col] == control_arm]

        t_conv = pd.to_numeric(treat[conversion_col], errors="coerce").dropna().to_numpy()
        c_conv = pd.to_numeric(control[conversion_col], errors="coerce").dropna().to_numpy()
        conv_uplift = t_conv.mean() - c_conv.mean()
        ci_low, ci_high = _bootstrap_diff(t_conv, c_conv, n_boot, seed)
        rows.append(
            {
                "section": "conversion_uplift",
                "arm": arm,
                "estimate": float(conv_uplift),
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "n_converters": int(t_conv.sum()),
                "median": np.nan,
                "iqr": np.nan,
                "p90": np.nan,
                "p99": np.nan,
                "notes": "diff_in_proportions",
            }
        )

        t_spend = pd.to_numeric(treat.loc[treat[conversion_col] == 1, spend_col], errors="coerce").dropna().to_numpy()
        c_spend = pd.to_numeric(control.loc[control[conversion_col] == 1, spend_col], errors="coerce").dropna().to_numpy()
        if len(t_spend) == 0 or len(c_spend) == 0:
            spend_uplift = np.nan
            ci_low, ci_high = np.nan, np.nan
        else:
            spend_uplift = t_spend.mean() - c_spend.mean()
            ci_low, ci_high = _bootstrap_diff(t_spend, c_spend, n_boot, seed)

        if len(t_spend) == 0:
            median, iqr, p90, p99 = (np.nan, np.nan, np.nan, np.nan)
        else:
            q1, q3 = np.quantile(t_spend, [0.25, 0.75])
            median = float(np.median(t_spend))
            iqr = float(q3 - q1)
            p90 = float(np.quantile(t_spend, 0.90))
            p99 = float(np.quantile(t_spend, 0.99))
        rows.append(
            {
                "section": "spend_among_converters",
                "arm": arm,
                "estimate": float(spend_uplift) if spend_uplift == spend_uplift else np.nan,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "n_converters": int(len(t_spend)),
                "median": median,
                "iqr": iqr,
                "p90": p90,
                "p99": p99,
                "notes": "conditional_on_conversion",
            }
        )

    return pd.DataFrame(rows)
