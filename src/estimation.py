from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


def bootstrap_ci(
    treat: np.ndarray,
    control: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_t = len(treat)
    n_c = len(control)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        t = rng.choice(treat, size=n_t, replace=True)
        c = rng.choice(control, size=n_c, replace=True)
        diffs[i] = t.mean() - c.mean()
    lower = float(np.percentile(diffs, 100 * (alpha / 2)))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return lower, upper


def estimate_diff_in_means(
    df: pd.DataFrame,
    outcome: str,
    arm_col: str,
    treat_arm: str,
    control_arm: str,
) -> Tuple[float, float, float, float, float]:
    treat = pd.to_numeric(df[df[arm_col] == treat_arm][outcome], errors="coerce").dropna().to_numpy()
    control = pd.to_numeric(df[df[arm_col] == control_arm][outcome], errors="coerce").dropna().to_numpy()
    if len(treat) == 0 or len(control) == 0:
        raise ValueError("Treat or control arm has no valid outcome values.")

    estimate = float(treat.mean() - control.mean())
    se = float(np.sqrt(treat.var(ddof=1) / len(treat) + control.var(ddof=1) / len(control)))
    if se == 0.0:
        ci_low = estimate
        ci_high = estimate
        p_value = 1.0
    else:
        z = estimate / se
        p_value = float(2 * (1 - norm.cdf(abs(z))))
        ci_low = float(estimate - norm.ppf(0.975) * se)
        ci_high = float(estimate + norm.ppf(0.975) * se)

    return estimate, se, ci_low, ci_high, p_value


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p_values = np.asarray(list(p_values), dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    prev = 0.0
    for i, idx in enumerate(order):
        adj = (m - i) * p_values[idx]
        adj = max(adj, prev)
        adjusted[idx] = min(adj, 1.0)
        prev = adjusted[idx]
    return adjusted


def summarize_outcomes(
    df: pd.DataFrame,
    outcomes: Iterable[str],
    arm_col: str,
    control_arm: str,
) -> pd.DataFrame:
    outcomes = list(outcomes)
    arms = sorted(a for a in df[arm_col].dropna().unique() if a != control_arm)
    rows: List[dict] = []
    for outcome in outcomes:
        for arm in arms:
            est, se, ci_low, ci_high, p_value = estimate_diff_in_means(df, outcome, arm_col, arm, control_arm)
            rows.append(
                {
                    "outcome": outcome,
                    "arm": arm,
                    "estimate": est,
                    "se": se,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)
