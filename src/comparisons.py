from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def mens_vs_womens(
    df: pd.DataFrame,
    outcome: str,
    arm_col: str = "arm",
    n_boot: int = 10000,
    seed: int = 0,
) -> Tuple[float, float, float, float]:
    mens = pd.to_numeric(df[df[arm_col] == "mens"][outcome], errors="coerce").dropna().to_numpy()
    womens = pd.to_numeric(df[df[arm_col] == "womens"][outcome], errors="coerce").dropna().to_numpy()
    if len(mens) == 0 or len(womens) == 0:
        raise ValueError("Mens or Womens arm has no valid values.")
    estimate = float(mens.mean() - womens.mean())
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        m = rng.choice(mens, size=len(mens), replace=True)
        w = rng.choice(womens, size=len(womens), replace=True)
        diffs[i] = m.mean() - w.mean()
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    p_value = float((np.sum(np.abs(diffs) >= abs(estimate)) + 1.0) / (len(diffs) + 1.0))
    return estimate, ci_low, ci_high, p_value
