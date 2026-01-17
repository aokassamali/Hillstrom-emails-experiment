from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chisquare


def check_srm(df: pd.DataFrame, arm_col: str) -> Tuple[float, float]:
    arms = df[arm_col].dropna().astype(str)
    counts = arms.value_counts().sort_index()
    total = counts.sum()
    if total == 0:
        raise ValueError("No arm assignments found for SRM check.")
    expected = np.full(len(counts), total / len(counts), dtype=float)
    chi2, p_value = chisquare(f_obs=counts.values, f_exp=expected)
    return float(chi2), float(p_value)


def _balance_design(df: pd.DataFrame, covariates: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in covariates if c in df.columns]
    if not cols:
        return pd.DataFrame(index=df.index)

    cat_cols = [c for c in cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in cols if c not in cat_cols]

    parts = []
    if num_cols:
        parts.append(df[num_cols].apply(pd.to_numeric, errors="coerce"))
    if cat_cols:
        parts.append(pd.get_dummies(df[cat_cols], prefix=cat_cols, prefix_sep=":", dummy_na=False))

    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def balance_table(
    df: pd.DataFrame, covariates: Iterable[str], arm_col: str, reference_arm: str
) -> pd.DataFrame:
    design = _balance_design(df, covariates)
    if design.empty:
        return pd.DataFrame(columns=["covariate", "arm", "smd", "abs_smd"])

    arms = df[arm_col].astype(str)
    ref_mask = arms == reference_arm
    if not ref_mask.any():
        raise ValueError("Reference arm not found in data.")

    rows = []
    for arm in sorted(arms.unique()):
        if arm == reference_arm:
            continue
        treat_mask = arms == arm
        for cov in design.columns:
            x_t = pd.to_numeric(design.loc[treat_mask, cov], errors="coerce")
            x_c = pd.to_numeric(design.loc[ref_mask, cov], errors="coerce")
            mean_t = x_t.mean()
            mean_c = x_c.mean()
            sd_t = x_t.std(ddof=1)
            sd_c = x_c.std(ddof=1)
            pooled = np.sqrt((sd_t ** 2 + sd_c ** 2) / 2.0)
            smd = np.nan
            if pooled and not np.isnan(pooled):
                smd = (mean_t - mean_c) / pooled
            rows.append({"covariate": cov, "arm": arm, "smd": smd, "abs_smd": abs(smd) if smd == smd else np.nan})

    return pd.DataFrame(rows)


def permutation_test_ate(
    df: pd.DataFrame,
    outcome: str,
    arm_col: str,
    treat_arm: str,
    control_arm: str,
    n_perm: int = 5000,
    seed: int = 0,
) -> Tuple[float, float]:
    subset = df[df[arm_col].isin([treat_arm, control_arm])].copy()
    if subset.empty:
        raise ValueError("No rows found for treat/control arms.")

    y = pd.to_numeric(subset[outcome], errors="coerce").to_numpy()
    arm = subset[arm_col].astype(str).to_numpy()

    treat_mask = arm == treat_arm
    control_mask = arm == control_arm
    if not treat_mask.any() or not control_mask.any():
        raise ValueError("Treat or control arm has no rows.")

    obs = y[treat_mask].mean() - y[control_mask].mean()

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(arm)
        t_mask = perm == treat_arm
        c_mask = perm == control_arm
        diffs[i] = y[t_mask].mean() - y[c_mask].mean()

    p_value = (np.sum(np.abs(diffs) >= abs(obs)) + 1.0) / (len(diffs) + 1.0)
    return float(obs), float(p_value)
