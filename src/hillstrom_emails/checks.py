from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chisquare


def data_sanity(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    data_cfg = cfg["data"]
    spend_col = data_cfg["spend_col"]
    visit_col = data_cfg["visit_col"]
    conversion_col = data_cfg["conversion_col"]

    issues = []
    if (df[spend_col] < 0).any():
        issues.append("spend_below_zero")

    for col in [visit_col, conversion_col]:
        invalid = ~df[col].isin([0, 1])
        if invalid.any():
            issues.append(f"{col}_not_binary")

    return issues


def srm_check(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    srm_cfg = cfg["srm"]
    expected = srm_cfg.get("expected_allocation")
    practical_delta_pp = srm_cfg.get("practical_delta_pp")

    arms = ["control", "mens", "womens"]
    counts = df["arm"].value_counts().reindex(arms, fill_value=0).astype(int)
    total = counts.sum()

    if expected is None:
        expected = {"control": 0.50, "mens": 0.25, "womens": 0.25}
        expected_is_default = True
    else:
        expected_is_default = False

    expected_counts = np.array([expected[a] * total for a in arms], dtype=float)
    chi2, p_value = chisquare(f_obs=counts.values, f_exp=expected_counts)

    deviation_pp = 100.0 * (counts.values / total - np.array([expected[a] for a in arms]))

    practical_flag = False
    if practical_delta_pp is not None:
        practical_flag = np.any(np.abs(deviation_pp) > float(practical_delta_pp))

    flagged = (p_value < 0.001) or practical_flag

    table = pd.DataFrame({
        "arm": arms,
        "observed": counts.values,
        "expected": expected_counts,
        "deviation_pp": deviation_pp,
        "chi2": chi2,
        "p_value": p_value,
        "flagged": flagged,
    })

    return {
        "table": table,
        "flagged": flagged,
        "expected_is_default": expected_is_default,
    }


def _balance_design(df: pd.DataFrame, covariates: List[str]) -> pd.DataFrame:
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


def covariate_balance(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, bool]:
    covariates = cfg["data"].get("balance_covariates", [])
    if not covariates:
        return pd.DataFrame(columns=["covariate", "arm", "smd", "abs_smd", "flag"]), False

    rows = []
    design = _balance_design(df, covariates)
    control_mask = df["arm"] == "control"

    for arm in ["mens", "womens"]:
        treat_mask = df["arm"] == arm
        for cov in design.columns:
            x_t = pd.to_numeric(design.loc[treat_mask, cov], errors="coerce")
            x_c = pd.to_numeric(design.loc[control_mask, cov], errors="coerce")
            mean_t = x_t.mean()
            mean_c = x_c.mean()
            sd_t = x_t.std(ddof=1)
            sd_c = x_c.std(ddof=1)
            pooled = np.sqrt((sd_t ** 2 + sd_c ** 2) / 2.0)
            smd = np.nan
            if pooled and not np.isnan(pooled):
                smd = (mean_t - mean_c) / pooled
            rows.append(
                {
                    "covariate": cov,
                    "arm": arm,
                    "smd": smd,
                    "abs_smd": abs(smd) if smd == smd else np.nan,
                }
            )

    balance = pd.DataFrame(rows)
    balance["flag"] = balance["abs_smd"] > 0.10
    any_flag = bool(balance["flag"].any()) if not balance.empty else False

    return balance, any_flag
