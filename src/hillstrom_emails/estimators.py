from __future__ import annotations

from typing import Any, Dict, Tuple, List
import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_profit(df: pd.DataFrame, margin: float, email_cost: float, spend_col: str) -> pd.Series:
    return margin * df[spend_col] - email_cost * df["emailed"]


def bootstrap_diff(df_control: pd.DataFrame, df_treat: pd.DataFrame, metric: str, iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    diffs = np.empty(iterations, dtype=float)
    n_c = len(df_control)
    n_t = len(df_treat)
    c_vals = df_control[metric].to_numpy()
    t_vals = df_treat[metric].to_numpy()
    for i in range(iterations):
        c_s = rng.choice(c_vals, size=n_c, replace=True)
        t_s = rng.choice(t_vals, size=n_t, replace=True)
        diffs[i] = t_s.mean() - c_s.mean()
    return diffs


def estimate_primary(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    margin = cfg["profit"]["margin"]
    email_cost = cfg["profit"]["email_cost"]
    spend_col = cfg["data"]["spend_col"]
    iterations = int(cfg["bootstrap"]["iterations"])
    seed = int(cfg["bootstrap"]["seed"])

    df = df.copy()
    df["profit"] = compute_profit(df, margin, email_cost, spend_col)

    control = df[df["arm"] == "control"]
    results = []
    p_values = []

    for arm in ["mens", "womens"]:
        treat = df[df["arm"] == arm]
        tau_hat = treat["profit"].mean() - control["profit"].mean()
        boot = bootstrap_diff(control, treat, "profit", iterations, seed)
        ci_lower, ci_upper = np.percentile(boot, [2.5, 97.5])
        p_value = (np.sum(boot <= 0.0) + 1.0) / (len(boot) + 1.0)
        p_values.append(p_value)
        results.append({
            "arm": arm,
            "n": int(len(treat)),
            "mean_profit": float(treat["profit"].mean()),
            "tau_hat": float(tau_hat),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "p_value_raw": float(p_value),
        })

    results_df = pd.DataFrame(results)
    return results_df, df


def ols_cross_check(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mens"] = (df["arm"] == "mens").astype(int)
    df["womens"] = (df["arm"] == "womens").astype(int)

    X = sm.add_constant(df[["mens", "womens"]])
    model = sm.OLS(df["profit"], X).fit(cov_type="HC3")

    rows = []
    for arm in ["mens", "womens"]:
        coef = model.params[arm]
        p_value = model.pvalues[arm] / 2.0 if coef > 0 else 1.0 - (model.pvalues[arm] / 2.0)
        rows.append({"arm": arm, "coef": float(coef), "p_value_one_sided": float(p_value)})

    return pd.DataFrame(rows)


def _adjustment_design(df: pd.DataFrame, covariates: List[str]) -> pd.DataFrame:
    cols = [c for c in covariates if c in df.columns]
    if not cols:
        return pd.DataFrame(index=df.index)

    cat_cols = [c for c in cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in cols if c not in cat_cols]

    parts = []
    if num_cols:
        parts.append(df[num_cols].apply(pd.to_numeric, errors="coerce"))
    if cat_cols:
        parts.append(pd.get_dummies(df[cat_cols], prefix=cat_cols, prefix_sep=":", drop_first=True, dummy_na=False))

    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def adjusted_ate(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    covariates = cfg["data"].get("adjustment_covariates", [])
    design = _adjustment_design(df, covariates)
    if design.empty:
        return pd.DataFrame(columns=["arm", "coef", "p_value_one_sided"])

    df = df.copy()
    df["mens"] = (df["arm"] == "mens").astype(int)
    df["womens"] = (df["arm"] == "womens").astype(int)

    X = pd.concat([df[["mens", "womens"]], design], axis=1)
    X = sm.add_constant(X)
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["profit"], errors="coerce")
    valid = ~(X.isna().any(axis=1) | y.isna())
    X = X.loc[valid]
    y = y.loc[valid]

    if X.empty:
        return pd.DataFrame(columns=["arm", "coef", "p_value_one_sided"])

    cols = list(X.columns)
    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)
    model = sm.OLS(y_np, X_np).fit(cov_type="HC3")

    rows = []
    for arm in ["mens", "womens"]:
        idx = cols.index(arm)
        coef = model.params[idx]
        p_value = model.pvalues[idx] / 2.0 if coef > 0 else 1.0 - (model.pvalues[idx] / 2.0)
        rows.append({"arm": arm, "coef": float(coef), "p_value_one_sided": float(p_value)})

    return pd.DataFrame(rows)


def guardrail_bootstrap(df_control: pd.DataFrame, df_treat: pd.DataFrame, metric: str, iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    diffs = np.empty(iterations, dtype=float)
    n_c = len(df_control)
    n_t = len(df_treat)
    c_vals = df_control[metric].to_numpy()
    t_vals = df_treat[metric].to_numpy()
    for i in range(iterations):
        c_s = rng.choice(c_vals, size=n_c, replace=True)
        t_s = rng.choice(t_vals, size=n_t, replace=True)
        diffs[i] = (t_s.mean() - c_s.mean()) * 100.0
    return diffs


def estimate_guardrails(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    iterations = int(cfg["bootstrap"]["iterations"])
    seed = int(cfg["bootstrap"]["seed"])
    visit_col = cfg["data"]["visit_col"]
    conversion_col = cfg["data"]["conversion_col"]
    deltas = cfg["guardrails"]

    control = df[df["arm"] == "control"]
    rows = []

    for arm in ["mens", "womens"]:
        treat = df[df["arm"] == arm]
        for metric, delta_pp in [("visit", deltas["visit_delta_pp"]), ("conversion", deltas["conversion_delta_pp"])]:
            col = visit_col if metric == "visit" else conversion_col
            delta_hat = (treat[col].mean() - control[col].mean()) * 100.0
            boot = guardrail_bootstrap(control, treat, col, iterations, seed)
            ci_lower, ci_upper = np.percentile(boot, [2.5, 97.5])
            passed = ci_lower >= float(delta_pp)
            rows.append({
                "arm": arm,
                "metric": metric,
                "delta_hat_pp": float(delta_hat),
                "ci_lower_pp": float(ci_lower),
                "ci_upper_pp": float(ci_upper),
                "threshold_pp": float(delta_pp),
                "pass": bool(passed),
            })

    return pd.DataFrame(rows)
