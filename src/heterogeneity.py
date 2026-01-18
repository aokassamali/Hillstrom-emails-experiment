from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from data import build_processed, clean_data, normalize_columns, validate_no_missing, validate_schema, validate_values
from hillstrom_emails.cleaning import normalize_arm
from hillstrom_emails.config import load_config

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import StratifiedKFold
except ImportError as exc:  # pragma: no cover - handled in runtime
    raise ImportError("scikit-learn is required for heterogeneity.py") from exc

ALLOWED_FEATURES = ["recency", "history", "history_segment", "zip_code", "newbie", "channel"]
FORBIDDEN_FEATURES = ["mens", "womens", "segment", "visit", "conversion", "spend"]
ARMS = ["control", "mens", "womens"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heterogeneity/targeting extension")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", default="reports/extension", help="Output directory")
    parser.add_argument("--zip-top-k", type=int, default=50, help="Top-K zip codes to keep")
    parser.add_argument("--model", choices=["rf", "gbr"], default="rf", help="Outcome model")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap iterations")
    parser.add_argument("--propensity", type=float, default=1 / 3, help="Treatment propensity")
    return parser.parse_args()


def _canonicalize_arms(df: pd.DataFrame, arm_col: str, arm_map: Dict[str, List[str]]) -> pd.DataFrame:
    df = df.copy()
    df["arm"] = df[arm_col].apply(lambda v: normalize_arm(v, arm_map))
    return df[df["arm"].notna()].copy()


def _load_data(cfg: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    data_cfg = cfg["data"]
    raw_dir = data_cfg.get("raw_dir")
    processed_path = data_cfg.get("processed_path", "data/processed/hillstrom_clean.csv")
    if raw_dir:
        try:
            processed = build_processed(Path(raw_dir), Path(processed_path))
            df = pd.read_csv(processed)
        except FileNotFoundError:
            raw_dir = None
    if not raw_dir:
        df = pd.read_csv(data_cfg["input_csv"])
        df = normalize_columns(df)
        validate_schema(df)
        validate_values(df)
        validate_no_missing(df)
        df = clean_data(df)
    return df


def _prepare_features(df: pd.DataFrame, zip_top_k: int) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    missing = [c for c in ALLOWED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing allowed features: {missing}")

    zip_series = df["zip_code"].astype(str).fillna("unknown")
    top_zip = set(zip_series.value_counts().nlargest(zip_top_k).index)
    df["zip_code_top"] = zip_series.where(zip_series.isin(top_zip), "other")

    feature_cols = ["recency", "history", "newbie", "history_segment", "channel", "zip_code_top"]
    feature_df = df[feature_cols].copy()
    feature_df["recency"] = pd.to_numeric(feature_df["recency"], errors="coerce").fillna(0.0)
    feature_df["history"] = pd.to_numeric(feature_df["history"], errors="coerce").fillna(0.0)
    feature_df["newbie"] = pd.to_numeric(feature_df["newbie"], errors="coerce").fillna(0.0)

    cat_cols = ["history_segment", "channel", "zip_code_top"]
    X = pd.get_dummies(feature_df, columns=cat_cols, drop_first=False)

    forbidden_present = [c for c in FORBIDDEN_FEATURES if c in X.columns]
    if forbidden_present:
        raise ValueError(f"Forbidden features leaked into X: {forbidden_present}")

    return X, list(X.columns)

def _make_model(model_name: str, seed: int):
    if model_name == "gbr":
        return GradientBoostingRegressor(random_state=seed)
    return RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=10,
        random_state=seed,
        n_jobs=-1,
    )


def _fit_t_learner(
    X_train: pd.DataFrame,
    T_train: pd.Series,
    Y_train: pd.Series,
    model_name: str,
    seed: int,
) -> Dict[str, object]:
    models: Dict[str, object] = {}
    for arm in ARMS:
        subset = T_train == arm
        if not subset.any():
            raise ValueError(f"No training rows for arm {arm}")
        model = _make_model(model_name, seed)
        model.fit(X_train.loc[subset], Y_train.loc[subset])
        models[arm] = model
    return models


def _predict_mu(models: Dict[str, object], X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mu_control": models["control"].predict(X),
            "mu_mens": models["mens"].predict(X),
            "mu_womens": models["womens"].predict(X),
        },
        index=X.index,
    )


def _cross_fitted_mu(
    X: pd.DataFrame,
    T: pd.Series,
    Y: pd.Series,
    model_name: str,
    seed: int,
) -> pd.DataFrame:
    mu = pd.DataFrame(index=X.index, columns=["mu_control", "mu_mens", "mu_womens"], dtype=float)
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, T):
        X_train = X.iloc[train_idx]
        T_train = T.iloc[train_idx]
        Y_train = Y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        models = _fit_t_learner(X_train, T_train, Y_train, model_name, seed)
        mu.iloc[test_idx] = _predict_mu(models, X_test)
    if mu.isna().any().any():
        raise ValueError("Cross-fitting failed to fill all mu predictions.")
    return mu


def _rng_for_key(seed: int, key: str) -> np.random.Generator:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    stream_int = int.from_bytes(digest[:4], "little", signed=False)
    ss = np.random.SeedSequence([seed, stream_int])
    return np.random.default_rng(ss)


def _get_propensity_map(propensity: float | Dict[str, float]) -> Dict[str, float]:
    if isinstance(propensity, dict):
        return {arm: float(propensity.get(arm, 0.0)) for arm in ARMS}
    return {arm: float(propensity) for arm in ARMS}


def _propensity_check(T: pd.Series, prop_map: Dict[str, float], tol: float = 0.02) -> pd.DataFrame:
    rows = []
    emp = T.value_counts(normalize=True)
    for arm in ARMS:
        empirical = float(emp.get(arm, 0.0))
        expected = float(prop_map.get(arm, 0.0))
        diff = empirical - expected
        rows.append(
            {
                "arm": arm,
                "empirical": empirical,
                "expected": expected,
                "diff": diff,
                "flag": abs(diff) > tol,
            }
        )
    return pd.DataFrame(rows)


def _profit_obs(spend: np.ndarray, T_obs: np.ndarray, margin: float, cost: float) -> np.ndarray:
    return margin * spend - cost * (T_obs != "control")


def _ipw_value(
    profit: np.ndarray,
    T_obs: np.ndarray,
    T_policy: np.ndarray,
    prop_map: Dict[str, float],
) -> float:
    weights = np.array([prop_map[t] for t in T_policy], dtype=float)
    matches = T_obs == T_policy
    return float(np.mean(profit * matches / weights))


def _dr_value(
    profit: np.ndarray,
    T_obs: np.ndarray,
    T_policy: np.ndarray,
    mu_profit: pd.DataFrame,
    prop_map: Dict[str, float],
) -> float:
    mu_policy = np.where(
        T_policy == "mens",
        mu_profit["mu_mens"].to_numpy(),
        np.where(T_policy == "womens", mu_profit["mu_womens"].to_numpy(), mu_profit["mu_control"].to_numpy()),
    )
    weights = np.array([prop_map[t] for t in T_policy], dtype=float)
    matches = T_obs == T_policy
    return float(np.mean(mu_policy + matches / weights * (profit - mu_policy)))


def _bootstrap_policy_value(
    estimator: str,
    profit: np.ndarray,
    T_obs: np.ndarray,
    T_policy: np.ndarray,
    mu_profit: pd.DataFrame,
    prop_map: Dict[str, float],
    n_boot: int,
    seed: int,
    key: str,
) -> Tuple[float, float, float]:
    rng = _rng_for_key(seed, key)
    n = len(profit)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if estimator == "dr":
            boot[i] = _dr_value(profit[idx], T_obs[idx], T_policy[idx], mu_profit.iloc[idx], prop_map)
        else:
            boot[i] = _ipw_value(profit[idx], T_obs[idx], T_policy[idx], prop_map)
    se = float(np.std(boot, ddof=1))
    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))
    return se, ci_low, ci_high


def sanity_check_policy_value(
    df_eval: pd.DataFrame,
    margin: float,
    cost: float,
    prop_map: Dict[str, float],
    tol: float = 1e-2,
) -> pd.DataFrame:
    rows = []
    profit = _profit_obs(df_eval["spend"].to_numpy(), df_eval["arm"].to_numpy(), margin, cost)
    for policy, arm in [
        ("treat_none", "control"),
        ("treat_all_mens", "mens"),
        ("treat_all_womens", "womens"),
    ]:
        assigned = np.full(len(df_eval), arm, dtype=object)
        ipw = _ipw_value(profit, df_eval["arm"].to_numpy(), assigned, prop_map)
        arm_mean = float(np.mean(profit[df_eval["arm"] == arm]))
        diff = abs(ipw - arm_mean)
        rows.append(
            {
                "policy": policy,
                "arm": arm,
                "ipw_value": ipw,
                "arm_mean": arm_mean,
                "abs_diff": diff,
                "flag": diff > tol,
                "notes": "ipw_vs_within_arm_mean",
            }
        )
    return pd.DataFrame(rows)

def _policy_assignments(dprofit_mens: np.ndarray, dprofit_womens: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(dprofit_mens)
    base = np.zeros(n, dtype=float)
    best_idx = np.argmax(np.vstack([base, dprofit_mens, dprofit_womens]), axis=0)
    best_map = np.array(["control", "mens", "womens"], dtype=object)
    best = best_map[best_idx]

    out: Dict[str, np.ndarray] = {
        "treat_none": np.full(n, "control", dtype=object),
        "treat_all_mens": np.full(n, "mens", dtype=object),
        "treat_all_womens": np.full(n, "womens", dtype=object),
        "send_or_not_mens": np.where(dprofit_mens > 0, "mens", "control"),
        "best_of_three": best,
    }

    top_scores = np.maximum(dprofit_mens, dprofit_womens)
    order = np.argsort(-top_scores)
    for k in [0.01, 0.05, 0.10, 0.20]:
        cutoff = int(np.ceil(k * n))
        assigned = np.full(n, "control", dtype=object)
        if cutoff > 0:
            idx = order[:cutoff]
            best_arm = np.where(dprofit_mens[idx] >= dprofit_womens[idx], "mens", "womens")
            assigned[idx] = best_arm
        out[f"top_k_best_of_three_{k:.2f}"] = assigned
    return out


def _policy_assignments_binary(dprofit_mens: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(dprofit_mens)
    out: Dict[str, np.ndarray] = {
        "treat_none": np.full(n, "control", dtype=object),
        "treat_all_mens": np.full(n, "mens", dtype=object),
        "send_or_not_mens": np.where(dprofit_mens > 0, "mens", "control"),
    }
    order = np.argsort(-dprofit_mens)
    for k in [0.01, 0.05, 0.10, 0.20, 0.50]:
        cutoff = int(np.ceil(k * n))
        assigned = np.full(n, "control", dtype=object)
        if cutoff > 0:
            assigned[order[:cutoff]] = "mens"
        out[f"top_k_mens_{k:.2f}"] = assigned
    return out


def _assignment_summary(assignments: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for policy, assigned in assignments.items():
        counts = pd.Series(assigned).value_counts(normalize=True)
        rows.append(
            {
                "policy": policy,
                "assigned_control": float(counts.get("control", 0.0)),
                "assigned_mens": float(counts.get("mens", 0.0)),
                "assigned_womens": float(counts.get("womens", 0.0)),
                "treated_rate": float(1.0 - counts.get("control", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _distribution_summary(values: np.ndarray, label: str) -> Dict[str, float]:
    return {
        "metric": label,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def _policy_value_table(
    assignments: Dict[str, np.ndarray],
    profit_obs: np.ndarray,
    T_obs: np.ndarray,
    mu_profit: pd.DataFrame,
    prop_map: Dict[str, float],
    n_boot: int,
    seed: int,
    notes: str,
) -> pd.DataFrame:
    rows = []
    for policy, assigned in assignments.items():
        treated_rate = float(np.mean(assigned != "control"))
        for estimator in ["ipw", "dr"]:
            value = (
                _dr_value(profit_obs, T_obs, assigned, mu_profit, prop_map)
                if estimator == "dr"
                else _ipw_value(profit_obs, T_obs, assigned, prop_map)
            )
            _, ci_low, ci_high = _bootstrap_policy_value(
                estimator,
                profit_obs,
                T_obs,
                assigned,
                mu_profit,
                prop_map,
                n_boot=n_boot,
                seed=seed,
                key=f"policy:{policy}|estimator:{estimator}",
            )
            rows.append(
                {
                    "policy": policy,
                    "estimator": estimator,
                    "value_hat": value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "treated_rate": treated_rate,
                    "notes": notes,
                }
            )
    return pd.DataFrame(rows)


def _dr_tau_mens_control(
    profit: np.ndarray,
    T_obs: np.ndarray,
    mu_profit: pd.DataFrame,
    prop_map: Dict[str, float],
) -> float:
    mu_mens = mu_profit["mu_mens"].to_numpy()
    mu_control = mu_profit["mu_control"].to_numpy()
    term_mens = (T_obs == "mens") / prop_map["mens"] * (profit - mu_mens)
    term_control = (T_obs == "control") / prop_map["control"] * (profit - mu_control)
    return float(np.mean((mu_mens - mu_control) + term_mens - term_control))


def _bootstrap_tau_mens_control(
    profit: np.ndarray,
    T_obs: np.ndarray,
    mu_profit: pd.DataFrame,
    prop_map: Dict[str, float],
    n_boot: int,
    seed: int,
    key: str,
) -> Tuple[float, float, float]:
    rng = _rng_for_key(seed, key)
    n = len(profit)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot[i] = _dr_tau_mens_control(profit[idx], T_obs[idx], mu_profit.iloc[idx], prop_map)
    se = float(np.std(boot, ddof=1))
    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))
    return se, ci_low, ci_high


def _decile_diagnostics(
    scores: np.ndarray,
    label: str,
    outcome: np.ndarray,
    T_obs: np.ndarray,
    mu_profit: pd.DataFrame,
    prop_map: Dict[str, float],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    df = pd.DataFrame({"score": scores})
    df["decile"] = pd.qcut(df["score"], 10, labels=False, duplicates="drop")
    rows = []
    for decile, group in df.groupby("decile"):
        idx = group.index.to_numpy()
        T_bin = T_obs[idx]
        y_bin = outcome[idx]
        mu_bin = mu_profit.iloc[idx]
        tau = _dr_tau_mens_control(y_bin, T_bin, mu_bin, prop_map)
        _, ci_low, ci_high = _bootstrap_tau_mens_control(
            y_bin,
            T_bin,
            mu_bin,
            prop_map,
            n_boot=n_boot,
            seed=seed,
            key=f"decile:{label}:{decile}",
        )
        rows.append(
            {
                "score_type": label,
                "decile": int(decile),
                "n": len(idx),
                "score_min": float(np.min(scores[idx])),
                "score_max": float(np.max(scores[idx])),
                "value_hat": tau,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "estimator": "dr",
            }
        )
    return pd.DataFrame(rows)


def _decile_monotone_summary(decile_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in decile_df.groupby("score_type"):
        g = group.sort_values("decile")
        diffs = np.diff(g["value_hat"].to_numpy())
        nondec = int(np.sum(diffs >= -1e-6))
        total = int(len(diffs))
        rows.append(
            {
                "score_type": label,
                "nondecreasing_steps": nondec,
                "total_steps": total,
                "nondecreasing_share": float(nondec / total) if total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_diff_mean(
    t_vals: np.ndarray,
    c_vals: np.ndarray,
    n_boot: int,
    seed: int,
    key: str,
) -> Tuple[float, float]:
    rng = _rng_for_key(seed, key)
    n_t = len(t_vals)
    n_c = len(c_vals)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        t = rng.choice(t_vals, size=n_t, replace=True)
        c = rng.choice(c_vals, size=n_c, replace=True)
        boot[i] = t.mean() - c.mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _add_quantile_bins(df: pd.DataFrame, col: str) -> pd.Series:
    values = pd.to_numeric(df[col], errors="coerce")
    try:
        bins = pd.qcut(values, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    except ValueError:
        bins = pd.cut(values, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    return bins.astype(str)


def build_subgroup_ate_mens_vs_control(
    df: pd.DataFrame,
    spend_col: str,
    margin: float,
    cost: float,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    df = df.copy()
    df = df[df["arm"].isin(["control", "mens"])].copy()
    df["recency_q"] = _add_quantile_bins(df, "recency")
    df["history_q"] = _add_quantile_bins(df, "history")

    group_specs = {
        "history_segment": "history_segment",
        "newbie": "newbie",
        "channel": "channel",
        "recency_quartile": "recency_q",
        "history_quartile": "history_q",
    }

    rows = []
    for label, col in group_specs.items():
        for level, group in df.groupby(col):
            t = group[group["arm"] == "mens"]
            c = group[group["arm"] == "control"]
            if t.empty or c.empty:
                continue
            t_spend = pd.to_numeric(t[spend_col], errors="coerce").fillna(0.0).to_numpy()
            c_spend = pd.to_numeric(c[spend_col], errors="coerce").fillna(0.0).to_numpy()
            t_profit = margin * t_spend - cost
            c_profit = margin * c_spend

            tau_spend = float(t_spend.mean() - c_spend.mean())
            tau_profit = float(t_profit.mean() - c_profit.mean())
            ci_spend_low, ci_spend_high = _bootstrap_diff_mean(
                t_spend, c_spend, n_boot, seed, key=f"subgroup:{label}:{level}:spend"
            )
            ci_profit_low, ci_profit_high = _bootstrap_diff_mean(
                t_profit, c_profit, n_boot, seed, key=f"subgroup:{label}:{level}:profit"
            )
            rows.append(
                {
                    "group": label,
                    "level": str(level),
                    "n": int(len(group)),
                    "n_mens": int(len(t)),
                    "n_control": int(len(c)),
                    "tau_hat_spend": tau_spend,
                    "ci_low_spend": ci_spend_low,
                    "ci_high_spend": ci_spend_high,
                    "tau_hat_profit": tau_profit,
                    "ci_low_profit": ci_profit_low,
                    "ci_high_profit": ci_profit_high,
                }
            )
    return pd.DataFrame(rows)

def _interaction_test(
    df: pd.DataFrame,
    group_col: str,
    outcome: str,
) -> float:
    d = df[df["arm"].isin(["control", "mens"])].copy()
    d["treat"] = (d["arm"] == "mens").astype(int)
    y = pd.to_numeric(d[outcome], errors="coerce").fillna(0.0).to_numpy()

    group = d[group_col].astype(str)
    dummies = pd.get_dummies(group, drop_first=True)
    if dummies.shape[1] == 0:
        return float("nan")
    X = pd.concat([d["treat"], dummies], axis=1)
    for col in dummies.columns:
        X[f"treat_x_{col}"] = d["treat"] * dummies[col]

    X = sm.add_constant(X, has_constant="add")
    X = X.astype(float)
    model = sm.OLS(y, X).fit()
    interaction_cols = [c for c in X.columns if c.startswith("treat_x_")]
    if not interaction_cols:
        return float("nan")
    test = model.f_test(" + ".join(interaction_cols) + " = 0")
    return float(test.pvalue)


def _interaction_tests(df: pd.DataFrame, spend_col: str, margin: float, cost: float) -> pd.DataFrame:
    d = df.copy()
    d["profit"] = margin * pd.to_numeric(d[spend_col], errors="coerce").fillna(0.0) - cost * (
        d["arm"] == "mens"
    ).astype(int)
    d["recency_q"] = _add_quantile_bins(d, "recency")
    d["history_q"] = _add_quantile_bins(d, "history")

    groups = {
        "history_segment": "history_segment",
        "newbie": "newbie",
        "channel": "channel",
        "recency_quartile": "recency_q",
        "history_quartile": "history_q",
    }
    rows = []
    for label, col in groups.items():
        rows.append(
            {
                "group": label,
                "p_value_spend": _interaction_test(d, col, spend_col),
                "p_value_profit": _interaction_test(d, col, "profit"),
                "notes": "mens_vs_control",
            }
        )
    return pd.DataFrame(rows)


def _diagnostics_verdict(
    subgroup_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
    decile_summary: pd.DataFrame,
) -> str:
    subgroup_range = subgroup_df.groupby("group")["tau_hat_profit"].agg(lambda s: float(s.max() - s.min()))
    max_range = float(subgroup_range.max()) if not subgroup_range.empty else 0.0
    min_p = float(interaction_df["p_value_profit"].min()) if not interaction_df.empty else float("nan")
    hte_detectable = (max_range >= 0.02) and (min_p < 0.1)

    proxy_scores = {"history", "recency"}
    model_scores = {"dprofit_mens", "cate_mens"}
    monotone = decile_summary.set_index("score_type")["nondecreasing_share"].to_dict()
    best_model = max(monotone.get(s, 0.0) for s in model_scores)
    best_proxy = max(monotone.get(s, 0.0) for s in proxy_scores)
    ranking_signal = best_model >= best_proxy + 0.2

    proceed = hte_detectable and ranking_signal

    lines = [
        "HTE diagnostics verdict",
        "",
        f"HTE detectable? {'Yes' if hte_detectable else 'No'}",
        f"- Max subgroup profit range: {max_range:.4f}",
        f"- Min interaction p-value (profit): {min_p:.4g}",
        "",
        f"Ranking signal? {'Yes' if ranking_signal else 'No'}",
        f"- Best model monotonicity share: {best_model:.2f}",
        f"- Best proxy monotonicity share: {best_proxy:.2f}",
        "",
        f"Proceed to causal forest? {'Yes' if proceed else 'No'}",
        "- Criteria: range>=0.02 and interaction p<0.1, plus model beats proxies by >=0.2.",
    ]
    return "\n".join(lines) + "\n"


def run_extension(
    cfg_path: str,
    output_dir: Path,
    zip_top_k: int,
    model_name: str,
    seed: int,
    n_boot: int,
    propensity: float,
) -> None:
    cfg = load_config(cfg_path)
    df = _load_data(cfg)
    df = _canonicalize_arms(df, cfg["data"]["arm_col"], cfg["data"]["arm_map"])

    X, _ = _prepare_features(df, zip_top_k)
    T = df["arm"].astype(str)
    Y = pd.to_numeric(df[cfg["data"]["spend_col"]], errors="coerce").fillna(0.0)

    prop_override = cfg.get("heterogeneity", {}).get("propensity")
    prop_map = _get_propensity_map(prop_override if prop_override is not None else propensity)

    mu = _cross_fitted_mu(X, T, Y, model_name, seed)
    cates = pd.DataFrame(
        {
            "cate_mens": mu["mu_mens"].to_numpy() - mu["mu_control"].to_numpy(),
            "cate_womens": mu["mu_womens"].to_numpy() - mu["mu_control"].to_numpy(),
        },
        index=mu.index,
    )

    margin = float(cfg["profit"]["margin"])
    cost = float(cfg["profit"]["email_cost"])
    dprofit_mens = margin * cates["cate_mens"].to_numpy() - cost
    dprofit_womens = margin * cates["cate_womens"].to_numpy() - cost

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prop_check = _propensity_check(T, prop_map)
    prop_check.to_csv(output_dir / "propensity_check.csv", index=False)

    # Diagnostics
    dist_rows = [
        _distribution_summary(cates["cate_mens"].to_numpy(), "cate_mens"),
        _distribution_summary(cates["cate_womens"].to_numpy(), "cate_womens"),
        _distribution_summary(np.maximum(dprofit_mens, dprofit_womens), "max_dprofit"),
    ]
    pd.DataFrame(dist_rows).to_csv(output_dir / "cate_distribution_summary.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 3.5))
    for i, (vals, title) in enumerate(
        [
            (cates["cate_mens"].to_numpy(), "CATE Mens vs Control"),
            (cates["cate_womens"].to_numpy(), "CATE Womens vs Control"),
            (np.maximum(dprofit_mens, dprofit_womens), "Max incremental profit"),
        ]
    ):
        ax = plt.subplot(1, 3, i + 1)
        ax.hist(vals, bins=30, color="#4C78A8", alpha=0.85)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_dir / "cate_distributions.png", dpi=150)
    plt.close()

    # Policy evaluation
    profit_obs = _profit_obs(Y.to_numpy(), T.to_numpy(), margin, cost)
    mu_profit = pd.DataFrame(
        {
            "mu_control": margin * mu["mu_control"].to_numpy(),
            "mu_mens": margin * mu["mu_mens"].to_numpy() - cost,
            "mu_womens": margin * mu["mu_womens"].to_numpy() - cost,
        },
        index=mu.index,
    )
    assignments = _policy_assignments(dprofit_mens, dprofit_womens)
    policy_df = _policy_value_table(
        assignments,
        profit_obs,
        T.to_numpy(),
        mu_profit,
        prop_map,
        n_boot=n_boot,
        seed=seed,
        notes="oof_crossfit",
    )
    policy_df.to_csv(output_dir / "policy_value.csv", index=False)

    assignment_df = _assignment_summary(assignments)
    assignment_df.to_csv(output_dir / "policy_assignment_summary.csv", index=False)

    # Policy value curve (best of three)
    curve_rows = []
    for k in [0.01, 0.05, 0.10, 0.20]:
        policy = f"top_k_best_of_three_{k:.2f}"
        subset = policy_df[policy_df["policy"] == policy]
        for _, row in subset.iterrows():
            curve_rows.append(
                {
                    "k": k,
                    "estimator": row["estimator"],
                    "value_hat": row["value_hat"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "treated_rate": row["treated_rate"],
                }
            )
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(output_dir / "policy_value_curve.csv", index=False)

    plt.figure(figsize=(6, 4))
    for estimator, group in curve_df.groupby("estimator"):
        group = group.sort_values("k")
        plt.plot(group["k"], group["value_hat"], marker="o", label=estimator)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Policy value vs top-k targeting")
    plt.xlabel("k fraction treated")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "policy_value_curve.png", dpi=150)
    plt.close()

    # Binary targeting (mens vs control)
    assignments_bin = _policy_assignments_binary(dprofit_mens)
    policy_bin = _policy_value_table(
        assignments_bin,
        profit_obs,
        T.to_numpy(),
        mu_profit,
        prop_map,
        n_boot=n_boot,
        seed=seed,
        notes="oof_crossfit_binary",
    )
    policy_bin.to_csv(output_dir / "policy_value_binary.csv", index=False)

    curve_bin_rows = []
    for k in [0.01, 0.05, 0.10, 0.20, 0.50]:
        policy = f"top_k_mens_{k:.2f}"
        subset = policy_bin[policy_bin["policy"] == policy]
        for _, row in subset.iterrows():
            curve_bin_rows.append(
                {
                    "k": k,
                    "estimator": row["estimator"],
                    "value_hat": row["value_hat"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "treated_rate": row["treated_rate"],
                }
            )
    curve_bin = pd.DataFrame(curve_bin_rows)
    curve_bin.to_csv(output_dir / "policy_value_binary_curve.csv", index=False)

    # Sanity check
    sanity_df = sanity_check_policy_value(
        pd.DataFrame({"arm": T, "spend": Y}),
        margin,
        cost,
        prop_map,
    )
    sanity_df.to_csv(output_dir / "sanity_checks_policy_value.csv", index=False)

    # Subgroup ATE table
    subgroup_df = build_subgroup_ate_mens_vs_control(
        df,
        spend_col=cfg["data"]["spend_col"],
        margin=margin,
        cost=cost,
        n_boot=n_boot,
        seed=seed,
    )
    subgroup_df.to_csv(output_dir / "subgroup_ate_mens_vs_control.csv", index=False)

    # Interaction tests
    interaction_df = _interaction_tests(df, cfg["data"]["spend_col"], margin, cost)
    interaction_df.to_csv(output_dir / "interaction_tests.csv", index=False)

    # Decile compare (mens vs control)
    deciles = []
    deciles.append(
        _decile_diagnostics(
            dprofit_mens,
            "dprofit_mens",
            profit_obs,
            T.to_numpy(),
            mu_profit,
            prop_map,
            n_boot=min(n_boot, 2000),
            seed=seed,
        )
    )
    deciles.append(
        _decile_diagnostics(
            cates["cate_mens"].to_numpy(),
            "cate_mens",
            profit_obs,
            T.to_numpy(),
            mu_profit,
            prop_map,
            n_boot=min(n_boot, 2000),
            seed=seed,
        )
    )
    deciles.append(
        _decile_diagnostics(
            pd.to_numeric(df["history"], errors="coerce").fillna(0.0).to_numpy(),
            "history",
            profit_obs,
            T.to_numpy(),
            mu_profit,
            prop_map,
            n_boot=min(n_boot, 2000),
            seed=seed,
        )
    )
    deciles.append(
        _decile_diagnostics(
            -pd.to_numeric(df["recency"], errors="coerce").fillna(0.0).to_numpy(),
            "recency",
            profit_obs,
            T.to_numpy(),
            mu_profit,
            prop_map,
            n_boot=min(n_boot, 2000),
            seed=seed,
        )
    )
    decile_df = pd.concat(deciles, ignore_index=True)
    decile_summary = _decile_monotone_summary(decile_df)
    decile_df = decile_df.merge(decile_summary, on="score_type", how="left")
    decile_df.to_csv(output_dir / "decile_compare.csv", index=False)

    plt.figure(figsize=(7, 4))
    for label, group in decile_df.groupby("score_type"):
        group = group.sort_values("decile")
        plt.plot(group["decile"], group["value_hat"], marker="o", label=label)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Decile uplift comparison (Mens vs Control)")
    plt.xlabel("Decile (low to high score)")
    plt.ylabel("Estimated uplift (profit)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "decile_compare.png", dpi=150)
    plt.close()

    # Verdict
    verdict = _diagnostics_verdict(subgroup_df, interaction_df, decile_summary)
    (output_dir / "diagnostics_verdict.md").write_text(verdict, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_extension(
        cfg_path=args.config,
        output_dir=Path(args.output),
        zip_top_k=args.zip_top_k,
        model_name=args.model,
        seed=args.seed,
        n_boot=args.bootstrap,
        propensity=args.propensity,
    )


if __name__ == "__main__":
    main()
