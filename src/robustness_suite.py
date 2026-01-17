from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class RobustRow:
    metric: str
    method: str
    arm: str
    tau_hat: float
    ci_lower: float
    ci_upper: float
    cap_value: float
    trim_threshold: float
    n_capped_t: int
    n_capped_c: int
    n_trim_t: int
    n_trim_c: int
    seed: int
    B: int
    notes: str


def _profit(spend: np.ndarray, emailed: np.ndarray, margin: float, email_cost: float) -> np.ndarray:
    return margin * spend - email_cost * emailed


def _bootstrap_resample(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(values, size=len(values), replace=True)


def _pooled_pos_threshold(values: np.ndarray, q: float) -> float:
    pos = values[values > 0]
    if len(pos) == 0:
        return 0.0
    return float(np.quantile(pos, q))


def _top_k_indices(spend: np.ndarray, ids: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=int)
    order = np.lexsort((ids, -spend))
    return order[:k]


def _pooled_trim_indices(spend: np.ndarray, ids: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=int)
    pos_mask = spend > 0
    spend_pos = spend[pos_mask]
    ids_pos = ids[pos_mask]
    if len(spend_pos) == 0:
        return np.array([], dtype=int)
    order = np.lexsort((ids_pos, -spend_pos))
    pos_indices = np.flatnonzero(pos_mask)
    return pos_indices[order[:k]]


def _remove_indices(values: np.ndarray, remove_idx: np.ndarray) -> np.ndarray:
    if len(remove_idx) == 0:
        return values
    mask = np.ones(len(values), dtype=bool)
    mask[remove_idx] = False
    return values[mask]


def _diff_mean(treat: np.ndarray, control: np.ndarray) -> float:
    return float(treat.mean() - control.mean())


def compute_robustness(
    df: pd.DataFrame,
    spend_col: str,
    id_col: str,
    margin: float,
    email_cost: float,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    df = df.copy()
    df["emailed"] = df["arm"].isin(["mens", "womens"]).astype(int)

    results: List[RobustRow] = []

    arms = ["mens", "womens"]
    control_arm = "control"

    # Precompute for baseline and diagnostics on full sample
    spend_all = pd.to_numeric(df[spend_col], errors="coerce").fillna(0.0).to_numpy()
    ids_all = df[id_col].astype(str).to_numpy()
    cap_p95 = _pooled_pos_threshold(spend_all, 0.95)
    cap_p975 = _pooled_pos_threshold(spend_all, 0.975)

    pos_mask = spend_all > 0
    n_pos_pooled = int(pos_mask.sum())
    k_trim = int(np.ceil(0.01 * n_pos_pooled)) if n_pos_pooled else 0
    trim_idx_pooled = _pooled_trim_indices(spend_all, ids_all, k_trim)
    trim_threshold = float(np.max(spend_all[trim_idx_pooled])) if k_trim > 0 else 0.0

    def bootstrap_diffs(t_vals: np.ndarray, c_vals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        diffs = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            t = _bootstrap_resample(t_vals, rng)
            c = _bootstrap_resample(c_vals, rng)
            diffs[i] = t.mean() - c.mean()
        return diffs

    for arm in arms:
        treat = df[df["arm"] == arm]
        control = df[df["arm"] == control_arm]

        t_spend = pd.to_numeric(treat[spend_col], errors="coerce").fillna(0.0).to_numpy()
        c_spend = pd.to_numeric(control[spend_col], errors="coerce").fillna(0.0).to_numpy()
        t_email = treat["emailed"].to_numpy()
        c_email = control["emailed"].to_numpy()
        t_id = treat[id_col].astype(str).to_numpy()
        c_id = control[id_col].astype(str).to_numpy()

        # Baseline
        t_profit = _profit(t_spend, t_email, margin, email_cost)
        c_profit = _profit(c_spend, c_email, margin, email_cost)
        rng = np.random.default_rng(seed)
        diffs = bootstrap_diffs(t_profit, c_profit, rng)
        results.append(
            RobustRow(
                metric="profit",
                method="baseline_mean",
                arm=arm,
                tau_hat=_diff_mean(t_profit, c_profit),
                ci_lower=float(np.percentile(diffs, 2.5)),
                ci_upper=float(np.percentile(diffs, 97.5)),
                cap_value=0.0,
                trim_threshold=0.0,
                n_capped_t=0,
                n_capped_c=0,
                n_trim_t=0,
                n_trim_c=0,
                seed=seed,
                B=n_boot,
                notes="primary_ate",
            )
        )
        rng = np.random.default_rng(seed)
        diffs = bootstrap_diffs(t_spend, c_spend, rng)
        results.append(
            RobustRow(
                metric="spend",
                method="baseline_mean",
                arm=arm,
                tau_hat=_diff_mean(t_spend, c_spend),
                ci_lower=float(np.percentile(diffs, 2.5)),
                ci_upper=float(np.percentile(diffs, 97.5)),
                cap_value=0.0,
                trim_threshold=0.0,
                n_capped_t=0,
                n_capped_c=0,
                n_trim_t=0,
                n_trim_c=0,
                seed=seed,
                B=n_boot,
                notes="primary_ate",
            )
        )

        # Winsorize pooled positive p95/p97.5
        for cap, label in [(cap_p95, "winsorize_pos_p95"), (cap_p975, "winsorize_pos_p975")]:
            t_cap = int(np.sum((t_spend > 0) & (t_spend > cap)))
            c_cap = int(np.sum((c_spend > 0) & (c_spend > cap)))
            t_spend_w = np.where(t_spend > cap, cap, t_spend)
            c_spend_w = np.where(c_spend > cap, cap, c_spend)
            t_profit_w = _profit(t_spend_w, t_email, margin, email_cost)
            c_profit_w = _profit(c_spend_w, c_email, margin, email_cost)
            rng = np.random.default_rng(seed)
            diffs = bootstrap_diffs(t_profit_w, c_profit_w, rng)
            results.append(
                RobustRow(
                    metric="profit",
                    method=label,
                    arm=arm,
                    tau_hat=_diff_mean(t_profit_w, c_profit_w),
                    ci_lower=float(np.percentile(diffs, 2.5)),
                    ci_upper=float(np.percentile(diffs, 97.5)),
                    cap_value=cap,
                    trim_threshold=0.0,
                    n_capped_t=t_cap,
                    n_capped_c=c_cap,
                    n_trim_t=0,
                    n_trim_c=0,
                    seed=seed,
                    B=n_boot,
                    notes="sensitivity;pooled_pos_cap",
                )
            )
            rng = np.random.default_rng(seed)
            diffs = bootstrap_diffs(t_spend_w, c_spend_w, rng)
            results.append(
                RobustRow(
                    metric="spend",
                    method=label,
                    arm=arm,
                    tau_hat=_diff_mean(t_spend_w, c_spend_w),
                    ci_lower=float(np.percentile(diffs, 2.5)),
                    ci_upper=float(np.percentile(diffs, 97.5)),
                    cap_value=cap,
                    trim_threshold=0.0,
                    n_capped_t=t_cap,
                    n_capped_c=c_cap,
                    n_trim_t=0,
                    n_trim_c=0,
                    seed=seed,
                    B=n_boot,
                    notes="sensitivity;pooled_pos_cap",
                )
            )

        # Trim pooled top 1% positive by rank
        t_trim_idx = _pooled_trim_indices(t_spend, t_id, k_trim) if k_trim > 0 else np.array([], dtype=int)
        c_trim_idx = _pooled_trim_indices(c_spend, c_id, k_trim) if k_trim > 0 else np.array([], dtype=int)
        t_spend_t = _remove_indices(t_spend, t_trim_idx)
        c_spend_t = _remove_indices(c_spend, c_trim_idx)
        t_profit_t = _profit(t_spend_t, np.ones_like(t_spend_t), margin, email_cost)
        c_profit_t = _profit(c_spend_t, np.zeros_like(c_spend_t), margin, email_cost)
        rng = np.random.default_rng(seed)
        diffs = bootstrap_diffs(t_profit_t, c_profit_t, rng)
        results.append(
            RobustRow(
                metric="profit",
                method="trim_pos_top_1pct",
                arm=arm,
                tau_hat=_diff_mean(t_profit_t, c_profit_t),
                ci_lower=float(np.percentile(diffs, 2.5)),
                ci_upper=float(np.percentile(diffs, 97.5)),
                cap_value=0.0,
                trim_threshold=trim_threshold,
                n_capped_t=0,
                n_capped_c=0,
                n_trim_t=len(t_trim_idx),
                n_trim_c=len(c_trim_idx),
                seed=seed,
                B=n_boot,
                notes="sensitivity;pooled_pos_rank_trim;ties_by_id",
            )
        )
        rng = np.random.default_rng(seed)
        diffs = bootstrap_diffs(t_spend_t, c_spend_t, rng)
        results.append(
            RobustRow(
                metric="spend",
                method="trim_pos_top_1pct",
                arm=arm,
                tau_hat=_diff_mean(t_spend_t, c_spend_t),
                ci_lower=float(np.percentile(diffs, 2.5)),
                ci_upper=float(np.percentile(diffs, 97.5)),
                cap_value=0.0,
                trim_threshold=trim_threshold,
                n_capped_t=0,
                n_capped_c=0,
                n_trim_t=len(t_trim_idx),
                n_trim_c=len(c_trim_idx),
                seed=seed,
                B=n_boot,
                notes="sensitivity;pooled_pos_rank_trim;ties_by_id",
            )
        )

        # log1p spend (secondary estimand)
        t_log = np.log1p(t_spend)
        c_log = np.log1p(c_spend)
        rng = np.random.default_rng(seed)
        diffs = bootstrap_diffs(t_log, c_log, rng)
        results.append(
            RobustRow(
                metric="spend",
                method="log1p_spend",
                arm=arm,
                tau_hat=_diff_mean(t_log, c_log),
                ci_lower=float(np.percentile(diffs, 2.5)),
                ci_upper=float(np.percentile(diffs, 97.5)),
                cap_value=0.0,
                trim_threshold=0.0,
                n_capped_t=0,
                n_capped_c=0,
                n_trim_t=0,
                n_trim_c=0,
                seed=seed,
                B=n_boot,
                notes="secondary_scale_not_dollars",
            )
        )

        # Remove top by rank within each arm
        for x in [0.001, 0.005, 0.01, 0.02]:
            t_k = int(np.ceil(x * len(t_spend)))
            c_k = int(np.ceil(x * len(c_spend)))
            t_idx = _top_k_indices(t_spend, t_id, t_k)
            c_idx = _top_k_indices(c_spend, c_id, c_k)
            t_spend_r = _remove_indices(t_spend, t_idx)
            c_spend_r = _remove_indices(c_spend, c_idx)
            t_profit_r = _profit(t_spend_r, np.ones_like(t_spend_r), margin, email_cost)
            c_profit_r = _profit(c_spend_r, np.zeros_like(c_spend_r), margin, email_cost)
            rng = np.random.default_rng(seed)
            diffs = bootstrap_diffs(t_profit_r, c_profit_r, rng)
            results.append(
                RobustRow(
                    metric="profit",
                    method=f"remove_top_by_rank_{x}",
                    arm=arm,
                    tau_hat=_diff_mean(t_profit_r, c_profit_r),
                    ci_lower=float(np.percentile(diffs, 2.5)),
                    ci_upper=float(np.percentile(diffs, 97.5)),
                    cap_value=0.0,
                    trim_threshold=0.0,
                    n_capped_t=0,
                    n_capped_c=0,
                    n_trim_t=len(t_idx),
                    n_trim_c=len(c_idx),
                    seed=seed,
                    B=n_boot,
                    notes="rank_remove_within_arm;ties_by_id",
                )
            )
            rng = np.random.default_rng(seed)
            diffs = bootstrap_diffs(t_spend_r, c_spend_r, rng)
            results.append(
                RobustRow(
                    metric="spend",
                    method=f"remove_top_by_rank_{x}",
                    arm=arm,
                    tau_hat=_diff_mean(t_spend_r, c_spend_r),
                    ci_lower=float(np.percentile(diffs, 2.5)),
                    ci_upper=float(np.percentile(diffs, 97.5)),
                    cap_value=0.0,
                    trim_threshold=0.0,
                    n_capped_t=0,
                    n_capped_c=0,
                    n_trim_t=len(t_idx),
                    n_trim_c=len(c_idx),
                    seed=seed,
                    B=n_boot,
                    notes="rank_remove_within_arm;ties_by_id",
                )
            )

    return pd.DataFrame([r.__dict__ for r in results])
