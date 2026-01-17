from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from bootstrap import baseline_stream, bootstrap_ci_from_diffs, bootstrap_diffs

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
    n_pos_t: int
    n_pos_c: int
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


def _rank_order(spend: np.ndarray, ids: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.lexsort((rows, ids, -spend))


def _top_k_indices(spend: np.ndarray, ids: np.ndarray, rows: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=int)
    order = _rank_order(spend, ids, rows)
    return order[:k]


def _pooled_trim_masks(
    spend_list: List[np.ndarray],
    ids_list: List[np.ndarray],
    rows_list: List[np.ndarray],
    k: int,
) -> Tuple[List[np.ndarray], float]:
    if k <= 0:
        return [np.zeros(len(s), dtype=bool) for s in spend_list], 0.0

    spend_all = np.concatenate(spend_list)
    ids_all = np.concatenate(ids_list)
    rows_all = np.concatenate(rows_list)
    origins = np.concatenate([np.full(len(s), idx) for idx, s in enumerate(spend_list)])
    idxs = np.concatenate([np.arange(len(s)) for s in spend_list])

    pos_mask = spend_all > 0
    if pos_mask.sum() == 0:
        return [np.zeros(len(s), dtype=bool) for s in spend_list], 0.0

    spend_pos = spend_all[pos_mask]
    ids_pos = ids_all[pos_mask]
    rows_pos = rows_all[pos_mask]
    origins_pos = origins[pos_mask]
    idxs_pos = idxs[pos_mask]

    order = _rank_order(spend_pos, ids_pos, rows_pos)
    k = min(k, len(order))
    top = order[:k]
    trim_threshold = float(spend_pos[order[k - 1]]) if k > 0 else 0.0

    trim_masks = [np.zeros(len(s), dtype=bool) for s in spend_list]
    for idx in top:
        origin = int(origins_pos[idx])
        pos_idx = int(idxs_pos[idx])
        trim_masks[origin][pos_idx] = True
    return trim_masks, trim_threshold


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
    margin: float,
    email_cost: float,
    n_boot: int,
    seed: int,
    id_col: str = "id",
) -> pd.DataFrame:
    df = df.copy()
    df["emailed"] = df["arm"].isin(["mens", "womens"]).astype(int)

    results: List[RobustRow] = []

    arms = ["mens", "womens"]
    control_arm = "control"

    # Precompute for baseline and diagnostics on full sample
    spend_all = pd.to_numeric(df[spend_col], errors="coerce").fillna(0.0).to_numpy()
    if id_col not in df.columns:
        df = df.reset_index(drop=True)
        df["__row_id__"] = df.index.astype(str)
        id_col = "__row_id__"
    ids_all = df[id_col].astype(str).to_numpy()
    rows_all = df.index.to_numpy()
    cap_p95 = _pooled_pos_threshold(spend_all, 0.95)
    cap_p975 = _pooled_pos_threshold(spend_all, 0.975)

    pos_mask = spend_all > 0
    n_pos_pooled = int(pos_mask.sum())
    k_trim = int(np.ceil(0.01 * n_pos_pooled)) if n_pos_pooled else 0
    pooled_trim_masks, trim_threshold = _pooled_trim_masks(
        [spend_all], [ids_all], [rows_all], k_trim
    )
    trim_idx_pooled = np.flatnonzero(pooled_trim_masks[0])

    def _bootstrap_diffs_method(
        t_spend: np.ndarray,
        t_email: np.ndarray,
        t_ids: np.ndarray,
        t_rows: np.ndarray,
        c_spend: np.ndarray,
        c_email: np.ndarray,
        c_ids: np.ndarray,
        c_rows: np.ndarray,
        o_spend: np.ndarray,
        o_ids: np.ndarray,
        o_rows: np.ndarray,
        method: str,
        metric: str,
    ) -> np.ndarray:
        diffs = np.empty(n_boot, dtype=float)
        rng = np.random.default_rng(seed)
        for i in range(n_boot):
            t_idx = rng.choice(len(t_spend), size=len(t_spend), replace=True)
            c_idx = rng.choice(len(c_spend), size=len(c_spend), replace=True)
            o_idx = rng.choice(len(o_spend), size=len(o_spend), replace=True)

            t_s = t_spend[t_idx]
            t_e = t_email[t_idx]
            t_id = t_ids[t_idx]
            t_row = t_rows[t_idx]
            c_s = c_spend[c_idx]
            c_e = c_email[c_idx]
            c_id = c_ids[c_idx]
            c_row = c_rows[c_idx]
            o_s = o_spend[o_idx]
            o_id = o_ids[o_idx]
            o_row = o_rows[o_idx]

            if method in ("winsorize_pos_p95", "winsorize_pos_p975"):
                q = 0.95 if method == "winsorize_pos_p95" else 0.975
                cap = _pooled_pos_threshold(np.concatenate([t_s, c_s, o_s]), q)
                t_s = np.where(t_s > cap, cap, t_s)
                c_s = np.where(c_s > cap, cap, c_s)
                if metric == "profit":
                    t_val = _profit(t_s, t_e, margin, email_cost)
                    c_val = _profit(c_s, c_e, margin, email_cost)
                else:
                    t_val = t_s
                    c_val = c_s
            elif method == "trim_pos_top_1pct":
                n_pos = int(np.sum(np.concatenate([t_s, c_s, o_s]) > 0))
                k = int(np.ceil(0.01 * n_pos)) if n_pos else 0
                trim_masks, _ = _pooled_trim_masks(
                    [t_s, c_s, o_s],
                    [t_id, c_id, o_id],
                    [t_row, c_row, o_row],
                    k,
                )
                t_s = t_s[~trim_masks[0]]
                t_e = t_e[~trim_masks[0]]
                c_s = c_s[~trim_masks[1]]
                c_e = c_e[~trim_masks[1]]
                if metric == "profit":
                    t_val = _profit(t_s, t_e, margin, email_cost)
                    c_val = _profit(c_s, c_e, margin, email_cost)
                else:
                    t_val = t_s
                    c_val = c_s
            elif method == "log1p_spend":
                t_val = np.log1p(t_s)
                c_val = np.log1p(c_s)
            elif method.startswith("remove_top_by_rank_"):
                x = float(method.split("_")[-1])
                t_k = int(np.ceil(x * len(t_s)))
                c_k = int(np.ceil(x * len(c_s)))
                t_idx_rm = _top_k_indices(t_s, t_id, t_row, t_k)
                c_idx_rm = _top_k_indices(c_s, c_id, c_row, c_k)
                t_s = _remove_indices(t_s, t_idx_rm)
                t_e = _remove_indices(t_e, t_idx_rm)
                c_s = _remove_indices(c_s, c_idx_rm)
                c_e = _remove_indices(c_e, c_idx_rm)
                if metric == "profit":
                    t_val = _profit(t_s, t_e, margin, email_cost)
                    c_val = _profit(c_s, c_e, margin, email_cost)
                else:
                    t_val = t_s
                    c_val = c_s
            else:
                if metric == "profit":
                    t_val = _profit(t_s, t_e, margin, email_cost)
                    c_val = _profit(c_s, c_e, margin, email_cost)
                else:
                    t_val = t_s
                    c_val = c_s

            diffs[i] = t_val.mean() - c_val.mean()
        return diffs

    for arm in arms:
        other_arm = "womens" if arm == "mens" else "mens"
        treat = df[df["arm"] == arm]
        control = df[df["arm"] == control_arm]
        other = df[df["arm"] == other_arm]

        t_spend = pd.to_numeric(treat[spend_col], errors="coerce").fillna(0.0).to_numpy()
        c_spend = pd.to_numeric(control[spend_col], errors="coerce").fillna(0.0).to_numpy()
        t_email = treat["emailed"].to_numpy()
        c_email = control["emailed"].to_numpy()
        t_id = treat[id_col].astype(str).to_numpy()
        c_id = control[id_col].astype(str).to_numpy()
        t_row = treat.index.to_numpy()
        c_row = control.index.to_numpy()
        o_spend = pd.to_numeric(other[spend_col], errors="coerce").fillna(0.0).to_numpy()
        o_id = other[id_col].astype(str).to_numpy()
        o_row = other.index.to_numpy()

        n_pos_t = int((t_spend > 0).sum())
        n_pos_c = int((c_spend > 0).sum())

        # Baseline
        t_profit = _profit(t_spend, t_email, margin, email_cost)
        c_profit = _profit(c_spend, c_email, margin, email_cost)
        diffs = bootstrap_diffs(t_profit, c_profit, n_boot=n_boot, seed=seed, stream=baseline_stream("profit", arm))
        ci_low, ci_high = bootstrap_ci_from_diffs(diffs)
        results.append(
            RobustRow(
                metric="profit",
                method="baseline_mean",
                arm=arm,
                tau_hat=_diff_mean(t_profit, c_profit),
                ci_lower=ci_low,
                ci_upper=ci_high,
                cap_value=0.0,
                trim_threshold=0.0,
                n_pos_t=n_pos_t,
                n_pos_c=n_pos_c,
                n_capped_t=0,
                n_capped_c=0,
                n_trim_t=0,
                n_trim_c=0,
                seed=seed,
                B=n_boot,
                notes="primary_ate",
            )
        )
        diffs = bootstrap_diffs(t_spend, c_spend, n_boot=n_boot, seed=seed, stream=baseline_stream("spend", arm))
        ci_low, ci_high = bootstrap_ci_from_diffs(diffs)
        results.append(
            RobustRow(
                metric="spend",
                method="baseline_mean",
                arm=arm,
                tau_hat=_diff_mean(t_spend, c_spend),
                ci_lower=ci_low,
                ci_upper=ci_high,
                cap_value=0.0,
                trim_threshold=0.0,
                n_pos_t=n_pos_t,
                n_pos_c=n_pos_c,
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
            diffs = _bootstrap_diffs_method(
                t_spend,
                t_email,
                t_id,
                t_row,
                c_spend,
                c_email,
                c_id,
                c_row,
                o_spend,
                o_id,
                o_row,
                label,
                "profit",
            )
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
                    n_pos_t=n_pos_t,
                    n_pos_c=n_pos_c,
                    n_capped_t=t_cap,
                    n_capped_c=c_cap,
                    n_trim_t=0,
                    n_trim_c=0,
                    seed=seed,
                    B=n_boot,
                    notes="sensitivity;pooled_pos_cap",
                )
            )
            diffs = _bootstrap_diffs_method(
                t_spend,
                t_email,
                t_id,
                t_row,
                c_spend,
                c_email,
                c_id,
                c_row,
                o_spend,
                o_id,
                o_row,
                label,
                "spend",
            )
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
                    n_pos_t=n_pos_t,
                    n_pos_c=n_pos_c,
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
        t_trim_mask = trim_idx_pooled[np.isin(trim_idx_pooled, treat.index)]
        c_trim_mask = trim_idx_pooled[np.isin(trim_idx_pooled, control.index)]
        t_trim_idx = treat.index.get_indexer(t_trim_mask)
        c_trim_idx = control.index.get_indexer(c_trim_mask)
        t_spend_t = _remove_indices(t_spend, t_trim_idx)
        t_email_t = _remove_indices(t_email, t_trim_idx)
        c_spend_t = _remove_indices(c_spend, c_trim_idx)
        c_email_t = _remove_indices(c_email, c_trim_idx)
        t_profit_t = _profit(t_spend_t, t_email_t, margin, email_cost)
        c_profit_t = _profit(c_spend_t, c_email_t, margin, email_cost)
        diffs = _bootstrap_diffs_method(
            t_spend,
            t_email,
            t_id,
            t_row,
            c_spend,
            c_email,
            c_id,
            c_row,
            o_spend,
            o_id,
            o_row,
            "trim_pos_top_1pct",
            "profit",
        )
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
                n_pos_t=n_pos_t,
                n_pos_c=n_pos_c,
                n_capped_t=0,
                n_capped_c=0,
                n_trim_t=len(t_trim_idx),
                n_trim_c=len(c_trim_idx),
                seed=seed,
                B=n_boot,
                notes="sensitivity;pooled_pos_rank_trim;ties_by_id",
            )
        )
        diffs = _bootstrap_diffs_method(
            t_spend,
            t_email,
            t_id,
            t_row,
            c_spend,
            c_email,
            c_id,
            c_row,
            o_spend,
            o_id,
            o_row,
            "trim_pos_top_1pct",
            "spend",
        )
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
                n_pos_t=n_pos_t,
                n_pos_c=n_pos_c,
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
        diffs = _bootstrap_diffs_method(
            t_spend,
            t_email,
            t_id,
            t_row,
            c_spend,
            c_email,
            c_id,
            c_row,
            o_spend,
            o_id,
            o_row,
            "log1p_spend",
            "spend",
        )
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
                n_pos_t=n_pos_t,
                n_pos_c=n_pos_c,
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
            t_idx = _top_k_indices(t_spend, t_id, t_row, t_k)
            c_idx = _top_k_indices(c_spend, c_id, c_row, c_k)
            t_spend_r = _remove_indices(t_spend, t_idx)
            t_email_r = _remove_indices(t_email, t_idx)
            c_spend_r = _remove_indices(c_spend, c_idx)
            c_email_r = _remove_indices(c_email, c_idx)
            t_profit_r = _profit(t_spend_r, t_email_r, margin, email_cost)
            c_profit_r = _profit(c_spend_r, c_email_r, margin, email_cost)
            diffs = _bootstrap_diffs_method(
                t_spend,
                t_email,
                t_id,
                t_row,
                c_spend,
                c_email,
                c_id,
                c_row,
                o_spend,
                o_id,
                o_row,
                f"remove_top_by_rank_{x}",
                "profit",
            )
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
                    n_pos_t=n_pos_t,
                    n_pos_c=n_pos_c,
                    n_capped_t=0,
                    n_capped_c=0,
                    n_trim_t=len(t_idx),
                    n_trim_c=len(c_idx),
                    seed=seed,
                    B=n_boot,
                    notes="rank_remove_within_arm;ties_by_id",
                )
            )
            diffs = _bootstrap_diffs_method(
                t_spend,
                t_email,
                t_id,
                t_row,
                c_spend,
                c_email,
                c_id,
                c_row,
                o_spend,
                o_id,
                o_row,
                f"remove_top_by_rank_{x}",
                "spend",
            )
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
                    n_pos_t=n_pos_t,
                    n_pos_c=n_pos_c,
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
