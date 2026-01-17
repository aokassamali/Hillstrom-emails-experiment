from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class RobustResult:
    method: str
    arm: str
    tau_hat: float
    ci_lower: float
    ci_upper: float
    notes: str


def _bootstrap_diff(
    treat: np.ndarray,
    control: np.ndarray,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_t = len(treat)
    n_c = len(control)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        t = rng.choice(treat, size=n_t, replace=True)
        c = rng.choice(control, size=n_c, replace=True)
        diffs[i] = t.mean() - c.mean()
    return diffs


def _ci_from_boot(diffs: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    return (
        float(np.percentile(diffs, 100 * (alpha / 2))),
        float(np.percentile(diffs, 100 * (1 - alpha / 2))),
    )


def _profit_from_spend(spend: np.ndarray, emailed: np.ndarray, margin: float, email_cost: float) -> np.ndarray:
    return margin * spend - email_cost * emailed


def _winsorize_upper(values: np.ndarray, cap: float) -> Tuple[np.ndarray, int]:
    capped = np.minimum(values, cap)
    return capped, int(np.sum(values > cap))


def _trim_upper(values: np.ndarray, trim_frac: float) -> Tuple[np.ndarray, int]:
    if trim_frac <= 0:
        return values, 0
    n = len(values)
    k = int(np.floor(n * trim_frac))
    if k <= 0:
        return values, 0
    threshold = np.partition(values, -k)[-k]
    trimmed = values[values <= threshold]
    return trimmed, int(n - len(trimmed))


def _topk_remove(values: np.ndarray, k: int) -> Tuple[np.ndarray, int]:
    if k <= 0:
        return values, 0
    if k >= len(values):
        return np.array([], dtype=float), len(values)
    threshold = np.partition(values, -k)[-k]
    kept = values[values < threshold]
    removed = len(values) - len(kept)
    return kept, removed


def compute_robustness(
    df: pd.DataFrame,
    spend_col: str,
    margin: float,
    email_cost: float,
    n_boot: int,
    seed: int,
    trim_frac: float = 0.01,
    topk_list: List[int] | None = None,
) -> pd.DataFrame:
    if topk_list is None:
        topk_list = [10, 50]

    df = df.copy()
    df["emailed"] = df["arm"].isin(["mens", "womens"]).astype(int)
    spend = pd.to_numeric(df[spend_col], errors="coerce").fillna(0.0).to_numpy()
    pooled_p99 = float(np.quantile(spend, 0.99))

    results: List[RobustResult] = []
    suspicious_flags: Dict[str, int] = {}

    def add_result(method: str, arm: str, tau_hat: float, diffs: np.ndarray, notes: str) -> None:
        ci_low, ci_high = _ci_from_boot(diffs)
        if abs(tau_hat + email_cost) < 1e-12 and method.startswith(("profit_winsor", "profit_trim")):
            suspicious_flags[method] = suspicious_flags.get(method, 0) + 1
            notes = (notes + ";FLAG_CONSTANT_MINUS_COST").strip(";")
        results.append(
            RobustResult(method=method, arm=arm, tau_hat=float(tau_hat), ci_lower=ci_low, ci_upper=ci_high, notes=notes)
        )

    for arm in ["mens", "womens"]:
        treat = df[df["arm"] == arm]
        control = df[df["arm"] == "control"]

        t_spend = pd.to_numeric(treat[spend_col], errors="coerce").fillna(0.0).to_numpy()
        c_spend = pd.to_numeric(control[spend_col], errors="coerce").fillna(0.0).to_numpy()
        t_email = treat["emailed"].to_numpy()
        c_email = control["emailed"].to_numpy()

        # Baseline (primary estimand)
        t_profit = _profit_from_spend(t_spend, t_email, margin, email_cost)
        c_profit = _profit_from_spend(c_spend, c_email, margin, email_cost)
        diffs = _bootstrap_diff(t_profit, c_profit, n_boot, seed)
        add_result("profit_baseline", arm, float(t_profit.mean() - c_profit.mean()), diffs, "primary_ate")

        diffs = _bootstrap_diff(t_spend, c_spend, n_boot, seed)
        add_result("spend_baseline", arm, float(t_spend.mean() - c_spend.mean()), diffs, "primary_ate")

        # Winsorization pooled p99
        t_spend_w, t_cap = _winsorize_upper(t_spend, pooled_p99)
        c_spend_w, c_cap = _winsorize_upper(c_spend, pooled_p99)
        t_profit_w = _profit_from_spend(t_spend_w, t_email, margin, email_cost)
        c_profit_w = _profit_from_spend(c_spend_w, c_email, margin, email_cost)
        notes = f"sensitivity;cap_p99={pooled_p99:.4f};capped_t={t_cap};capped_c={c_cap}"
        diffs = _bootstrap_diff(t_profit_w, c_profit_w, n_boot, seed)
        add_result("profit_winsor_p99", arm, float(t_profit_w.mean() - c_profit_w.mean()), diffs, notes)
        diffs = _bootstrap_diff(t_spend_w, c_spend_w, n_boot, seed)
        add_result("spend_winsor_p99", arm, float(t_spend_w.mean() - c_spend_w.mean()), diffs, notes)

        # Trim upper tail
        t_spend_t, t_trim = _trim_upper(t_spend, trim_frac)
        c_spend_t, c_trim = _trim_upper(c_spend, trim_frac)
        t_profit_t = _profit_from_spend(t_spend_t, np.ones_like(t_spend_t), margin, email_cost)
        c_profit_t = _profit_from_spend(c_spend_t, np.zeros_like(c_spend_t), margin, email_cost)
        notes = f"sensitivity;trim_upper={trim_frac};trim_t={t_trim};trim_c={c_trim}"
        diffs = _bootstrap_diff(t_profit_t, c_profit_t, n_boot, seed)
        add_result("profit_trim_upper", arm, float(t_profit_t.mean() - c_profit_t.mean()), diffs, notes)
        diffs = _bootstrap_diff(t_spend_t, c_spend_t, n_boot, seed)
        add_result("spend_trim_upper", arm, float(t_spend_t.mean() - c_spend_t.mean()), diffs, notes)

        # log1p spend (secondary scale)
        t_log = np.log1p(t_spend)
        c_log = np.log1p(c_spend)
        diffs = _bootstrap_diff(t_log, c_log, n_boot, seed)
        add_result("log1p_spend", arm, float(t_log.mean() - c_log.mean()), diffs, "secondary_scale")

        # Top-k sensitivity (pooled within each arm)
        for k in topk_list:
            t_spend_k, t_removed = _topk_remove(t_spend, k)
            c_spend_k, c_removed = _topk_remove(c_spend, k)
            if len(t_spend_k) == 0 or len(c_spend_k) == 0:
                continue
            t_profit_k = _profit_from_spend(t_spend_k, np.ones_like(t_spend_k), margin, email_cost)
            c_profit_k = _profit_from_spend(c_spend_k, np.zeros_like(c_spend_k), margin, email_cost)
            notes = f"sensitivity;topk_removed={k};removed_t={t_removed};removed_c={c_removed}"
            diffs = _bootstrap_diff(t_profit_k, c_profit_k, n_boot, seed)
            add_result(f"profit_topk_{k}", arm, float(t_profit_k.mean() - c_profit_k.mean()), diffs, notes)
            diffs = _bootstrap_diff(t_spend_k, c_spend_k, n_boot, seed)
            add_result(f"spend_topk_{k}", arm, float(t_spend_k.mean() - c_spend_k.mean()), diffs, notes)

    # If suspicious constants across multiple arms, add a global flag row
    if any(count >= 2 for count in suspicious_flags.values()):
        for method, count in suspicious_flags.items():
            results.append(
                RobustResult(
                    method=method,
                    arm="__flag__",
                    tau_hat=float("nan"),
                    ci_lower=float("nan"),
                    ci_upper=float("nan"),
                    notes=f"FLAG_CONSTANT_MINUS_COST;count={count}",
                )
            )

    return pd.DataFrame([r.__dict__ for r in results])
