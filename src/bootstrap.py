from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np


def _rng_for_stream(seed: int, stream: str, n_t: int, n_c: int) -> np.random.Generator:
    key = f"{stream}|n_t={n_t}|n_c={n_c}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    stream_int = int.from_bytes(digest[:4], "little", signed=False)
    ss = np.random.SeedSequence([seed, stream_int])
    return np.random.default_rng(ss)


def bootstrap_diffs(
    treat: np.ndarray,
    control: np.ndarray,
    n_boot: int,
    seed: int,
    stream: str,
) -> np.ndarray:
    n_t = len(treat)
    n_c = len(control)
    rng = _rng_for_stream(seed, stream, n_t, n_c)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        t = rng.choice(treat, size=n_t, replace=True)
        c = rng.choice(control, size=n_c, replace=True)
        diffs[i] = t.mean() - c.mean()
    return diffs


def bootstrap_ci_from_diffs(diffs: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    if len(diffs) == 0:
        return float("nan"), float("nan")
    lower = float(np.percentile(diffs, 100 * (alpha / 2)))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return lower, upper


def bootstrap_ci(
    treat: np.ndarray,
    control: np.ndarray,
    n_boot: int,
    seed: int,
    stream: str,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    diffs = bootstrap_diffs(treat, control, n_boot, seed, stream)
    return bootstrap_ci_from_diffs(diffs, alpha=alpha)


def bootstrap_p_one_sided(diffs: np.ndarray) -> float:
    if len(diffs) == 0:
        return float("nan")
    return float((np.sum(diffs <= 0.0) + 1.0) / (len(diffs) + 1.0))


def baseline_stream(metric: str, arm: str) -> str:
    return f"baseline|metric={metric}|arm={arm}"
