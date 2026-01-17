from __future__ import annotations

import numpy as np


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
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
