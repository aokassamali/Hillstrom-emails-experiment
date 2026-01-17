import numpy as np
import pandas as pd

from estimation import bootstrap_ci, estimate_diff_in_means, holm_adjust, summarize_outcomes


def test_holm_adjust():
    p = [0.01, 0.04]
    adj = holm_adjust(p)
    assert np.allclose(adj, np.array([0.02, 0.04]))


def test_bootstrap_ci_constant():
    t = np.array([1.0, 1.0, 1.0])
    c = np.array([1.0, 1.0, 1.0])
    low, high = bootstrap_ci(t, c, n_boot=200, seed=1)
    assert low == 0.0
    assert high == 0.0


def test_estimate_diff_in_means():
    df = pd.DataFrame({"arm": ["control"] * 3 + ["treat"] * 3, "y": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]})
    est, se, ci_low, ci_high, p = estimate_diff_in_means(df, "y", "arm", "treat", "control")
    assert est == 1.0
    assert se == 0.0
    assert ci_low == 1.0
    assert ci_high == 1.0
    assert p == 1.0


def test_summarize_outcomes():
    df = pd.DataFrame(
        {
            "arm": ["control", "treat", "control", "treat"],
            "y": [0.0, 1.0, 0.0, 1.0],
            "z": [1.0, 1.0, 1.0, 1.0],
        }
    )
    summary = summarize_outcomes(df, ["y", "z"], "arm", "control")
    assert set(summary["outcome"]) == {"y", "z"}
    assert set(summary["arm"]) == {"treat"}
