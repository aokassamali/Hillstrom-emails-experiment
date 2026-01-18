import numpy as np
import pandas as pd

from heterogeneity import (
    FORBIDDEN_FEATURES,
    _ipw_value,
    _policy_assignments_binary,
    _prepare_features,
    build_subgroup_ate_mens_vs_control,
    sanity_check_policy_value,
)


def _synthetic_df(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    history = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "recency": rng.integers(1, 10, size=n),
            "history": history,
            "history_segment": rng.choice(["1", "2", "3"], size=n),
            "zip_code": rng.choice(["A", "B", "C"], size=n),
            "newbie": rng.integers(0, 2, size=n),
            "channel": rng.choice(["Phone", "Web"], size=n),
            "mens": rng.integers(0, 2, size=n),
            "womens": rng.integers(0, 2, size=n),
            "segment": rng.choice(["mens e-mail", "womens e-mail", "no e-mail"], size=n),
            "visit": rng.integers(0, 2, size=n),
            "conversion": rng.integers(0, 2, size=n),
        }
    )
    arms = rng.choice(["control", "mens", "womens"], size=n)
    base = 1.0 + 0.2 * history
    effect_mens = np.where(history > 0, 1.5, -0.2)
    effect_womens = np.where(history <= 0, 1.2, -0.3)
    spend = base + rng.normal(scale=0.1, size=n)
    spend = spend + np.where(arms == "mens", effect_mens, 0.0)
    spend = spend + np.where(arms == "womens", effect_womens, 0.0)
    df["arm"] = arms
    df["spend"] = spend
    return df


def test_forbidden_features_excluded():
    df = _synthetic_df()
    X, _ = _prepare_features(df, zip_top_k=2)
    assert not any(col in X.columns for col in FORBIDDEN_FEATURES)


def test_sanity_check_policy_value_constant():
    df = pd.DataFrame(
        {
            "arm": ["control", "mens", "womens"] * 4,
            "spend": [1.0, 2.0, 3.0] * 4,
        }
    )
    prop_map = {"control": 1 / 3, "mens": 1 / 3, "womens": 1 / 3}
    out = sanity_check_policy_value(df, margin=1.0, cost=0.0, prop_map=prop_map, tol=1e-6)
    assert not out["flag"].any()


def test_top_k_mens_signal():
    rng = np.random.default_rng(123)
    n = 4000
    scores = rng.uniform(-1.0, 1.0, size=n)
    arms = rng.choice(["control", "mens", "womens"], size=n)
    spend = np.where(arms == "mens", scores, 0.0)
    prop_map = {"control": 1 / 3, "mens": 1 / 3, "womens": 1 / 3}
    assignments = _policy_assignments_binary(scores)
    profit = spend
    value_top_10 = _ipw_value(profit, arms, assignments["top_k_mens_0.10"], prop_map)
    value_top_50 = _ipw_value(profit, arms, assignments["top_k_mens_0.50"], prop_map)
    assert value_top_10 > value_top_50


def test_subgroup_ate_columns():
    df = _synthetic_df()
    out = build_subgroup_ate_mens_vs_control(df, "spend", margin=0.4, cost=0.01, n_boot=50, seed=7)
    expected = {
        "group",
        "level",
        "n",
        "n_mens",
        "n_control",
        "tau_hat_spend",
        "ci_low_spend",
        "ci_high_spend",
        "tau_hat_profit",
        "ci_low_profit",
        "ci_high_profit",
    }
    assert expected.issubset(set(out.columns))
