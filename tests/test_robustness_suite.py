import pandas as pd

from robustness_suite import compute_robustness


def test_robustness_not_constant_minus_cost():
    df = pd.DataFrame(
        {
            "arm": ["control"] * 5 + ["mens"] * 5 + ["womens"] * 5,
            "spend": [0, 0, 0, 0, 0, 10, 12, 8, 9, 11, 20, 22, 18, 19, 21],
        }
    )
    out = compute_robustness(
        df,
        spend_col="spend",
        margin=0.40,
        email_cost=0.01,
        n_boot=200,
        seed=1,
    )
    flagged = out["notes"].fillna("").str.contains("FLAG_CONSTANT_MINUS_COST").any()
    assert not flagged
