import pandas as pd

from comparisons import mens_vs_womens


def test_mens_vs_womens_ci_p_consistent():
    df = pd.DataFrame(
        {
            "arm": ["mens"] * 5 + ["womens"] * 5,
            "y": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        }
    )
    est, ci_low, ci_high, p_two, p_gt_0, B, seed = mens_vs_womens(df, "y", "arm", n_boot=500, seed=1)
    if ci_low > 0:
        assert p_two < 0.1
