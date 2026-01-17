import pandas as pd

from experiment_checks import check_srm, balance_table, permutation_test_ate


def test_check_srm_balanced():
    df = pd.DataFrame({"arm": ["A", "B", "C", "A", "B", "C"]})
    chi2, p_value = check_srm(df, "arm")
    assert chi2 >= 0.0
    assert p_value > 0.5


def test_balance_table_smd_zero():
    df = pd.DataFrame(
        {
            "arm": ["control", "treat", "control", "treat"],
            "x": [1.0, 1.0, 2.0, 2.0],
            "cat": ["U", "U", "U", "U"],
        }
    )
    tbl = balance_table(df, ["x", "cat"], "arm", "control")
    assert not tbl.empty
    assert (tbl["abs_smd"] == 0.0).all()


def test_permutation_test_ate_zero_effect():
    df = pd.DataFrame(
        {
            "arm": ["control"] * 5 + ["treat"] * 5,
            "y": [1.0] * 10,
        }
    )
    ate, p_value = permutation_test_ate(df, "y", "arm", "treat", "control", n_perm=500, seed=1)
    assert abs(ate) < 1e-9
    assert p_value > 0.9
