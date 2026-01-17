import pandas as pd

from data_summary import build_summary_tables


def test_build_summary_tables():
    df = pd.DataFrame(
        {
            "segment": ["Mens E-Mail", "No E-Mail"],
            "visit": [1, 0],
            "conversion": [0, 0],
            "spend": [0.0, 0.0],
        }
    )
    tables = build_summary_tables(df)
    assert "schema" in tables
    assert "missingness" in tables
    assert "arm_counts" in tables
    assert "outcome_summary" in tables
