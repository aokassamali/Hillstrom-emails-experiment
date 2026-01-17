import pandas as pd
from hillstrom_emails.estimators import compute_profit


def test_compute_profit():
    df = pd.DataFrame({"spend": [10.0], "emailed": [1]})
    profit = compute_profit(df, 0.40, 0.01, "spend")
    assert float(profit.iloc[0]) == 3.99
