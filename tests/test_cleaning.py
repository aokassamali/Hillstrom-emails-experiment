import pandas as pd
from hillstrom_emails.cleaning import prepare_data


def test_prepare_data_exclusions():
    df = pd.DataFrame({
        "segment": ["Mens E-Mail", "No E-Mail", None],
        "spend": [10.0, -1.0, 5.0],
        "visit": [1, 0, 1],
        "conversion": [1, 0, 0],
        "customer_id": [1, 2, 3],
    })
    cfg = {
        "data": {
            "arm_col": "segment",
            "spend_col": "spend",
            "visit_col": "visit",
            "conversion_col": "conversion",
            "id_col": "customer_id",
            "arm_map": {
                "control": ["no e-mail"],
                "mens": ["mens e-mail"],
                "womens": ["womens e-mail"],
            },
            "duplicate_policy": "drop_all",
        }
    }
    cleaned, report = prepare_data(df, cfg)
    assert len(cleaned) == 1
    assert report[report["metric"] == "rows_missing_arm"]["count"].iloc[0] == 1
