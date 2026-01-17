from hillstrom_emails.config import load_config


def test_srm_expected_allocation_default():
    cfg = load_config("config/config.yaml")
    expected = cfg["srm"].get("expected_allocation", {})
    assert round(expected.get("control", 0), 6) == round(1 / 3, 6)
    assert round(expected.get("mens", 0), 6) == round(1 / 3, 6)
    assert round(expected.get("womens", 0), 6) == round(1 / 3, 6)


def test_missingness_check_passes():
    cfg = load_config("config/config.yaml")
    required = [
        "recency",
        "history_segment",
        "history",
        "mens",
        "womens",
        "zip_code",
        "newbie",
        "channel",
        "segment",
        "visit",
        "conversion",
        "spend",
    ]
    data = {col: [1] for col in required}
    from data import validate_no_missing, normalize_columns
    import pandas as pd

    df = normalize_columns(pd.DataFrame(data))
    validate_no_missing(df)
