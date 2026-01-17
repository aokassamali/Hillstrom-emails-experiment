import pandas as pd

from data import normalize_columns, validate_schema, validate_values


def test_validate_schema_ok():
    df = pd.DataFrame(
        {
            "recency": [1],
            "history_segment": ["1) $0 - $100"],
            "history": [50.0],
            "mens": [1],
            "womens": [0],
            "zip_code": ["Urban"],
            "newbie": [0],
            "channel": ["Web"],
            "segment": ["Mens E-Mail"],
            "visit": [1],
            "conversion": [0],
            "spend": [0.0],
        }
    )
    df = normalize_columns(df)
    validate_schema(df)


def test_validate_values_rejects_spend():
    df = pd.DataFrame(
        {
            "recency": [1],
            "history_segment": ["1) $0 - $100"],
            "history": [50.0],
            "mens": [1],
            "womens": [0],
            "zip_code": ["Urban"],
            "newbie": [0],
            "channel": ["Web"],
            "segment": ["Mens E-Mail"],
            "visit": [1],
            "conversion": [0],
            "spend": [-1.0],
        }
    )
    df = normalize_columns(df)
    try:
        validate_values(df)
        assert False, "Expected validation error"
    except ValueError:
        assert True
