import pandas as pd

from two_part import summarize_two_part


def test_two_part_contains_converter_distribution():
    df = pd.DataFrame(
        {
            "arm": ["control", "control", "mens", "mens"],
            "conversion": [1, 0, 1, 0],
            "spend": [10.0, 0.0, 12.0, 0.0],
        }
    )
    out = summarize_two_part(df, "arm", "conversion", "spend", "control", n_boot=200, seed=1)
    assert (out["section"] == "converter_distribution").any()
