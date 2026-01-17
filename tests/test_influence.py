import pandas as pd

from influence import build_influence_table


def test_influence_top_share_not_all_one():
    df = pd.DataFrame(
        {
            "arm": ["control"] * 10 + ["mens"] * 10 + ["womens"] * 10,
            "spend": [0, 0, 0, 0, 0, 0, 0, 0, 0, 100] + [0, 0, 0, 0, 0, 0, 0, 0, 0, 200] + [0, 0, 0, 0, 0, 0, 0, 0, 0, 300],
            "id": [f"id{i}" for i in range(30)],
        }
    )
    out = build_influence_table(
        df,
        spend_col="spend",
        arm_col="arm",
        control_arm="control",
        id_col="id",
        margin=0.40,
        email_cost=0.01,
    )
    subset = out[(out["x_removed"] == 0.01) & (out["section"] == "top_share")]
    assert not (subset["top_share"] == 1.0).all()
