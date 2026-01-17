from __future__ import annotations

from typing import Any, Dict, Tuple
import pandas as pd


def normalize_arm(value: Any, arm_map: Dict[str, list]) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    for key, values in arm_map.items():
        for v in values:
            if text == str(v).strip().lower():
                return key
    return None


def prepare_data(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg = cfg["data"]
    arm_col = data_cfg["arm_col"]
    spend_col = data_cfg["spend_col"]
    visit_col = data_cfg["visit_col"]
    conversion_col = data_cfg["conversion_col"]
    id_col = data_cfg.get("id_col")
    arm_map = data_cfg["arm_map"]
    duplicate_policy = data_cfg.get("duplicate_policy", "drop_all")

    df = df.copy()

    cleaning_counts = []
    cleaning_counts.append({"metric": "rows_raw", "count": int(len(df))})

    df["arm"] = df[arm_col].apply(lambda v: normalize_arm(v, arm_map))
    missing_arm = df["arm"].isna()
    cleaning_counts.append({"metric": "rows_missing_arm", "count": int(missing_arm.sum())})
    df = df.loc[~missing_arm].copy()

    negative_spend = df[spend_col] < 0
    cleaning_counts.append({"metric": "rows_negative_spend", "count": int(negative_spend.sum())})
    df = df.loc[~negative_spend].copy()

    if id_col and id_col in df.columns:
        if duplicate_policy == "keep_first":
            duplicates = df[id_col].duplicated(keep="first")
            cleaning_counts.append({"metric": "rows_duplicate_ids_dropped", "count": int(duplicates.sum())})
            df = df.loc[~duplicates].copy()
        else:
            duplicates = df[id_col].duplicated(keep=False)
            cleaning_counts.append({"metric": "rows_duplicate_ids_dropped", "count": int(duplicates.sum())})
            df = df.loc[~duplicates].copy()
    else:
        cleaning_counts.append({"metric": "rows_duplicate_ids_dropped", "count": 0})

    spend_na = df[spend_col].isna()
    if spend_na.any():
        df.loc[spend_na, spend_col] = 0.0
    cleaning_counts.append({"metric": "rows_spend_na_set_to_zero", "count": int(spend_na.sum())})

    df["emailed"] = df["arm"].isin(["mens", "womens"]).astype(int)

    cleaning_counts.append({"metric": "rows_clean", "count": int(len(df))})

    cleaning_df = pd.DataFrame(cleaning_counts)
    return df, cleaning_df
