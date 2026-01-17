from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

REQUIRED_COLUMNS = [
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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def validate_schema(df: pd.DataFrame, required_cols: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_values(df: pd.DataFrame) -> None:
    issues: List[str] = []
    for col in ["visit", "conversion", "mens", "womens", "newbie"]:
        if col in df.columns:
            vals = df[col].dropna()
            bad = vals[~vals.isin([0, 1])]
            if not bad.empty:
                issues.append(f"{col}_not_binary")

    if "spend" in df.columns and (df["spend"] < 0).any():
        issues.append("spend_below_zero")

    if "segment" in df.columns:
        allowed = {"mens e-mail", "womens e-mail", "no e-mail"}
        seg = df["segment"].dropna().astype(str).str.strip().str.lower()
        if not seg.isin(allowed).all():
            issues.append("segment_unexpected")

    if issues:
        raise ValueError(f"Validation failed: {issues}")


def validate_no_missing(df: pd.DataFrame, required_cols: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = df[required_cols].isna().sum()
    if missing.any():
        bad = missing[missing > 0].to_dict()
        raise ValueError(f"Missing values in required columns: {bad}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "segment" in df.columns:
        df = df[df["segment"].notna()].copy()
    if "spend" in df.columns:
        df = df[df["spend"] >= 0].copy()
        df["spend"] = df["spend"].fillna(0.0)
    return df


def load_raw_from_dir(raw_dir: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    csvs = sorted(raw_dir.glob("*.csv"))
    if len(csvs) == 0:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    if len(csvs) > 1:
        raise ValueError(f"Multiple CSV files found in {raw_dir}: {csvs}")
    return pd.read_csv(csvs[0])


def write_processed(df: pd.DataFrame, processed_path: Path) -> Path:
    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    return processed_path


def build_processed(raw_dir: Path, processed_path: Path) -> Path:
    df_raw = load_raw_from_dir(raw_dir)
    df_raw = normalize_columns(df_raw)
    validate_schema(df_raw)
    validate_values(df_raw)
    validate_no_missing(df_raw)
    df_clean = clean_data(df_raw)
    return write_processed(df_clean, processed_path)
