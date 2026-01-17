from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def _pick_arm_col(df: pd.DataFrame) -> str:
    for col in ["arm", "segment", "Segment"]:
        if col in df.columns:
            return col
    raise ValueError("No arm/segment column found for summaries.")


def summarize_schema(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df)).fillna(0.0)
    return pd.DataFrame(
        {"column": missing_count.index, "missing_count": missing_count.values, "missing_pct": missing_pct.values}
    )


def summarize_arm_counts(df: pd.DataFrame) -> pd.DataFrame:
    arm_col = _pick_arm_col(df)
    arm = df[arm_col].astype(str).str.strip().str.lower()
    counts = arm.value_counts().reset_index()
    counts.columns = ["arm", "count"]
    return counts


def summarize_outcomes_by_arm(df: pd.DataFrame) -> pd.DataFrame:
    arm_col = _pick_arm_col(df)
    arm = df[arm_col].astype(str).str.strip().str.lower()
    df = df.copy()
    df["__arm__"] = arm

    visit_col = "visit" if "visit" in df.columns else "Visit"
    conversion_col = "conversion" if "conversion" in df.columns else "Conversion"
    spend_col = "spend" if "spend" in df.columns else "Spend"

    grouped = df.groupby("__arm__", dropna=False)
    summary = grouped.agg(
        n=("__arm__", "size"),
        visit_rate=(visit_col, "mean"),
        conversion_rate=(conversion_col, "mean"),
        mean_spend=(spend_col, "mean"),
    ).reset_index()
    summary = summary.rename(columns={"__arm__": "arm"})
    return summary


def build_summary_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "schema": summarize_schema(df),
        "missingness": summarize_missingness(df),
        "arm_counts": summarize_arm_counts(df),
        "outcome_summary": summarize_outcomes_by_arm(df),
    }


def _to_md_table(df: pd.DataFrame) -> str:
    df = df.copy()
    if "missing_pct" in df.columns:
        df["missing_pct"] = df["missing_pct"].astype(float).round(4)
    for col in ["visit_rate", "conversion_rate", "mean_spend"]:
        if col in df.columns:
            df[col] = df[col].astype(float).round(4)

    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def write_summary(df: pd.DataFrame, output_dir: Path) -> Tuple[Path, Path]:
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    tables = build_summary_tables(df)
    long_frames = []
    for section, frame in tables.items():
        temp = frame.copy()
        temp.insert(0, "section", section)
        long_frames.append(temp)
    summary_long = pd.concat(long_frames, ignore_index=True, sort=False)
    csv_path = tables_dir / "data_summary.csv"
    summary_long.to_csv(csv_path, index=False)

    md_path = output_dir / "data_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Data Summary\n\n")
        f.write("## Schema\n")
        f.write(_to_md_table(tables["schema"]))
        f.write("\n\n## Missingness\n")
        f.write(_to_md_table(tables["missingness"]))
        f.write("\n\n## Arm Counts\n")
        f.write(_to_md_table(tables["arm_counts"]))
        f.write("\n\n## Outcome Summary by Arm\n")
        f.write(_to_md_table(tables["outcome_summary"]))
        f.write("\n")

    return csv_path, md_path
