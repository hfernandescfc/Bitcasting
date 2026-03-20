from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def build_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages per column."""
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = ((missing / len(df)) * 100).round(2)
    return pd.DataFrame({"n_missing": missing, "pct_missing": missing_pct})


def dataset_snapshot(df: pd.DataFrame) -> pd.Series:
    """Return a compact structural summary of the dataframe."""
    snapshot = {
        "rows": len(df),
        "columns": len(df.columns),
        "start": df.index.min() if len(df.index) else None,
        "end": df.index.max() if len(df.index) else None,
    }
    return pd.Series(snapshot)


def top_columns_by_missing(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the top columns by missing percentage."""
    return build_missing_report(df).head(top_n)


def class_balance(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Return counts and percentages for a binary target."""
    counts = df[target_column].value_counts(dropna=False).sort_index()
    percentages = ((counts / counts.sum()) * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": percentages})


def feature_importance_frame(importance_map: Mapping[str, float], top_n: int = 20) -> pd.DataFrame:
    """Convert a feature-importance dict into a sorted dataframe."""
    frame = (
        pd.DataFrame({"feature": list(importance_map.keys()), "importance": list(importance_map.values())})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return frame
