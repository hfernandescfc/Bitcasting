from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def compare_series_snapshots(raw_df: pd.DataFrame, transformed_df: pd.DataFrame) -> pd.DataFrame:
    """Compare row and column counts between raw and transformed datasets."""
    return pd.DataFrame(
        [
            {
                "dataset": "raw",
                "rows": len(raw_df),
                "columns": len(raw_df.columns),
                "start": raw_df.index.min() if len(raw_df.index) else None,
                "end": raw_df.index.max() if len(raw_df.index) else None,
            },
            {
                "dataset": "transformed",
                "rows": len(transformed_df),
                "columns": len(transformed_df.columns),
                "start": transformed_df.index.min() if len(transformed_df.index) else None,
                "end": transformed_df.index.max() if len(transformed_df.index) else None,
            },
        ]
    )


def analyze_targets(
    df: pd.DataFrame,
    periods: Sequence[int] = (1, 3, 7, 14, 30),
) -> pd.DataFrame:
    """Summarize multiclass target distributions for each forecast period."""
    rows: list[dict[str, float | int]] = []
    for period in periods:
        column_name = f"target_{period}d"
        target = df[column_name]
        rows.append(
            {
                "period": period,
                "total_signals": int(len(target)),
                "sell_signals": int((target == 0).sum()),
                "hold_signals": int((target == 1).sum()),
                "buy_signals": int((target == 2).sum()),
                "sell_ratio": float((target == 0).mean()),
                "hold_ratio": float((target == 1).mean()),
                "buy_ratio": float((target == 2).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("period")


def target_correlation_matrix(
    df: pd.DataFrame,
    periods: Sequence[int] = (1, 3, 7, 14, 30),
) -> pd.DataFrame:
    """Return target correlation matrix across forecast windows."""
    columns = [f"target_{period}d" for period in periods]
    return df[columns].corr()


def analyze_threshold_effects(
    df: pd.DataFrame,
    target_builder,
    periods: Sequence[int] = (1, 3, 7, 14, 30),
    thresholds: Sequence[float] = tuple(np.arange(0.5, 2.25, 0.25)),
) -> pd.DataFrame:
    """Analyze how class distribution changes across threshold multipliers."""
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        threshold_df = target_builder(df, periods=periods, threshold_multiplier=threshold)
        for period in periods:
            column = f"target_{period}d"
            counts = threshold_df[column].value_counts(normalize=True).to_dict()
            rows.append(
                {
                    "threshold": float(threshold),
                    "period": int(period),
                    "sell_ratio": float(counts.get(0, 0.0)),
                    "hold_ratio": float(counts.get(1, 0.0)),
                    "buy_ratio": float(counts.get(2, 0.0)),
                }
            )
    return pd.DataFrame(rows)
