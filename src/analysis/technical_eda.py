from __future__ import annotations

import pandas as pd


def compare_feature_frames(base_df: pd.DataFrame, technical_df: pd.DataFrame) -> pd.DataFrame:
    """Compare the base OHLCV frame to the engineered technical frame."""
    return pd.DataFrame(
        [
            {
                "dataset": "base",
                "rows": len(base_df),
                "columns": len(base_df.columns),
                "start": base_df.index.min() if len(base_df.index) else None,
                "end": base_df.index.max() if len(base_df.index) else None,
            },
            {
                "dataset": "technical",
                "rows": len(technical_df),
                "columns": len(technical_df.columns),
                "start": technical_df.index.min() if len(technical_df.index) else None,
                "end": technical_df.index.max() if len(technical_df.index) else None,
            },
        ]
    )
