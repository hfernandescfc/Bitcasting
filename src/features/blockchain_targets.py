from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 7, 14, 30)


def add_direction_targets(
    df: pd.DataFrame,
    price_column: str = "PriceUSD",
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    include_base_target: bool = True,
) -> pd.DataFrame:
    """Create close and direction targets for all requested horizons."""
    if price_column not in df.columns:
        raise KeyError(f"Missing required price column: {price_column}")

    prepared = df.copy()
    prices = prepared[price_column]

    if include_base_target:
        prepared["Close_t+1"] = prices.shift(-1)
        prepared["Change"] = prepared["Close_t+1"] - prices
        prepared["Direction"] = (prepared["Change"] > 0).astype(int)

    for horizon in horizons:
        close_column = f"Close_t+{horizon}"
        direction_column = f"Direction_t+{horizon}"
        prepared[close_column] = prices.shift(-horizon)
        prepared[direction_column] = (prepared[close_column] > prices).astype(int)

    return prepared


def split_feature_and_target_columns(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and target series excluding helper columns."""
    target_like_columns = [
        column
        for column in df.columns
        if column.startswith("Direction_t+")
        or column.startswith("Close_t+")
        or column in {"Direction", "Change"}
    ]
    feature_columns = [column for column in df.columns if column not in set(target_like_columns)]
    return df[feature_columns].copy(), df[target_column].copy()
