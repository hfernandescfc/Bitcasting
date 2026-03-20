from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


DEFAULT_DROP_COLUMNS = (
    "principal_market_usd",
    "principal_market_price_usd",
    "CapMrktEstUSD",
)


@dataclass(frozen=True)
class BlockchainPreprocessingConfig:
    start_date: str | None = "2017-08-17"
    end_date: str | None = None
    time_column: str = "time"
    price_column: str = "PriceUSD"
    drop_columns: tuple[str, ...] = DEFAULT_DROP_COLUMNS
    drop_object_columns: bool = True
    sort_index: bool = True
    drop_all_na_rows: bool = False


def prepare_blockchain_dataframe(
    df: pd.DataFrame,
    config: BlockchainPreprocessingConfig | None = None,
) -> pd.DataFrame:
    """Normalize the raw blockchain dataframe into a modeling-ready base table."""
    config = config or BlockchainPreprocessingConfig()
    prepared = df.copy()

    if config.time_column not in prepared.columns:
        raise KeyError(f"Missing required time column: {config.time_column}")
    if config.price_column not in prepared.columns:
        raise KeyError(f"Missing required price column: {config.price_column}")

    prepared[config.time_column] = pd.to_datetime(prepared[config.time_column], errors="coerce")
    prepared = prepared.dropna(subset=[config.time_column])

    if config.start_date is not None:
        prepared = prepared[prepared[config.time_column] >= pd.Timestamp(config.start_date)]
    if config.end_date is not None:
        prepared = prepared[prepared[config.time_column] <= pd.Timestamp(config.end_date)]

    columns_to_drop = [column for column in config.drop_columns if column in prepared.columns]
    if columns_to_drop:
        prepared = prepared.drop(columns=columns_to_drop)

    prepared = prepared.set_index(config.time_column)

    if config.sort_index:
        prepared = prepared.sort_index()

    if config.drop_object_columns:
        object_columns = prepared.select_dtypes(include=["object"]).columns
        if len(object_columns) > 0:
            prepared = prepared.drop(columns=list(object_columns))

    if config.drop_all_na_rows:
        prepared = prepared.dropna()

    return prepared


def restrict_date_range(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Filter a dataframe with a DatetimeIndex by an optional date interval."""
    filtered = df.copy()
    if start_date is not None:
        filtered = filtered[filtered.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        filtered = filtered[filtered.index <= pd.Timestamp(end_date)]
    return filtered


def drop_rows_with_missing_targets(df: pd.DataFrame, target_columns: Iterable[str]) -> pd.DataFrame:
    """Remove rows missing any of the requested target columns."""
    return df.dropna(subset=list(target_columns))
