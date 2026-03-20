from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_PRICE_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume BTC",
    "Volume USDT",
    "tradecount",
)


@dataclass(frozen=True)
class HistoricalPreprocessingConfig:
    date_column: str = "Date"
    symbol_column: str = "Symbol"
    drop_symbol_column: bool = True
    drop_first_column: bool = True
    sort_index: bool = True


def prepare_historical_market_dataframe(
    df: pd.DataFrame,
    config: HistoricalPreprocessingConfig | None = None,
) -> pd.DataFrame:
    """Normalize the Binance market dataframe into a daily OHLCV table."""
    config = config or HistoricalPreprocessingConfig()
    prepared = df.copy()

    if config.drop_first_column and len(prepared.columns) > 0:
        prepared = prepared.iloc[:, 1:]

    if config.drop_symbol_column and config.symbol_column in prepared.columns:
        prepared = prepared.drop(columns=[config.symbol_column])

    if config.date_column not in prepared.columns:
        raise KeyError(f"Missing required date column: {config.date_column}")

    prepared[config.date_column] = pd.to_datetime(prepared[config.date_column], errors="coerce")
    prepared = prepared.dropna(subset=[config.date_column]).set_index(config.date_column)

    if config.sort_index:
        prepared = prepared.sort_index()

    return prepared


def add_price_direction_targets(
    df: pd.DataFrame,
    price_column: str = "Close",
    horizons: Sequence[int] = (1,),
) -> pd.DataFrame:
    """Add close and direction targets to the raw historical dataframe."""
    if price_column not in df.columns:
        raise KeyError(f"Missing required price column: {price_column}")

    prepared = df.copy()
    if 1 in horizons:
        prepared["Close_t+1"] = prepared[price_column].shift(-1)
        prepared["Close_Diff"] = prepared[price_column].diff()
        prepared["Direction_t+1"] = np.where(prepared["Close_t+1"] > prepared[price_column], 1, 0)

    for horizon in horizons:
        close_column = f"Close_t+{horizon}"
        direction_column = f"Direction_t+{horizon}"
        prepared[close_column] = prepared[price_column].shift(-horizon)
        prepared[direction_column] = np.where(prepared[close_column] > prepared[price_column], 1, 0)

    return prepared
