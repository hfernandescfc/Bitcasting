from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_MA_WINDOWS: tuple[int, ...] = (3, 5, 7, 20, 50, 100)
DEFAULT_EMA_SPANS: tuple[int, ...] = (3, 5, 7, 20, 50, 100)
DEFAULT_TECHNICAL_HORIZONS: tuple[int, ...] = (1, 3, 7, 14, 30)

DEFAULT_TECHNICAL_FEATURES: tuple[str, ...] = (
    "High",
    "S3",
    "Upper_Band",
    "Close",
    "STD",
    "Lower_Band",
    "Volume",
    "Low",
    "OBV",
    "MA_50",
    "S2",
    "R2",
    "Pivot",
    "MA_100",
    "R3",
    "MA_5",
    "MA_3",
    "MA_7",
    "Change",
    "Open",
    "RSI_14",
    "MA_20",
    "EMA_3",
    "R1",
    "S1",
    "EMA_5",
    "EMA_7",
    "EMA_20",
    "EMA_50",
    "EMA_100",
)

REDUCED_TECHNICAL_FEATURES: tuple[str, ...] = (
    "High",
    "S3",
    "Upper_Band",
    "Close",
    "STD",
    "Lower_Band",
    "Volume",
    "Low",
    "OBV",
    "MA_50",
    "S2",
    "R2",
    "Pivot",
    "MA_100",
    "R3",
    "MA_5",
    "MA_7",
    "Change",
    "Open",
    "RSI_14",
    "MA_20",
    "R1",
    "S1",
    "EMA_5",
    "EMA_7",
    "EMA_20",
    "EMA_50",
    "EMA_100",
)

OPTIMIZED_XGB_FEATURES: tuple[str, ...] = (
    "High",
    "S3",
    "Upper_Band",
    "Close",
    "STD",
    "Lower_Band",
    "Volume",
    "Low",
    "OBV",
    "MA_50",
    "S2",
    "R2",
    "Pivot",
    "MA_100",
    "R3",
    "MA_5",
    "MA_3",
    "MA_7",
    "Change",
    "Open",
    "RSI_14",
    "MA_20",
    "EMA_3",
    "R1",
    "S1",
)


def calculate_moving_average(df: pd.DataFrame, column: str, window: int) -> pd.Series:
    return df[column].rolling(window=window).mean()


def calculate_exponential_moving_average(df: pd.DataFrame, column: str, span: int) -> pd.Series:
    return df[column].ewm(span=span, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


def calculate_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std_dev: int = 2,
) -> pd.DataFrame:
    prepared = df.copy()
    prepared["SMA"] = prepared["Close"].rolling(window=window).mean()
    prepared["STD"] = prepared["Close"].rolling(window=window).std()
    prepared["Upper_Band"] = prepared["SMA"] + (prepared["STD"] * num_std_dev)
    prepared["Lower_Band"] = prepared["SMA"] - (prepared["STD"] * num_std_dev)
    return prepared


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["Change"] = prepared["Close"].diff()
    prepared["Direction"] = prepared["Change"].apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
    prepared["OBV"] = (prepared["Direction"] * prepared["Volume"]).cumsum()
    return prepared


def calculate_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    pivot_points = pd.DataFrame(index=df.index, columns=["Pivot", "R1", "S1", "R2", "S2", "R3", "S3"])
    pivot_points["Pivot"] = (df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)) / 3
    pivot_points["R1"] = 2 * pivot_points["Pivot"] - df["Low"].shift(1)
    pivot_points["S1"] = 2 * pivot_points["Pivot"] - df["High"].shift(1)
    pivot_points["R2"] = pivot_points["Pivot"] + (df["High"].shift(1) - df["Low"].shift(1))
    pivot_points["S2"] = pivot_points["Pivot"] - (df["High"].shift(1) - df["Low"].shift(1))
    pivot_points["R3"] = pivot_points["R2"] + (df["High"].shift(1) - df["Low"].shift(1))
    pivot_points["S3"] = pivot_points["S2"] - (df["High"].shift(1) - df["Low"].shift(1))
    return pivot_points


def add_technical_indicators(
    df: pd.DataFrame,
    ma_windows: Sequence[int] = DEFAULT_MA_WINDOWS,
    ema_spans: Sequence[int] = DEFAULT_EMA_SPANS,
    rsi_window: int = 14,
) -> pd.DataFrame:
    prepared = df.copy()
    for window in ma_windows:
        prepared[f"MA_{window}"] = calculate_moving_average(prepared, "Close", window)
    for span in ema_spans:
        prepared[f"EMA_{span}"] = calculate_exponential_moving_average(prepared, "Close", span)
    prepared["RSI_14"] = calculate_rsi(prepared, window=rsi_window)
    prepared = calculate_bollinger_bands(prepared)
    prepared = calculate_obv(prepared)
    prepared = prepared.join(calculate_pivot_points(prepared))
    return prepared


def add_technical_targets(
    df: pd.DataFrame,
    horizons: Sequence[int] = DEFAULT_TECHNICAL_HORIZONS,
    price_column: str = "Close",
) -> pd.DataFrame:
    prepared = df.copy()
    prepared["Close_t+1"] = prepared[price_column].shift(-1)
    prepared["Direction_t+1"] = np.where(prepared["Close_t+1"] > prepared[price_column], 1, 0)

    for horizon in horizons:
        close_column = f"Close_t+{horizon}"
        direction_column = f"Direction_t+{horizon}"
        prepared[close_column] = prepared[price_column].shift(-horizon)
        prepared[direction_column] = np.where(prepared[close_column] > prepared[price_column], 1, 0)

    return prepared
