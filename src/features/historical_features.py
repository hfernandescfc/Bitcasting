from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_HISTORICAL_FEATURES: tuple[str, ...] = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume BTC",
    "Volume USDT",
    "tradecount",
)


def log_diff_transform(series: pd.Series) -> pd.Series:
    """Apply the log-difference transform used in the notebook."""
    return np.log(series + 1).diff()


def build_log_diff_dataset(
    df: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_HISTORICAL_FEATURES,
    keep_close_column: bool = True,
    close_column: str = "Close",
) -> pd.DataFrame:
    """Create the transformed log-difference dataset."""
    transformed = pd.DataFrame(index=df.index)
    for feature in feature_columns:
        transformed[f"log_diff_{feature.lower().replace(' ', '_')}"] = log_diff_transform(df[feature])
    if keep_close_column and close_column in df.columns:
        transformed[close_column] = df[close_column]
    return transformed.dropna().copy()


def add_log_diff_targets(
    df: pd.DataFrame,
    horizons: Sequence[int] = (1, 3, 7, 14, 30),
    base_column: str = "log_diff_close",
) -> pd.DataFrame:
    """Create future close and direction targets from transformed close returns."""
    if base_column not in df.columns:
        raise KeyError(f"Missing required base column: {base_column}")

    prepared = df.copy()
    for horizon in horizons:
        close_column = f"Close_t+{horizon}"
        direction_column = f"Direction_t+{horizon}"
        prepared[close_column] = prepared[base_column].shift(-horizon)
        prepared[direction_column] = np.where(prepared[close_column] > 0, 1, 0)
    return prepared


def create_multi_period_targets(
    df: pd.DataFrame,
    periods: Sequence[int] = (1, 3, 7, 14, 30),
    threshold_multiplier: float = 0.75,
    price_column: str = "Close",
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Create multiclass trading targets using dynamic volatility thresholds."""
    if price_column not in df.columns:
        raise KeyError(f"Missing required price column: {price_column}")

    prepared = df.copy()
    for period in periods:
        returns = prepared[price_column].pct_change(periods=period)
        volatility = returns.rolling(window=volatility_window).std()
        upper_threshold = threshold_multiplier * volatility
        lower_threshold = -threshold_multiplier * volatility

        target = np.ones(len(returns), dtype=int)
        target[returns > upper_threshold] = 2
        target[returns < lower_threshold] = 0
        prepared[f"target_{period}d"] = target

    return prepared


def create_lagged_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    n_lags: int,
) -> pd.DataFrame:
    """Append lagged versions of each feature column."""
    prepared = df.copy()
    for feature in feature_columns:
        for lag in range(1, n_lags + 1):
            prepared[f"{feature}_t-{lag}"] = prepared[feature].shift(lag)
    return prepared


def build_lag_feature_columns(
    feature_columns: Sequence[str],
    n_lags: int,
    include_current: bool = True,
) -> list[str]:
    """Build the ordered list of lagged feature columns used in modeling."""
    lagged = [f"{feature}_t-{lag}" for feature in feature_columns for lag in range(1, n_lags + 1)]
    if include_current:
        lagged += list(feature_columns)
    return lagged


def prepare_lagged_dataset(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    n_lags: int,
    target_column: str,
) -> pd.DataFrame:
    """Create the lagged dataset and remove rows missing the requested target."""
    prepared = create_lagged_features(df, feature_columns, n_lags)
    return prepared.dropna(subset=[target_column]).dropna().copy()
