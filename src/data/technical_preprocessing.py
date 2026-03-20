from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TechnicalPreprocessingConfig:
    date_column: str = "Date"
    volume_column: str = "Volume USDT"
    renamed_volume_column: str = "Volume"
    drop_symbol_column: str = "Symbol"


def prepare_technical_market_dataframe(
    df: pd.DataFrame,
    config: TechnicalPreprocessingConfig | None = None,
) -> pd.DataFrame:
    """Normalize OHLCV data for technical-indicator generation."""
    config = config or TechnicalPreprocessingConfig()
    prepared = df.copy()

    if config.date_column not in prepared.columns:
        raise KeyError(f"Missing required date column: {config.date_column}")

    prepared[config.date_column] = pd.to_datetime(prepared[config.date_column], errors="coerce")
    prepared = prepared.dropna(subset=[config.date_column])

    if config.drop_symbol_column in prepared.columns:
        prepared = prepared.drop(columns=[config.drop_symbol_column])

    selected_columns = [config.date_column, "Open", "High", "Low", "Close", config.volume_column]
    missing = [column for column in selected_columns if column not in prepared.columns]
    if missing:
        raise KeyError(f"Missing required columns for technical dataset: {missing}")

    prepared = prepared[selected_columns].rename(columns={config.volume_column: config.renamed_volume_column})
    prepared = prepared.set_index(config.date_column).sort_index()
    return prepared
