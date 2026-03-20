from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_historical_csvs(data_dir: str | Path) -> list[Path]:
    """Return historical CSV files sorted by name."""
    directory = Path(data_dir).expanduser().resolve()
    return sorted(directory.glob("*.csv"))


def load_historical_csvs(data_dir: str | Path, header: int = 1) -> dict[str, pd.DataFrame]:
    """Load all historical Binance CSV files from a directory."""
    datasets: dict[str, pd.DataFrame] = {}
    for csv_path in list_historical_csvs(data_dir):
        datasets[csv_path.stem] = pd.read_csv(csv_path, header=header)
    return datasets


def load_historical_asset_dataframe(
    data_dir: str | Path,
    asset_name: str,
    header: int = 1,
) -> pd.DataFrame:
    """Load a single historical asset dataframe by CSV stem."""
    csv_path = Path(data_dir).expanduser().resolve() / f"{asset_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found for asset '{asset_name}': {csv_path}")
    return pd.read_csv(csv_path, header=header)
