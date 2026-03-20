from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_csv_files(data_dir: str | Path) -> list[Path]:
    """Return CSV files sorted by name."""
    directory = Path(data_dir).expanduser().resolve()
    return sorted(directory.glob("*.csv"))


def load_blockchain_csvs(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all blockchain CSV files from a directory."""
    datasets: dict[str, pd.DataFrame] = {}
    for csv_path in list_csv_files(data_dir):
        datasets[csv_path.stem] = pd.read_csv(csv_path)
    return datasets


def load_asset_dataframe(data_dir: str | Path, asset_name: str) -> pd.DataFrame:
    """Load a single asset dataframe by CSV stem."""
    csv_path = Path(data_dir).expanduser().resolve() / f"{asset_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found for asset '{asset_name}': {csv_path}")
    return pd.read_csv(csv_path)
