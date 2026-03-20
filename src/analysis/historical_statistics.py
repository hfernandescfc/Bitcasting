from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import friedmanchisquare, rankdata

try:
    from scikit_posthocs import posthoc_nemenyi_friedman
except ImportError:  # pragma: no cover
    posthoc_nemenyi_friedman = None


def pivot_metric_by_horizon(results_df: pd.DataFrame, metric: str = "f1_score") -> pd.DataFrame:
    """Pivot model metrics by horizon for statistical comparison."""
    return results_df.pivot(index="horizon", columns="model", values=metric).sort_index()


def run_friedman_test(results_df: pd.DataFrame, metric: str = "f1_score") -> dict[str, Any]:
    """Run Friedman test across models using horizon as repeated measure."""
    pivoted = pivot_metric_by_horizon(results_df, metric=metric)
    stat, p_value = friedmanchisquare(*[pivoted[column].values for column in pivoted.columns])
    return {"statistic": float(stat), "p_value": float(p_value), "pivoted": pivoted}


def run_nemenyi_test(results_df: pd.DataFrame, metric: str = "f1_score") -> pd.DataFrame:
    """Run the Nemenyi post-hoc test when scikit-posthocs is available."""
    if posthoc_nemenyi_friedman is None:
        raise ImportError("scikit-posthocs must be installed to run the Nemenyi test.")
    pivoted = pivot_metric_by_horizon(results_df, metric=metric)
    return posthoc_nemenyi_friedman(pivoted)


def average_model_ranks(results_df: pd.DataFrame, metric: str = "f1_score", ascending: bool = False) -> pd.Series:
    """Compute average model ranks across horizons."""
    pivoted = pivot_metric_by_horizon(results_df, metric=metric)
    ranked = pivoted.apply(lambda row: pd.Series(rankdata(row, method="average") if ascending else rankdata(-row, method="average"), index=row.index), axis=1)
    return ranked.mean(axis=0).sort_values()
