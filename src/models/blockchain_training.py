from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.data.blockchain_preprocessing import restrict_date_range
from src.features.blockchain_targets import split_feature_and_target_columns

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None


DEFAULT_IMPORTANCE_THRESHOLDS = {
    "XGBoost": 0.015,
    "LightGBM": 40.0,
}


@dataclass
class FoldArtifacts:
    y_true: pd.Series
    y_pred: np.ndarray
    confusion: np.ndarray


def build_default_models(random_state: int = 42) -> dict[str, Any]:
    """Return the default model registry used by the notebook."""
    models: dict[str, Any] = {}
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
        )
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(random_state=random_state, verbosity=-1)
    if not models:
        raise ImportError("xgboost or lightgbm must be installed to train the models.")
    return models


def evaluate_time_series_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
    n_splits: int = 5,
    scale_features: bool = True,
) -> tuple[dict[str, float], FoldArtifacts]:
    """Evaluate one classifier with rolling time-series splits."""
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    last_fold: FoldArtifacts | None = None
    splitter = TimeSeriesSplit(n_splits=n_splits)

    for train_index, test_index in splitter.split(X):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        estimator = clone(model)
        if scale_features:
            scaler = StandardScaler()
            X_train_values = scaler.fit_transform(X_train)
            X_test_values = scaler.transform(X_test)
        else:
            X_train_values = X_train.to_numpy()
            X_test_values = X_test.to_numpy()

        estimator.fit(X_train_values, y_train)
        y_pred = estimator.predict(X_test_values)

        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        last_fold = FoldArtifacts(
            y_true=y_test,
            y_pred=np.asarray(y_pred),
            confusion=confusion_matrix(y_test, y_pred),
        )

    if last_fold is None:
        raise ValueError(
            "TimeSeriesSplit did not produce any valid folds. "
            "Check class balance, dataset length, or reduce the number of splits."
        )

    mean_metrics = {f"mean_{name}": float(np.mean(values)) for name, values in metrics.items()}
    mean_metrics.update({f"std_{name}": float(np.std(values)) for name, values in metrics.items()})
    return mean_metrics, last_fold


def get_feature_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    scale_features: bool = True,
) -> pd.DataFrame:
    """Fit a model once and return feature importances."""
    estimator = clone(model)
    if scale_features:
        scaler = StandardScaler()
        X_values = scaler.fit_transform(X)
    else:
        X_values = X.to_numpy()
    estimator.fit(X_values, y)

    if not hasattr(estimator, "feature_importances_"):
        raise AttributeError(f"Model {type(estimator).__name__} does not expose feature_importances_.")

    importance_df = pd.DataFrame(
        {"feature": X.columns.tolist(), "importance": estimator.feature_importances_}
    ).sort_values("importance", ascending=False)
    return importance_df.reset_index(drop=True)


def select_features_by_threshold(
    importance_df: pd.DataFrame,
    threshold: float,
) -> list[str]:
    """Select features above a threshold, with a fallback to the top 10."""
    selected = importance_df.loc[importance_df["importance"] > threshold, "feature"].tolist()
    if selected:
        return selected
    return importance_df.head(min(10, len(importance_df)))["feature"].tolist()


def evaluate_models_by_horizon(
    df: pd.DataFrame,
    horizons: tuple[int, ...] | list[int],
    models: dict[str, Any] | None = None,
    n_splits: int = 5,
    scale_features: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, FoldArtifacts]]]:
    """Evaluate multiple models across multiple forecast horizons."""
    models = models or build_default_models()
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, FoldArtifacts]] = {name: {} for name in models}

    for model_name, model in models.items():
        for horizon in horizons:
            target_column = f"Direction_t+{horizon}"
            horizon_df = df.dropna(subset=[target_column]).copy()
            X, y = split_feature_and_target_columns(horizon_df, target_column)
            metrics, last_fold = evaluate_time_series_classifier(
                X=X,
                y=y,
                model=model,
                n_splits=n_splits,
                scale_features=scale_features,
            )
            rows.append(
                {
                    "model": model_name,
                    "horizon": f"t+{horizon}",
                    "n_rows": len(horizon_df),
                    "n_features": X.shape[1],
                    **metrics,
                }
            )
            artifacts[model_name][f"t+{horizon}"] = last_fold

    return pd.DataFrame(rows), artifacts


def evaluate_models_with_feature_selection(
    df: pd.DataFrame,
    horizons: tuple[int, ...] | list[int],
    models: dict[str, Any] | None = None,
    importance_thresholds: dict[str, float] | None = None,
    start_date: str | None = "2017-11-21",
    end_date: str | None = "2024-01-04",
    n_splits: int = 5,
    scale_features: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Run feature selection per model and horizon, then evaluate again with the reduced feature set."""
    models = models or build_default_models()
    thresholds = importance_thresholds or DEFAULT_IMPORTANCE_THRESHOLDS
    common_df = restrict_date_range(df, start_date=start_date, end_date=end_date)

    rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {name: {} for name in models}

    for model_name, model in models.items():
        threshold = thresholds.get(model_name, 0.0)
        for horizon in horizons:
            target_column = f"Direction_t+{horizon}"
            horizon_df = common_df.dropna(subset=[target_column]).copy()
            X, y = split_feature_and_target_columns(horizon_df, target_column)

            importance_df = get_feature_importance(model=model, X=X, y=y, scale_features=scale_features)
            selected_features = select_features_by_threshold(importance_df, threshold)
            X_selected = X[selected_features].copy()

            metrics, last_fold = evaluate_time_series_classifier(
                X=X_selected,
                y=y,
                model=model,
                n_splits=n_splits,
                scale_features=scale_features,
            )

            rows.append(
                {
                    "model": model_name,
                    "horizon": f"t+{horizon}",
                    "n_rows": len(horizon_df),
                    "n_features": len(selected_features),
                    **metrics,
                }
            )
            artifacts[model_name][f"t+{horizon}"] = {
                "selected_features": selected_features,
                "importance": importance_df,
                "last_fold": last_fold,
            }

    return pd.DataFrame(rows), artifacts
