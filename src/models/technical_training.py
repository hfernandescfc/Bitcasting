from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import Normalizer, StandardScaler

from src.models.blockchain_training import build_default_models

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


def evaluate_technical_models_by_horizon(
    df: pd.DataFrame,
    feature_columns: list[str],
    horizons: list[int] | tuple[int, ...],
    models: dict[str, Any] | None = None,
    normalize_features: bool = True,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Evaluate models across horizons and collect last-fold confusion matrices."""
    models = models or build_default_models()
    scaler = Normalizer() if normalize_features else None
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {name: {} for name in models}

    for model_name, model_instance in models.items():
        for horizon in horizons:
            target_col = f"Direction_t+{horizon}"
            horizon_df = df.dropna(subset=[target_col]).copy()
            X = horizon_df[feature_columns]
            y = horizon_df[target_col]
            X_values = scaler.fit_transform(X) if scaler is not None else X.to_numpy()

            splitter = TimeSeriesSplit(n_splits=n_splits)
            metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
            last_confusion = None

            for train_index, test_index in splitter.split(X_values):
                X_train, X_test = X_values[train_index], X_values[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    continue

                estimator = clone(model_instance)
                estimator.fit(X_train, y_train)
                predictions = estimator.predict(X_test)

                metrics["accuracy"].append(float(accuracy_score(y_test, predictions)))
                metrics["precision"].append(float(precision_score(y_test, predictions, zero_division=0)))
                metrics["recall"].append(float(recall_score(y_test, predictions, zero_division=0)))
                metrics["f1"].append(float(f1_score(y_test, predictions, zero_division=0)))
                last_confusion = confusion_matrix(y_test, predictions)

            rows.append(
                {
                    "model": model_name,
                    "horizon": f"t+{horizon}",
                    "mean_accuracy": float(np.mean(metrics["accuracy"])) if metrics["accuracy"] else np.nan,
                    "mean_precision": float(np.mean(metrics["precision"])) if metrics["precision"] else np.nan,
                    "mean_recall": float(np.mean(metrics["recall"])) if metrics["recall"] else np.nan,
                    "mean_f1": float(np.mean(metrics["f1"])) if metrics["f1"] else np.nan,
                }
            )
            artifacts[model_name][f"t+{horizon}"] = {"confusion": last_confusion}

    return pd.DataFrame(rows), artifacts


def evaluate_feature_importance_by_horizon(
    df: pd.DataFrame,
    feature_columns: list[str],
    horizons: list[int] | tuple[int, ...],
    models: dict[str, Any] | None = None,
    normalize_features: bool = True,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, dict[int, np.ndarray]], dict[str, dict[str, Any]]]:
    """Evaluate models and aggregate feature importance across folds."""
    models = models or build_default_models()
    scaler = Normalizer() if normalize_features else None
    rows: list[dict[str, Any]] = []
    importance_dict: dict[str, dict[int, np.ndarray]] = {name: {} for name in models}
    artifacts: dict[str, dict[str, Any]] = {name: {} for name in models}

    for model_name, model_instance in models.items():
        for horizon in horizons:
            target_col = f"Direction_t+{horizon}"
            horizon_df = df.dropna(subset=[target_col]).copy()
            X = horizon_df[feature_columns]
            y = horizon_df[target_col]
            X_values = scaler.fit_transform(X) if scaler is not None else X.to_numpy()

            splitter = TimeSeriesSplit(n_splits=n_splits)
            metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
            feature_importance = np.zeros(len(feature_columns))
            last_confusion = None

            for train_index, test_index in splitter.split(X_values):
                X_train, X_test = X_values[train_index], X_values[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    continue

                estimator = clone(model_instance)
                estimator.fit(X_train, y_train)
                predictions = estimator.predict(X_test)
                if hasattr(estimator, "feature_importances_"):
                    feature_importance += estimator.feature_importances_

                metrics["accuracy"].append(float(accuracy_score(y_test, predictions)))
                metrics["precision"].append(float(precision_score(y_test, predictions, zero_division=0)))
                metrics["recall"].append(float(recall_score(y_test, predictions, zero_division=0)))
                metrics["f1"].append(float(f1_score(y_test, predictions, zero_division=0)))
                last_confusion = confusion_matrix(y_test, predictions)

            if metrics["accuracy"]:
                feature_importance = feature_importance / len(metrics["accuracy"])
            importance_dict[model_name][horizon] = feature_importance
            rows.append(
                {
                    "model": model_name,
                    "horizon": f"t+{horizon}",
                    "mean_accuracy": float(np.mean(metrics["accuracy"])) if metrics["accuracy"] else np.nan,
                    "mean_precision": float(np.mean(metrics["precision"])) if metrics["precision"] else np.nan,
                    "mean_recall": float(np.mean(metrics["recall"])) if metrics["recall"] else np.nan,
                    "mean_f1": float(np.mean(metrics["f1"])) if metrics["f1"] else np.nan,
                }
            )
            artifacts[model_name][f"t+{horizon}"] = {"confusion": last_confusion}

    return pd.DataFrame(rows), importance_dict, artifacts


def collect_best_fold_outputs(
    df: pd.DataFrame,
    feature_columns: list[str],
    horizons: list[int] | tuple[int, ...],
    models: dict[str, Any],
    start_date: str = "2017-11-21",
    end_date: str = "2024-01-04",
    normalize_features: bool = True,
    n_splits: int = 5,
) -> dict[str, dict[str, Any]]:
    """Store predictions, probabilities and actuals from the best fold per model/horizon."""
    scaler = Normalizer() if normalize_features else None
    max_horizon = max(horizons)
    common_df = df.iloc[:-max_horizon].copy()
    common_df = common_df[(common_df.index >= start_date) & (common_df.index <= end_date)]
    output_storage: dict[str, dict[str, Any]] = {model_name: {} for model_name in models}

    for model_name, model_instance in models.items():
        for horizon in horizons:
            target_col = f"Direction_t+{horizon}"
            horizon_df = common_df.dropna(subset=[target_col]).copy()
            X = horizon_df[feature_columns]
            y = horizon_df[target_col]
            X_values = scaler.fit_transform(X) if scaler is not None else X.to_numpy()

            splitter = TimeSeriesSplit(n_splits=n_splits)
            best_fold_score = -np.inf
            best_predictions = None
            best_probabilities = None
            best_actuals = None

            for train_index, test_index in splitter.split(X_values):
                X_train, X_test = X_values[train_index], X_values[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    continue

                estimator = clone(model_instance)
                estimator.fit(X_train, y_train)
                predictions = estimator.predict(X_test)
                score = float(accuracy_score(y_test, predictions))
                if score > best_fold_score:
                    best_fold_score = score
                    best_predictions = predictions
                    best_actuals = y_test.to_numpy()
                    best_probabilities = estimator.predict_proba(X_test)[:, 1] if hasattr(estimator, "predict_proba") else None

            output_storage[model_name][f"t+{horizon}"] = {
                "predictions": best_predictions,
                "probabilities": best_probabilities,
                "actuals": best_actuals,
                "best_fold_score": best_fold_score,
            }

    return output_storage


def optimize_xgboost_technical(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "Direction_t+1",
    n_splits: int = 5,
    n_iter: int = 100,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, float], np.ndarray, pd.DataFrame]:
    """Run RandomizedSearchCV for the technical XGBoost model and return final artifacts."""
    if XGBClassifier is None:
        raise ImportError("xgboost must be installed to optimize the technical model.")

    prepared = df.dropna(subset=[target_column] + feature_columns).copy()
    X = prepared[feature_columns]
    y = prepared[target_column]
    splitter = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()

    search = RandomizedSearchCV(
        estimator=XGBClassifier(random_state=42),
        param_distributions={
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [3, 4, 5, 6, 7, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 2, 3, 4, 5],
        },
        n_iter=n_iter,
        cv=splitter,
        verbose=0,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(scaler.fit_transform(X), y)
    best_model = search.best_estimator_

    accuracies: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    last_confusion = None

    for train_index, test_index in splitter.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        fold_scaler = StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train)
        X_test_scaled = fold_scaler.transform(X_test)
        best_model.fit(X_train_scaled, y_train)
        predictions = best_model.predict(X_test_scaled)

        accuracies.append(float(accuracy_score(y_test, predictions)))
        precisions.append(float(precision_score(y_test, predictions, zero_division=0)))
        recalls.append(float(recall_score(y_test, predictions, zero_division=0)))
        f1_scores.append(float(f1_score(y_test, predictions, zero_division=0)))
        last_confusion = confusion_matrix(y_test, predictions)

    metrics = {
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else np.nan,
        "mean_precision": float(np.mean(precisions)) if precisions else np.nan,
        "mean_recall": float(np.mean(recalls)) if recalls else np.nan,
        "mean_f1": float(np.mean(f1_scores)) if f1_scores else np.nan,
    }
    importance_df = pd.DataFrame(
        {"feature": feature_columns, "importance": best_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return search.best_params_, pd.DataFrame(search.cv_results_), metrics, last_confusion, importance_df
