from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.svm import SVC

from src.features.historical_features import build_lag_feature_columns, create_lagged_features

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None


DEFAULT_TARGET_HORIZONS = (1, 3, 7, 14, 30)
DEFAULT_FEATURE_COLUMNS = (
    "log_diff_open",
    "log_diff_high",
    "log_diff_low",
    "log_diff_close",
    "log_diff_volume_btc",
    "log_diff_volume_usdt",
    "log_diff_tradecount",
)


def build_default_multiclass_models(random_state: int = 42) -> dict[str, Any]:
    """Return the default model family used in the new reference notebook."""
    models: dict[str, Any] = {
        "SVM": SVC(random_state=random_state),
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=random_state),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(random_state=random_state)
    return models


def build_default_param_spaces() -> dict[str, dict[str, Any]]:
    """Return the default randomized-search spaces from the notebook."""
    models = build_default_multiclass_models()
    spaces: dict[str, dict[str, Any]] = {}
    if "XGBoost" in models:
        spaces["XGBoost"] = {
            "model": models["XGBoost"],
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
            },
        }
    spaces["SVM"] = {
        "model": models["SVM"],
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        },
    }
    spaces["Logistic Regression"] = {
        "model": models["Logistic Regression"],
        "params": {
            "C": [0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs", "saga"],
        },
    }
    spaces["Random Forest"] = {
        "model": models["Random Forest"],
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    }
    return spaces


def build_xgb_lgbm_models(random_state: int = 42) -> dict[str, Any]:
    """Return the binary models used in the lag-output experiment."""
    models: dict[str, Any] = {}
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=random_state)
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(random_state=random_state)
    return models


def prepare_lagged_modeling_data(
    df: pd.DataFrame,
    feature_columns: list[str],
    max_lags: int,
    target_column: str,
) -> pd.DataFrame:
    """Create lagged features and drop rows with missing values."""
    lagged = create_lagged_features(df, feature_columns, max_lags)
    data = lagged.dropna(subset=[target_column]).dropna().copy()
    return data


def build_feature_matrix_and_target(
    data: pd.DataFrame,
    feature_columns: list[str],
    n_lags: int,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build X and y for a given lag count."""
    feature_names = build_lag_feature_columns(feature_columns, n_lags)
    X = pd.concat([data[[name for name in feature_names if name in data.columns]]], axis=1)
    y = data[target_column]
    return X, y


def _time_series_split_generator(X: pd.DataFrame, n_splits: int) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


def find_best_lags(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    max_lags: int,
    models: dict[str, Any],
    n_splits: int = 5,
) -> dict[str, dict[str, float | int]]:
    """Find the best lag count for each model using mean temporal accuracy."""
    splitter = _time_series_split_generator(data, n_splits)
    best_lags: dict[str, dict[str, float | int]] = {}

    for model_name, model in models.items():
        best_score = -np.inf
        best_num_lags = 1

        for num_lags in range(1, max_lags + 1):
            X, y = build_feature_matrix_and_target(data, feature_columns, num_lags, target_column)
            scores: list[float] = []

            for train_idx, test_idx in splitter.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    continue

                estimator = clone(model)
                estimator.fit(X_train, y_train)
                predictions = estimator.predict(X_test)
                scores.append(float(accuracy_score(y_test, predictions)))

            if not scores:
                continue

            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_num_lags = num_lags

        best_lags[model_name] = {"best_lags": best_num_lags, "best_score": best_score}

    return best_lags


def tune_models_with_best_lags(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    best_lags: dict[str, dict[str, float | int]],
    models_and_params: dict[str, dict[str, Any]],
    n_splits: int = 5,
    n_iter: int = 10,
) -> dict[str, dict[str, Any]]:
    """Tune model hyperparameters after lag selection."""
    splitter = _time_series_split_generator(data, n_splits)
    results: dict[str, dict[str, Any]] = {}

    for model_name, model_info in models_and_params.items():
        num_lags = int(best_lags[model_name]["best_lags"])
        X, y = build_feature_matrix_and_target(data, feature_columns, num_lags, target_column)
        search = RandomizedSearchCV(
            estimator=model_info["model"],
            param_distributions=model_info["params"],
            n_iter=n_iter,
            scoring="f1_weighted",
            cv=splitter,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X, y)
        results[model_name] = {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "best_lags": num_lags,
        }

    return results


def tune_models_with_cv(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    max_lags: int,
    models_and_params: dict[str, dict[str, Any]],
    n_splits: int = 5,
    n_iter: int = 10,
) -> dict[str, dict[str, Any]]:
    """Jointly search lag count and hyperparameters for each model."""
    splitter = _time_series_split_generator(data, n_splits)
    results: dict[str, dict[str, Any]] = {}

    for num_lags in range(1, max_lags + 1):
        X, y = build_feature_matrix_and_target(data, feature_columns, num_lags, target_column)
        for model_name, model_info in models_and_params.items():
            if model_name not in results:
                results[model_name] = {"best_score": -np.inf, "best_params": None, "best_lags": None}

            search = RandomizedSearchCV(
                estimator=model_info["model"],
                param_distributions=model_info["params"],
                n_iter=n_iter,
                scoring="f1_weighted",
                cv=splitter,
                random_state=42,
                n_jobs=-1,
            )
            search.fit(X, y)

            if float(search.best_score_) > float(results[model_name]["best_score"]):
                results[model_name] = {
                    "best_params": search.best_params_,
                    "best_score": float(search.best_score_),
                    "best_lags": num_lags,
                }

    return results


def tune_binary_lags_with_cv(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    max_lags: int,
    model: Any,
    n_splits: int = 5,
) -> tuple[int, float, list[float]]:
    """Tune lag count for binary Direction_t+n targets using mean CV accuracy."""
    splitter = _time_series_split_generator(data, n_splits)
    accuracies_by_lag: dict[int, list[float]] = {lag: [] for lag in range(1, max_lags + 1)}

    for lag in range(1, max_lags + 1):
        X, y = build_feature_matrix_and_target(data, feature_columns, lag, target_column)
        for train_idx, test_idx in splitter.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue

            estimator = clone(model)
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_test)
            accuracies_by_lag[lag].append(float(accuracy_score(y_test, predictions)))

    mean_accuracies = [float(np.mean(accuracies_by_lag[lag])) if accuracies_by_lag[lag] else np.nan for lag in range(1, max_lags + 1)]
    best_lag = int(np.nanargmax(mean_accuracies) + 1)
    best_accuracy = float(np.nanmax(mean_accuracies))
    return best_lag, best_accuracy, mean_accuracies


def evaluate_best_model_configs(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_configs: dict[int, dict[str, dict[str, Any]]],
    best_model_name: str = "XGBoost",
    n_splits: int = 5,
) -> dict[int, dict[str, Any]]:
    """Evaluate a chosen model config per horizon and aggregate confusion matrices."""
    results: dict[int, dict[str, Any]] = {}

    for horizon, configs in target_configs.items():
        target_column = f"target_{horizon}d"
        config = configs[best_model_name]
        n_lags = int(config["best_lags"])
        X, y = build_feature_matrix_and_target(data, feature_columns, n_lags, target_column)
        splitter = _time_series_split_generator(X, n_splits)

        if best_model_name == "XGBoost":
            estimator_factory = XGBClassifier
        elif best_model_name == "SVM":
            estimator_factory = SVC
        elif best_model_name == "Logistic Regression":
            estimator_factory = LogisticRegression
        elif best_model_name == "Random Forest":
            estimator_factory = RandomForestClassifier
        else:
            raise ValueError(f"Unsupported model: {best_model_name}")

        all_scores: list[float] = []
        all_matrices: list[np.ndarray] = []

        for train_idx, test_idx in splitter.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue

            model = estimator_factory(random_state=42, **config["best_params"])
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            all_scores.append(float(accuracy_score(y_test, predictions)))
            all_matrices.append(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

        results[horizon] = {
            "model_name": best_model_name,
            "best_params": config["best_params"],
            "best_lags": n_lags,
            "mean_accuracy": float(np.mean(all_scores)) if all_scores else np.nan,
            "confusion_matrix": np.sum(all_matrices, axis=0) if all_matrices else np.zeros((3, 3), dtype=int),
        }

    return results


def build_classification_metrics_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    results_by_horizon: dict[int, dict[str, dict[str, Any]]],
    train_ratio: float = 0.7,
) -> pd.DataFrame:
    """Compute holdout metrics for each model and horizon using tuned configs."""
    rows: list[dict[str, Any]] = []

    for horizon, horizon_results in results_by_horizon.items():
        target_column = f"target_{horizon}d"
        for model_name, config in horizon_results.items():
            n_lags = int(config["best_lags"])
            X, y = build_feature_matrix_and_target(df, feature_columns, n_lags, target_column)
            data = pd.concat([X, y], axis=1).dropna()
            X = data.drop(columns=[target_column])
            y = data[target_column]

            split_index = int(len(X) * train_ratio)
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

            if model_name == "XGBoost":
                model = XGBClassifier(random_state=42, **config["best_params"])
            elif model_name == "SVM":
                model = SVC(random_state=42, **config["best_params"])
            elif model_name == "Logistic Regression":
                model = LogisticRegression(random_state=42, max_iter=1000, **config["best_params"])
            elif model_name == "Random Forest":
                model = RandomForestClassifier(random_state=42, **config["best_params"])
            else:
                continue

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rows.append(
                {
                    "horizon": horizon,
                    "model": model_name,
                    "accuracy": float(accuracy_score(y_test, predictions)),
                    "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
                    "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
                    "f1_score": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
                    "best_num_lags": n_lags,
                    "best_params": config["best_params"],
                }
            )

    return pd.DataFrame(rows)


@dataclass
class StrategySimulationResult:
    horizon: int
    accuracy: float
    total_return_strategy: float
    total_return_buy_hold: float
    curve_strategy: np.ndarray
    curve_buy_hold: np.ndarray


def simulate_strategy(
    df: pd.DataFrame,
    horizon: int,
    feature_columns: list[str],
    best_params_lags: dict[int, dict[str, dict[str, Any]]],
    train_ratio: float = 0.7,
) -> StrategySimulationResult:
    """Simulate the XGBoost trading strategy from the notebook."""
    config = best_params_lags[horizon]["XGBoost"]
    best_params = config["best_params"]
    best_lags = int(config["best_lags"])

    lagged_features = [f"{col}_t-{lag}" for col in feature_columns for lag in range(1, best_lags + 1)]
    df_lagged = create_lagged_features(df, feature_columns, best_lags).dropna()

    X = df_lagged[lagged_features + feature_columns]
    y = df_lagged[f"target_{horizon}d"]
    close_today = df_lagged["Close"]
    close_future = df_lagged["Close"].shift(-horizon)

    valid_idx = X.index.intersection(close_future.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    close_today = close_today.loc[valid_idx]
    close_future = close_future.loc[valid_idx]

    split_index = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    close_today_test = close_today.iloc[split_index:]
    close_future_test = close_future.iloc[split_index:]

    model = XGBClassifier(random_state=42, **best_params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    strategy_returns: list[float] = []
    buy_hold_returns: list[float] = []
    for idx, signal in enumerate(predictions):
        p_now = float(close_today_test.iloc[idx])
        p_future = float(close_future_test.iloc[idx])
        if signal == 2:
            strategy_return = (p_future / p_now) - 1
        elif signal == 0:
            strategy_return = (p_now / p_future) - 1
        else:
            strategy_return = 0.0
        strategy_returns.append(strategy_return)
        buy_hold_returns.append((p_future / p_now) - 1)

    curve_strategy = np.cumprod([1 + value for value in strategy_returns])
    curve_buy_hold = np.cumprod([1 + value for value in buy_hold_returns])
    return StrategySimulationResult(
        horizon=horizon,
        accuracy=float(accuracy_score(y_test, predictions)),
        total_return_strategy=float(curve_strategy[-1] - 1),
        total_return_buy_hold=float(curve_buy_hold[-1] - 1),
        curve_strategy=curve_strategy,
        curve_buy_hold=curve_buy_hold,
    )


def get_fixed_test_set(data: pd.DataFrame, test_size: float = 0.2) -> pd.Index:
    """Return the fixed test indices used by the binary output notebook block."""
    n_test = int(len(data) * test_size)
    return data.index[-n_test:]


def tune_lags_with_fixed_test(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    max_lags: int,
    test_indices: pd.Index,
    n_splits: int,
    model: Any,
) -> tuple[int, float, np.ndarray | None, np.ndarray | None]:
    """Tune lag count with a fixed holdout test set and rolling train windows."""
    accuracies: list[float] = []
    best_predictions: np.ndarray | None = None
    best_probabilities: np.ndarray | None = None

    for lag in range(1, max_lags + 1):
        X, y = build_feature_matrix_and_target(data, feature_columns, lag, target_column)
        fold_accuracies: list[float] = []
        splitter = _time_series_split_generator(data.loc[: test_indices[0]], n_splits)
        current_predictions: np.ndarray | None = None
        current_probabilities: np.ndarray | None = None

        for train_idx, _ in splitter.split(data.loc[: test_indices[0]]):
            train_data = data.iloc[train_idx]
            test_data = data.loc[test_indices]
            X_train, y_train = build_feature_matrix_and_target(train_data, feature_columns, lag, target_column)
            X_test, y_test = build_feature_matrix_and_target(test_data, feature_columns, lag, target_column)
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue

            estimator = clone(model)
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_test)
            probabilities = estimator.predict_proba(X_test)[:, 1] if hasattr(estimator, "predict_proba") else None
            fold_accuracies.append(float(accuracy_score(y_test, predictions)))
            current_predictions = np.asarray(predictions)
            current_probabilities = probabilities

        mean_accuracy = float(np.mean(fold_accuracies)) if fold_accuracies else np.nan
        accuracies.append(mean_accuracy)
        current_best = np.nanmax(accuracies)
        if np.isfinite(mean_accuracy) and mean_accuracy >= current_best:
            best_predictions = current_predictions
            best_probabilities = current_probabilities

    best_lag = int(np.nanargmax(accuracies) + 1)
    best_accuracy = float(np.nanmax(accuracies))
    return best_lag, best_accuracy, best_predictions, best_probabilities
