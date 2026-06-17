from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

libomp_path = Path("/opt/homebrew/opt/libomp/lib")
if libomp_path.exists() and "DYLD_LIBRARY_PATH" not in os.environ:
    os.environ["DYLD_LIBRARY_PATH"] = str(libomp_path)

import joblib  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .config import ensure_dir, load_config, setup_logging
from .features import FeatureEngineer
from .utils import set_global_seed, write_json

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


class IsolationForestFraudClassifier(BaseEstimator, ClassifierMixin):
    """Wrap IsolationForest scores so the serving layer can call predict_proba."""

    def __init__(self, model: Any | None = None, score_min: float | None = None, score_max: float | None = None):
        self.model = model
        self.score_min = score_min
        self.score_max = score_max

    def fit(self, X: Any, y: Any | None = None):
        from sklearn.ensemble import IsolationForest

        if self.model is None:
            self.model = IsolationForest(random_state=42)
        self.model.fit(X)
        scores = self.model.score_samples(X)
        self.score_min = float(np.nanmin(scores))
        self.score_max = float(np.nanmax(scores))
        if self.score_max <= self.score_min:
            self.score_max = self.score_min + 1.0
        return self

    def score_samples(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("IsolationForestFraudClassifier has not been fitted.")
        return self.model.score_samples(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        scores = self.score_samples(X)
        probability = (self.score_max - scores) / (self.score_max - self.score_min)
        probability = np.clip(probability, 0.0, 1.0)
        return np.column_stack([1.0 - probability, probability])

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


@dataclass
class ModelRunResult:
    model_name: str
    model: Any
    params: dict[str, Any]
    metrics: dict[str, float]
    predictions: pd.DataFrame
    threshold: float
    artifacts: dict[str, Path]


def _safe_smote(X: pd.DataFrame, y: pd.Series, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    class_counts = y.value_counts()
    minority_count = int(class_counts.min())
    if minority_count <= 1:
        raise ValueError("SMOTE requires at least two examples in the minority class after train/test split.")
    k_neighbors = min(5, max(1, minority_count - 1))
    sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, index=np.arange(len(y_resampled)))


def _split_features(df: pd.DataFrame, feature_names: list[str], target: str, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    training_cfg = config["training"]
    test_size = float(training_cfg.get("test_size", 0.2))
    stratify = training_cfg.get("stratify", True) and df[target].nunique() > 1
    stratify_column = df[target] if stratify else None
    X = df[feature_names]
    y = df[target].astype(int)
    metadata = df[["transaction_id", "user_id", "transaction_time", target]].copy()
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        metadata,
        test_size=test_size,
        random_state=int(config["project"].get("random_state", 42)),
        stratify=stratify_column,
    )
    return X_train, X_test, y_train, y_test, meta_test.reset_index(drop=True)


def _best_f1_threshold(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> tuple[float, np.ndarray]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_values = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
    if len(thresholds) == 0:
        return 0.5, (probabilities >= 0.5).astype(int)
    best_idx = int(np.nanargmax(f1_values[:-1]))
    threshold = float(thresholds[best_idx])
    return threshold, (probabilities >= threshold).astype(int)


def _predict_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.score_samples(X)
    normalized = (scores - np.nanmin(scores)) / (np.nanmax(scores) - np.nanmin(scores))
    return 1.0 - normalized


def _evaluate(model: Any, X_test: pd.DataFrame, y_test: pd.Series, meta_test: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame, float]:
    probabilities = _predict_probabilities(model, X_test)
    threshold, predicted = _best_f1_threshold(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    metrics = {
        "auc_roc": float(roc_auc_score(y_test, probabilities)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "pr_auc": float(auc(recall, precision)) if len(recall) > 1 else 0.0,
    }
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    metrics.update({"true_negatives": int(tn), "false_positives": int(fp), "false_negatives": int(fn), "true_positives": int(tp)})
    predictions = pd.DataFrame(
        {
            "transaction_id": meta_test["transaction_id"].values,
            "user_id": meta_test["user_id"].values,
            "transaction_time": meta_test["transaction_time"].values,
            "is_fraud": y_test.astype(int).values,
            "fraud_probability": probabilities,
            "predicted_label": predicted,
            "decision": np.where(predicted == 1, "fraud", "legitimate"),
            "threshold": threshold,
        }
    )
    return metrics, predictions, threshold


def _log_curves(metrics_dir: Path, y_true: pd.Series, probabilities: np.ndarray, prefix: str) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    if plt is None:
        return artifacts

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    pr_path = metrics_dir / f"{prefix}_precision_recall_curve.png"
    plt.figure(figsize=(6, 5))
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{prefix} Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()
    artifacts["precision_recall_curve"] = pr_path

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_path = metrics_dir / f"{prefix}_roc_curve.png"
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, probabilities):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    artifacts["roc_curve"] = roc_path

    curve_df = pd.DataFrame({"recall": recall, "precision": precision})
    curve_path = metrics_dir / f"{prefix}_precision_recall_curve.csv"
    curve_df.to_csv(curve_path, index=False)
    artifacts["precision_recall_curve_csv"] = curve_path
    return artifacts


def _log_importance(model: Any, feature_names: list[str], metrics_dir: Path, prefix: str) -> Path | None:
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    elif hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
        importance = np.asarray(model.model.feature_importances_)
    if importance is None:
        return None
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)
    path = metrics_dir / f"{prefix}_feature_importance.csv"
    importance_df.to_csv(path, index=False)
    if plt is not None:
        plt.figure(figsize=(8, 8))
        importance_df.head(25).iloc[::-1].plot.barh(x="feature", y="importance", legend=False)
        plt.title(f"{prefix} Top Feature Importances")
        plt.tight_layout()
        plt.savefig(metrics_dir / f"{prefix}_feature_importance.png")
        plt.close()
    return path


def _xgboost_objective(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series, seed: int):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 220),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "random_state": seed,
            "n_jobs": 2,
            "tree_method": "hist",
            "eval_metric": "logloss",
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_valid)[:, 1]
        return float(roc_auc_score(y_valid, probabilities))

    return objective


def _lightgbm_objective(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series, seed: int):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 220),
            "num_leaves": trial.suggest_int("num_leaves", 16, 96),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "random_state": seed,
            "n_jobs": 2,
            "verbosity": -1,
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_valid)[:, 1]
        return float(roc_auc_score(y_valid, probabilities))

    return objective


def _isolation_forest_objective(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series, seed: int):
    from sklearn.ensemble import IsolationForest

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 240),
            "max_samples": trial.suggest_categorical("max_samples", ["auto", 256, 512, 1024]),
            "contamination": trial.suggest_float("contamination", 0.0005, 0.01, log=True),
            "random_state": seed,
            "n_jobs": 2,
        }
        wrapper = IsolationForestFraudClassifier(IsolationForest(**params))
        wrapper.fit(X_train)
        probabilities = wrapper.predict_proba(X_valid)[:, 1]
        return float(roc_auc_score(y_valid, probabilities))

    return objective


def _tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    n_trials: int,
    seed: int,
) -> dict[str, Any]:
    if model_name == "xgboost":
        objective = _xgboost_objective(X_train, y_train, X_valid, y_valid, seed)
    elif model_name == "lightgbm":
        objective = _lightgbm_objective(X_train, y_train, X_valid, y_valid, seed)
    elif model_name == "isolation_forest":
        objective = _isolation_forest_objective(X_train, y_train, X_valid, y_valid, seed)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_value": float(study.best_value), "best_params": dict(study.best_params)}


def _build_model(model_name: str, params: dict[str, Any], seed: int) -> Any:
    params = dict(params)
    if model_name == "xgboost":
        scale_pos_weight = params.pop("scale_pos_weight", 1.0)
        base = {
            "random_state": seed,
            "n_jobs": 2,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight,
        }
        return XGBClassifier(**{**params, **base})
    if model_name == "lightgbm":
        base = {"random_state": seed, "n_jobs": 2, "verbosity": -1}
        return LGBMClassifier(**{**params, **base})
    if model_name == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        return IsolationForestFraudClassifier(IsolationForest(random_state=seed, **params))
    raise ValueError(f"Unsupported model: {model_name}")


def _default_params(model_name: str, y_train: pd.Series) -> dict[str, Any]:
    if model_name == "xgboost":
        fraud_rate = float((y_train == 1).mean())
        return {
            "n_estimators": 160,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 3,
            "reg_lambda": 3.0,
            "scale_pos_weight": (1.0 - fraud_rate) / max(fraud_rate, 1e-6),
        }
    if model_name == "lightgbm":
        fraud_rate = float((y_train == 1).mean())
        return {
            "n_estimators": 160,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_samples": 40,
            "reg_lambda": 3.0,
            "class_weight": "balanced",
        }
    if model_name == "isolation_forest":
        return {"n_estimators": 160, "max_samples": "auto", "contamination": 0.002}
    raise ValueError(model_name)


def _train_one_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    meta_test: pd.DataFrame,
    feature_names: list[str],
    config: dict[str, Any],
    smote_X_train: pd.DataFrame | None,
    smote_y_train: pd.Series | None,
) -> ModelRunResult:
    project_cfg = config["project"]
    training_cfg = config["training"]
    seed = int(project_cfg.get("random_state", 42))
    n_trials = int(training_cfg.get("n_trials", 20))
    tune = bool(training_cfg.get("tune", True))
    artifacts_dir = Path(project_cfg["artifacts_dir"])
    metrics_dir = ensure_dir(artifacts_dir / "metrics" / model_name)
    run_name = f"fraud_detection_{model_name}"

    X_fit = smote_X_train if smote_X_train is not None and model_name != "isolation_forest" else X_train
    y_fit = smote_y_train if smote_y_train is not None and model_name != "isolation_forest" else y_train
    X_tune_train, X_tune_valid, y_tune_train, y_tune_valid = train_test_split(
        X_fit,
        y_fit,
        test_size=0.2,
        random_state=seed,
        stratify=y_fit if y_fit.nunique() > 1 else None,
    )

    params = _default_params(model_name, y_fit)
    tuning_summary: dict[str, Any] = {}
    if tune and n_trials > 0:
        tuning_summary = _tune_model(model_name, X_tune_train, y_tune_train, X_tune_valid, y_tune_valid, n_trials, seed)
        params.update(tuning_summary.get("best_params", {}))

    model = _build_model(model_name, params, seed)
    model.fit(X_fit, y_fit)
    metrics, predictions, threshold = _evaluate(model, X_test, y_test, meta_test)
    metrics.update(tuning_summary)
    metrics["model_name"] = model_name
    metrics["n_train_rows"] = int(len(X_fit))
    metrics["n_test_rows"] = int(len(X_test))
    metrics["smote_applied"] = bool(smote_X_train is not None and model_name != "isolation_forest")
    metrics["classification_report"] = classification_report(y_test, predictions["predicted_label"], output_dict=True, zero_division=0)

    write_json(metrics_dir / "metrics.json", metrics)
    predictions.to_parquet(metrics_dir / "predictions.parquet", index=False)
    artifacts = _log_curves(metrics_dir, y_test, predictions["fraud_probability"].to_numpy(), model_name)
    importance_path = _log_importance(model, feature_names, metrics_dir, model_name)
    if importance_path is not None:
        artifacts["feature_importance"] = importance_path

    model_path = ensure_dir(Path(project_cfg["models_dir"]) / model_name) / "model.joblib"
    joblib.dump(model, model_path)
    artifacts["model_path"] = model_path

    mlflow.set_tracking_uri(project_cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(training_cfg.get("mlflow_experiment_name", "real-time-fraud-detection"))
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**params, **tuning_summary.get("best_params", {})})
        scalar_metrics = {key: value for key, value in metrics.items() if isinstance(value, (int, float, np.floating))}
        mlflow.log_metrics(scalar_metrics)
        for key, path in artifacts.items():
            if Path(path).is_file():
                mlflow.log_artifact(str(path), artifact_path=f"{model_name}_{key}")
        mlflow.log_artifact(str(metrics_dir / "metrics.json"), artifact_path=model_name)
        mlflow.log_artifact(str(model_path), artifact_path=model_name)

    return ModelRunResult(
        model_name=model_name,
        model=model,
        params=params,
        metrics=metrics,
        predictions=predictions,
        threshold=threshold,
        artifacts=artifacts,
    )


def train_models(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    project_cfg = config["project"]
    training_cfg = config["training"]
    feature_cfg = config["feature_engineering"]
    seed = int(project_cfg.get("random_state", 42))
    set_global_seed(seed)

    features_path = Path(feature_cfg.get("output_path", "data/processed/transactions_features.parquet"))
    feature_config_path = Path(feature_cfg.get("feature_config_path", Path(project_cfg["artifacts_dir"]) / "feature_config.joblib"))
    df = pd.read_parquet(features_path)
    feature_config = joblib.load(feature_config_path)
    engineer = FeatureEngineer.from_config(feature_config)
    feature_names = list(engineer.feature_names)
    target = str(training_cfg.get("target", "is_fraud"))

    X_train, X_test, y_train, y_test, meta_test = _split_features(df, feature_names, target, config)
    smote_X_train = None
    smote_y_train = None
    if bool(training_cfg.get("smote", True)):
        smote_X_train, smote_y_train = _safe_smote(X_train, y_train, int(training_cfg.get("smote_random_state", seed)))

    results: list[ModelRunResult] = []
    for model_name in training_cfg.get("models", ["xgboost", "lightgbm", "isolation_forest"]):
        result = _train_one_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            meta_test=meta_test,
            feature_names=feature_names,
            config=config,
            smote_X_train=smote_X_train,
            smote_y_train=smote_y_train,
        )
        results.append(result)

    best_result = max(results, key=lambda item: item.metrics.get("auc_roc", 0.0))
    best_model_path = ensure_dir(Path(project_cfg["models_dir"])) / "best_model.joblib"
    best_artifact = {
        "model_name": best_result.model_name,
        "model": best_result.model,
        "threshold": float(best_result.threshold),
        "feature_config": feature_config,
        "feature_names": feature_names,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    joblib.dump(best_artifact, best_model_path)

    all_metrics = {result.model_name: result.metrics for result in results}
    summary = {
        "best_model": best_result.model_name,
        "best_auc_roc": float(best_result.metrics.get("auc_roc", 0.0)),
        "best_f1": float(best_result.metrics.get("f1", 0.0)),
        "best_threshold": float(best_result.threshold),
        "metrics": all_metrics,
        "best_model_path": str(best_model_path),
        "best_artifact_path": str(best_model_path),
    }
    metrics_path = ensure_dir(Path(project_cfg["artifacts_dir"])) / "training_metrics.json"
    write_json(metrics_path, summary)

    combined_predictions = pd.concat([result.predictions.assign(model_name=result.model_name) for result in results], ignore_index=True)
    predictions_path = ensure_dir(Path(project_cfg["artifacts_dir"])) / "predictions.parquet"
    combined_predictions.to_parquet(predictions_path, index=False)

    print(
        f"Trained {len(results)} models. Best model: {best_result.model_name} "
        f"with AUC-ROC {best_result.metrics.get('auc_roc', 0.0):.4f} and F1 {best_result.metrics.get('f1', 0.0):.4f}."
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate fraud models.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    setup_logging()
    train_models(args.config)


if __name__ == "__main__":
    main()