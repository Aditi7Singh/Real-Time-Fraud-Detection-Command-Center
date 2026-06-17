from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config, setup_logging
from .utils import read_json


def summarize_metrics(metrics_path: str | Path) -> pd.DataFrame:
    metrics = read_json(metrics_path)
    rows = []
    for model_name, payload in metrics.get("metrics", {}).items():
        rows.append(
            {
                "model_name": model_name,
                "auc_roc": payload.get("auc_roc"),
                "pr_auc": payload.get("pr_auc"),
                "f1": payload.get("f1"),
                "precision": payload.get("precision"),
                "recall": payload.get("recall"),
                "false_positives": payload.get("false_positives"),
                "false_negatives": payload.get("false_negatives"),
                "true_positives": payload.get("true_positives"),
                "threshold": payload.get("threshold"),
            }
        )
    return pd.DataFrame(rows).sort_values("auc_roc", ascending=False)


def summarize_predictions(predictions_path: str | Path, false_positive_cost: float = 25.0) -> pd.DataFrame:
    predictions = pd.read_parquet(predictions_path)
    predictions["false_positive"] = ((predictions["predicted_label"] == 1) & (predictions["is_fraud"] == 0)).astype(int)
    predictions["false_negative"] = ((predictions["predicted_label"] == 0) & (predictions["is_fraud"] == 1)).astype(int)
    predictions["false_positive_cost"] = predictions["false_positive"] * false_positive_cost
    predictions["transaction_time"] = pd.to_datetime(predictions["transaction_time"], format="mixed")

    rows = []
    for model_name, group in predictions.groupby("model_name"):
        positives = group[group["predicted_label"] == 1]
        fraud = group[group["is_fraud"] == 1]
        rows.append(
            {
                "model_name": model_name,
                "rows": len(group),
                "fraud_rate": group["is_fraud"].mean(),
                "predicted_fraud_rate": group["predicted_label"].mean(),
                "precision": (positives["is_fraud"] == 1).mean() if len(positives) else 0.0,
                "recall": (fraud["predicted_label"] == 1).mean() if len(fraud) else 0.0,
                "false_positive_cost": group["false_positive_cost"].sum(),
            }
        )
    return pd.DataFrame(rows)


def evaluate(config_path: str | Path = "config/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    project_cfg = config["project"]
    metrics_path = Path(project_cfg["artifacts_dir"]) / "training_metrics.json"
    predictions_path = Path(project_cfg["artifacts_dir"]) / "predictions.parquet"
    false_positive_cost = float(config.get("monitoring", {}).get("false_positive_cost", 25.0))

    metrics_summary = summarize_metrics(metrics_path)
    prediction_summary = summarize_predictions(predictions_path, false_positive_cost=false_positive_cost)
    print("Model metrics:")
    print(metrics_summary.to_string(index=False))
    print("\nPrediction summary:")
    print(prediction_summary.to_string(index=False))
    return metrics_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fraud model evaluation artifacts.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    setup_logging()
    evaluate(args.config)


if __name__ == "__main__":
    main()
