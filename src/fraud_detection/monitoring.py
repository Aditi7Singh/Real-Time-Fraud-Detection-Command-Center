from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_dir, load_config, setup_logging
from .utils import write_json


def _precision_recall_f1(group: pd.DataFrame) -> pd.Series:
    tp = int(((group["predicted_label"] == 1) & (group["is_fraud"] == 1)).sum())
    fp = int(((group["predicted_label"] == 1) & (group["is_fraud"] == 0)).sum())
    fn = int(((group["predicted_label"] == 0) & (group["is_fraud"] == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return pd.Series({"precision": precision, "recall": recall, "f1": f1, "false_positives": fp, "false_negatives": fn})


def build_fraud_rate_over_time(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    data["transaction_time"] = pd.to_datetime(data["transaction_time"], format="mixed")
    data["time_bucket"] = data["transaction_time"].dt.floor("h")
    return (
        data.groupby(["time_bucket", "model_name"], as_index=False)
        .agg(
            transactions=("transaction_id", "count"),
            frauds=("is_fraud", "sum"),
            predicted_frauds=("predicted_label", "sum"),
            avg_fraud_probability=("fraud_probability", "mean"),
        )
        .assign(fraud_rate=lambda d: d["frauds"] / d["transactions"], predicted_fraud_rate=lambda d: d["predicted_frauds"] / d["transactions"])
    )


def build_false_positive_cost_tracker(predictions: pd.DataFrame, false_positive_cost: float) -> pd.DataFrame:
    data = predictions.copy()
    data["transaction_time"] = pd.to_datetime(data["transaction_time"], format="mixed")
    data["day"] = data["transaction_time"].dt.date
    data["false_positive"] = ((data["predicted_label"] == 1) & (data["is_fraud"] == 0)).astype(int)
    data["false_positive_cost"] = data["false_positive"] * false_positive_cost
    return (
        data.groupby(["day", "model_name"], as_index=False)
        .agg(
            transactions=("transaction_id", "count"),
            false_positives=("false_positive", "sum"),
            false_positive_cost=("false_positive_cost", "sum"),
            false_positive_rate=("false_positive", "mean"),
        )
    )


def build_precision_recall_trends(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    data["transaction_time"] = pd.to_datetime(data["transaction_time"], format="mixed")
    data["day"] = data["transaction_time"].dt.date
    rows = []
    for (day, model_name), group in data.groupby(["day", "model_name"], sort=False):
        row = _precision_recall_f1(group)
        row["day"] = day
        row["model_name"] = model_name
        rows.append(row)
    columns = ["day", "model_name", "precision", "recall", "f1", "false_positives", "false_negatives"]
    return pd.DataFrame(rows, columns=columns)


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.to_numeric(actual, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if expected.empty or actual.empty:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.nanquantile(expected, quantiles))
    if len(edges) < 3:
        return 0.0
    expected_counts = np.histogram(expected, bins=edges)[0] + 1e-6
    actual_counts = np.histogram(actual, bins=edges)[0] + 1e-6
    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def build_model_drift(features: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    feature_data = features.drop(columns=["transaction_time"], errors="ignore")
    data = feature_data.merge(
        predictions[["transaction_id", "model_name", "transaction_time"]],
        on="transaction_id",
        how="inner",
    )
    data["transaction_time"] = pd.to_datetime(data["transaction_time"], format="mixed")
    cutoff = data["transaction_time"].quantile(0.8)
    baseline = data[data["transaction_time"] <= cutoff]
    latest = data[data["transaction_time"] > cutoff]
    exclude = {"transaction_id", "user_id", "transaction_time", "is_fraud", "merchant_category", "location", "device_type"}
    feature_columns = [column for column in features.columns if column not in exclude and pd.api.types.is_numeric_dtype(features[column])]
    rows = []
    for model_name in sorted(predictions["model_name"].unique()):
        latest_model = latest[latest["model_name"] == model_name]
        baseline_model = baseline[baseline["model_name"] == model_name] if "model_name" in baseline.columns else baseline
        for feature in feature_columns[:80]:
            if feature not in baseline_model or feature not in latest_model:
                continue
            baseline_mean = float(baseline_model[feature].mean())
            latest_mean = float(latest_model[feature].mean())
            baseline_std = float(baseline_model[feature].std() or 1.0)
            latest_std = float(latest_model[feature].std() or 1.0)
            rows.append(
                {
                    "model_name": model_name,
                    "feature": feature,
                    "baseline_mean": baseline_mean,
                    "latest_mean": latest_mean,
                    "mean_shift": latest_mean - baseline_mean,
                    "baseline_std": baseline_std,
                    "latest_std": latest_std,
                    "std_shift": latest_std - baseline_std,
                    "psi": _psi(baseline_model[feature], latest_model[feature]),
                }
            )
    return pd.DataFrame(rows)


def build_powerbi_assets(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    ensure_dir(output_dir)
    for name, table in tables.items():
        table.to_parquet(output_dir / f"{name}.parquet", index=False)

    model_json: dict[str, Any] = {
        "name": "Real-Time Fraud Detection Dashboard",
        "tables": [
            {
                "name": "FraudRateOverTime",
                "path": "fraud_rate_over_time.parquet",
                "measures": ["fraud_rate", "predicted_fraud_rate", "avg_fraud_probability"],
            },
            {
                "name": "FalsePositiveCostTracker",
                "path": "false_positive_cost_tracker.parquet",
                "measures": ["false_positive_cost", "false_positive_rate", "false_positives"],
            },
            {
                "name": "ModelDrift",
                "path": "model_drift.parquet",
                "measures": ["psi", "mean_shift", "std_shift"],
            },
            {
                "name": "PrecisionRecallTrends",
                "path": "precision_recall_trends.parquet",
                "measures": ["precision", "recall", "f1"],
            },
        ],
        "recommended_visuals": [
            "Line chart: fraud_rate over time by model_name",
            "Clustered column chart: false_positive_cost by day and model_name",
            "Heat map or table: top PSI drift indicators by feature",
            "Line chart: precision/recall trends by day and model_name",
        ],
    }
    write_json(output_dir / "powerbi_dashboard_model.json", model_json)

    report_json: dict[str, Any] = {
        "pages": [
            {
                "name": "Fraud Operations",
                "visuals": ["Fraud rate over time", "Predicted fraud rate", "Average fraud probability"],
            },
            {
                "name": "Cost Control",
                "visuals": ["False positive cost by day", "False positive rate", "False positive count"],
            },
            {
                "name": "Model Health",
                "visuals": ["PSI drift by feature", "Mean/std feature shifts", "Precision and recall trends"],
            },
        ]
    }
    write_json(output_dir / "powerbi_dashboard_report.json", report_json)


def generate_monitoring_tables(config_path: str | Path = "config/config.yaml") -> dict[str, Path]:
    config = load_config(config_path)
    project_cfg = config["project"]
    monitoring_cfg = config.get("monitoring", {})
    predictions_path = Path(monitoring_cfg.get("predictions_path", Path(project_cfg["artifacts_dir"]) / "predictions.parquet"))
    output_dir = ensure_dir(monitoring_cfg.get("output_dir", Path(project_cfg["artifacts_dir"]) / "powerbi"))
    false_positive_cost = float(monitoring_cfg.get("false_positive_cost", 25.0))

    predictions = pd.read_parquet(predictions_path)
    fraud_rate = build_fraud_rate_over_time(predictions)
    false_positive_cost_tracker = build_false_positive_cost_tracker(predictions, false_positive_cost)
    precision_recall_trends = build_precision_recall_trends(predictions)

    feature_path = Path(monitoring_cfg.get("parquet_path", config["feature_engineering"].get("output_path")))
    features = pd.read_parquet(feature_path) if feature_path.exists() else pd.DataFrame()
    model_drift = build_model_drift(features, predictions) if not features.empty else pd.DataFrame()

    tables = {
        "fraud_rate_over_time": fraud_rate,
        "false_positive_cost_tracker": false_positive_cost_tracker,
        "model_drift": model_drift,
        "precision_recall_trends": precision_recall_trends,
    }
    build_powerbi_assets(output_dir, tables)
    paths = {name: output_dir / f"{name}.parquet" for name in tables}
    print(f"Power BI monitoring tables written to {output_dir}: {sorted(paths)}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Power BI monitoring tables.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    setup_logging()
    generate_monitoring_tables(args.config)


if __name__ == "__main__":
    main()
