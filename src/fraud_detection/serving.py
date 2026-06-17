from __future__ import annotations

import argparse
import threading
import uuid
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_config, setup_logging
from .features import FeatureEngineer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TransactionRequest(BaseModel):
    transaction_id: str | None = Field(default=None, description="Optional stable transaction identifier.")
    user_id: str = Field(..., description="Customer or account identifier.")
    transaction_time: str = Field(..., description="ISO-8601 transaction timestamp.")
    amount: float = Field(..., ge=0, description="Transaction amount in account currency.")
    merchant_category: str = Field(..., description="Merchant category code/name.")
    location: str = Field(..., description="Merchant or transaction location code.")
    device_type: str = Field(..., description="Device/channel type.")

    def to_frame(self) -> pd.DataFrame:
        payload = self.model_dump()
        payload["transaction_id"] = payload["transaction_id"] or f"REALTIME_{uuid.uuid4().hex}"
        return pd.DataFrame([payload])


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: str
    threshold: float
    model_name: str


class FraudPredictor:
    def __init__(self, model_path: str | Path, feature_config_path: str | Path, history_path: str | Path, max_history_rows: int = 250_000):
        self.model_path = Path(model_path)
        self.feature_config_path = Path(feature_config_path)
        self.history_path = Path(history_path)
        self.max_history_rows = max_history_rows
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        if not self.feature_config_path.exists():
            raise FileNotFoundError(f"Feature config not found: {self.feature_config_path}")

        artifact = joblib.load(self.model_path)
        if isinstance(artifact, dict) and "model" in artifact:
            self.model = artifact["model"]
            self.feature_config = artifact.get("feature_config")
            self.feature_names = list(artifact.get("feature_names", []))
            self.threshold = float(artifact.get("threshold", 0.5))
            self.model_name = str(artifact.get("model_name", "unknown"))
        else:
            self.model = artifact
            self.feature_config = joblib.load(self.feature_config_path)
            engineer = FeatureEngineer.from_config(self.feature_config)
            self.feature_names = list(engineer.feature_names)
            self.threshold = 0.5
            self.model_name = "unknown"

        self.feature_engineer = FeatureEngineer.from_config(self.feature_config)
        self.history = self._load_history()
        self.lock = threading.Lock()

    def _load_history(self) -> pd.DataFrame:
        if self.history_path.exists():
            history = pd.read_parquet(self.history_path)
            required = {"transaction_id", "user_id", "transaction_time", "amount", "merchant_category", "location", "device_type"}
            if required.issubset(history.columns):
                return history.sort_values("transaction_time").tail(self.max_history_rows)
        return pd.DataFrame(columns=["transaction_id", "user_id", "transaction_time", "amount", "merchant_category", "location", "device_type"])

    def predict(self, request: TransactionRequest) -> PredictionResponse:
        transaction = request.to_frame()
        transaction_id = str(transaction.loc[0, "transaction_id"])
        with self.lock:
            history = pd.concat([self.history, transaction], ignore_index=True)
            features = self.feature_engineer.transform(history)
            if set(self.feature_names).difference(features.columns):
                missing = sorted(set(self.feature_names).difference(features.columns))
                raise RuntimeError(f"Feature mismatch after engineering. Missing: {missing[:10]}")
            row_mask = features["transaction_id"].astype(str) == transaction_id
            if not row_mask.any():
                raise RuntimeError("Engineered feature row for the submitted transaction was not found.")
            X = features.loc[row_mask, self.feature_names].tail(1)
            probabilities = self.model.predict_proba(X)
            probability = float(probabilities[0, 1])
            predicted_label = int(probability >= self.threshold)
            decision = "fraud" if predicted_label == 1 else "legitimate"
            self.history = history.tail(self.max_history_rows)

        return PredictionResponse(
            transaction_id=str(transaction.loc[0, "transaction_id"]),
            fraud_probability=round(probability, 6),
            decision=decision,
            threshold=self.threshold,
            model_name=self.model_name,
        )


def create_app(config_path: str | Path = "config/config.yaml") -> FastAPI:
    config = load_config(config_path)
    project_cfg = config["project"]
    serving_cfg = config.get("serving", {})
    feature_cfg = config.get("feature_engineering", {})
    model_path = serving_cfg.get("model_path", Path(project_cfg["models_dir"]) / "best_model.joblib")
    feature_config_path = serving_cfg.get("feature_config_path", Path(project_cfg["artifacts_dir"]) / "feature_config.joblib")
    history_path = feature_cfg.get("history_path", Path(project_cfg["artifacts_dir"]) / "history.parquet")
    predictor = FraudPredictor(model_path, feature_config_path, history_path)

    app = FastAPI(title="Real-Time Fraud Detection API", version="0.1.0")
    app.state.predictor = predictor

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_name": predictor.model_name, "threshold": predictor.threshold}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: TransactionRequest) -> PredictionResponse:
        try:
            return predictor.predict(request)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve fraud scoring API.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    serving_cfg = config.get("serving", {})
    setup_logging()
    app = create_app(args.config)
    uvicorn.run(
        app,
        host=serving_cfg.get("host", "0.0.0.0"),
        port=int(serving_cfg.get("port", 8000)),
    )


if __name__ == "__main__":
    main()
