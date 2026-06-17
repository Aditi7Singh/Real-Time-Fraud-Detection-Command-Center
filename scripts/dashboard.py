from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fraud_detection.utils import read_json
ARTIFACTS = Path(os.environ.get("FRAUD_ARTIFACTS_DIR", ROOT / "artifacts"))
POWERBI = Path(os.environ.get("FRAUD_POWERBI_DIR", ROOT / "monitoring" / "powerbi"))


def load_optional_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def predict_from_api(payload: dict) -> dict | None:
    try:
        response = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


st.set_page_config(page_title="Fraud Detection Command Center", layout="wide")
st.title("Real-Time Fraud Detection Command Center")
st.caption("Model performance, operational cost, drift indicators, and live scoring in one interface.")

metrics_path = ARTIFACTS / "training_metrics.json"
predictions_path = ARTIFACTS / "predictions.parquet"
metrics = read_json(metrics_path) if metrics_path.exists() else {}
predictions = load_optional_parquet(predictions_path)

sidebar = st.sidebar
sidebar.header("Controls")
api_enabled = sidebar.checkbox("Use live API for prediction", value=False)

if metrics:
    sidebar.success(f"Best model: {metrics.get('best_model', 'unknown')}")
    sidebar.metric("Best AUC-ROC", f"{metrics.get('best_auc_roc', 0):.3f}")
    sidebar.metric("Best F1", f"{metrics.get('best_f1', 0):.3f}")

col1, col2, col3, col4 = st.columns(4)
if not predictions.empty:
    col1.metric("Scored transactions", f"{len(predictions):,}")
    col2.metric("Observed fraud rate", f"{predictions['is_fraud'].mean():.2%}")
    col3.metric("Predicted fraud rate", f"{predictions['predicted_label'].mean():.2%}")
    col4.metric("False positives", f"{int(((predictions['predicted_label'] == 1) & (predictions['is_fraud'] == 0)).sum()):,}")
else:
    col1.metric("Scored transactions", "0")
    col2.metric("Observed fraud rate", "—")
    col3.metric("Predicted fraud rate", "—")
    col4.metric("False positives", "—")

st.subheader("Live transaction scoring")
with st.form("predict_form", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    user_id = c1.text_input("User ID", value="U000001")
    amount = c2.number_input("Amount", min_value=0.0, value=125.50)
    merchant_category = c3.selectbox(
        "Merchant category",
        ["grocery", "fuel", "restaurant", "travel", "electronics", "fashion", "entertainment", "healthcare", "online_marketplace", "subscription", "crypto_exchange", "jewelry", "wire_transfer", "gaming", "adult_entertainment"],
    )
    c4, c5, c6 = st.columns(3)
    location = c4.text_input("Location", value="US-NY")
    device_type = c5.selectbox("Device type", ["ios_phone", "android_phone", "desktop_web", "mobile_web", "pos_terminal", "atm", "new_device", "emulator"])
    transaction_time = c6.text_input("Transaction time", value="2024-01-02T10:00:00")
    submitted = st.form_submit_button("Score transaction")

if submitted:
    payload = {
        "user_id": user_id,
        "transaction_time": transaction_time,
        "amount": float(amount),
        "merchant_category": merchant_category,
        "location": location,
        "device_type": device_type,
    }
    if api_enabled:
        result = predict_from_api(payload)
        if result:
            st.success(f"Decision: **{result['decision']}**")
            st.metric("Fraud probability", f"{result['fraud_probability']:.4f}")
            st.json(result)
        else:
            st.warning("API is not reachable. Start it with `python scripts/serve.py`.")
    else:
        st.info("Enable the live API checkbox to score through FastAPI, or run the batch pipeline to refresh model artifacts.")

st.subheader("Model quality")
if not predictions.empty:
    model_quality = predictions.groupby("model_name").agg(
        rows=("transaction_id", "count"),
        fraud_rate=("is_fraud", "mean"),
        predicted_fraud_rate=("predicted_label", "mean"),
        avg_probability=("fraud_probability", "mean"),
    )
    st.dataframe(model_quality, use_container_width=True)
else:
    st.info("Run `python scripts/run_pipeline.py` to generate prediction metrics.")

st.subheader("Monitoring trends")
fraud_rate = load_optional_parquet(POWERBI / "fraud_rate_over_time.parquet")
false_positive_cost = load_optional_parquet(POWERBI / "false_positive_cost_tracker.parquet")
precision_recall = load_optional_parquet(POWERBI / "precision_recall_trends.parquet")
drift = load_optional_parquet(POWERBI / "model_drift.parquet")

chart1, chart2 = st.columns(2)
with chart1:
    if not fraud_rate.empty:
        st.line_chart(fraud_rate.set_index("time_bucket")[["fraud_rate", "predicted_fraud_rate"]])
    else:
        st.info("Fraud rate trend table is not available yet.")
with chart2:
    if not false_positive_cost.empty:
        st.bar_chart(false_positive_cost.set_index("day")["false_positive_cost"])
    else:
        st.info("False positive cost table is not available yet.")

quality1, quality2 = st.columns(2)
with quality1:
    if not precision_recall.empty:
        st.line_chart(precision_recall.set_index("day")[["precision", "recall", "f1"]])
    else:
        st.info("Precision/recall trend table is not available yet.")
with quality2:
    if not drift.empty:
        st.dataframe(drift.sort_values("psi", ascending=False).head(20), use_container_width=True)
    else:
        st.info("Drift table is not available yet.")
