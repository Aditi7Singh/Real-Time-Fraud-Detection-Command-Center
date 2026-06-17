# Real-Time Fraud Detection System

End-to-end fraud detection pipeline with synthetic/Kaggle ingestion, 40+ engineered features, SMOTE-balanced supervised training, XGBoost/LightGBM/Isolation Forest modeling, MLflow experiment tracking, FastAPI real-time scoring, Docker deployment, and Power BI monitoring assets.
<img width="1452" height="813" alt="Screenshot 2026-06-17 at 12 05 06 PM" src="https://github.com/user-attachments/assets/b083483e-314c-43de-9818-c5849a5e0fd4" />


## Stack

- Python, pandas, NumPy, scikit-learn
- XGBoost, LightGBM, Isolation Forest
- SMOTE via `imbalanced-learn`
- Optuna hyperparameter tuning
- MLflow experiment tracking
- FastAPI + Uvicorn serving
- Docker + Docker Compose
- Power BI Parquet dashboard inputs

## Project layout

- [`scripts/ingest.py`](scripts/ingest.py): generate synthetic transactions or normalize Kaggle/custom CSV input.
- [`scripts/engineer_features.py`](scripts/engineer_features.py): create 40+ behavioral, velocity, temporal, ratio, and risk features.
- [`scripts/train.py`](scripts/train.py): train/tune XGBoost, LightGBM, and Isolation Forest with SMOTE on the training split only.
- [`scripts/evaluate.py`](scripts/evaluate.py): summarize evaluation metrics and prediction artifacts.
- [`scripts/generate_monitoring.py`](scripts/generate_monitoring.py): export Power BI monitoring tables.
- [`scripts/serve.py`](scripts/serve.py): run FastAPI scoring service.
- [`scripts/run_pipeline.py`](scripts/run_pipeline.py): orchestrate the full batch pipeline.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

python scripts/run_pipeline.py
```

Optional beautiful Streamlit interface:

```bash
streamlit run scripts/dashboard.py
```

For a faster smoke test, temporarily set `ingestion.synthetic_n_rows` to a smaller value such as `50000` and `training.n_trials` to `3`.

## Ingestion

Synthetic data is the default and creates 1M transactions at a 0.17% fraud rate.

```bash
python scripts/ingest.py
```

To use the Kaggle Credit Card Fraud Detection dataset, place it at `data/raw/creditcard.csv` and set:

```yaml
ingestion:
  source: kaggle
  kaggle_path: data/raw/creditcard.csv
```

## Feature engineering

```bash
python scripts/engineer_features.py
```

Outputs:

- `data/processed/transactions_features.parquet`
- `artifacts/feature_config.joblib`
- `artifacts/history.parquet`

Feature families include:

- Behavioral: user spend count/sum/avg/median/std/range and uniqueness features.
- Velocity: transaction counts, amount sums/averages/std, and category/location/device uniqueness over 1h, 24h, and 7d.
- Temporal: hour, day of week, month, day, night/weekend flags, and time since last transaction.
- Ratio: amount versus user, merchant category, location, device type, and global baselines.
- Risk flags: high-risk category/location/device, new user/category/location/device, and high velocity.

## Training and evaluation

```bash
python scripts/train.py
python scripts/evaluate.py
```

Training behavior:

1. Split train/test before resampling.
2. Apply SMOTE only to the training split for XGBoost and LightGBM.
3. Tune hyperparameters with Optuna.
4. Track parameters, metrics, PR/ROC curves, feature importance, and model artifacts in MLflow.
5. Save the best model to `models/best_model.joblib`.

## Real-time scoring API

Start the service after training:

```bash
python scripts/serve.py
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "U000123",
    "transaction_time": "2026-06-16T10:15:00",
    "amount": 1250.75,
    "merchant_category": "electronics",
    "location": "US-NY",
    "device_type": "ios_phone"
  }'
```

Example response:

```json
{
  "transaction_id": "REALTIME_abc123",
  "fraud_probability": 0.913421,
  "decision": "fraud",
  "threshold": 0.5,
  "model_name": "xgboost"
}
```

## Docker

```bash
docker compose up --build
```

The API listens on `http://localhost:8000`.

## Power BI monitoring

After training:

```bash
python scripts/generate_monitoring.py
```

Then import the generated Parquet files from [`monitoring/powerbi`](monitoring/powerbi) into Power BI Desktop. Use [`monitoring/powerbi/powerbi_dashboard_report.json`](monitoring/powerbi/powerbi_dashboard_report.json) as the visual layout guide.

For a lightweight browser interface, run:

```bash
streamlit run scripts/dashboard.py
```

The Streamlit dashboard displays model KPIs, live scoring form, fraud-rate trends, false-positive cost, precision/recall trends, and top drift indicators.
