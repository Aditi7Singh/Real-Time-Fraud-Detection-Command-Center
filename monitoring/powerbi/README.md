# Power BI Fraud Monitoring Dashboard

Run the training pipeline first so `artifacts/predictions.parquet` and `data/processed/transactions_features.parquet` exist.

```bash
python scripts/generate_monitoring.py
```

This creates Parquet fact tables for Power BI:

- `fraud_rate_over_time.parquet`: hourly fraud rate, predicted fraud rate, and average fraud probability.
- `false_positive_cost_tracker.parquet`: daily false positives, false positive rate, and operational cost.
- `model_drift.parquet`: PSI, mean shift, and standard-deviation shift for monitored numeric features.
- `precision_recall_trends.parquet`: daily precision, recall, and F1 by model.

Recommended Power BI setup:

1. Open Power BI Desktop.
2. Use **Get Data → Folder** and point to this directory, or import each Parquet table.
3. Create relationships on `model_name` and time keys (`time_bucket` or `day`).
4. Use the JSON files as dashboard design references:
   - [`powerbi_dashboard_model.json`](powerbi_dashboard_model.json)
   - [`powerbi_dashboard_report.json`](powerbi_dashboard_report.json)

Suggested visuals:

- Line chart: live fraud rate over time by `model_name`.
- Clustered column chart: false positive cost by day.
- Table or heat map: top PSI drift indicators.
- Line chart: precision and recall trends by day.
