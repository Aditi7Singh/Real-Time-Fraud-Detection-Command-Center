from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ensure_parent, load_config, setup_logging
from .utils import set_global_seed

MERCHANT_CATEGORIES = np.array(
    [
        "grocery",
        "fuel",
        "restaurant",
        "travel",
        "electronics",
        "fashion",
        "entertainment",
        "healthcare",
        "online_marketplace",
        "subscription",
        "crypto_exchange",
        "jewelry",
        "wire_transfer",
        "gaming",
        "adult_entertainment",
    ]
)
LOCATIONS = np.array(
    [
        "US-NY",
        "US-CA",
        "US-TX",
        "US-FL",
        "US-IL",
        "GB-LND",
        "CA-ON",
        "DE-BE",
        "FR-IDF",
        "IN-MH",
        "SG-01",
        "AE-DU",
        "BR-SP",
        "NG-LA",
        "RU-MOW",
    ]
)
DEVICE_TYPES = np.array(["ios_phone", "android_phone", "desktop_web", "mobile_web", "pos_terminal", "atm", "new_device", "emulator"])
HIGH_RISK_CATEGORIES = np.array(["crypto_exchange", "jewelry", "wire_transfer", "gaming", "adult_entertainment", "electronics", "travel"])
HIGH_RISK_LOCATIONS = np.array(["NG-LA", "RU-MOW", "AE-DU", "SG-01", "BR-SP"])
RARE_DEVICES = np.array(["new_device", "emulator"])


def _write_csv_in_chunks(df: pd.DataFrame, output_path: str | Path, chunk_size: int) -> None:
    output = ensure_parent(output_path)
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        chunk.to_csv(output, mode="w" if start == 0 else "a", header=start == 0, index=False)


def generate_synthetic_transactions(config: dict, overwrite: bool = True) -> pd.DataFrame:
    ingestion_cfg = config["ingestion"]
    project_cfg = config["project"]
    n_rows = int(ingestion_cfg.get("synthetic_n_rows", 1_000_000))
    fraud_rate = float(ingestion_cfg.get("fraud_rate", 0.0017))
    chunk_size = int(ingestion_cfg.get("chunk_size", 100_000))
    output_path = Path(ingestion_cfg.get("output_path", Path(project_cfg["processed_dir"]) / "transactions.csv"))
    seed = int(project_cfg.get("random_state", 42))
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    if output_path.exists() and not overwrite:
        return pd.read_csv(output_path)

    fraud_n = max(1, int(round(n_rows * fraud_rate)))
    normal_n = n_rows - fraud_n
    n_users = max(25_000, n_rows // 4)

    user_numbers = rng.integers(1, n_users + 1, size=n_rows)
    user_ids = np.array([f"U{number:06d}" for number in user_numbers], dtype=object)

    start_date = pd.Timestamp("2024-01-01T00:00:00")
    end_date = start_date + pd.Timedelta(days=180)
    user_base_dates = pd.date_range(start_date, end_date, periods=n_users, tz=None)
    base_for_user = user_base_dates[user_numbers - 1]
    interarrival_seconds = rng.exponential(scale=5.5 * 24 * 3600, size=n_rows)
    transaction_time = pd.Series(pd.to_datetime(base_for_user + pd.to_timedelta(interarrival_seconds, unit="s")).floor("s"))

    amount = np.clip(rng.lognormal(mean=3.35, sigma=0.85, size=n_rows), 0.50, 5_000).round(2)
    merchant_category = rng.choice(
        MERCHANT_CATEGORIES,
        size=n_rows,
        p=np.array([0.20, 0.14, 0.13, 0.06, 0.08, 0.08, 0.06, 0.07, 0.10, 0.05, 0.006, 0.006, 0.006, 0.006, 0.006]),
    )
    location = rng.choice(
        LOCATIONS,
        size=n_rows,
        p=np.array([0.18, 0.16, 0.13, 0.10, 0.07, 0.06, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025, 0.02, 0.015, 0.015]),
    )
    device_type = rng.choice(DEVICE_TYPES, size=n_rows, p=np.array([0.24, 0.28, 0.18, 0.14, 0.08, 0.05, 0.015, 0.015]))

    is_fraud = np.zeros(n_rows, dtype=np.int8)
    fraud_idx = rng.choice(n_rows, size=fraud_n, replace=False)
    is_fraud[fraud_idx] = 1

    amount[fraud_idx] = np.clip(rng.lognormal(mean=5.05, sigma=0.65, size=fraud_n), 75.00, 25_000).round(2)
    category_mix = rng.random(fraud_n)
    merchant_category[fraud_idx] = np.where(
        category_mix < 0.72,
        rng.choice(HIGH_RISK_CATEGORIES, size=fraud_n),
        merchant_category[fraud_idx],
    )
    location_mix = rng.random(fraud_n)
    location[fraud_idx] = np.where(location_mix < 0.35, rng.choice(HIGH_RISK_LOCATIONS, size=fraud_n), location[fraud_idx])
    device_mix = rng.random(fraud_n)
    device_type[fraud_idx] = np.where(device_mix < 0.28, rng.choice(RARE_DEVICES, size=fraud_n), device_type[fraud_idx])

    new_user_mask = rng.random(fraud_n) < 0.08
    if new_user_mask.any():
        user_ids[fraud_idx[new_user_mask]] = np.array(
            [f"U{n_users + offset + 1:06d}" for offset in range(int(new_user_mask.sum()))],
            dtype=object,
        )

    fraud_hours = rng.choice(np.array([0, 1, 2, 3, 4, 5, 22, 23]), size=fraud_n)
    fraud_minutes = rng.integers(0, 60, size=fraud_n)
    fraud_dates = pd.to_datetime(transaction_time[fraud_idx]).dt.floor("D")
    transaction_time[fraud_idx] = fraud_dates + pd.to_timedelta(fraud_hours, unit="h") + pd.to_timedelta(fraud_minutes, unit="m")

    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i + 1:09d}" for i in range(n_rows)],
            "user_id": user_ids,
            "transaction_time": transaction_time,
            "amount": amount,
            "merchant_category": merchant_category,
            "location": location,
            "device_type": device_type,
            "is_fraud": is_fraud,
        }
    )
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    _write_csv_in_chunks(df, output_path, chunk_size)
    return df


def load_kaggle_creditcard(path: str | Path, output_path: str | Path, chunk_size: int = 100_000) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Time", "Amount", "Class"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Kaggle creditcard CSV must contain columns: {sorted(missing)}")

    start = pd.Timestamp("2024-01-01T00:00:00")
    normalized = pd.DataFrame(
        {
            "transaction_id": [f"KAGGLE{i + 1:09d}" for i in range(len(df))],
            "user_id": [f"U{(int(t) // 3600) + 1:06d}" for t in df["Time"].astype(int)],
            "transaction_time": start + pd.to_timedelta(df["Time"].astype(float), unit="s"),
            "amount": df["Amount"].astype(float).round(2),
            "merchant_category": "card_present",
            "location": "unknown",
            "device_type": "card_present",
            "is_fraud": df["Class"].astype(np.int8),
        }
    )
    _write_csv_in_chunks(normalized, output_path, chunk_size)
    return normalized


def ingest(config_path: str | Path = "config/config.yaml", overwrite: bool = True) -> pd.DataFrame:
    config = load_config(config_path)
    ingestion_cfg = config["ingestion"]
    source = str(ingestion_cfg.get("source", "synthetic")).lower()
    output_path = Path(ingestion_cfg.get("output_path", "data/raw/transactions.csv"))

    if source == "synthetic":
        df = generate_synthetic_transactions(config, overwrite=overwrite)
    elif source == "kaggle":
        kaggle_path = Path(ingestion_cfg.get("kaggle_path", "data/raw/creditcard.csv"))
        df = load_kaggle_creditcard(kaggle_path, output_path, int(ingestion_cfg.get("chunk_size", 100_000)))
    elif source == "file":
        file_path = Path(ingestion_cfg.get("file_path"))
        df = pd.read_csv(file_path)
        _write_csv_in_chunks(df, output_path, int(ingestion_cfg.get("chunk_size", 100_000)))
    else:
        raise ValueError(f"Unsupported ingestion source: {source}")

    fraud_rate = float(df["is_fraud"].mean()) if "is_fraud" in df else np.nan
    print(f"Ingested {len(df):,} transactions to {output_path} with fraud rate {fraud_rate:.4%}.")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw fraud transactions.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-overwrite", action="store_true", help="Reuse an existing raw CSV when present.")
    args = parser.parse_args()
    setup_logging()
    ingest(args.config, overwrite=not args.no_overwrite)


if __name__ == "__main__":
    main()
