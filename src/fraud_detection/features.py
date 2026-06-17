from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import ensure_dir, ensure_parent, load_config, setup_logging

MISSING_CATEGORY = "__MISSING__"
ROLLING_WINDOWS = {"1h": "1h", "24h": "24h", "7d": "7D"}


@dataclass
class FeatureEngineer:
    categorical_columns: list[str]
    raw_columns: list[str]
    rolling_windows: dict[str, str]
    global_amount_mean: float = 0.0
    global_amount_median: float = 0.0
    global_amount_std: float = 1.0
    category_values: dict[str, list[str]] | None = None
    numeric_features: list[str] | None = None
    feature_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.category_values = self.category_values or {}
        self.numeric_features = self.numeric_features or build_numeric_feature_names(self.rolling_windows)
        self.feature_names = self.feature_names or []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = self._prepare_dataframe(df)
        self._fit_category_values(prepared)
        self._fit_global_stats(prepared)
        transformed = self._add_features(prepared)
        encoded = self._encode_categorical(transformed)
        self.feature_names = self.numeric_features + self._encoded_column_names(encoded)
        return encoded

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.category_values or not self.feature_names:
            raise RuntimeError("FeatureEngineer must be fitted with fit_transform before transform.")
        prepared = self._prepare_dataframe(df)
        transformed = self._add_features(prepared)
        encoded = self._encode_categorical(transformed)
        return encoded

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Cannot engineer features for an empty dataframe.")

        data = df.copy()
        required = {"transaction_id", "user_id", "transaction_time", "amount", "merchant_category", "location", "device_type"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Missing required raw columns: {sorted(missing)}")

        data["transaction_id"] = data["transaction_id"].astype(str)
        data["user_id"] = data["user_id"].astype(str)
        data["transaction_time"] = pd.to_datetime(data["transaction_time"], utc=False, format="mixed")
        data["amount"] = pd.to_numeric(data["amount"], errors="coerce").fillna(0.0).clip(lower=0.0)
        for column in self.categorical_columns:
            data[column] = data[column].astype("string").fillna(MISSING_CATEGORY).astype(str)

        if "is_fraud" in data.columns:
            data["is_fraud"] = pd.to_numeric(data["is_fraud"], errors="coerce").fillna(0).astype(np.int8)

        sort_columns = ["user_id", "transaction_time", "transaction_id"]
        return data.sort_values(sort_columns).reset_index(drop=True)

    def _fit_category_values(self, df: pd.DataFrame) -> None:
        self.category_values = {
            column: sorted(df[column].dropna().astype(str).unique().tolist()) + [MISSING_CATEGORY]
            for column in self.categorical_columns
        }

    def _fit_global_stats(self, df: pd.DataFrame) -> None:
        self.global_amount_mean = float(df["amount"].mean())
        self.global_amount_median = float(df["amount"].median())
        std = float(df["amount"].std())
        self.global_amount_std = std if std > 0 else 1.0

    @staticmethod
    def _group_expanding_stats(values: pd.Series, group_ids: pd.Series, name: str) -> pd.Series:
        return (
            values.groupby(group_ids, sort=False)
            .expanding(min_periods=1)
            .agg(name)
            .reset_index(level=0, drop=True)
            .sort_index()
        )

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data = self._add_temporal_features(data)
        data = self._add_global_features(data)
        data = self._add_user_profile_features(data)
        data = self._add_group_profile_features(data)
        data = self._add_rolling_features(data)
        data = self._add_risk_features(data)
        return data

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        time = data["transaction_time"]
        data["hour_of_day"] = time.dt.hour.astype(float)
        data["day_of_week"] = time.dt.dayofweek.astype(float)
        data["month_of_year"] = time.dt.month.astype(float)
        data["day_of_month"] = time.dt.day.astype(float)
        data["is_night"] = ((time.dt.hour < 6) | (time.dt.hour >= 22)).astype(float)
        data["is_weekend"] = (time.dt.dayofweek >= 5).astype(float)
        data["time_since_last_transaction_min"] = (
            data.groupby("user_id")["transaction_time"].diff().dt.total_seconds().div(60.0).fillna(10_080.0)
        )
        data["time_since_last_transaction_log"] = np.log1p(data["time_since_last_transaction_min"])
        return data

    def _add_global_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        amount = data["amount"].astype(float)
        data["amount_log1p"] = np.log1p(amount)
        data["amount_zscore_global"] = (amount - self.global_amount_mean) / self.global_amount_std
        data["amount_vs_global_median"] = np.where(
            self.global_amount_median > 0, amount / self.global_amount_median, 0.0
        )
        return data

    def _add_user_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        amount_before = data.groupby("user_id", sort=False)["amount"].shift(1)
        user_stats = pd.DataFrame(
            {
                "user_transaction_count": self._group_expanding_stats(amount_before, data["user_id"], "count").fillna(0.0),
                "user_total_spend": self._group_expanding_stats(amount_before, data["user_id"], "sum").fillna(0.0),
                "user_avg_spend": self._group_expanding_stats(amount_before, data["user_id"], "mean").fillna(0.0),
                "user_median_spend": self._group_expanding_stats(amount_before, data["user_id"], "median").fillna(0.0),
                "user_std_spend": self._group_expanding_stats(amount_before, data["user_id"], "std").fillna(0.0),
                "user_max_spend": self._group_expanding_stats(amount_before, data["user_id"], "max").fillna(0.0),
                "user_min_spend": self._group_expanding_stats(amount_before, data["user_id"], "min").fillna(0.0),
            },
            index=data.index,
        )
        user_stats["user_amount_range"] = user_stats["user_max_spend"] - user_stats["user_min_spend"]

        for column, output in [
            ("merchant_category", "user_unique_merchant_categories"),
            ("location", "user_unique_locations"),
            ("device_type", "user_unique_device_types"),
        ]:
            unique_count = pd.Series(0.0, index=data.index)
            for value in self.category_values.get(column, []):
                if value == MISSING_CATEGORY:
                    continue
                seen_before = data[column].eq(value).groupby(data["user_id"], sort=False).cumsum().shift(fill_value=0)
                unique_count = unique_count.add(seen_before.gt(0).astype(float))
            user_stats[output] = unique_count

        data = pd.concat([data, user_stats], axis=1)
        data["amount_vs_user_avg"] = np.where(data["user_avg_spend"] > 0, data["amount"] / data["user_avg_spend"], 0.0)
        data["amount_vs_user_median"] = np.where(data["user_median_spend"] > 0, data["amount"] / data["user_median_spend"], 0.0)
        return data

    def _add_group_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        for column in ["merchant_category", "location", "device_type"]:
            amount_before = df.groupby(column, sort=False)["amount"].shift(1)
            stats = pd.DataFrame(
                {
                    f"{column}_txn_count": self._group_expanding_stats(amount_before, df[column], "count").fillna(0.0),
                    f"{column}_avg_amount": self._group_expanding_stats(amount_before, df[column], "mean").fillna(0.0),
                    f"{column}_median_amount": self._group_expanding_stats(amount_before, df[column], "median").fillna(0.0),
                    f"{column}_std_amount": self._group_expanding_stats(amount_before, df[column], "std").fillna(0.0),
                },
                index=data.index,
            )
            data = pd.concat([data, stats], axis=1)
            data[f"amount_vs_{column}_avg"] = np.where(
                data[f"{column}_avg_amount"] > 0, data["amount"] / data[f"{column}_avg_amount"], 0.0
            )
            data[f"amount_vs_{column}_median"] = np.where(
                data[f"{column}_median_amount"] > 0, data["amount"] / data[f"{column}_median_amount"], 0.0
            )
        return data

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        grouped = data.groupby("user_id", sort=False)
        code_columns: dict[str, str] = {}
        for column, output_prefix in [
            ("merchant_category", "merchant"),
            ("location", "location"),
            ("device_type", "device"),
        ]:
            code_column = f"__{column}_code"
            data[code_column] = pd.factorize(data[column].astype(str))[0]
            code_columns[column] = code_column

        grouped = data.groupby("user_id", sort=False)
        for label, window in self.rolling_windows.items():
            rolling = grouped.rolling(window, on="transaction_time")
            data[f"txn_count_{label}"] = rolling["amount"].count().reset_index(level=0, drop=True).to_numpy()
            data[f"amount_sum_{label}"] = rolling["amount"].sum().reset_index(level=0, drop=True).to_numpy()
            data[f"amount_avg_{label}"] = rolling["amount"].mean().reset_index(level=0, drop=True).to_numpy()
            data[f"amount_std_{label}"] = rolling["amount"].std().reset_index(level=0, drop=True).to_numpy()

            for output_prefix, code_column in code_columns.items():
                data[f"{output_prefix}_nunique_{label}"] = (
                    grouped.rolling(window, on="transaction_time")[code_column]
                    .apply(lambda values: len(set(values)), raw=True)
                    .reset_index(level=0, drop=True)
                    .to_numpy()
                )

        data = data.drop(columns=list(code_columns.values()))
        grouped = data.groupby("user_id", sort=False)

        fill_columns = [column for column in data.columns if column.startswith("amount_std_")]
        data[fill_columns] = data[fill_columns].fillna(0.0)
        return data

    def _add_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["is_high_risk_category"] = data["merchant_category"].isin(
            ["crypto_exchange", "jewelry", "wire_transfer", "gaming", "adult_entertainment", "electronics", "travel"]
        ).astype(float)
        data["is_high_risk_location"] = data["location"].isin(["NG-LA", "RU-MOW", "AE-DU", "SG-01", "BR-SP"]).astype(float)
        data["is_rare_device"] = data["device_type"].isin(["new_device", "emulator"]).astype(float)
        data["is_new_user"] = (data["user_transaction_count"] <= 1).astype(float)
        data["is_new_merchant_category_for_user"] = (
            data.groupby("user_id")["merchant_category"].cumcount() == 0
        ).astype(float)
        data["is_new_location_for_user"] = (data.groupby("user_id")["location"].cumcount() == 0).astype(float)
        data["is_new_device_for_user"] = (data.groupby("user_id")["device_type"].cumcount() == 0).astype(float)
        data["high_velocity_1h"] = (data["txn_count_1h"] >= 3).astype(float)
        data["high_velocity_24h"] = (data["txn_count_24h"] >= 6).astype(float)
        data["high_amount_ratio"] = (data["amount_vs_user_avg"] >= 4.0).astype(float)
        return data

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.category_values is None:
            raise RuntimeError("Category values have not been fitted.")
        data = df.copy()
        encoded_parts = [data.drop(columns=self.categorical_columns, errors="ignore")]
        for column in self.categorical_columns:
            dummies = pd.get_dummies(
                data[column].fillna(MISSING_CATEGORY).astype(str),
                prefix=column,
                dtype=float,
            )
            dummy_columns = [f"{column}_{category}" for category in self.category_values[column]]
            for dummy_column in dummy_columns:
                if dummy_column not in dummies.columns:
                    dummies[dummy_column] = 0.0
            dummies = dummies[dummy_columns]
            encoded_parts.append(dummies)
        encoded = pd.concat(encoded_parts, axis=1)
        numeric_missing = [column for column in self.numeric_features or [] if column not in encoded.columns]
        for column in numeric_missing:
            encoded[column] = 0.0
        encoded = encoded.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return encoded

    def _encoded_column_names(self, df: pd.DataFrame) -> list[str]:
        excluded = set(self.raw_columns + ["is_fraud"] + self.numeric_features)
        return [column for column in df.columns if column not in excluded]

    def to_config(self) -> dict[str, Any]:
        return {
            "categorical_columns": self.categorical_columns,
            "raw_columns": self.raw_columns,
            "rolling_windows": self.rolling_windows,
            "global_amount_mean": self.global_amount_mean,
            "global_amount_median": self.global_amount_median,
            "global_amount_std": self.global_amount_std,
            "category_values": self.category_values or {},
            "numeric_features": self.numeric_features or build_numeric_feature_names(self.rolling_windows),
            "feature_names": self.feature_names or [],
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FeatureEngineer":
        return cls(
            categorical_columns=list(config["categorical_columns"]),
            raw_columns=list(config["raw_columns"]),
            rolling_windows=dict(config.get("rolling_windows", ROLLING_WINDOWS)),
            global_amount_mean=float(config.get("global_amount_mean", 0.0)),
            global_amount_median=float(config.get("global_amount_median", 0.0)),
            global_amount_std=float(config.get("global_amount_std", 1.0)),
            category_values={key: list(value) for key, value in config.get("category_values", {}).items()},
            numeric_features=list(config.get("numeric_features", build_numeric_feature_names(config.get("rolling_windows", ROLLING_WINDOWS)))),
            feature_names=list(config.get("feature_names", [])),
        )


def build_numeric_feature_names(windows: dict[str, str] | None = None) -> list[str]:
    windows = windows or ROLLING_WINDOWS
    names = [
        "hour_of_day",
        "day_of_week",
        "month_of_year",
        "day_of_month",
        "is_night",
        "is_weekend",
        "time_since_last_transaction_min",
        "time_since_last_transaction_log",
        "amount_log1p",
        "amount_zscore_global",
        "amount_vs_global_median",
        "user_transaction_count",
        "user_total_spend",
        "user_avg_spend",
        "user_median_spend",
        "user_std_spend",
        "user_max_spend",
        "user_min_spend",
        "user_amount_range",
        "user_unique_merchant_categories",
        "user_unique_locations",
        "user_unique_device_types",
        "amount_vs_user_avg",
        "amount_vs_user_median",
        "merchant_category_txn_count",
        "merchant_category_avg_amount",
        "merchant_category_median_amount",
        "merchant_category_std_amount",
        "amount_vs_merchant_category_avg",
        "amount_vs_merchant_category_median",
        "location_txn_count",
        "location_avg_amount",
        "location_median_amount",
        "location_std_amount",
        "amount_vs_location_avg",
        "amount_vs_location_median",
        "device_type_txn_count",
        "device_type_avg_amount",
        "device_type_median_amount",
        "device_type_std_amount",
        "amount_vs_device_type_avg",
        "amount_vs_device_type_median",
        "is_high_risk_category",
        "is_high_risk_location",
        "is_rare_device",
        "is_new_user",
        "is_new_merchant_category_for_user",
        "is_new_location_for_user",
        "is_new_device_for_user",
        "high_velocity_1h",
        "high_velocity_24h",
        "high_amount_ratio",
    ]
    for label in windows:
        names.extend(
            [
                f"txn_count_{label}",
                f"amount_sum_{label}",
                f"amount_avg_{label}",
                f"amount_std_{label}",
                f"merchant_nunique_{label}",
                f"location_nunique_{label}",
                f"device_nunique_{label}",
            ]
        )
    return names


def default_engineer() -> FeatureEngineer:
    return FeatureEngineer(
        categorical_columns=["merchant_category", "location", "device_type"],
        raw_columns=["transaction_id", "user_id", "transaction_time", "amount", "merchant_category", "location", "device_type", "is_fraud"],
        rolling_windows=ROLLING_WINDOWS,
    )


def engineer_features(config_path: str | Path = "config/config.yaml") -> tuple[pd.DataFrame, FeatureEngineer]:
    config = load_config(config_path)
    project_cfg = config["project"]
    feature_cfg = config["feature_engineering"]
    raw_path = Path(feature_cfg.get("input_path", "data/raw/transactions.csv"))
    output_path = Path(feature_cfg.get("output_path", "data/processed/transactions_features.parquet"))
    artifacts_dir = ensure_dir(project_cfg["artifacts_dir"])
    feature_config_path = Path(feature_cfg.get("feature_config_path", artifacts_dir / "feature_config.joblib"))
    history_path = Path(feature_cfg.get("history_path", artifacts_dir / "history.parquet"))

    raw = pd.read_csv(raw_path)
    engineer = default_engineer()
    features = engineer.fit_transform(raw)
    ensure_parent(output_path)
    features.to_parquet(output_path, index=False, compression=feature_cfg.get("parquet_compression", "snappy"))
    joblib.dump(engineer.to_config(), feature_config_path)
    raw.sort_values("transaction_time").to_parquet(history_path, index=False, compression="snappy")
    print(
        f"Engineered {len(features):,} rows and {len(engineer.feature_names):,} features to {output_path}. "
        f"Feature config saved to {feature_config_path}."
    )
    return features, engineer


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fraud detection features.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    setup_logging()
    engineer_features(args.config)


if __name__ == "__main__":
    main()
