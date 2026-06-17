from __future__ import annotations

import argparse

from .config import setup_logging
from .features import engineer_features
from .ingestion import ingest
from .monitoring import generate_monitoring_tables
from .training import train_models


def run_pipeline(config_path: str = "config/config.yaml", skip_ingestion: bool = False, skip_monitoring: bool = False) -> None:
    if not skip_ingestion:
        ingest(config_path, overwrite=True)
    engineer_features(config_path)
    train_models(config_path)
    if not skip_monitoring:
        generate_monitoring_tables(config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end fraud detection pipeline.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-ingestion", action="store_true", help="Reuse existing raw transactions.csv.")
    parser.add_argument("--skip-monitoring", action="store_true", help="Skip Power BI table generation.")
    args = parser.parse_args()
    setup_logging()
    run_pipeline(args.config, skip_ingestion=args.skip_ingestion, skip_monitoring=args.skip_monitoring)


if __name__ == "__main__":
    main()
