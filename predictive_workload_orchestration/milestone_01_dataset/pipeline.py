"""
milestone_01_dataset/pipeline.py
==================================
Milestone 1 orchestrator — ties together all M1 sub-modules.

Execution order:
  1. [Optional] Download dataset
  2. Load raw machine_usage.csv
  3. Clean data
  4. Aggregate to 5-min cluster-level time series
  5. Engineer features
  6. Run EDA and write reports
  7. Save processed_dataset.csv

Usage:
    # From project root:
    python -m milestone_01_dataset.pipeline

    # With download step:
    python -m milestone_01_dataset.pipeline --download

    # Limit rows (for testing):
    python -m milestone_01_dataset.pipeline --nrows 50000

    # Skip EDA (faster iteration):
    python -m milestone_01_dataset.pipeline --skip-eda
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to sys.path so imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logger import get_logger
from config.settings import PROCESSED_DIR, PROCESSED_DATASET_FILENAME
from milestone_01_dataset.aggregator import WorkloadAggregator
from milestone_01_dataset.cleaner import DataCleaner
from milestone_01_dataset.downloader import DatasetDownloader
from milestone_01_dataset.eda import EDAReport
from milestone_01_dataset.feature_engineer import FeatureEngineer
from milestone_01_dataset.loader import RawDataLoader

logger = get_logger(__name__)


class DatasetPipeline:
    """
    Orchestrates the complete Milestone 1 data preparation pipeline.

    Parameters
    ----------
    download : bool
        If True, download the dataset before processing.
    nrows : int, optional
        Limit the number of rows loaded (for development/testing).
    skip_eda : bool
        If True, skip the EDA report generation step.
    """

    def __init__(
        self,
        download: bool = False,
        nrows: int = None,
        skip_eda: bool = False,
    ) -> None:
        self.download = download
        self.nrows = nrows
        self.skip_eda = skip_eda

        # Instantiate all sub-modules
        self.downloader = DatasetDownloader()
        self.loader = RawDataLoader()
        self.cleaner = DataCleaner()
        self.aggregator = WorkloadAggregator()
        self.feature_engineer = FeatureEngineer()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the full pipeline and return the processed DataFrame.

        Returns
        -------
        pd.DataFrame
            The fully processed dataset, also saved to disk.
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("MILESTONE 1 — DATASET PREPARATION PIPELINE STARTED")
        logger.info("=" * 60)

        # Step 1: Download (optional)
        if self.download:
            logger.info("[1/6] Downloading dataset...")
            self.downloader.download_all()
        else:
            logger.info("[1/6] Skipping download (use --download to enable).")

        # Step 2: Load
        logger.info("[2/6] Loading raw data...")
        raw_df = self.loader.load_machine_usage(nrows=self.nrows)

        # Step 3: Clean
        logger.info("[3/6] Cleaning data...")
        cleaned_df = self.cleaner.clean(raw_df)

        # Step 4: Aggregate
        logger.info("[4/6] Aggregating to 5-minute cluster-level intervals...")
        aggregated_df = self.aggregator.aggregate(cleaned_df)

        # Step 5: Feature engineering
        logger.info("[5/6] Engineering features...")
        processed_df = self.feature_engineer.engineer(aggregated_df)

        # Step 6: EDA
        if not self.skip_eda:
            logger.info("[6/6] Generating EDA report and plots...")
            eda = EDAReport(processed_df)
            eda.run()
        else:
            logger.info("[6/6] EDA skipped.")

        # Step 7: Save
        output_path = PROCESSED_DIR / PROCESSED_DATASET_FILENAME
        processed_df.to_csv(output_path)
        logger.info("Processed dataset saved: %s", output_path)
        logger.info("  Shape: %s rows × %s columns", *processed_df.shape)

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("MILESTONE 1 COMPLETE in %.1f seconds.", elapsed)
        logger.info("=" * 60)

        return processed_df

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Run post-pipeline validation checks.

        Validates:
        - No missing values in numeric columns
        - Timestamps are continuous (no gaps)
        - Expected features are all present
        - Shape is reasonable (at least 1000 rows)

        Returns
        -------
        bool
            True if all checks pass.
        """
        logger.info("Running validation checks...")
        passed = True

        # Check 1: No missing values
        numeric_missing = df.select_dtypes(include=["number"]).isnull().sum().sum()
        if numeric_missing > 0:
            logger.error("FAIL — %d missing numeric values.", numeric_missing)
            passed = False
        else:
            logger.info("PASS — No missing numeric values.")

        # Check 2: Continuous timestamps
        if isinstance(df.index, pd.DatetimeIndex):
            diffs = pd.Series(df.index).diff().dropna()
            expected = pd.Timedelta("5min")
            irregular = (diffs != expected).sum()
            if irregular:
                logger.error("FAIL — %d irregular timestamp intervals.", irregular)
                passed = False
            else:
                logger.info("PASS — Timestamps are continuous.")

        # Check 3: Minimum rows
        if len(df) < 1000:
            logger.error("FAIL — Only %d rows; expected >= 1000.", len(df))
            passed = False
        else:
            logger.info("PASS — Row count: %d.", len(df))

        # Check 4: Required feature columns present
        required = [
            "cpu_util_percent",
            "mem_util_percent",
            "cpu_util_percent_roll_mean_12",
            "cpu_util_percent_lag_1",
            "hour_sin",
            "load_intensity",
        ]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            logger.error("FAIL — Missing required columns: %s", missing_cols)
            passed = False
        else:
            logger.info("PASS — All required feature columns present.")

        # Check 5: CPU values in [0, 100]
        cpu = df["cpu_util_percent"]
        if (cpu < 0).any() or (cpu > 100).any():
            logger.error("FAIL — cpu_util_percent contains out-of-range values.")
            passed = False
        else:
            logger.info("PASS — cpu_util_percent in [0, 100].")

        result = "ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"
        logger.info("Validation result: %s", result)
        return passed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 1: Dataset Preparation Pipeline"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset files before processing.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limit number of rows loaded (for testing).",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip EDA report generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    pipeline = DatasetPipeline(
        download=args.download,
        nrows=args.nrows,
        skip_eda=args.skip_eda,
    )
    df = pipeline.run()
    pipeline.validate(df)
