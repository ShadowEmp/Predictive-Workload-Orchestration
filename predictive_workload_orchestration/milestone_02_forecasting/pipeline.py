"""
milestone_02_forecasting/pipeline.py
======================================
Milestone 2 orchestrator — ties together trainer, evaluator, and merger.

Execution order:
  1. Load processed_dataset.csv (M1 output)
  2. Train Prophet model
  3. Evaluate forecast accuracy (hold-out + optional cross-validation)
  4. Merge forecast into processed_dataset → forecast_dataset.csv
  5. Log summary

Usage:
    # From project root:
    python3 -m milestone_02_forecasting.pipeline

    # Skip cross-validation (faster):
    python3 -m milestone_02_forecasting.pipeline --skip-cv

    # Specify custom processed data directory:
    python3 -m milestone_02_forecasting.pipeline --data-dir data/processed
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logger import get_logger
from config.settings import PROCESSED_DIR, PROCESSED_DATASET_FILENAME
from milestone_02_forecasting.evaluator import ForecastEvaluator
from milestone_02_forecasting.merger import ForecastMerger
from milestone_02_forecasting.trainer import ProphetTrainer

logger = get_logger(__name__)


class ForecastingPipeline:
    """
    Orchestrates the complete Milestone 2 forecasting pipeline.

    Parameters
    ----------
    processed_dir : Path
        Directory containing processed_dataset.csv from M1.
    run_cv : bool
        If True, run Prophet cross-validation during evaluation.
    """

    def __init__(
        self,
        processed_dir: Path = PROCESSED_DIR,
        run_cv: bool = True,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.run_cv = run_cv

        self.trainer = ProphetTrainer(processed_dir=self.processed_dir)
        self.merger = ForecastMerger(processed_dir=self.processed_dir)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the full M2 pipeline.

        Returns
        -------
        pd.DataFrame
            forecast_dataset.csv as a DataFrame.
        """
        start = time.time()
        logger.info("=" * 60)
        logger.info("MILESTONE 2 — WORKLOAD FORECASTING PIPELINE STARTED")
        logger.info("=" * 60)

        # Step 1: Train
        logger.info("[1/4] Training Prophet model...")
        model, forecast_df, test_df = self.trainer.run()

        # Step 2: Evaluate
        logger.info("[2/4] Evaluating forecast accuracy...")
        evaluator = ForecastEvaluator(
            forecast_df=forecast_df,
            test_df=test_df,
            model=model,
        )
        metrics = evaluator.run(run_cv=self.run_cv)

        # Step 3: Load processed dataset for merging
        logger.info("[3/4] Loading processed_dataset.csv for merge...")
        processed_df = self._load_processed(self.processed_dir)

        # Step 4: Merge and save forecast_dataset.csv
        logger.info("[4/4] Merging forecast into processed dataset...")
        merged_df = self.merger.merge(processed_df, forecast_df)

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info("MILESTONE 2 COMPLETE in %.1f seconds.", elapsed)
        logger.info(
            "  MAE : %.4f %%CPU | RMSE : %.4f %%CPU | MAPE : %.4f %%",
            metrics.get("mae", float("nan")),
            metrics.get("rmse", float("nan")),
            metrics.get("mape", float("nan")),
        )
        logger.info("=" * 60)

        return merged_df

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Post-pipeline validation checks.

        Validates:
        - forecast_dataset.csv has 'yhat' column
        - yhat values are in a physically plausible range [0, 200]
        - forecast_error column exists
        - No fully empty forecast section

        Returns
        -------
        bool
            True if all checks pass.
        """
        logger.info("Running M2 validation checks...")
        passed = True

        # Check 1: yhat column present
        if "yhat" not in df.columns:
            logger.error("FAIL — 'yhat' column missing from forecast_dataset.")
            passed = False
        else:
            logger.info("PASS — 'yhat' column present.")

        # Check 2: yhat in plausible range
        if "yhat" in df.columns:
            yhat_valid = df["yhat"].notna()
            if yhat_valid.sum() == 0:
                logger.error("FAIL — All yhat values are NaN.")
                passed = False
            else:
                out_of_range = (
                    (df.loc[yhat_valid, "yhat"] < -20) |
                    (df.loc[yhat_valid, "yhat"] > 200)
                ).sum()
                if out_of_range:
                    logger.warning(
                        "WARNING — %d yhat values outside [-20, 200]. "
                        "Model may be extrapolating beyond training data.",
                        out_of_range,
                    )
                else:
                    logger.info("PASS — yhat values in plausible range.")

        # Check 3: forecast_error column present
        if "forecast_error" not in df.columns:
            logger.error("FAIL — 'forecast_error' column missing.")
            passed = False
        else:
            logger.info("PASS — 'forecast_error' column present.")

        # Check 4: Coverage — at least 50% of rows should have a forecast
        if "yhat" in df.columns:
            coverage = df["yhat"].notna().mean()
            if coverage < 0.5:
                logger.error(
                    "FAIL — Forecast covers only %.1f%% of rows (expected >= 50%%).",
                    coverage * 100,
                )
                passed = False
            else:
                logger.info("PASS — Forecast coverage: %.1f%%.", coverage * 100)

        result = "ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"
        logger.info("M2 Validation result: %s", result)
        return passed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_processed(processed_dir: Path) -> pd.DataFrame:
        path = processed_dir / PROCESSED_DATASET_FILENAME
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 2: Workload Forecasting Pipeline"
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip Prophet cross-validation (faster run).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory containing processed_dataset.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    pipeline = ForecastingPipeline(
        processed_dir=args.data_dir,
        run_cv=not args.skip_cv,
    )
    df = pipeline.run()
    pipeline.validate(df)
