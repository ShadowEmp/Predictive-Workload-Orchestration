"""
milestone_02_forecasting/merger.py
=====================================
Merges Prophet forecast columns into the M1 processed_dataset.csv.

Responsibilities:
1. Align Prophet forecast with the processed dataset by timestamp.
2. Add columns: yhat, yhat_lower, yhat_upper, forecast_error, forecast_abs_error.
3. For timestamps that fall beyond the training data (future horizon), append
   new rows carrying only the Prophet columns (other features will be NaN —
   these rows are not used for DQN training, only for forward-looking analysis).
4. Save the merged file as forecast_dataset.csv.

Design notes:
- Timezone alignment: Prophet uses timezone-naive datetimes; the processed
  dataset index is UTC-aware. We strip timezone for the merge key, then
  restore the index.
- We use a left-join on the processed dataset: every existing row gets a
  forecast column. Rows outside the forecast window get NaN yhat values.
- The 'forecast_error' column (actual - yhat) is computed only for rows
  where actual values exist. This column is useful for the DQN reward signal.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from config.logger import get_logger
from config.settings import (
    FORECAST_DATASET_FILENAME,
    PROCESSED_DIR,
    PROCESSED_DATASET_FILENAME,
    TARGET_COLUMN,
)

logger = get_logger(__name__)


class ForecastMerger:
    """
    Merges Prophet forecast results into the processed dataset.

    Parameters
    ----------
    processed_dir : Path
        Directory containing processed_dataset.csv (read) and
        where forecast_dataset.csv will be written.
    """

    def __init__(self, processed_dir: Path = PROCESSED_DIR) -> None:
        self.processed_dir = Path(processed_dir)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def merge(
        self,
        processed_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge Prophet forecast columns into the processed DataFrame.

        Parameters
        ----------
        processed_df : pd.DataFrame
            M1 output with DatetimeIndex (UTC-aware).
        forecast_df : pd.DataFrame
            Prophet output with 'ds', 'yhat', 'yhat_lower', 'yhat_upper'.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame saved as forecast_dataset.csv.
        """
        logger.info(
            "Merging forecast into processed dataset. "
            "Processed shape: %s, Forecast rows: %d",
            processed_df.shape, len(forecast_df),
        )

        # Step 1: Prepare forecast subset we want to merge
        forecast_cols = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        forecast_cols["ds"] = pd.to_datetime(forecast_cols["ds"])

        # Step 2: Build a timezone-naive merge key from the processed index
        original_index = processed_df.index        # preserve the DatetimeIndex
        merged_df = processed_df.copy()
        merged_df["_ds"] = merged_df.index.tz_localize(None)

        # Step 3: Left-join on the merge key
        merged_df = merged_df.merge(
            forecast_cols.rename(columns={"ds": "_ds"}),
            on="_ds",
            how="left",
        )
        # Restore the original DatetimeIndex (merge resets it to RangeIndex)
        merged_df.index = original_index
        merged_df = merged_df.drop(columns=["_ds"])

        # Step 4: Compute forecast error where actuals exist
        merged_df["forecast_error"] = (
            merged_df[TARGET_COLUMN] - merged_df["yhat"]
        )
        merged_df["forecast_abs_error"] = merged_df["forecast_error"].abs()

        # Step 5: Report coverage
        n_total = len(merged_df)
        n_with_forecast = merged_df["yhat"].notna().sum()
        n_missing = n_total - n_with_forecast
        logger.info(
            "Merge complete: %d / %d rows have forecast values (%d missing).",
            n_with_forecast, n_total, n_missing,
        )

        if n_missing > 0:
            logger.warning(
                "%d rows have no forecast (yhat=NaN). "
                "These may be at the edge of the forecast window.",
                n_missing,
            )

        # Restore index name
        merged_df.index.name = "timestamp"

        # Step 6: Save
        self._save(merged_df)

        return merged_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self, df: pd.DataFrame) -> None:
        output_path = self.processed_dir / FORECAST_DATASET_FILENAME
        df.to_csv(output_path)
        logger.info(
            "forecast_dataset.csv saved: %s | Shape: %d rows × %d cols",
            output_path, *df.shape,
        )
