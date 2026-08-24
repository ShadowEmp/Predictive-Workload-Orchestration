"""
milestone_01_dataset/cleaner.py
=================================
Cleans raw machine_usage data before aggregation.

Responsibilities:
1. Remove duplicate (machine_id, timestamp) pairs.
2. Clip physically impossible utilization values (< 0 or > 100).
3. Log a missing-value report before and after filling.
4. Apply forward-fill then backward-fill to handle missing values.
5. Flag time windows with excessive consecutive missing values.

Design note:
- Cleaning operates on raw per-machine data BEFORE aggregation, so that
  per-machine anomalies don't corrupt the cluster-level signal.
"""

from typing import Tuple

import pandas as pd

from config.logger import get_logger
from config.settings import (
    AGGREGATE_COLUMNS,
    MAX_CONSECUTIVE_MISSING,
    MISSING_VALUE_STRATEGY,
)

logger = get_logger(__name__)

# Columns that must remain in [0, 100]
PERCENT_COLUMNS = ["cpu_util_percent", "mem_util_percent", "disk_io_percent"]

# Columns that must be non-negative
NONNEG_COLUMNS = ["mem_gps", "mkpi", "net_in", "net_out"]


class DataCleaner:
    """
    Cleans a raw machine_usage DataFrame.

    Parameters
    ----------
    strategy : str
        Missing-value strategy.  Currently only ``'ffill_bfill'`` is supported.
    max_consecutive_missing : int
        If a machine has more than this many consecutive NaN steps, log a
        warning (these are likely machine outages, not sensor dropouts).
    """

    def __init__(
        self,
        strategy: str = MISSING_VALUE_STRATEGY,
        max_consecutive_missing: int = MAX_CONSECUTIVE_MISSING,
    ) -> None:
        self.strategy = strategy
        self.max_consecutive = max_consecutive_missing

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full cleaning pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Output of RawDataLoader.load_machine_usage().

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame ready for aggregation.
        """
        logger.info("Starting data cleaning. Input shape: %s", df.shape)

        df = self._remove_duplicates(df)
        df = self._clip_physical_bounds(df)
        self._log_missing_report(df, stage="before fill")
        df = self._flag_excessive_missing(df)
        df = self._fill_missing(df)
        self._log_missing_report(df, stage="after fill")
        self._assert_no_missing(df)

        logger.info("Cleaning complete. Output shape: %s", df.shape)
        return df

    def get_missing_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame summarising missing values per column."""
        summary = pd.DataFrame({
            "missing_count": df.isnull().sum(),
            "missing_pct": (df.isnull().mean() * 100).round(2),
        })
        return summary[summary["missing_count"] > 0]

    # ------------------------------------------------------------------
    # Cleaning steps
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact duplicate (machine_id, timestamp) rows."""
        before = len(df)
        df = df.drop_duplicates(subset=["machine_id", "timestamp"])
        removed = before - len(df)
        if removed:
            logger.warning("Removed %d duplicate rows.", removed)
        else:
            logger.debug("No duplicate rows found.")
        return df

    @staticmethod
    def _clip_physical_bounds(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clip utilization percentages to [0, 100] and other signals to [0, ∞).

        Values outside these ranges are physically impossible and indicate
        sensor errors or data corruption.
        """
        df = df.copy()
        for col in PERCENT_COLUMNS:
            if col in df.columns:
                clipped = ((df[col] < 0) | (df[col] > 100)).sum()
                if clipped:
                    logger.warning("Clipping %d out-of-range values in '%s'.", clipped, col)
                df[col] = df[col].clip(lower=0.0, upper=100.0)

        for col in NONNEG_COLUMNS:
            if col in df.columns:
                neg = (df[col] < 0).sum()
                if neg:
                    logger.warning("Clipping %d negative values in '%s'.", neg, col)
                df[col] = df[col].clip(lower=0.0)

        return df

    def _flag_excessive_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Log machines with more than `max_consecutive` consecutive NaN rows.

        These likely indicate machine downtime, not sensor dropout. They are
        logged but not dropped — the aggregation step handles them by producing
        NaN windows that are then filled.
        """
        for col in AGGREGATE_COLUMNS:
            if col not in df.columns:
                continue
            # Group by machine_id and count max consecutive NaNs
            def max_consec_nan(series: pd.Series) -> int:
                null_mask = series.isnull()
                # cumsum trick: count consecutive True values
                groups = null_mask.ne(null_mask.shift()).cumsum()
                return null_mask.groupby(groups).sum().max()

            per_machine = (
                df.groupby("machine_id")[col]
                .apply(max_consec_nan)
                .rename("max_consec_nan")
            )
            flagged = per_machine[per_machine > self.max_consecutive]
            if not flagged.empty:
                logger.warning(
                    "Column '%s': %d machines have >%d consecutive NaN steps "
                    "(possible outages): %s",
                    col,
                    len(flagged),
                    self.max_consecutive,
                    flagged.index.tolist()[:5],  # show first 5
                )
        return df

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured missing-value fill strategy."""
        if self.strategy != "ffill_bfill":
            raise NotImplementedError(
                f"Missing value strategy '{self.strategy}' is not implemented. "
                "Only 'ffill_bfill' is currently supported."
            )

        df = df.copy()
        # Fill within each machine's time series independently
        df = df.sort_values(["machine_id", "timestamp"])

        for col in AGGREGATE_COLUMNS:
            if col not in df.columns:
                continue
            df[col] = (
                df.groupby("machine_id")[col]
                .transform(lambda s: s.ffill().bfill())
            )

        logger.debug("Applied ffill + bfill per machine.")
        return df

    @staticmethod
    def _log_missing_report(df: pd.DataFrame, stage: str) -> None:
        """Log the count and percentage of NaN values per column."""
        missing = df[AGGREGATE_COLUMNS].isnull().sum()
        total = len(df)
        report_lines = [f"Missing values report ({stage}):"]
        for col, count in missing.items():
            pct = count * 100 / total if total > 0 else 0
            report_lines.append(f"  {col}: {count} ({pct:.2f}%)")
        logger.info("\n".join(report_lines))

    @staticmethod
    def _assert_no_missing(df: pd.DataFrame) -> None:
        """
        Assert that AGGREGATE_COLUMNS have no remaining NaN values.

        A residual NaN after ffill+bfill means the column is entirely NaN for
        some machine (all samples are missing), which should not happen in the
        Alibaba trace.  If it does, the pipeline should fail loudly.
        """
        remaining = df[AGGREGATE_COLUMNS].isnull().sum()
        truly_missing = remaining[remaining > 0]
        if not truly_missing.empty:
            raise ValueError(
                f"NaN values remain after filling: {truly_missing.to_dict()}\n"
                "This may indicate a machine with no valid samples at all."
            )
        logger.info("No missing values remain after cleaning.")
