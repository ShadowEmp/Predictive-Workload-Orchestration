"""
milestone_01_dataset/aggregator.py
=====================================
Aggregates cleaned per-machine data into cluster-level 5-minute intervals.

Design notes:
- Per-machine data is first resampled to the target frequency, then averaged
  across all machines to get a cluster-level signal.
- We use mean aggregation for utilization percentages (see settings.py).
- The resulting time-series index is validated to be continuous (no gaps).
- A gap report is produced before gap-filling so anomalies are documented.
"""

from typing import Optional

import pandas as pd

from config.logger import get_logger
from config.settings import (
    AGGREGATION_FREQ,
    AGGREGATION_FUNCTIONS,
    AGGREGATE_COLUMNS,
)

logger = get_logger(__name__)


class WorkloadAggregator:
    """
    Aggregates per-machine cleaned data to a cluster-level time series.

    Parameters
    ----------
    freq : str
        Pandas resample frequency string (e.g. ``'5min'``).
    agg_functions : dict
        Column → aggregation function mapping.
    """

    def __init__(
        self,
        freq: str = AGGREGATION_FREQ,
        agg_functions: Optional[dict] = None,
    ) -> None:
        self.freq = freq
        self.agg_functions = agg_functions or AGGREGATION_FUNCTIONS

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert per-machine cleaned data → cluster-level 5-min time series.

        Pipeline:
        1. Resample each machine's data to `freq` (fills intra-machine gaps).
        2. Average across all machines at each time window.
        3. Validate continuity of the resulting index.
        4. Fill any residual gaps in the cluster-level series.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned DataFrame from DataCleaner, must have 'timestamp' column.

        Returns
        -------
        pd.DataFrame
            Cluster-level time series indexed by timestamp, one row per window.
        """
        logger.info("Starting aggregation. Input shape: %s", df.shape)

        # Step 1: Set timestamp as index for resampling
        df = df.set_index("timestamp").sort_index()

        # Step 2: Resample per machine then average across machines
        agg_cols = [c for c in AGGREGATE_COLUMNS if c in df.columns]
        cluster_df = self._resample_and_average(df, agg_cols)

        # Step 3: Check for gaps in the aggregated index
        gap_report = self._find_gaps(cluster_df)
        if gap_report:
            logger.warning(
                "Found %d gaps in the cluster-level time series (before fill).",
                len(gap_report),
            )
            for gap in gap_report[:5]:  # log first 5
                logger.warning("  Gap: %s → %s (%s)", gap["start"], gap["end"], gap["length"])

        # Step 4: Reindex to a perfectly continuous range, then fill
        cluster_df = self._enforce_continuous_index(cluster_df)

        logger.info(
            "Aggregation complete. Output shape: %s | Range: %s → %s",
            cluster_df.shape,
            cluster_df.index.min(),
            cluster_df.index.max(),
        )
        return cluster_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resample_and_average(
        self, df: pd.DataFrame, agg_cols: list
    ) -> pd.DataFrame:
        """
        For each machine, resample to `freq`.  Then compute the mean across
        machines at each time step to get the cluster-level signal.

        Two-step approach avoids the issue of machines with different sampling
        rates biasing the cluster average.
        """
        machine_groups = df.groupby("machine_id")
        resampled_list = []

        for machine_id, machine_df in machine_groups:
            # Resample this machine's metrics to the target frequency
            resampled = machine_df[agg_cols].resample(self.freq).agg(self.agg_functions)
            resampled_list.append(resampled)

        logger.info(
            "Resampled %d machines at %s frequency.", len(resampled_list), self.freq
        )

        if not resampled_list:
            raise ValueError("No machines found in the dataset.")

        # Stack all machines and compute cluster-level mean at each timestamp
        all_machines = pd.concat(resampled_list, axis=0)
        cluster_series = all_machines.groupby(level=0).mean()
        cluster_series.index.name = "timestamp"

        return cluster_series

    def _find_gaps(self, df: pd.DataFrame) -> list:
        """
        Identify gaps in the DatetimeIndex larger than the expected frequency.

        Returns a list of dicts: [{start, end, length}, ...]
        """
        if len(df) < 2:
            return []

        expected_delta = pd.to_timedelta(
            pd.tseries.frequencies.to_offset(self.freq)
        )
        gaps = []

        diffs = pd.Series(df.index).diff().dropna()
        for i, diff in enumerate(diffs):
            if diff > expected_delta:
                start = df.index[i]
                end = df.index[i + 1] if i + 1 < len(df.index) else df.index[i]
                gaps.append({"start": start, "end": end, "length": diff})

        return gaps

    def _enforce_continuous_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reindex to a perfectly regular date range, then fill forward.

        This ensures downstream modules (Prophet, RL env) receive a
        continuous, evenly-spaced time series with no implicit gaps.
        """
        full_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=self.freq,
            tz=df.index.tz,
        )
        n_original = len(df)
        df = df.reindex(full_range)
        df.index.name = "timestamp"
        n_added = len(df) - n_original
        if n_added:
            logger.info("Added %d synthetic rows to fill index gaps.", n_added)

        # Fill synthetic rows using ffill then bfill
        df = df.ffill().bfill()

        return df
