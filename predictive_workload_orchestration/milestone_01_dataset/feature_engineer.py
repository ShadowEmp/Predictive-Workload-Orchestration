"""
milestone_01_dataset/feature_engineer.py
==========================================
Computes all engineered features on the aggregated cluster-level time series.

Feature groups:
1. Rolling statistics  — trend and volatility awareness
2. Lag features        — temporal memory
3. Cyclic time encoding — daily and weekly seasonality
4. Trend signal        — direction of change
5. Load intensity label — categorical bucketing for EDA and reward shaping

Design note:
- Features are computed in a fixed order so that derived features (e.g. lags
  of rolling means) are always based on finalized base features.
- The first max(lag, window) rows will contain NaN for rolling/lag features.
  We drop these warm-up rows at the end so the saved dataset is fully clean.
- All feature names are documented in FEATURE_DOCUMENTATION.md (written by eda.py).
"""

import numpy as np
import pandas as pd

from config.logger import get_logger
from config.settings import (
    LAG_STEPS,
    LOAD_INTENSITY_LABELS,
    LOAD_INTENSITY_THRESHOLDS,
    ROLLING_WINDOWS,
    TARGET_COLUMN,
)

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Adds engineered features to an aggregated cluster-level DataFrame.

    Parameters
    ----------
    target_col : str
        Primary signal column (default: ``cpu_util_percent``).
    rolling_windows : list[int]
        Rolling window sizes in steps.
    lag_steps : list[int]
        Lag offsets in steps.
    load_thresholds : list[float]
        Percentile thresholds for load intensity buckets.
    load_labels : list[str]
        Labels for each bucket.
    """

    def __init__(
        self,
        target_col: str = TARGET_COLUMN,
        rolling_windows: list = None,
        lag_steps: list = None,
        load_thresholds: list = None,
        load_labels: list = None,
    ) -> None:
        self.target_col = target_col
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS
        self.lag_steps = lag_steps or LAG_STEPS
        self.load_thresholds = load_thresholds or LOAD_INTENSITY_THRESHOLDS
        self.load_labels = load_labels or LOAD_INTENSITY_LABELS

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run full feature engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Aggregated cluster time series from WorkloadAggregator.

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame with all engineered features.
        """
        logger.info("Starting feature engineering. Input shape: %s", df.shape)

        df = df.copy()
        df = self._add_rolling_features(df)
        df = self._add_lag_features(df)
        df = self._add_cyclic_time_features(df)
        df = self._add_trend_features(df)
        df = self._add_load_intensity(df)
        df = self._drop_warmup_rows(df)

        logger.info(
            "Feature engineering complete. Output shape: %s | Features: %d",
            df.shape,
            len(df.columns),
        )
        self._log_feature_summary(df)
        return df

    def get_feature_names(self, df: pd.DataFrame) -> list:
        """Return all feature column names (excluding timestamp if it's the index)."""
        return df.columns.tolist()

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling mean and standard deviation over multiple windows.

        Rationale: Rolling mean captures the recent trend; rolling std
        captures volatility.  Both are important inputs to the RL agent
        for determining if the current load spike is sustained or transient.
        """
        for window in self.rolling_windows:
            col_mean = f"{self.target_col}_roll_mean_{window}"
            col_std = f"{self.target_col}_roll_std_{window}"
            df[col_mean] = df[self.target_col].rolling(window=window, min_periods=1).mean()
            df[col_std] = df[self.target_col].rolling(window=window, min_periods=1).std().fillna(0)
            logger.debug("Added rolling features: %s, %s", col_mean, col_std)

        # Also add rolling features for memory utilization
        if "mem_util_percent" in df.columns:
            for window in self.rolling_windows:
                col_mean = f"mem_util_percent_roll_mean_{window}"
                df[col_mean] = df["mem_util_percent"].rolling(window=window, min_periods=1).mean()

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lag values of the target column.

        Rationale: Auto-scaling decisions depend on recent history.  Lag-1
        (5 min ago), lag-3 (15 min), lag-6 (30 min), lag-12 (1 hour) give
        the DQN agent a temporal window without requiring an LSTM.
        """
        for lag in self.lag_steps:
            col_name = f"{self.target_col}_lag_{lag}"
            df[col_name] = df[self.target_col].shift(lag)
            logger.debug("Added lag feature: %s", col_name)
        return df

    def _add_cyclic_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sin/cos encoding of hour-of-day and day-of-week.

        Rationale: Time-of-day and day-of-week are the dominant drivers of
        cloud workload patterns.  Linear encoding (e.g. hour=23) is wrong
        because 23 and 0 are adjacent.  Sin/cos wraps the cycle correctly
        and is compatible with both Prophet (as regressors) and neural nets.
        """
        if not hasattr(df.index, "hour"):
            logger.warning(
                "Index does not have .hour attribute. "
                "Cyclic features require a DatetimeIndex."
            )
            return df

        # Hour of day (period = 24 hours = 288 steps at 5-min resolution)
        hour = df.index.hour + df.index.minute / 60.0
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        # Day of week (period = 7 days)
        dow = df.index.dayofweek  # Monday=0, Sunday=6
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

        # Binary business-hours flag (08:00 – 20:00)
        df["is_business_hours"] = (
            (df.index.hour >= 8) & (df.index.hour < 20)
        ).astype(int)

        # Binary weekend flag
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

        logger.debug("Added cyclic time features.")
        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        First difference (trend direction) of the target column.

        Rationale: The DQN agent needs to know whether load is rising or
        falling, not just its current value.  The first difference provides
        this direction signal.  Positive = rising, negative = falling.
        """
        df[f"{self.target_col}_trend"] = df[self.target_col].diff().fillna(0)

        # Also compute a smoothed rate-of-change (diff of 12-step rolling mean)
        roll_col = f"{self.target_col}_roll_mean_{self.rolling_windows[0]}"
        if roll_col in df.columns:
            df[f"{self.target_col}_trend_smooth"] = df[roll_col].diff().fillna(0)

        logger.debug("Added trend features.")
        return df

    def _add_load_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorical load intensity label based on percentile thresholds.

        Percentiles are computed on the TRAINING data only.  During
        inference / replay, the same thresholds from settings.py are used.

        Labels: low | medium | high | critical
        Thresholds: [25th, 50th, 75th] percentiles of cpu_util_percent.
        """
        thresholds = [
            df[self.target_col].quantile(p / 100)
            for p in self.load_thresholds
        ]
        logger.info(
            "Load intensity thresholds (percentiles %s): %s",
            self.load_thresholds,
            [f"{t:.2f}" for t in thresholds],
        )

        conditions = [
            df[self.target_col] <= thresholds[0],
            (df[self.target_col] > thresholds[0]) & (df[self.target_col] <= thresholds[1]),
            (df[self.target_col] > thresholds[1]) & (df[self.target_col] <= thresholds[2]),
            df[self.target_col] > thresholds[2],
        ]
        df["load_intensity"] = np.select(conditions, self.load_labels, default="unknown")

        # Also store the numeric intensity level (0–3) for the RL agent
        label_map = {label: i for i, label in enumerate(self.load_labels)}
        df["load_intensity_code"] = df["load_intensity"].map(label_map)

        return df

    def _drop_warmup_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop the initial rows that contain NaN from lag/rolling features.

        The warm-up period is the maximum of the largest rolling window and
        the largest lag step.  Rows before this point do not have valid
        feature values for all features.
        """
        warmup = max(max(self.rolling_windows), max(self.lag_steps))
        n_before = len(df)
        df = df.iloc[warmup:].copy()
        logger.info(
            "Dropped %d warm-up rows (warmup period = %d steps = %d minutes).",
            n_before - len(df),
            warmup,
            warmup * 5,
        )
        return df

    @staticmethod
    def _log_feature_summary(df: pd.DataFrame) -> None:
        """Log a summary of all feature columns."""
        logger.info("Feature summary:")
        for col in df.columns:
            logger.info(
                "  %-45s  dtype=%-10s  min=%8.3f  max=%8.3f  null=%d",
                col,
                str(df[col].dtype),
                df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else float("nan"),
                df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else float("nan"),
                df[col].isnull().sum(),
            )
