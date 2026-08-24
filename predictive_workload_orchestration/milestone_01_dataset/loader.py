"""
milestone_01_dataset/loader.py
================================
Loads raw CSV files from disk and validates their schema.

Design notes:
- Loading is separated from cleaning to respect the Single Responsibility Principle.
- Schema validation catches missing columns early, before expensive operations.
- The loader returns a typed pandas DataFrame with correct dtypes enforced.
- Timestamps in Alibaba Cluster Trace are Unix seconds (integer); we convert
  them to pandas Timestamps with a known epoch.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from config.logger import get_logger
from config.settings import (
    MACHINE_USAGE_COLUMNS,
    MACHINE_META_COLUMNS,
    RAW_DIR,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Epoch: Alibaba Cluster Trace 2018 timestamps are seconds offset from
# 2018-01-01 00:00:00 UTC.  This is documented in the dataset README.
# ---------------------------------------------------------------------------
ALIBABA_EPOCH = pd.Timestamp("2018-01-01 00:00:00", tz="UTC")


class RawDataLoader:
    """
    Loads and validates Alibaba Cluster Trace raw CSV files.

    Parameters
    ----------
    raw_dir : Path
        Directory containing downloaded CSV files.
    """

    def __init__(self, raw_dir: Path = RAW_DIR) -> None:
        self.raw_dir = Path(raw_dir)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_machine_usage(
        self,
        filename: str = "machine_usage.csv",
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load machine_usage.csv.

        Parameters
        ----------
        filename : str
            Filename within raw_dir.
        nrows : int, optional
            Limit rows loaded (useful for quick tests).

        Returns
        -------
        pd.DataFrame
            DataFrame with parsed timestamps and validated schema.
        """
        path = self.raw_dir / filename
        logger.info("Loading machine_usage from: %s", path)

        df = self._read_csv(path, expected_columns=MACHINE_USAGE_COLUMNS, nrows=nrows)
        df = self._parse_timestamp(df)
        df = self._enforce_numeric(
            df,
            columns=[
                "cpu_util_percent",
                "mem_util_percent",
                "mem_gps",
                "mkpi",
                "net_in",
                "net_out",
                "disk_io_percent",
            ],
        )
        logger.info(
            "machine_usage loaded: %d rows, %d columns, time range: %s → %s",
            len(df),
            len(df.columns),
            df["timestamp"].min(),
            df["timestamp"].max(),
        )
        return df

    def load_machine_meta(
        self,
        filename: str = "machine_meta.csv",
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load machine_meta.csv (static metadata)."""
        path = self.raw_dir / filename
        logger.info("Loading machine_meta from: %s", path)
        df = self._read_csv(path, expected_columns=MACHINE_META_COLUMNS, nrows=nrows)
        df = self._parse_timestamp(df)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_csv(
        self,
        path: Path,
        expected_columns: List[str],
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Read a CSV file, assign column names, and validate schema.

        Alibaba Cluster Trace CSVs have NO header row — columns are assigned
        positionally using the expected schema.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}\n"
                "Run the downloader first: python -m milestone_01_dataset.pipeline"
            )

        df = pd.read_csv(
            path,
            header=None,
            names=expected_columns,
            nrows=nrows,
            low_memory=False,
        )

        logger.debug("Loaded %d rows from %s", len(df), path.name)
        self._validate_schema(df, expected_columns, path.name)
        return df

    @staticmethod
    def _validate_schema(df: pd.DataFrame, expected: List[str], filename: str) -> None:
        """Raise ValueError if any expected column is missing."""
        missing = set(expected) - set(df.columns)
        if missing:
            raise ValueError(
                f"Schema validation failed for {filename}: "
                f"missing columns {missing}"
            )
        logger.debug("Schema validation passed for %s", filename)

    @staticmethod
    def _parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert integer Unix-second offset (from Alibaba epoch) to Timestamp.

        The Alibaba Cluster Trace README states timestamps are seconds elapsed
        since the start of the trace period.  We anchor to 2018-01-01 UTC.
        """
        df = df.copy()
        df["timestamp"] = ALIBABA_EPOCH + pd.to_timedelta(
            df["time_stamp"].astype(float), unit="s"
        )
        return df

    @staticmethod
    def _enforce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Coerce specified columns to float, turning non-numeric values to NaN."""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
