"""
tests/test_milestone_01.py
============================
Unit and integration tests for Milestone 1: Dataset Preparation.

Test strategy:
- Every test uses synthetic in-memory data, never the real dataset.
  This makes tests fast, reproducible, and CI-friendly.
- Tests are grouped by module (one class per module).
- Each test has a single assertion to make failures easy to diagnose.

Run:
    cd predictive_workload_orchestration
    python -m pytest tests/test_milestone_01.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from milestone_01_dataset.aggregator import WorkloadAggregator
from milestone_01_dataset.cleaner import DataCleaner
from milestone_01_dataset.feature_engineer import FeatureEngineer
from milestone_01_dataset.loader import RawDataLoader

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EPOCH = pd.Timestamp("2018-01-01 00:00:00", tz="UTC")


def make_raw_df(n_machines: int = 5, n_steps: int = 500) -> pd.DataFrame:
    """
    Build a synthetic raw DataFrame that mimics machine_usage.csv output
    from RawDataLoader (after timestamp parsing).
    """
    rng = np.random.default_rng(42)
    rows = []
    base_time = EPOCH
    step_seconds = 300  # 5 minutes

    for m in range(n_machines):
        for t in range(n_steps):
            rows.append({
                "machine_id": f"machine_{m:03d}",
                "time_stamp": t * step_seconds,
                "timestamp": base_time + pd.Timedelta(seconds=t * step_seconds),
                "cpu_util_percent": float(rng.uniform(10, 90)),
                "mem_util_percent": float(rng.uniform(20, 80)),
                "mem_gps": float(rng.uniform(0, 5)),
                "mkpi": float(rng.uniform(0, 100)),
                "net_in": float(rng.uniform(0, 50)),
                "net_out": float(rng.uniform(0, 50)),
                "disk_io_percent": float(rng.uniform(0, 40)),
            })

    return pd.DataFrame(rows)


def make_aggregated_df(n_steps: int = 500) -> pd.DataFrame:
    """Build a synthetic aggregated cluster-level DataFrame."""
    rng = np.random.default_rng(42)
    index = pd.date_range(
        start=EPOCH,
        periods=n_steps,
        freq="5min",
        tz="UTC",
    )
    df = pd.DataFrame({
        "cpu_util_percent": rng.uniform(10, 90, n_steps).astype(float),
        "mem_util_percent": rng.uniform(20, 80, n_steps).astype(float),
        "net_in": rng.uniform(0, 50, n_steps).astype(float),
        "net_out": rng.uniform(0, 50, n_steps).astype(float),
        "disk_io_percent": rng.uniform(0, 40, n_steps).astype(float),
    }, index=index)
    df.index.name = "timestamp"
    return df


# ===========================================================================
# Tests: DataCleaner
# ===========================================================================

class TestDataCleaner:

    def setup_method(self):
        self.cleaner = DataCleaner()
        self.raw = make_raw_df()

    def test_no_missing_after_clean(self):
        """clean() should produce zero NaN values in aggregate columns."""
        cleaned = self.cleaner.clean(self.raw)
        from config.settings import AGGREGATE_COLUMNS
        missing = cleaned[AGGREGATE_COLUMNS].isnull().sum().sum()
        assert missing == 0, f"Expected 0 NaNs, got {missing}"

    def test_cpu_clipped_to_100(self):
        """cpu_util_percent values must not exceed 100 after cleaning."""
        df = self.raw.copy()
        df.loc[0, "cpu_util_percent"] = 150.0  # inject out-of-range
        cleaned = self.cleaner.clean(df)
        assert cleaned["cpu_util_percent"].max() <= 100.0

    def test_cpu_clipped_to_zero(self):
        """cpu_util_percent must not be negative after cleaning."""
        df = self.raw.copy()
        df.loc[0, "cpu_util_percent"] = -10.0
        cleaned = self.cleaner.clean(df)
        assert cleaned["cpu_util_percent"].min() >= 0.0

    def test_duplicates_removed(self):
        """Duplicate (machine_id, timestamp) rows should be removed."""
        df = pd.concat([self.raw, self.raw.iloc[:10]], ignore_index=True)
        cleaned = self.cleaner.clean(df)
        assert len(cleaned) == len(self.raw)

    def test_ffill_fills_nan(self):
        """Forward-fill should handle injected NaN values (partial, not entire machine)."""
        df = self.raw.copy()
        # Inject NaN on only SOME rows for machine_000 (rows 5–20),
        # leaving valid data before and after so ffill/bfill can fill them.
        mask = (df["machine_id"] == "machine_000")
        indices = df.index[mask].tolist()
        df.loc[indices[5:20], "cpu_util_percent"] = np.nan
        cleaned = self.cleaner.clean(df)
        assert cleaned["cpu_util_percent"].isnull().sum() == 0


# ===========================================================================
# Tests: WorkloadAggregator
# ===========================================================================

class TestWorkloadAggregator:

    def setup_method(self):
        self.aggregator = WorkloadAggregator()
        self.raw = make_raw_df(n_machines=3, n_steps=300)

    def test_output_is_dataframe(self):
        """aggregate() should return a pd.DataFrame."""
        result = self.aggregator.aggregate(self.raw)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_datetime_index(self):
        """Output index should be DatetimeIndex."""
        result = self.aggregator.aggregate(self.raw)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_output_index_is_continuous(self):
        """Aggregated time series should have no gaps."""
        result = self.aggregator.aggregate(self.raw)
        diffs = pd.Series(result.index).diff().dropna()
        expected = pd.Timedelta("5min")
        assert (diffs == expected).all(), "Gaps found in aggregated time series."

    def test_cpu_in_valid_range(self):
        """Aggregated CPU should remain in [0, 100]."""
        result = self.aggregator.aggregate(self.raw)
        assert result["cpu_util_percent"].between(0, 100).all()

    def test_no_missing_after_aggregation(self):
        """No NaN values should remain in the aggregated output."""
        result = self.aggregator.aggregate(self.raw)
        assert result.isnull().sum().sum() == 0


# ===========================================================================
# Tests: FeatureEngineer
# ===========================================================================

class TestFeatureEngineer:

    def setup_method(self):
        self.fe = FeatureEngineer()
        self.agg = make_aggregated_df(n_steps=500)

    def test_output_has_more_columns(self):
        """Feature engineering should add new columns."""
        result = self.fe.engineer(self.agg)
        assert len(result.columns) > len(self.agg.columns)

    def test_rolling_mean_columns_exist(self):
        """Rolling mean columns should be present."""
        result = self.fe.engineer(self.agg)
        from config.settings import ROLLING_WINDOWS
        for w in ROLLING_WINDOWS:
            col = f"cpu_util_percent_roll_mean_{w}"
            assert col in result.columns, f"Missing column: {col}"

    def test_lag_columns_exist(self):
        """Lag columns should be present."""
        result = self.fe.engineer(self.agg)
        from config.settings import LAG_STEPS
        for lag in LAG_STEPS:
            col = f"cpu_util_percent_lag_{lag}"
            assert col in result.columns, f"Missing column: {col}"

    def test_cyclic_time_columns_exist(self):
        """Cyclic time encoding columns should be present."""
        result = self.fe.engineer(self.agg)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_load_intensity_column_exists(self):
        """load_intensity column should be present."""
        result = self.fe.engineer(self.agg)
        assert "load_intensity" in result.columns

    def test_load_intensity_valid_labels(self):
        """load_intensity should only contain valid labels."""
        result = self.fe.engineer(self.agg)
        valid = {"low", "medium", "high", "critical"}
        actual = set(result["load_intensity"].unique())
        assert actual.issubset(valid), f"Invalid labels: {actual - valid}"

    def test_no_missing_values_after_warmup_drop(self):
        """No NaN should remain after warm-up rows are dropped."""
        result = self.fe.engineer(self.agg)
        numeric_missing = result.select_dtypes(include=["number"]).isnull().sum().sum()
        assert numeric_missing == 0, f"{numeric_missing} NaNs remain."

    def test_hour_sin_in_range(self):
        """hour_sin should be in [-1, 1]."""
        result = self.fe.engineer(self.agg)
        assert result["hour_sin"].between(-1, 1).all()

    def test_warmup_rows_dropped(self):
        """Output should have fewer rows than input (warm-up removed)."""
        result = self.fe.engineer(self.agg)
        from config.settings import ROLLING_WINDOWS, LAG_STEPS
        warmup = max(max(ROLLING_WINDOWS), max(LAG_STEPS))
        assert len(result) == len(self.agg) - warmup

    def test_trend_column_exists(self):
        """cpu_util_percent_trend column should be present."""
        result = self.fe.engineer(self.agg)
        assert "cpu_util_percent_trend" in result.columns

    def test_is_weekend_binary(self):
        """is_weekend should contain only 0 and 1."""
        result = self.fe.engineer(self.agg)
        assert set(result["is_weekend"].unique()).issubset({0, 1})


# ===========================================================================
# Tests: DatasetPipeline (integration, no download)
# ===========================================================================

class TestDatasetPipelineIntegration:
    """
    Integration test: run the full pipeline on synthetic data by monkey-patching
    the loader to return synthetic data instead of reading a real file.
    """

    def test_pipeline_produces_processed_csv(self, tmp_path, monkeypatch):
        """
        Full pipeline (without download) should produce a valid CSV.
        """
        # Patch loader to return synthetic data
        synthetic_raw = make_raw_df(n_machines=3, n_steps=300)

        from milestone_01_dataset import pipeline as p
        monkeypatch.setattr(
            p.RawDataLoader,
            "load_machine_usage",
            lambda self, **kw: synthetic_raw,
        )
        monkeypatch.setattr(p, "PROCESSED_DIR", tmp_path)

        pipeline_obj = p.DatasetPipeline(download=False, skip_eda=True)
        df = pipeline_obj.run()

        assert df is not None
        assert len(df) > 0
        assert "cpu_util_percent" in df.columns

    def test_pipeline_validation_passes(self, tmp_path, monkeypatch):
        """Validation should pass on correctly processed synthetic data."""
        # Use enough steps so after aggregation + warmup drop we exceed 1000 rows.
        # 3 machines × 2000 steps at 5-min = 2000 aggregated rows → 1964 after warmup.
        synthetic_raw = make_raw_df(n_machines=3, n_steps=2000)

        from milestone_01_dataset import pipeline as p
        monkeypatch.setattr(
            p.RawDataLoader,
            "load_machine_usage",
            lambda self, **kw: synthetic_raw,
        )
        monkeypatch.setattr(p, "PROCESSED_DIR", tmp_path)

        pipeline_obj = p.DatasetPipeline(download=False, skip_eda=True)
        df = pipeline_obj.run()
        result = pipeline_obj.validate(df)

        assert result is True
