"""
tests/test_milestone_02.py
============================
Unit and integration tests for Milestone 2: Workload Forecasting.

Strategy:
- All tests use synthetic in-memory data.
- Prophet is NOT actually trained in unit tests (too slow for CI).
  We mock the Prophet model and inject synthetic forecast DataFrames.
- Only the integration test trains a real (tiny) Prophet model.
- Tests are grouped by module: Trainer, Evaluator, Merger, Pipeline.

Run:
    cd predictive_workload_orchestration
    python3 -m pytest tests/test_milestone_02.py -v
"""

import sys
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from milestone_02_forecasting.evaluator import ForecastEvaluator
from milestone_02_forecasting.merger import ForecastMerger
from milestone_02_forecasting.trainer import ProphetTrainer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

EPOCH = pd.Timestamp("2018-01-01 00:00:00", tz="UTC")
N_STEPS = 2000   # ~7 days at 5-min resolution — enough for Prophet


def make_processed_df(n: int = N_STEPS) -> pd.DataFrame:
    """Synthetic processed_dataset.csv DataFrame with DatetimeIndex (UTC)."""
    rng = np.random.default_rng(0)
    index = pd.date_range(start=EPOCH, periods=n, freq="5min", tz="UTC")
    # Add a mild daily sinusoidal pattern so Prophet has something to detect
    hours = np.arange(n) / 12.0   # 12 steps per hour
    cpu = 50 + 15 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 3, n)
    cpu = np.clip(cpu, 0, 100)
    df = pd.DataFrame({
        "cpu_util_percent": cpu,
        "mem_util_percent": rng.uniform(30, 70, n),
        "net_in": rng.uniform(0, 50, n),
        "net_out": rng.uniform(0, 50, n),
        "disk_io_percent": rng.uniform(0, 30, n),
        "hour_sin": np.sin(2 * np.pi * (np.arange(n) % 288) / 288),
        "load_intensity": ["low"] * n,
    }, index=index)
    df.index.name = "timestamp"
    return df


def make_prophet_df(n: int = N_STEPS) -> pd.DataFrame:
    """Synthetic Prophet-format {ds, y} DataFrame (timezone-naive)."""
    rng = np.random.default_rng(1)
    index = pd.date_range(start="2018-01-01", periods=n, freq="5min")
    hours = np.arange(n) / 12.0
    cpu = 50 + 15 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 3, n)
    return pd.DataFrame({"ds": index, "y": np.clip(cpu, 0, 100)})


def make_forecast_df(n: int = N_STEPS) -> pd.DataFrame:
    """Synthetic Prophet output forecast DataFrame."""
    rng = np.random.default_rng(2)
    index = pd.date_range(start="2018-01-01", periods=n, freq="5min")
    yhat = 50 + 15 * np.sin(2 * np.pi * np.arange(n) / 288) + rng.normal(0, 1, n)
    return pd.DataFrame({
        "ds": index,
        "yhat": yhat,
        "yhat_lower": yhat - 5,
        "yhat_upper": yhat + 5,
        "trend": yhat * 0.9,
        "weekly": rng.normal(0, 1, n),
        "daily": rng.normal(0, 2, n),
    })


def make_test_df(prophet_df: pd.DataFrame, train_frac: float = 0.8) -> pd.DataFrame:
    """Extract the test split from a prophet_df."""
    split = int(len(prophet_df) * train_frac)
    return prophet_df.iloc[split:].copy()


# ===========================================================================
# Tests: ProphetTrainer
# ===========================================================================

class TestProphetTrainer:

    def setup_method(self):
        self.trainer = ProphetTrainer()

    def test_prepare_prophet_df_columns(self):
        """Prepared DataFrame should have exactly 'ds' and 'y' columns."""
        processed = make_processed_df(500)
        result = self.trainer._prepare_prophet_df(processed)
        assert list(result.columns) == ["ds", "y"], (
            f"Expected ['ds', 'y'], got {list(result.columns)}"
        )

    def test_prepare_prophet_df_no_timezone(self):
        """Prophet requires timezone-naive 'ds'."""
        processed = make_processed_df(500)
        result = self.trainer._prepare_prophet_df(processed)
        assert result["ds"].dt.tz is None, "ds column should be timezone-naive."

    def test_prepare_prophet_df_no_nan(self):
        """Prepared DataFrame should have no NaN in 'y'."""
        processed = make_processed_df(500)
        result = self.trainer._prepare_prophet_df(processed)
        assert result["y"].isna().sum() == 0

    def test_split_preserves_temporal_order(self):
        """Train split must end before test split begins."""
        prophet_df = make_prophet_df(500)
        train, test = self.trainer._split(prophet_df)
        assert train["ds"].max() < test["ds"].min(), (
            "Train period should end before test period begins."
        )

    def test_split_fraction(self):
        """Train fraction should be approximately PROPHET_TRAIN_FRACTION."""
        prophet_df = make_prophet_df(1000)
        train, test = self.trainer._split(prophet_df)
        actual_fraction = len(train) / len(prophet_df)
        assert abs(actual_fraction - 0.8) < 0.01

    def test_split_no_overlap(self):
        """Train and test sets must not share any rows."""
        prophet_df = make_prophet_df(500)
        train, test = self.trainer._split(prophet_df)
        overlap = pd.merge(train, test, on="ds")
        assert len(overlap) == 0, "Train and test have overlapping timestamps."

    def test_make_future_df_length(self):
        """Future DataFrame should have exactly total_steps rows."""
        prophet_df = make_prophet_df(200)
        trainer = ProphetTrainer(forecast_horizon_steps=50)
        future = trainer._make_future_df(None, prophet_df, total_steps=300)
        assert len(future) == 300

    def test_make_future_df_freq(self):
        """Future DataFrame should have 5-minute intervals."""
        prophet_df = make_prophet_df(100)
        trainer = ProphetTrainer()
        future = trainer._make_future_df(None, prophet_df, total_steps=100)
        diffs = pd.Series(future["ds"]).diff().dropna()
        expected = pd.Timedelta("5min")
        assert (diffs == expected).all(), "Future df does not have 5-min intervals."

    def test_save_and_load_model(self, tmp_path):
        """Save and load a picklable object through ProphetTrainer.load_model."""
        import pickle

        # Use a plain dict as a stand-in — MagicMock is not picklable
        fake_model = {"name": "mock_prophet", "type": "prophet"}
        model_path = tmp_path / "prophet_model.pkl"

        with open(model_path, "wb") as fh:
            pickle.dump(fake_model, fh)

        loaded = ProphetTrainer.load_model(model_path)
        assert loaded["name"] == "mock_prophet"


# ===========================================================================
# Tests: ForecastEvaluator
# ===========================================================================

class TestForecastEvaluator:

    def setup_method(self):
        self.prophet_df = make_prophet_df(N_STEPS)
        self.forecast_df = make_forecast_df(N_STEPS)
        self.test_df = make_test_df(self.prophet_df)
        self.mock_model = MagicMock()

    def test_align_returns_dataframe(self, tmp_path):
        """_align_forecast_with_actuals should return a DataFrame."""
        ev = ForecastEvaluator(
            self.forecast_df, self.test_df, self.mock_model,
            reports_dir=tmp_path, plots_dir=tmp_path / "plots",
        )
        aligned = ev._align_forecast_with_actuals()
        assert isinstance(aligned, pd.DataFrame)

    def test_align_has_required_columns(self, tmp_path):
        """Aligned DataFrame must have y, yhat, yhat_lower, yhat_upper."""
        ev = ForecastEvaluator(
            self.forecast_df, self.test_df, self.mock_model,
            reports_dir=tmp_path, plots_dir=tmp_path / "plots",
        )
        aligned = ev._align_forecast_with_actuals()
        for col in ["y", "yhat", "yhat_lower", "yhat_upper"]:
            assert col in aligned.columns, f"Missing column: {col}"

    def test_compute_metrics_returns_three_keys(self, tmp_path):
        """Metric dictionary must have mae, rmse, mape."""
        ev = ForecastEvaluator(
            self.forecast_df, self.test_df, self.mock_model,
            reports_dir=tmp_path, plots_dir=tmp_path / "plots",
        )
        aligned = ev._align_forecast_with_actuals()
        metrics = ev._compute_metrics(aligned)
        for key in ["mae", "rmse", "mape"]:
            assert key in metrics, f"Missing metric: {key}"

    def test_mae_is_nonnegative(self, tmp_path):
        """MAE must always be >= 0."""
        ev = ForecastEvaluator(
            self.forecast_df, self.test_df, self.mock_model,
            reports_dir=tmp_path, plots_dir=tmp_path / "plots",
        )
        aligned = ev._align_forecast_with_actuals()
        metrics = ev._compute_metrics(aligned)
        assert metrics["mae"] >= 0

    def test_rmse_geq_mae(self, tmp_path):
        """RMSE >= MAE always (due to squaring)."""
        ev = ForecastEvaluator(
            self.forecast_df, self.test_df, self.mock_model,
            reports_dir=tmp_path, plots_dir=tmp_path / "plots",
        )
        aligned = ev._align_forecast_with_actuals()
        metrics = ev._compute_metrics(aligned)
        assert metrics["rmse"] >= metrics["mae"], (
            f"RMSE ({metrics['rmse']:.4f}) < MAE ({metrics['mae']:.4f})"
        )

    def test_perfect_forecast_zero_mae(self, tmp_path):
        """Perfect forecast (yhat == y) should give MAE = 0."""
        perfect_forecast = self.test_df.rename(columns={"y": "yhat"}).copy()
        perfect_forecast["yhat_lower"] = perfect_forecast["yhat"] - 1
        perfect_forecast["yhat_upper"] = perfect_forecast["yhat"] + 1

        ev = ForecastEvaluator(
            perfect_forecast, self.test_df, self.mock_model,
            reports_dir=tmp_path, plots_dir=tmp_path / "plots",
        )
        aligned = ev._align_forecast_with_actuals()
        metrics = ev._compute_metrics(aligned)
        assert metrics["mae"] < 1e-10, f"Expected ~0 MAE, got {metrics['mae']}"

    def test_mape_guarded_against_zero_actuals(self, tmp_path):
        """MAPE computation must not raise ZeroDivisionError."""
        zero_test = self.test_df.copy()
        zero_test["y"] = 0  # all zeros

        ev = ForecastEvaluator(
            self.forecast_df, zero_test, self.mock_model,
            reports_dir=tmp_path, plots_dir=tmp_path / "plots",
        )
        aligned = ev._align_forecast_with_actuals()
        # Should not raise; MAPE should be NaN when all actuals are zero
        metrics = ev._compute_metrics(aligned)
        assert "mape" in metrics  # key must exist (value may be nan)

    def test_write_accuracy_report_creates_file(self, tmp_path):
        """Accuracy report file should be created after run (no CV)."""
        ev = ForecastEvaluator(
            self.forecast_df, self.test_df, self.mock_model,
            reports_dir=tmp_path, plots_dir=tmp_path / "plots",
        )
        ev._aligned = ev._align_forecast_with_actuals()
        ev._metrics = ev._compute_metrics(ev._aligned)
        ev._write_accuracy_report(run_cv=False)

        from config.settings import FORECAST_ACCURACY_REPORT_FILENAME
        assert (tmp_path / FORECAST_ACCURACY_REPORT_FILENAME).exists()


# ===========================================================================
# Tests: ForecastMerger
# ===========================================================================

class TestForecastMerger:

    def setup_method(self):
        self.processed_df = make_processed_df(N_STEPS)
        self.forecast_df = make_forecast_df(N_STEPS)

    def test_merge_returns_dataframe(self, tmp_path):
        """merge() should return a DataFrame."""
        merger = ForecastMerger(processed_dir=tmp_path)
        result = merger.merge(self.processed_df, self.forecast_df)
        assert isinstance(result, pd.DataFrame)

    def test_merge_adds_yhat_column(self, tmp_path):
        """Merged DataFrame must contain 'yhat' column."""
        merger = ForecastMerger(processed_dir=tmp_path)
        result = merger.merge(self.processed_df, self.forecast_df)
        assert "yhat" in result.columns

    def test_merge_adds_forecast_error_column(self, tmp_path):
        """Merged DataFrame must contain 'forecast_error' column."""
        merger = ForecastMerger(processed_dir=tmp_path)
        result = merger.merge(self.processed_df, self.forecast_df)
        assert "forecast_error" in result.columns

    def test_merge_preserves_row_count(self, tmp_path):
        """Merged DataFrame should have same number of rows as processed_df."""
        merger = ForecastMerger(processed_dir=tmp_path)
        result = merger.merge(self.processed_df, self.forecast_df)
        assert len(result) == len(self.processed_df)

    def test_merge_preserves_original_columns(self, tmp_path):
        """All original columns from processed_df must still be present."""
        merger = ForecastMerger(processed_dir=tmp_path)
        result = merger.merge(self.processed_df, self.forecast_df)
        for col in self.processed_df.columns:
            assert col in result.columns, f"Original column missing: {col}"

    def test_forecast_error_computation(self, tmp_path):
        """forecast_error should equal cpu_util_percent - yhat."""
        merger = ForecastMerger(processed_dir=tmp_path)
        result = merger.merge(self.processed_df, self.forecast_df)
        mask = result["yhat"].notna()
        expected_error = result.loc[mask, "cpu_util_percent"] - result.loc[mask, "yhat"]
        actual_error = result.loc[mask, "forecast_error"]
        pd.testing.assert_series_equal(
            actual_error.reset_index(drop=True),
            expected_error.reset_index(drop=True),
            check_names=False,
        )

    def test_merge_saves_csv(self, tmp_path):
        """merge() should write forecast_dataset.csv to processed_dir."""
        merger = ForecastMerger(processed_dir=tmp_path)
        merger.merge(self.processed_df, self.forecast_df)
        from config.settings import FORECAST_DATASET_FILENAME
        assert (tmp_path / FORECAST_DATASET_FILENAME).exists()


# ===========================================================================
# Integration test: Full pipeline with a real (tiny) Prophet model
# ===========================================================================

class TestForecastingPipelineIntegration:

    @pytest.mark.slow
    def test_full_pipeline_on_synthetic_data(self, tmp_path):
        """
        End-to-end integration test.

        Trains a real Prophet model on synthetic data (2000 steps ≈ 7 days),
        evaluates, merges, and validates output.

        Marked @pytest.mark.slow — skipped in fast CI runs.
        Run explicitly: pytest -m slow
        """
        # Write synthetic processed_dataset.csv to tmp_path
        processed_df = make_processed_df(N_STEPS)
        processed_df.to_csv(tmp_path / "processed_dataset.csv")

        from milestone_02_forecasting.pipeline import ForecastingPipeline

        pipeline = ForecastingPipeline(
            processed_dir=tmp_path,
            run_cv=False,   # skip CV in integration test for speed
        )
        result_df = pipeline.run()

        # Validate
        assert "yhat" in result_df.columns
        assert "forecast_error" in result_df.columns
        assert len(result_df) > 0

        # Run validation method
        passed = pipeline.validate(result_df)
        assert passed is True
