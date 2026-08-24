"""
milestone_02_forecasting/trainer.py
=====================================
Trains a Facebook Prophet model on the processed cluster-level CPU time series.

Responsibilities:
1. Load processed_dataset.csv from M1.
2. Prepare the Prophet-format DataFrame: {ds: timestamp, y: cpu_util_percent}.
3. Configure and fit the Prophet model.
4. Persist the trained model to disk (pickle).
5. Return in-sample fitted values for merging.

Design notes:
- Prophet requires columns named exactly 'ds' (datetime) and 'y' (target).
- We suppress Prophet/Stan verbose logging to keep our pipeline logs readable.
- The train/test split is done by time (not random) to respect temporal order.
- Timezone handling: Prophet does not support timezone-aware datetimes.
  We strip the UTC timezone before passing to Prophet and restore it after.
- The fitted model uses all training data so the forecast starts from the
  end of the training period.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from config.logger import get_logger
from config.settings import (
    PROCESSED_DIR,
    PROCESSED_DATASET_FILENAME,
    PROPHET_CHANGEPOINT_PRIOR_SCALE,
    PROPHET_DAILY_FOURIER_ORDER,
    PROPHET_FORECAST_HORIZON_STEPS,
    PROPHET_INTERVAL_WIDTH,
    PROPHET_MODEL_FILENAME,
    PROPHET_SEASONALITY_PRIOR_SCALE,
    PROPHET_TRAIN_FRACTION,
    PROPHET_WEEKLY_FOURIER_ORDER,
    TARGET_COLUMN,
)

logger = get_logger(__name__)

# Suppress Prophet and Stan's verbose output
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)


class ProphetTrainer:
    """
    Trains a Prophet model on the cluster CPU utilization time series.

    Parameters
    ----------
    processed_dir : Path
        Directory containing processed_dataset.csv.
    train_fraction : float
        Fraction of data to use for training (remainder is held out for eval).
    forecast_horizon_steps : int
        Number of 5-min steps to forecast beyond the training window.
    changepoint_prior_scale : float
        Flexibility of trend change-points (lower = smoother trend).
    seasonality_prior_scale : float
        Flexibility of seasonality components.
    daily_fourier_order : int
        Number of Fourier terms for daily seasonality.
    weekly_fourier_order : int
        Number of Fourier terms for weekly seasonality.
    interval_width : float
        Width of the uncertainty interval (e.g. 0.95 = 95%).
    """

    def __init__(
        self,
        processed_dir: Path = PROCESSED_DIR,
        train_fraction: float = PROPHET_TRAIN_FRACTION,
        forecast_horizon_steps: int = PROPHET_FORECAST_HORIZON_STEPS,
        changepoint_prior_scale: float = PROPHET_CHANGEPOINT_PRIOR_SCALE,
        seasonality_prior_scale: float = PROPHET_SEASONALITY_PRIOR_SCALE,
        daily_fourier_order: int = PROPHET_DAILY_FOURIER_ORDER,
        weekly_fourier_order: int = PROPHET_WEEKLY_FOURIER_ORDER,
        interval_width: float = PROPHET_INTERVAL_WIDTH,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.train_fraction = train_fraction
        self.forecast_horizon_steps = forecast_horizon_steps
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.daily_fourier_order = daily_fourier_order
        self.weekly_fourier_order = weekly_fourier_order
        self.interval_width = interval_width

        self._model = None          # Prophet model after fit
        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._prophet_df: Optional[pd.DataFrame] = None  # Full {ds, y}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> Tuple["Prophet", pd.DataFrame, pd.DataFrame]:  # noqa: F821
        """
        Full training pipeline.

        Returns
        -------
        model : Prophet
            Fitted Prophet model.
        forecast_df : pd.DataFrame
            Forecast over the full horizon (training + future steps).
        test_df : pd.DataFrame
            Held-out test data for evaluation.
        """
        logger.info("=" * 50)
        logger.info("PROPHET TRAINER STARTED")
        logger.info("=" * 50)

        processed_df = self._load_processed_data()
        prophet_df = self._prepare_prophet_df(processed_df)
        train_prophet_df, test_prophet_df = self._split(prophet_df)

        logger.info(
            "Train set: %d rows (%s → %s)",
            len(train_prophet_df),
            train_prophet_df["ds"].min(),
            train_prophet_df["ds"].max(),
        )
        logger.info(
            "Test set: %d rows (%s → %s)",
            len(test_prophet_df),
            test_prophet_df["ds"].min(),
            test_prophet_df["ds"].max(),
        )

        model = self._build_model()
        logger.info("Fitting Prophet model...")
        model.fit(train_prophet_df)
        logger.info("Prophet model fitted.")

        self._model = model
        self._train_df = train_prophet_df
        self._test_df = test_prophet_df
        self._prophet_df = prophet_df

        # Generate forecast: cover training period + test period + extra horizon
        total_steps = len(prophet_df) + self.forecast_horizon_steps
        future_df = self._make_future_df(model, train_prophet_df, total_steps)
        forecast_df = model.predict(future_df)

        logger.info(
            "Forecast generated: %d rows (%s → %s)",
            len(forecast_df),
            forecast_df["ds"].min(),
            forecast_df["ds"].max(),
        )

        self._save_model(model)

        return model, forecast_df, test_prophet_df

    def get_train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return the train/test split DataFrames (call after run())."""
        if self._train_df is None:
            raise RuntimeError("Call run() before get_train_test_split().")
        return self._train_df, self._test_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_processed_data(self) -> pd.DataFrame:
        """Load M1 output processed_dataset.csv."""
        path = self.processed_dir / PROCESSED_DATASET_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"processed_dataset.csv not found at: {path}\n"
                "Run Milestone 1 pipeline first: "
                "python3 -m milestone_01_dataset.pipeline"
            )
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        logger.info("Loaded processed_dataset.csv: %d rows × %d cols", *df.shape)
        return df

    def _prepare_prophet_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert processed DataFrame to Prophet format {ds, y}.

        Prophet requires:
        - 'ds': datetime column (timezone-naive)
        - 'y': numeric target column
        """
        prophet_df = pd.DataFrame({
            "ds": df.index.tz_localize(None),   # strip timezone (Prophet limitation)
            "y": df[TARGET_COLUMN].values,
        })
        prophet_df = prophet_df.dropna(subset=["y"]).reset_index(drop=True)
        logger.info(
            "Prophet DataFrame prepared: %d rows, y range [%.2f, %.2f]",
            len(prophet_df),
            prophet_df["y"].min(),
            prophet_df["y"].max(),
        )
        return prophet_df

    def _split(
        self, prophet_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Temporal train/test split (no shuffling — time order must be preserved)."""
        split_idx = int(len(prophet_df) * self.train_fraction)
        return prophet_df.iloc[:split_idx].copy(), prophet_df.iloc[split_idx:].copy()

    def _build_model(self) -> "Prophet":  # noqa: F821
        """Construct and configure the Prophet model."""
        # Import here to allow the module to load even if Prophet is not installed
        # (tests mock it out)
        from prophet import Prophet

        model = Prophet(
            growth="linear",
            seasonality_mode="additive",
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            interval_width=self.interval_width,
            daily_seasonality=False,    # we add custom below
            weekly_seasonality=False,   # we add custom below
            yearly_seasonality=False,   # only 8 days of data — would overfit
        )

        # Custom seasonalities with configured Fourier order
        model.add_seasonality(
            name="daily",
            period=1,           # 1 day
            fourier_order=self.daily_fourier_order,
            mode="additive",
        )
        model.add_seasonality(
            name="weekly",
            period=7,           # 7 days
            fourier_order=self.weekly_fourier_order,
            mode="additive",
        )

        logger.info(
            "Prophet model configured: daily_fourier=%d, weekly_fourier=%d, "
            "changepoint_prior=%.3f, seasonality_prior=%.1f",
            self.daily_fourier_order,
            self.weekly_fourier_order,
            self.changepoint_prior_scale,
            self.seasonality_prior_scale,
        )
        return model

    def _make_future_df(
        self,
        model: "Prophet",  # noqa: F821
        train_df: pd.DataFrame,
        total_steps: int,
    ) -> pd.DataFrame:
        """
        Create the future DataFrame for prediction.

        We build it manually from the training start to avoid Prophet's
        make_future_dataframe() adding business-day logic that doesn't apply
        to cluster traces.

        Parameters
        ----------
        model : Prophet
            Fitted model (used only to access the ds column).
        train_df : pd.DataFrame
            Training data (to get the start timestamp).
        total_steps : int
            Total number of 5-min steps to generate.
        """
        start = train_df["ds"].min()
        future_index = pd.date_range(
            start=start,
            periods=total_steps,
            freq="5min",
        )
        future_df = pd.DataFrame({"ds": future_index})
        return future_df

    def _save_model(self, model: "Prophet") -> None:  # noqa: F821
        """Persist the trained model to disk using pickle."""
        model_path = self.processed_dir / PROPHET_MODEL_FILENAME
        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)
        logger.info("Prophet model saved: %s", model_path)

    @staticmethod
    def load_model(model_path: Path) -> "Prophet":  # noqa: F821
        """Load a previously saved Prophet model from disk."""
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)
        logger.info("Prophet model loaded from: %s", model_path)
        return model
