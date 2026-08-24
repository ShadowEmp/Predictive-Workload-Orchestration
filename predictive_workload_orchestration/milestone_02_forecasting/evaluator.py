"""
milestone_02_forecasting/evaluator.py
=======================================
Evaluates Prophet forecast accuracy.

Responsibilities:
1. Align forecast with actual test data and compute MAE, RMSE, MAPE.
2. Run Prophet cross-validation (rolling origin evaluation).
3. Write a text accuracy report.
4. Generate 5 evaluation plots.

Design notes:
- Two evaluation modes:
    (a) Hold-out evaluation — fast, uses the 20% test split from training.
    (b) Cross-validation — slower, uses Prophet's built-in CV engine.
        CV is optional because on small datasets or during testing it can
        be slow. Use --skip-cv flag to skip it.
- MAPE formula: mean(|actual - forecast| / |actual|) × 100
  We guard against division by zero by dropping rows where actual = 0.
- All plots use the Agg backend for headless compatibility.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from config.logger import get_logger
from config.settings import (
    FORECAST_ACCURACY_REPORT_FILENAME,
    FORECAST_PLOTS_DIR,
    PROPHET_CV_HORIZON,
    PROPHET_CV_INITIAL,
    PROPHET_CV_PERIOD,
    REPORTS_DIR,
    TARGET_COLUMN,
)

logger = get_logger(__name__)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

FORECAST_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class ForecastEvaluator:
    """
    Evaluates Prophet forecast quality using hold-out metrics and cross-validation.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Output of Prophet.predict() — contains 'ds', 'yhat', 'yhat_lower', 'yhat_upper'.
    test_df : pd.DataFrame
        Hold-out test data in Prophet format: {'ds', 'y'}.
    model : Prophet
        Fitted Prophet model (needed for cross-validation).
    reports_dir : Path
        Where to write the accuracy report.
    plots_dir : Path
        Where to write evaluation plots.
    """

    def __init__(
        self,
        forecast_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model,
        reports_dir: Path = REPORTS_DIR,
        plots_dir: Path = FORECAST_PLOTS_DIR,
    ) -> None:
        self.forecast_df = forecast_df.copy()
        self.test_df = test_df.copy()
        self.model = model
        self.reports_dir = Path(reports_dir)
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._metrics: Dict[str, float] = {}
        self._aligned: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, run_cv: bool = True) -> Dict[str, float]:
        """
        Full evaluation pipeline.

        Parameters
        ----------
        run_cv : bool
            If True, run Prophet cross-validation (takes longer).

        Returns
        -------
        Dict[str, float]
            Dictionary of metric name → value.
        """
        logger.info("Starting forecast evaluation.")

        self._aligned = self._align_forecast_with_actuals()
        self._metrics = self._compute_metrics(self._aligned)

        self._log_metrics()

        if run_cv:
            logger.info("Running cross-validation (this may take a few minutes)...")
            cv_df = self._run_cross_validation()
            cv_metrics = self._compute_cv_metrics(cv_df)
            self._metrics.update({f"cv_{k}": v for k, v in cv_metrics.items()})
            self._plot_cv_metrics(cv_df)
        else:
            logger.info("Cross-validation skipped.")

        self._plot_actual_vs_forecast()
        self._plot_forecast_overview()
        self._plot_residuals()
        self._plot_components()
        self._write_accuracy_report(run_cv=run_cv)

        logger.info("Evaluation complete.")
        return self._metrics

    def get_aligned_df(self) -> pd.DataFrame:
        """Return the aligned (forecast vs actual) DataFrame."""
        if self._aligned is None:
            raise RuntimeError("Call run() first.")
        return self._aligned

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _align_forecast_with_actuals(self) -> pd.DataFrame:
        """
        Inner-join forecast_df with test_df on 'ds'.

        Both DataFrames must have a 'ds' column (timezone-naive datetime).
        """
        # Ensure ds is in the same format
        forecast = self.forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        forecast["ds"] = pd.to_datetime(forecast["ds"])
        test = self.test_df[["ds", "y"]].copy()
        test["ds"] = pd.to_datetime(test["ds"])

        aligned = pd.merge(test, forecast, on="ds", how="inner")
        logger.info(
            "Aligned %d rows for evaluation (test has %d, forecast has %d).",
            len(aligned), len(test), len(forecast),
        )

        if aligned.empty:
            raise ValueError(
                "No overlapping timestamps between forecast and test data. "
                "Check that the forecast horizon covers the test period."
            )
        return aligned

    @staticmethod
    def _compute_metrics(aligned: pd.DataFrame) -> Dict[str, float]:
        """Compute MAE, RMSE, MAPE on the aligned DataFrame."""
        actual = aligned["y"].values
        predicted = aligned["yhat"].values
        residuals = actual - predicted

        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        # MAPE: guard against zero actuals
        nonzero_mask = actual != 0
        if nonzero_mask.sum() == 0:
            mape = float("nan")
        else:
            mape = float(
                np.mean(np.abs(residuals[nonzero_mask] / actual[nonzero_mask])) * 100
            )

        return {"mae": mae, "rmse": rmse, "mape": mape}

    def _run_cross_validation(self) -> pd.DataFrame:
        """
        Run Prophet's built-in rolling-origin cross-validation.

        Returns the cross-validation DataFrame with columns:
        ds, yhat, yhat_lower, yhat_upper, y, cutoff
        """
        from prophet.diagnostics import cross_validation

        cv_df = cross_validation(
            self.model,
            initial=PROPHET_CV_INITIAL,
            period=PROPHET_CV_PERIOD,
            horizon=PROPHET_CV_HORIZON,
            parallel=None,  # sequential to avoid multiprocessing issues in some envs
        )
        logger.info(
            "Cross-validation complete: %d rows, %d unique cutoffs.",
            len(cv_df),
            cv_df["cutoff"].nunique(),
        )
        return cv_df

    @staticmethod
    def _compute_cv_metrics(cv_df: pd.DataFrame) -> Dict[str, float]:
        """Compute aggregate MAE, RMSE, MAPE across all CV cutpoints."""
        actual = cv_df["y"].values
        predicted = cv_df["yhat"].values
        residuals = actual - predicted

        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        nonzero_mask = actual != 0
        mape = (
            float(np.mean(np.abs(residuals[nonzero_mask] / actual[nonzero_mask])) * 100)
            if nonzero_mask.sum() > 0
            else float("nan")
        )

        logger.info("CV metrics — MAE: %.4f, RMSE: %.4f, MAPE: %.4f%%", mae, rmse, mape)
        return {"mae": mae, "rmse": rmse, "mape": mape}

    def _log_metrics(self) -> None:
        """Log the computed metrics."""
        logger.info("Hold-out evaluation metrics:")
        for name, value in self._metrics.items():
            logger.info("  %-10s : %.4f", name.upper(), value)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _write_accuracy_report(self, run_cv: bool) -> None:
        report_path = self.reports_dir / FORECAST_ACCURACY_REPORT_FILENAME
        lines = [
            "=" * 60,
            "FORECAST ACCURACY REPORT — Prophet CPU Utilization",
            "=" * 60,
            "",
            "[1] HOLD-OUT EVALUATION (last 20% of data)",
            f"  Evaluation rows : {len(self._aligned)}",
            f"  MAE             : {self._metrics.get('mae', float('nan')):.4f} %CPU",
            f"  RMSE            : {self._metrics.get('rmse', float('nan')):.4f} %CPU",
            f"  MAPE            : {self._metrics.get('mape', float('nan')):.4f} %",
            "",
        ]

        if run_cv:
            lines += [
                "[2] CROSS-VALIDATION (rolling origin)",
                f"  Initial window  : {PROPHET_CV_INITIAL}",
                f"  Period          : {PROPHET_CV_PERIOD}",
                f"  Horizon         : {PROPHET_CV_HORIZON}",
                f"  CV MAE          : {self._metrics.get('cv_mae', float('nan')):.4f} %CPU",
                f"  CV RMSE         : {self._metrics.get('cv_rmse', float('nan')):.4f} %CPU",
                f"  CV MAPE         : {self._metrics.get('cv_mape', float('nan')):.4f} %",
                "",
            ]
        else:
            lines += ["[2] CROSS-VALIDATION: skipped", ""]

        lines += [
            "[3] METRIC DEFINITIONS",
            "  MAE  = mean(|actual - forecast|)",
            "  RMSE = sqrt(mean((actual - forecast)^2))",
            "  MAPE = mean(|actual - forecast| / |actual|) * 100",
            "  Lower is better for all metrics.",
            "",
            "=" * 60,
        ]

        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        logger.info("Accuracy report written: %s", report_path)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _plot_actual_vs_forecast(self) -> None:
        """Actual vs forecast on the test period."""
        aligned = self._aligned
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(aligned["ds"], aligned["y"],
                color="#1f77b4", linewidth=1.0, label="Actual", alpha=0.9)
        ax.plot(aligned["ds"], aligned["yhat"],
                color="#e74c3c", linewidth=1.0, label="Forecast (yhat)", alpha=0.9)
        ax.fill_between(
            aligned["ds"],
            aligned["yhat_lower"],
            aligned["yhat_upper"],
            alpha=0.2, color="#e74c3c", label="95% interval",
        )
        ax.set_title("Actual vs Forecast — CPU Utilization (Test Period)")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("CPU Utilization (%)")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
        self._save_plot(fig, "actual_vs_forecast.png")

    def _plot_forecast_overview(self) -> None:
        """Full forecast including training period + future horizon."""
        from prophet.plot import plot as prophet_plot
        fig = prophet_plot(self.model, self.forecast_df, figsize=(16, 6))
        fig.axes[0].set_title("Prophet Forecast — Full Horizon")
        fig.axes[0].set_ylabel("CPU Utilization (%)")
        self._save_plot(fig, "forecast_overview.png")

    def _plot_residuals(self) -> None:
        """Residual plot: actual − forecast over the test period."""
        aligned = self._aligned
        residuals = aligned["y"] - aligned["yhat"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Time-series of residuals
        axes[0].plot(aligned["ds"], residuals, color="#2ecc71", linewidth=0.7, alpha=0.8)
        axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
        axes[0].set_title("Residuals Over Time (Actual − Forecast)")
        axes[0].set_ylabel("Residual (%CPU)")
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        # Residual histogram
        axes[1].hist(residuals, bins=40, color="#2ecc71", edgecolor="white", alpha=0.8)
        axes[1].axvline(0, color="black", linestyle="--", linewidth=0.8)
        axes[1].set_title("Residual Distribution")
        axes[1].set_xlabel("Residual (%CPU)")
        axes[1].set_ylabel("Frequency")

        plt.tight_layout()
        self._save_plot(fig, "residual_plot.png")

    def _plot_components(self) -> None:
        """Prophet component decomposition plot (trend + seasonalities)."""
        from prophet.plot import plot_components as prophet_plot_components
        fig = prophet_plot_components(self.model, self.forecast_df, figsize=(12, 8))
        fig.suptitle("Prophet Components — Trend and Seasonality", y=1.01)
        self._save_plot(fig, "forecast_components.png")

    def _plot_cv_metrics(self, cv_df: pd.DataFrame) -> None:
        """MAE and MAPE across CV horizon (how accuracy degrades with horizon)."""
        from prophet.diagnostics import performance_metrics

        perf = performance_metrics(cv_df)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # MAE vs horizon
        axes[0].plot(perf["horizon"].dt.total_seconds() / 3600,
                     perf["mae"], marker="o", markersize=3, color="#1f77b4")
        axes[0].set_title("MAE vs Horizon")
        axes[0].set_xlabel("Horizon (hours)")
        axes[0].set_ylabel("MAE (%CPU)")
        axes[0].grid(True, alpha=0.3)

        # RMSE vs horizon
        axes[1].plot(perf["horizon"].dt.total_seconds() / 3600,
                     perf["rmse"], marker="o", markersize=3, color="#e74c3c")
        axes[1].set_title("RMSE vs Horizon")
        axes[1].set_xlabel("Horizon (hours)")
        axes[1].set_ylabel("RMSE (%CPU)")
        axes[1].grid(True, alpha=0.3)

        # MAPE vs horizon (if available)
        if "mape" in perf.columns:
            axes[2].plot(perf["horizon"].dt.total_seconds() / 3600,
                         perf["mape"] * 100, marker="o", markersize=3, color="#2ecc71")
            axes[2].set_title("MAPE vs Horizon")
            axes[2].set_xlabel("Horizon (hours)")
            axes[2].set_ylabel("MAPE (%)")
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_plot(fig, "cv_metrics.png")

    def _save_plot(self, fig: plt.Figure, filename: str) -> None:
        path = self.plots_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Plot saved: %s", path)
