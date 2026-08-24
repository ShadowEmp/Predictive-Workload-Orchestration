"""
research/phase12/prophet_eval.py
==================================
Phase 12-A: Baseline evaluation of the trained Prophet model.
Metrics: MAE, RMSE, MAPE, R²
Plots: actual vs forecast, residuals, error distribution, rolling error
"""

import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.logger import get_logger
from config.settings import (
    PROCESSED_DIR,
    FORECAST_DATASET_FILENAME,
    PROPHET_MODEL_FILENAME,
    RESULTS_BASELINE,
    PROPHET_TRAIN_FRACTION,
)
from research.experiment_tracker import log_experiment

logger = get_logger(__name__)
PLOTS_DIR = RESULTS_BASELINE / "prophet_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class ProphetEvaluator:

    def __init__(self):
        self.forecast_df = None
        self.train_df = None
        self.test_df = None
        self.metrics = {}

    def run(self) -> dict:
        logger.info("=== Phase 12-A: Prophet Baseline Evaluation ===")
        t0 = time.time()

        self._load_data()
        self._compute_metrics()
        self._plot_actual_vs_forecast()
        self._plot_residuals()
        self._plot_error_distribution()
        self._plot_rolling_error()
        self._save_metrics_csv()
        self._write_report()

        elapsed = time.time() - t0
        log_experiment(
            phase="12A", model="Prophet", dataset="Alibaba-v2018",
            hyperparameters={
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10.0,
                "seasonality_mode": "additive",
                "daily_fourier": 10,
                "weekly_fourier": 5,
                "train_fraction": PROPHET_TRAIN_FRACTION,
            },
            eval_time_s=elapsed,
            mae=self.metrics["mae"],
            rmse=self.metrics["rmse"],
            mape=self.metrics["mape"],
            r2=self.metrics["r2"],
            notes="Phase 12-A baseline Prophet evaluation",
        )
        logger.info("Phase 12-A complete. MAE=%.4f RMSE=%.4f MAPE=%.4f R2=%.4f",
                    self.metrics["mae"], self.metrics["rmse"],
                    self.metrics["mape"], self.metrics["r2"])
        return self.metrics

    # ------------------------------------------------------------------

    def _load_data(self):
        path = PROCESSED_DIR / FORECAST_DATASET_FILENAME
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        n = len(df)
        split = int(n * PROPHET_TRAIN_FRACTION)
        self.train_df = df.iloc[:split]
        self.test_df  = df.iloc[split:]
        self.forecast_df = df
        logger.info("Loaded forecast dataset: %d rows (train=%d test=%d)", n, split, n-split)

    def _compute_metrics(self):
        test = self.test_df
        actual    = test["cpu_util_percent"].values
        predicted = test["yhat"].values
        mask = ~np.isnan(predicted)
        actual, predicted = actual[mask], predicted[mask]

        mae  = float(mean_absolute_error(actual, predicted))
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        r2   = float(r2_score(actual, predicted))
        nonzero = actual != 0
        mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)

        self.metrics = {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2,
                        "n_test": int(mask.sum())}
        logger.info("Metrics — MAE:%.4f  RMSE:%.4f  MAPE:%.4f%%  R²:%.4f",
                    mae, rmse, mape, r2)

    def _plot_actual_vs_forecast(self):
        test = self.test_df.dropna(subset=["yhat"])
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(test.index, test["cpu_util_percent"], color="#1f77b4",
                lw=1.0, alpha=0.9, label="Actual CPU (%)")
        ax.plot(test.index, test["yhat"], color="#e74c3c",
                lw=1.0, alpha=0.9, label="Prophet Forecast (yhat)")
        if "yhat_lower" in test.columns:
            ax.fill_between(test.index, test["yhat_lower"], test["yhat_upper"],
                            alpha=0.15, color="#e74c3c", label="95% interval")
        ax.set_title("Phase 12-A — Actual vs Prophet Forecast (Test Period)",
                     fontsize=12)
        ax.set_xlabel("Timestamp"); ax.set_ylabel("CPU Utilization (%)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        self._save(fig, "actual_vs_forecast")

    def _plot_residuals(self):
        test = self.test_df.dropna(subset=["yhat"])
        residuals = test["cpu_util_percent"] - test["yhat"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(test.index, residuals, color="#2ecc71", lw=0.7, alpha=0.8)
        axes[0].axhline(0, color="black", ls="--", lw=0.8)
        axes[0].set_title("Residuals Over Time (Actual − Forecast)")
        axes[0].set_ylabel("Residual (%CPU)"); axes[0].grid(True, alpha=0.3)
        axes[1].hist(residuals, bins=40, color="#2ecc71", edgecolor="white", alpha=0.85)
        axes[1].axvline(0, color="black", ls="--", lw=0.8)
        axes[1].set_title("Residual Distribution")
        axes[1].set_xlabel("Residual (%CPU)"); axes[1].set_ylabel("Count")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, "residual_plot")

    def _plot_error_distribution(self):
        test = self.test_df.dropna(subset=["yhat"])
        abs_err = (test["cpu_util_percent"] - test["yhat"]).abs()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(abs_err, bins=40, color="#9b59b6", edgecolor="white", alpha=0.85,
                cumulative=False, density=True)
        ax.axvline(abs_err.mean(), color="#e74c3c", ls="--", lw=1.5,
                   label=f"Mean = {abs_err.mean():.2f}")
        ax.axvline(abs_err.median(), color="#f39c12", ls="--", lw=1.5,
                   label=f"Median = {abs_err.median():.2f}")
        ax.set_title("Prediction Error Distribution (|Actual − Forecast|)")
        ax.set_xlabel("Absolute Error (%CPU)"); ax.set_ylabel("Density")
        ax.legend(); ax.grid(True, alpha=0.3)
        self._save(fig, "error_distribution")

    def _plot_rolling_error(self):
        test = self.test_df.dropna(subset=["yhat"]).copy()
        test["abs_err"] = (test["cpu_util_percent"] - test["yhat"]).abs()
        test["rolling_mae"] = test["abs_err"].rolling(12, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(test.index, test["abs_err"], color="#bdc3c7", lw=0.5,
                alpha=0.6, label="Step error")
        ax.plot(test.index, test["rolling_mae"], color="#e74c3c", lw=1.5,
                label="Rolling MAE (1h window)")
        ax.set_title("Rolling Forecast Error (12-step = 1 hour window)")
        ax.set_xlabel("Timestamp"); ax.set_ylabel("Absolute Error (%CPU)")
        ax.legend(); ax.grid(True, alpha=0.3)
        self._save(fig, "rolling_error")

    def _save_metrics_csv(self):
        path = RESULTS_BASELINE / "prophet_metrics.csv"
        pd.DataFrame([self.metrics]).to_csv(path, index=False)
        logger.info("Prophet metrics saved: %s", path)

    def _write_report(self):
        m = self.metrics
        report = f"""# Prophet Baseline Evaluation Report

**Phase:** 12-A  
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  
**Dataset:** Alibaba Cluster Trace 2018  
**Model:** Facebook Prophet 1.1.5  

## Objective
Evaluate the baseline Prophet forecasting model on the held-out test set
(final 20% of the 8-day trace) without any hyperparameter modifications.

## Configuration
| Parameter | Value |
|-----------|-------|
| Seasonality mode | Additive |
| changepoint_prior_scale | 0.05 |
| seasonality_prior_scale | 10.0 |
| Daily Fourier order | 10 |
| Weekly Fourier order | 5 |
| Train fraction | 80% (1,814 rows) |
| Test fraction | 20% ({m['n_test']} rows) |

## Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | **{m['mae']:.4f} %CPU** | Mean absolute forecast error |
| **RMSE** | **{m['rmse']:.4f} %CPU** | Root mean squared error (penalises spikes) |
| **MAPE** | **{m['mape']:.4f} %** | Mean absolute percentage error |
| **R²** | **{m['r2']:.4f}** | Coefficient of determination |

## Analysis

**MAE = {m['mae']:.4f} %CPU** — On average the Prophet forecast deviates from actual
CPU utilization by {m['mae']:.2f} percentage points. For cloud auto-scaling where
SLA is set at 85% CPU, this error margin is {"acceptable" if m['mae'] < 10 else "large"}.

**RMSE = {m['rmse']:.4f} %CPU** — The higher RMSE compared to MAE indicates that
Prophet {"occasionally misses sudden workload spikes" if m['rmse'] > m['mae'] * 1.2 else "has relatively consistent errors"}.
This is expected for 5-minute resolution cloud traces.

**MAPE = {m['mape']:.4f}%** — {"Good" if m['mape'] < 15 else "Moderate"} percentage accuracy.
Values below 20% are generally acceptable for cloud workload forecasting.

**R² = {m['r2']:.4f}** — Prophet explains {"most" if m['r2'] > 0.7 else "some"} of the
variance in CPU utilization, capturing the daily and weekly seasonality patterns.

## Plots Generated
- `actual_vs_forecast.png` — Test period actual vs predicted CPU
- `residual_plot.png` — Residuals over time and distribution
- `error_distribution.png` — Absolute error density
- `rolling_error.png` — 1-hour rolling MAE over the test period

## Conclusion
The baseline Prophet model {"performs well" if m['r2'] > 0.6 else "shows limitations"} on
the Alibaba cluster trace. {"The seasonality patterns (daily business hours and weekly cycles) are captured effectively." if m['r2'] > 0.5 else "Improvements may be possible through hyperparameter tuning (Phase 13)."}
Hyperparameter tuning in Phase 13 will attempt to reduce MAE further.

## Limitations
- Only 8 days of training data — insufficient to capture yearly seasonality.
- Prophet cannot account for sudden unexpected load spikes.
- No external regressors used in this baseline.
"""
        (RESULTS_BASELINE / "prophet_evaluation_report.md").write_text(report)
        logger.info("prophet_evaluation_report.md written.")

    @staticmethod
    def _save(fig, name):
        fig.savefig(PLOTS_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Plot saved: %s.png", name)
