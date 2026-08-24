"""
milestone_01_dataset/eda.py
============================
Exploratory Data Analysis (EDA) module.

Produces:
1. A text-based EDA report (data/reports/eda_report.txt)
2. Multiple plots saved to data/reports/eda_plots/
3. A feature documentation Markdown file (data/reports/FEATURE_DOCUMENTATION.md)

Design notes:
- All plots use matplotlib with a non-interactive backend (Agg) so the module
  can run in headless environments (CI, server, Docker).
- The EDA report is machine-readable (structured sections) so it can be parsed
  by downstream validation scripts.
- We never call plt.show() — plots are always saved to disk.
"""

import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from config.logger import get_logger
from config.settings import (
    EDA_PLOTS_DIR,
    EDA_REPORT_FILENAME,
    REPORTS_DIR,
    TARGET_COLUMN,
)

logger = get_logger(__name__)

# Ensure EDA plots directory exists
EDA_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class EDAReport:
    """
    Generates EDA report and visualisations for the processed dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The fully processed dataset (output of FeatureEngineer.engineer()).
    reports_dir : Path
        Directory to write report files.
    plots_dir : Path
        Directory to write plot image files.
    target_col : str
        Primary signal column name.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        reports_dir: Path = REPORTS_DIR,
        plots_dir: Path = EDA_PLOTS_DIR,
        target_col: str = TARGET_COLUMN,
    ) -> None:
        self.df = df.copy()
        self.reports_dir = Path(reports_dir)
        self.plots_dir = Path(plots_dir)
        self.target_col = target_col
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full EDA pipeline."""
        logger.info("Starting EDA report generation.")

        self._write_text_report()
        self._plot_cpu_utilization_overview()
        self._plot_distribution()
        self._plot_daily_seasonality()
        self._plot_weekly_seasonality()
        self._plot_rolling_features()
        self._plot_lag_correlation()
        self._plot_load_intensity_distribution()
        self._plot_correlation_heatmap()
        self._write_feature_documentation()

        logger.info("EDA complete. Reports saved to: %s", self.reports_dir)

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------

    def _write_text_report(self) -> None:
        report_path = self.reports_dir / EDA_REPORT_FILENAME
        df = self.df

        lines = [
            "=" * 70,
            "EDA REPORT — Alibaba Cluster Trace (Processed Dataset)",
            "=" * 70,
            "",
            "[1] DATASET OVERVIEW",
            f"  Rows                : {len(df):,}",
            f"  Columns             : {len(df.columns)}",
            f"  Time range start    : {df.index.min()}",
            f"  Time range end      : {df.index.max()}",
            f"  Duration            : {df.index.max() - df.index.min()}",
            f"  Sampling frequency  : 5 minutes",
            "",
            "[2] TARGET COLUMN STATISTICS (cpu_util_percent)",
        ]

        if self.target_col in df.columns:
            s = df[self.target_col].describe()
            for stat, val in s.items():
                lines.append(f"  {stat:<20}: {val:.4f}")

        lines += [
            "",
            "[3] ALL COLUMN STATISTICS",
        ]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            s = df[col].describe()
            lines.append(f"  Column: {col}")
            for stat, val in s.items():
                lines.append(f"    {stat:<18}: {val:.4f}")
            lines.append("")

        lines += [
            "[4] MISSING VALUE REPORT",
        ]
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        if missing_cols.empty:
            lines.append("  No missing values detected. Dataset is clean.")
        else:
            for col, count in missing_cols.items():
                lines.append(f"  {col}: {count} ({count/len(df)*100:.2f}%)")

        lines += [
            "",
            "[5] LOAD INTENSITY DISTRIBUTION",
        ]
        if "load_intensity" in df.columns:
            dist = df["load_intensity"].value_counts(normalize=True) * 100
            for label, pct in dist.items():
                lines.append(f"  {label:<10}: {pct:.2f}%")

        lines += [
            "",
            "[6] TIMESTAMP CONTINUITY CHECK",
        ]
        if isinstance(df.index, pd.DatetimeIndex):
            diffs = pd.Series(df.index).diff().dropna()
            expected = pd.Timedelta("5min")
            gaps = diffs[diffs != expected]
            if gaps.empty:
                lines.append("  PASS — Timestamps are perfectly continuous.")
            else:
                lines.append(f"  WARNING — {len(gaps)} irregular intervals detected.")

        lines += ["", "=" * 70, "END OF REPORT", "=" * 70]

        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        logger.info("Text EDA report written: %s", report_path)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _plot_cpu_utilization_overview(self) -> None:
        """Full time-series plot of CPU utilization."""
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(self.df.index, self.df[self.target_col], linewidth=0.5, color="#1f77b4", alpha=0.8)
        ax.set_title("Cluster CPU Utilization — Full Trace (5-min intervals)")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("CPU Utilization (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        self._save_plot(fig, "cpu_utilization_overview.png")

    def _plot_distribution(self) -> None:
        """Histogram + KDE of CPU utilization."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        data = self.df[self.target_col].dropna()

        axes[0].hist(data, bins=50, color="#1f77b4", edgecolor="white", alpha=0.8)
        axes[0].set_title("CPU Utilization Distribution")
        axes[0].set_xlabel("CPU Utilization (%)")
        axes[0].set_ylabel("Frequency")

        axes[1].boxplot(data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#1f77b4", alpha=0.6))
        axes[1].set_title("CPU Utilization Box Plot")
        axes[1].set_ylabel("CPU Utilization (%)")

        plt.tight_layout()
        self._save_plot(fig, "cpu_distribution.png")

    def _plot_daily_seasonality(self) -> None:
        """Average CPU utilization by hour of day."""
        df = self.df.copy()
        df["hour"] = df.index.hour
        hourly = df.groupby("hour")[self.target_col].agg(["mean", "std"])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hourly.index, hourly["mean"], color="#1f77b4", marker="o", markersize=3)
        ax.fill_between(
            hourly.index,
            hourly["mean"] - hourly["std"],
            hourly["mean"] + hourly["std"],
            alpha=0.2,
            color="#1f77b4",
        )
        ax.set_title("Daily Seasonality — Average CPU Utilization by Hour of Day")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("CPU Utilization (%)")
        ax.set_xticks(range(0, 24))
        ax.grid(True, alpha=0.3)
        self._save_plot(fig, "daily_seasonality.png")

    def _plot_weekly_seasonality(self) -> None:
        """Average CPU utilization by day of week."""
        df = self.df.copy()
        df["dow"] = df.index.dayofweek
        daily = df.groupby("dow")[self.target_col].mean()
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(daily)), daily.values, color="#1f77b4", alpha=0.7)
        ax.set_xticks(range(7))
        ax.set_xticklabels(day_labels[:len(daily)])
        ax.set_title("Weekly Seasonality — Average CPU Utilization by Day of Week")
        ax.set_ylabel("CPU Utilization (%)")
        ax.grid(axis="y", alpha=0.3)
        self._save_plot(fig, "weekly_seasonality.png")

    def _plot_rolling_features(self) -> None:
        """Overlay of raw CPU vs rolling means."""
        fig, ax = plt.subplots(figsize=(16, 4))
        sample = self.df.iloc[:2016]  # 1 week

        ax.plot(sample.index, sample[self.target_col],
                linewidth=0.5, color="gray", alpha=0.6, label="Raw")

        from config.settings import ROLLING_WINDOWS
        colors = ["#e74c3c", "#2ecc71"]
        for win, color in zip(ROLLING_WINDOWS, colors):
            col = f"{self.target_col}_roll_mean_{win}"
            if col in sample.columns:
                ax.plot(sample.index, sample[col], linewidth=1.2,
                        color=color, label=f"Roll mean {win}×5min")

        ax.set_title("CPU Utilization vs Rolling Means (first 1 week)")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("CPU Utilization (%)")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        self._save_plot(fig, "rolling_features.png")

    def _plot_lag_correlation(self) -> None:
        """Autocorrelation plot of the target column."""
        from pandas.plotting import autocorrelation_plot
        fig, ax = plt.subplots(figsize=(12, 4))
        autocorrelation_plot(self.df[self.target_col].dropna(), ax=ax)
        ax.set_title("Autocorrelation — CPU Utilization")
        ax.set_xlim(0, 288 * 2)  # 2 days
        self._save_plot(fig, "autocorrelation.png")

    def _plot_load_intensity_distribution(self) -> None:
        """Pie chart of load intensity label distribution."""
        if "load_intensity" not in self.df.columns:
            return
        dist = self.df["load_intensity"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            dist.values,
            labels=dist.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"],
        )
        ax.set_title("Load Intensity Distribution")
        self._save_plot(fig, "load_intensity_distribution.png")

    def _plot_correlation_heatmap(self) -> None:
        """Correlation heatmap of all numeric features."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        # Limit to key features for readability
        key_cols = [c for c in numeric_df.columns if not c.startswith("lag")][:15]
        corr = numeric_df[key_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(key_cols)))
        ax.set_yticks(range(len(key_cols)))
        ax.set_xticklabels(key_cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(key_cols, fontsize=7)
        plt.colorbar(im, ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        plt.tight_layout()
        self._save_plot(fig, "correlation_heatmap.png")

    # ------------------------------------------------------------------
    # Feature documentation
    # ------------------------------------------------------------------

    def _write_feature_documentation(self) -> None:
        """Write a Markdown file documenting every feature column."""
        doc_path = self.reports_dir / "FEATURE_DOCUMENTATION.md"

        FEATURE_DESCRIPTIONS = {
            "cpu_util_percent": "Mean CPU utilization (%) across all cluster machines in the 5-min window.",
            "mem_util_percent": "Mean memory utilization (%) across all machines.",
            "net_in": "Mean inbound network bandwidth (MB/s).",
            "net_out": "Mean outbound network bandwidth (MB/s).",
            "disk_io_percent": "Mean disk I/O utilization (%).",
            "cpu_util_percent_roll_mean_12": "Rolling mean of CPU over 12 steps (1 hour).",
            "cpu_util_percent_roll_mean_36": "Rolling mean of CPU over 36 steps (3 hours).",
            "cpu_util_percent_roll_std_12": "Rolling std of CPU over 12 steps (volatility).",
            "cpu_util_percent_roll_std_36": "Rolling std of CPU over 36 steps.",
            "mem_util_percent_roll_mean_12": "Rolling mean of memory over 1 hour.",
            "mem_util_percent_roll_mean_36": "Rolling mean of memory over 3 hours.",
            "cpu_util_percent_lag_1": "CPU utilization 1 step (5 min) ago.",
            "cpu_util_percent_lag_3": "CPU utilization 3 steps (15 min) ago.",
            "cpu_util_percent_lag_6": "CPU utilization 6 steps (30 min) ago.",
            "cpu_util_percent_lag_12": "CPU utilization 12 steps (1 hour) ago.",
            "hour_sin": "Sine encoding of hour-of-day (cyclic, period=24h).",
            "hour_cos": "Cosine encoding of hour-of-day.",
            "dow_sin": "Sine encoding of day-of-week (cyclic, period=7d).",
            "dow_cos": "Cosine encoding of day-of-week.",
            "is_business_hours": "1 if 08:00–20:00, else 0.",
            "is_weekend": "1 if Saturday or Sunday, else 0.",
            "cpu_util_percent_trend": "First difference of CPU utilization (rate of change).",
            "cpu_util_percent_trend_smooth": "First difference of 1-hour rolling mean (smooth trend).",
            "load_intensity": "Categorical label: low / medium / high / critical.",
            "load_intensity_code": "Numeric encoding of load_intensity (0=low, 3=critical).",
        }

        lines = [
            "# Feature Documentation",
            "",
            "Generated automatically by `milestone_01_dataset/eda.py`.",
            "",
            "## Dataset",
            "- Source: Alibaba Cluster Trace 2018 (`machine_usage.csv`)",
            "- Aggregation: 5-minute intervals, mean across all machines",
            "- Rows after warm-up removal: dependent on trace length",
            "",
            "## Feature Catalogue",
            "",
            "| Feature | Type | Description |",
            "|---------|------|-------------|",
        ]

        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            desc = FEATURE_DESCRIPTIONS.get(col, "Engineered feature.")
            lines.append(f"| `{col}` | {dtype} | {desc} |")

        lines += [
            "",
            "## Notes",
            "- All percentage features are clipped to [0, 100].",
            "- Rolling and lag features use only past data (no lookahead).",
            "- Warm-up rows (first 36 steps) are removed before saving.",
            "- Cyclic time features use sin/cos to preserve neighborhood of hours/days.",
        ]

        with open(doc_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        logger.info("Feature documentation written: %s", doc_path)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _save_plot(self, fig: plt.Figure, filename: str) -> None:
        path = self.plots_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Plot saved: %s", path)
