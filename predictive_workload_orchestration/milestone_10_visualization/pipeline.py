"""
milestone_10_visualization/pipeline.py
=========================================
Milestone 10 CLI — generates all publication-quality result figures.

Reads all output files produced by M1-M9 and generates 15 plots
saved to data/reports/final_plots/.

Usage:
    python -m milestone_10_visualization.pipeline
    python -m milestone_10_visualization.pipeline --no-training   (skip M4/M5 plots if logs missing)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logger import get_logger
from config.settings import (
    EXPERIMENT_RESULTS_DIR,
    EXPERIMENT_METRICS_FILENAME,
    CLOUDSIM_RESULTS_DIR,
    CLOUDSIM_METRICS_FILENAME,
    PROCESSED_DIR,
    FORECAST_DATASET_FILENAME,
    REPORTS_DIR,
    VIZ_PLOTS_DIR,
    DQN_TRAINING_LOG_FILENAME,
    HARS_LOG_FILENAME,
)
from milestone_10_visualization.plotter import ResultsPlotter

logger = get_logger(__name__)


class VisualizationPipeline:
    """
    Loads all results from previous milestones and generates
    publication-quality plots for the final report.
    """

    def __init__(self, skip_training_plots: bool = False) -> None:
        self.skip_training = skip_training_plots
        VIZ_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        self.plotter = ResultsPlotter(plots_dir=VIZ_PLOTS_DIR)
        self._plots_generated = []
        self._plots_skipped = []

    def run(self) -> None:
        logger.info("=" * 60)
        logger.info("MILESTONE 10 — VISUALIZATION PIPELINE STARTED")
        logger.info("  Output dir: %s", VIZ_PLOTS_DIR)
        logger.info("=" * 60)

        # --- Training plots (M4/M5) ---
        if not self.skip_training:
            self._plot_training()
            self._plot_hars()

        # --- Forecast accuracy (M2) ---
        self._plot_forecast()

        # --- Strategy comparison (M9) ---
        self._plot_experiments()

        # --- CloudSim infrastructure metrics (M8) ---
        self._plot_cloudsim()

        # --- Summary ---
        logger.info("=" * 60)
        logger.info("MILESTONE 10 COMPLETE")
        logger.info("  Generated : %d plots", len(self._plots_generated))
        if self._plots_skipped:
            logger.info(
                "  Skipped   : %d (missing input files) — run earlier milestones to generate",
                len(self._plots_skipped),
            )
        logger.info("  All plots : %s", VIZ_PLOTS_DIR)
        logger.info("=" * 60)

        # Print file list
        for p in sorted(VIZ_PLOTS_DIR.glob("*.png")):
            logger.info("  %s", p.name)

    # ------------------------------------------------------------------
    # Section: Training (M4/M5)
    # ------------------------------------------------------------------

    def _plot_training(self) -> None:
        path = REPORTS_DIR / DQN_TRAINING_LOG_FILENAME
        if not path.exists():
            logger.warning("Training log not found (%s) — skipping training plots.", path.name)
            self._plots_skipped += ["training_reward", "training_loss", "epsilon_curve"]
            return

        df = pd.read_csv(path)
        logger.info("Loaded training log: %d episodes.", len(df))

        self._try("training_reward",  lambda: self.plotter.plot_training_reward(df))
        self._try("training_loss",    lambda: self.plotter.plot_training_loss(df))
        self._try("epsilon_curve",    lambda: self.plotter.plot_epsilon_curve(df))

    # ------------------------------------------------------------------
    # Section: HARS (M5)
    # ------------------------------------------------------------------

    def _plot_hars(self) -> None:
        path = REPORTS_DIR / HARS_LOG_FILENAME
        if not path.exists():
            logger.warning("HARS log not found (%s) — skipping HARS plots.", path.name)
            self._plots_skipped += ["difficulty_score", "reward_weights"]
            return

        df = pd.read_csv(path)
        logger.info("Loaded HARS log: %d episodes.", len(df))

        self._try("difficulty_score", lambda: self.plotter.plot_difficulty_score(df))
        self._try("reward_weights",   lambda: self.plotter.plot_reward_weights(df))

    # ------------------------------------------------------------------
    # Section: Forecast (M2)
    # ------------------------------------------------------------------

    def _plot_forecast(self) -> None:
        path = PROCESSED_DIR / FORECAST_DATASET_FILENAME
        if not path.exists():
            logger.warning("forecast_dataset.csv not found — skipping forecast plot.")
            self._plots_skipped.append("forecast_accuracy")
            return

        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        logger.info("Loaded forecast dataset: %d rows.", len(df))
        self._try("forecast_accuracy", lambda: self.plotter.plot_forecast_accuracy(df))

    # ------------------------------------------------------------------
    # Section: Experiments (M9)
    # ------------------------------------------------------------------

    def _plot_experiments(self) -> None:
        path = EXPERIMENT_RESULTS_DIR / EXPERIMENT_METRICS_FILENAME
        if not path.exists():
            logger.warning(
                "experiment_results.csv not found — skipping strategy comparison plots."
            )
            self._plots_skipped += [
                "sla_violations", "cost_comparison", "energy_consumption",
                "vm_count", "scaling_timeline", "strategy_radar",
            ]
            return

        df = pd.read_csv(path)
        logger.info(
            "Loaded experiment results: %d rows, %d strategies.",
            len(df), df["strategy"].nunique(),
        )

        self._try("strategy_comparisons", lambda: self.plotter.plot_all_strategy_comparisons(df))
        self._try("strategy_radar",       lambda: self.plotter.plot_radar_chart(df))

    # ------------------------------------------------------------------
    # Section: CloudSim (M8)
    # ------------------------------------------------------------------

    def _plot_cloudsim(self) -> None:
        path = CLOUDSIM_RESULTS_DIR / CLOUDSIM_METRICS_FILENAME
        if not path.exists():
            logger.warning(
                "cloudsim_metrics.csv not found — skipping CloudSim plots."
            )
            self._plots_skipped.append("response_time")
            return

        df = pd.read_csv(path)
        logger.info("Loaded CloudSim metrics: %d steps.", len(df))
        self._try("response_time", lambda: self.plotter.plot_response_time(df))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _try(self, name: str, fn) -> None:
        """Run a plot function, catching and logging any errors."""
        try:
            fn()
            self._plots_generated.append(name)
        except Exception as exc:
            logger.error("Failed to generate '%s': %s", name, exc)
            self._plots_skipped.append(name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Milestone 10: Generate all publication-quality result plots"
    )
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Skip training/HARS plots (use if M4/M5 logs are not available).",
    )
    args = parser.parse_args()

    pipeline = VisualizationPipeline(skip_training_plots=args.no_training)
    pipeline.run()
