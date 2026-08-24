"""
milestone_10_visualization/plotter.py
=======================================
ResultsPlotter: generates all publication-quality figures.

Plots produced:
  1.  training_reward.png       — DQN episode reward curve
  2.  training_loss.png         — DQN training loss curve
  3.  epsilon_curve.png         — Exploration epsilon decay
  4.  difficulty_score.png      — HARS difficulty score over episodes
  5.  reward_weights.png        — HARS reward weight evolution
  6.  cpu_utilization.png       — CPU utilization comparison across strategies
  7.  vm_count.png              — VM count over time per strategy
  8.  queue_length.png          — Queue length over time
  9.  energy_consumption.png    — Energy consumption comparison
  10. cost_comparison.png       — Cumulative cost comparison
  11. sla_violations.png        — SLA violation rate per strategy
  12. response_time.png         — Response time distribution
  13. forecast_accuracy.png     — Prophet actual vs forecast
  14. scaling_timeline.png      — Scaling actions timeline (replay)
  15. strategy_radar.png        — Radar chart: all metrics per strategy
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

from config.logger import get_logger
from config.settings import VIZ_DPI, VIZ_FIGSIZE_WIDE, VIZ_FIGSIZE_SQUARE, VIZ_PLOTS_DIR

logger = get_logger(__name__)

STRATEGY_COLORS = {
    "static":           "#95a5a6",
    "threshold":        "#3498db",
    "prophet":          "#2ecc71",
    "dqn":              "#e74c3c",
    "prophet_dqn":      "#9b59b6",
    "prophet_dqn_hars": "#f39c12",
}

VIZ_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class ResultsPlotter:
    """
    Generates all final result plots.

    Parameters
    ----------
    plots_dir : Path
        Directory where plots are saved.
    dpi : int
        Plot resolution.
    """

    def __init__(
        self,
        plots_dir: Path = VIZ_PLOTS_DIR,
        dpi: int = VIZ_DPI,
    ) -> None:
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    # ------------------------------------------------------------------
    # Training plots (M4)
    # ------------------------------------------------------------------

    def plot_training_reward(self, training_log: pd.DataFrame) -> None:
        """Episode reward curve with smoothed overlay."""
        fig, ax = plt.subplots(figsize=VIZ_FIGSIZE_WIDE)
        raw = training_log["episode_reward"].values
        ax.plot(raw, alpha=0.3, color="#e74c3c", linewidth=0.8, label="Raw")

        # Smoothed (rolling mean window=20)
        if len(raw) >= 20:
            smoothed = pd.Series(raw).rolling(20).mean()
            ax.plot(smoothed, color="#e74c3c", linewidth=2.0, label="Smoothed (20-ep)")

        ax.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_title("DQN Training — Episode Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Episode Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save(fig, "training_reward.png")

    def plot_training_loss(self, training_log: pd.DataFrame) -> None:
        """Mean training loss per episode."""
        fig, ax = plt.subplots(figsize=VIZ_FIGSIZE_WIDE)
        loss = training_log["mean_loss"].replace(0, np.nan).dropna()
        ax.semilogy(loss.index, loss.values, color="#3498db", linewidth=0.8, alpha=0.7)
        ax.set_title("DQN Training — Mean Loss (log scale)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("MSE Loss (log scale)")
        ax.grid(True, alpha=0.3, which="both")
        self._save(fig, "training_loss.png")

    def plot_epsilon_curve(self, training_log: pd.DataFrame) -> None:
        """Epsilon decay over training episodes."""
        fig, ax = plt.subplots(figsize=VIZ_FIGSIZE_WIDE)
        ax.plot(training_log["epsilon"], color="#2ecc71", linewidth=1.5)
        ax.set_title("Exploration — Epsilon Decay")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon (exploration probability)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        self._save(fig, "epsilon_curve.png")

    # ------------------------------------------------------------------
    # HARS plots (M5)
    # ------------------------------------------------------------------

    def plot_difficulty_score(self, hars_log: pd.DataFrame) -> None:
        """HARS difficulty score evolution."""
        fig, ax = plt.subplots(figsize=VIZ_FIGSIZE_WIDE)
        ax.plot(hars_log["episode"], hars_log["difficulty"],
                color="#f39c12", linewidth=1.5, label="Difficulty Score")
        ax.fill_between(hars_log["episode"], 0, hars_log["difficulty"],
                        alpha=0.15, color="#f39c12")
        ax.set_ylim(0, 1.05)
        ax.set_title("HARS — Difficulty Score Evolution")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Difficulty Score")
        ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5, linewidth=0.8)
        ax.grid(True, alpha=0.3)
        self._save(fig, "difficulty_score.png")

    def plot_reward_weights(self, hars_log: pd.DataFrame) -> None:
        """Stacked area plot of reward weight evolution."""
        fig, ax = plt.subplots(figsize=VIZ_FIGSIZE_WIDE)
        weight_cols = [c for c in hars_log.columns if c.startswith("w_")]
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
        labels = [c.replace("w_", "") for c in weight_cols]

        data = [hars_log[c].values for c in weight_cols]
        ax.stackplot(hars_log["episode"], data, labels=labels,
                     colors=colors[:len(weight_cols)], alpha=0.8)
        ax.set_title("HARS — Reward Weight Evolution")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        self._save(fig, "reward_weights.png")

    # ------------------------------------------------------------------
    # Strategy comparison plots (M9)
    # ------------------------------------------------------------------

    def plot_strategy_comparison(
        self,
        results_df: pd.DataFrame,
        metric: str,
        title: str,
        ylabel: str,
        filename: str,
    ) -> None:
        """Generic bar chart comparing strategies on one metric."""
        agg = results_df.groupby("strategy")[metric].mean().reset_index()
        strategies = agg["strategy"].tolist()
        values     = agg[metric].tolist()
        colors     = [STRATEGY_COLORS.get(s, "#7f8c8d") for s in strategies]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(strategies, values, color=colors, edgecolor="white", alpha=0.9)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Strategy")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        self._save(fig, filename)

    def plot_all_strategy_comparisons(self, results_df: pd.DataFrame) -> None:
        """Generate all strategy comparison bar charts."""
        comparisons = [
            ("episode_reward",      "Episode Reward by Strategy",         "Mean Episode Reward",     "cpu_utilization.png"),
            ("sla_violation_rate",  "SLA Violation Rate by Strategy",      "Mean SLA Violation Rate", "sla_violations.png"),
            ("estimated_cost",      "Estimated Cost by Strategy (USD)",    "Mean Cost (USD)",         "cost_comparison.png"),
            ("estimated_energy_kwh","Energy Consumption by Strategy (kWh)","Mean Energy (kWh)",       "energy_consumption.png"),
            ("scale_events",        "Scaling Frequency by Strategy",       "Mean Scale Events",       "scaling_timeline.png"),
            ("mean_vm_count",       "Mean VM Count by Strategy",           "Mean VM Count",           "vm_count.png"),
            ("final_queue_length",  "Queue Length by Strategy",            "Mean Queue Length",       "queue_length.png"),
        ]
        for metric, title, ylabel, filename in comparisons:
            if metric in results_df.columns:
                self.plot_strategy_comparison(results_df, metric, title, ylabel, filename)

    def plot_radar_chart(self, results_df: pd.DataFrame) -> None:
        """Radar / spider chart comparing strategies on 5 normalised metrics."""
        metrics = [
            "episode_reward", "sla_violation_rate",
            "estimated_cost", "scale_events", "mean_vm_count",
        ]
        available = [m for m in metrics if m in results_df.columns]
        if len(available) < 3:
            logger.warning("Not enough metrics for radar chart.")
            return

        agg = results_df.groupby("strategy")[available].mean()

        # Normalise each metric to [0,1] (invert cost/sla/scale — lower is better)
        normed = agg.copy()
        invert = {"sla_violation_rate", "estimated_cost", "scale_events"}
        for col in available:
            col_min, col_max = normed[col].min(), normed[col].max()
            rng = col_max - col_min
            if rng > 0:
                normed[col] = (normed[col] - col_min) / rng
                if col in invert:
                    normed[col] = 1 - normed[col]
            else:
                normed[col] = 0.5

        n = len(available)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        for strategy, row in normed.iterrows():
            values = row[available].tolist()
            values += values[:1]
            color = STRATEGY_COLORS.get(strategy, "#7f8c8d")
            ax.plot(angles, values, color=color, linewidth=2, label=strategy)
            ax.fill(angles, values, color=color, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", "\n") for m in available], size=8)
        ax.set_ylim(0, 1)
        ax.set_title("Strategy Comparison — Normalised Metrics", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        self._save(fig, "strategy_radar.png")

    # ------------------------------------------------------------------
    # Forecast accuracy plot (M2)
    # ------------------------------------------------------------------

    def plot_forecast_accuracy(
        self, forecast_df: pd.DataFrame, n_samples: int = 500
    ) -> None:
        """Actual vs Prophet forecast on a sample window."""
        if "yhat" not in forecast_df.columns or "cpu_util_percent" not in forecast_df.columns:
            logger.warning("forecast_df missing yhat or cpu_util_percent. Skipping plot.")
            return
        sample = forecast_df.dropna(subset=["yhat"]).tail(n_samples)
        fig, ax = plt.subplots(figsize=VIZ_FIGSIZE_WIDE)
        ax.plot(range(len(sample)), sample["cpu_util_percent"].values,
                color="#1f77b4", linewidth=1.0, alpha=0.9, label="Actual")
        ax.plot(range(len(sample)), sample["yhat"].values,
                color="#e74c3c", linewidth=1.0, alpha=0.9, label="Prophet yhat")
        if "yhat_lower" in sample.columns:
            ax.fill_between(range(len(sample)),
                            sample["yhat_lower"], sample["yhat_upper"],
                            alpha=0.15, color="#e74c3c")
        ax.set_title("Prophet Forecast Accuracy (last 500 steps)")
        ax.set_xlabel("Step")
        ax.set_ylabel("CPU Utilization (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save(fig, "forecast_accuracy.png")

    def plot_response_time(self, cloudsim_df: pd.DataFrame) -> None:
        """Response time distribution from CloudSim metrics."""
        if "response_time_ms" not in cloudsim_df.columns:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        rt = cloudsim_df["response_time_ms"].dropna()
        axes[0].plot(rt.values, color="#9b59b6", linewidth=0.8, alpha=0.8)
        axes[0].set_title("Response Time Over Time")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Response Time (ms)")
        axes[1].hist(rt, bins=40, color="#9b59b6", edgecolor="white", alpha=0.8)
        axes[1].set_title("Response Time Distribution")
        axes[1].set_xlabel("Response Time (ms)")
        axes[1].set_ylabel("Frequency")
        plt.tight_layout()
        self._save(fig, "response_time.png")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = self.plots_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Plot saved: %s", path)
