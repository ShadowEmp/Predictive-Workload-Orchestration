"""
research/phase16/visualization.py — Phase 16: Publication-quality plots in PNG/PDF/SVG.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from config.logger import get_logger
from config.settings import (
    RESULTS_VIZ, VIZ_FORMATS, VIZ_DPI,
    REPORTS_DIR, RESULTS_ABLATION, RESULTS_STATS,
    CLOUDSIM_RESULTS_DIR, CLOUDSIM_METRICS_FILENAME,
    PROCESSED_DIR, FORECAST_DATASET_FILENAME,
    HARS_LOG_FILENAME, DQN_TRAINING_LOG_FILENAME,
    EXPERIMENT_RESULTS_DIR, EXPERIMENT_METRICS_FILENAME,
)

logger = get_logger(__name__)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": VIZ_DPI,
    "axes.spines.top": False, "axes.spines.right": False,
})

STRATEGY_COLORS = {
    "static":"#95a5a6","threshold":"#3498db","prophet":"#2ecc71",
    "dqn":"#e74c3c","prophet_dqn":"#9b59b6","prophet_dqn_hars":"#f39c12",
}


def _save(fig, name: str, outdir: Path = RESULTS_VIZ):
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in VIZ_FORMATS:
        p = outdir / f"{name}.{fmt}"
        fig.savefig(p, dpi=VIZ_DPI, bbox_inches="tight",
                    format=fmt, transparent=(fmt == "svg"))
    plt.close(fig)
    logger.info("Saved: %s [%s]", name, ", ".join(VIZ_FORMATS))


class PublicationVisualizer:

    def run(self):
        logger.info("=== Phase 16: Publication-Quality Visualization ===")
        self._training_dashboard()
        self._hars_dashboard()
        self._forecast_dashboard()
        self._cloudsim_dashboard()
        self._comparison_dashboard()
        self._statistical_dashboard()
        logger.info("Phase 16 complete. Plots in: %s", RESULTS_VIZ)

    # ------------------------------------------------------------------
    # Training (M4/M5)
    # ------------------------------------------------------------------
    def _training_dashboard(self):
        path = REPORTS_DIR / DQN_TRAINING_LOG_FILENAME
        if not path.exists(): return
        df = pd.read_csv(path)
        rewards = df["episode_reward"].values
        rolling = pd.Series(rewards).rolling(20, min_periods=1).mean()

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

        # Reward
        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(df["episode"], rewards, color="#bdc3c7", lw=0.5, alpha=0.5, label="Raw reward")
        ax0.plot(df["episode"], rolling, color="#e74c3c", lw=2.0, label="Rolling mean (20 ep)")
        ax0.axhline(0, color="black", ls="--", lw=0.7, alpha=0.4)
        ax0.fill_between(df["episode"], 0, rolling, where=rolling>0, alpha=0.1, color="#2ecc71")
        ax0.fill_between(df["episode"], 0, rolling, where=rolling<0, alpha=0.1, color="#e74c3c")
        ax0.set_xlabel("Episode"); ax0.set_ylabel("Reward")
        ax0.set_title("DQN Training — Episode Reward with 20-Episode Rolling Mean")
        ax0.legend(); ax0.grid(True, alpha=0.25)

        # Loss
        ax1 = fig.add_subplot(gs[1, 0])
        loss = df["mean_loss"].replace(0, np.nan)
        ax1.semilogy(df["episode"], loss, color="#3498db", lw=0.9, alpha=0.85)
        ax1.set_xlabel("Episode"); ax1.set_ylabel("MSE Loss (log scale)")
        ax1.set_title("Training Loss"); ax1.grid(True, alpha=0.25, which="both")

        # Epsilon
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(df["episode"], df["epsilon"], color="#2ecc71", lw=1.5)
        ax2.set_xlabel("Episode"); ax2.set_ylabel("Epsilon")
        ax2.set_title("Exploration — Epsilon Decay"); ax2.grid(True, alpha=0.25)

        # SLA violations
        ax3 = fig.add_subplot(gs[2, 0])
        sla_roll = pd.Series(df["sla_violations"]).rolling(20, min_periods=1).mean()
        ax3.plot(df["episode"], sla_roll, color="#f39c12", lw=1.5)
        ax3.set_xlabel("Episode"); ax3.set_ylabel("SLA Violations (rolling)")
        ax3.set_title("SLA Violations per Episode"); ax3.grid(True, alpha=0.25)

        # Scale events
        ax4 = fig.add_subplot(gs[2, 1])
        scale_roll = pd.Series(df["scale_events"]).rolling(20, min_periods=1).mean()
        ax4.plot(df["episode"], scale_roll, color="#9b59b6", lw=1.5)
        ax4.set_xlabel("Episode"); ax4.set_ylabel("Scale Events (rolling)")
        ax4.set_title("Scaling Activity per Episode"); ax4.grid(True, alpha=0.25)

        fig.suptitle("DQN Training Summary Dashboard", fontsize=14, fontweight="bold")
        _save(fig, "training_dashboard")

    # ------------------------------------------------------------------
    # HARS
    # ------------------------------------------------------------------
    def _hars_dashboard(self):
        path = REPORTS_DIR / HARS_LOG_FILENAME
        if not path.exists(): return
        df = pd.read_csv(path)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # Difficulty
        axes[0,0].plot(df["episode"], df["difficulty"], color="#f39c12", lw=1.5)
        axes[0,0].fill_between(df["episode"], 0, df["difficulty"], alpha=0.15, color="#f39c12")
        axes[0,0].axhline(0.5, ls="--", color="gray", lw=0.8)
        axes[0,0].set_title("HARS Difficulty Score Evolution")
        axes[0,0].set_ylabel("Difficulty [0–1]"); axes[0,0].set_ylim(0, 1.05)
        axes[0,0].grid(True, alpha=0.25)

        # Stacked weights
        cols = ["w_util","w_sla","w_cost","w_scale"]
        colors_w = ["#3498db","#e74c3c","#2ecc71","#9b59b6"]
        labels_w = ["Util","SLA","Cost","Scale"]
        axes[0,1].stackplot(df["episode"],
                            [df[c] for c in cols],
                            labels=labels_w, colors=colors_w, alpha=0.8)
        axes[0,1].set_title("Reward Weight Evolution (Stacked)")
        axes[0,1].set_ylabel("Weight"); axes[0,1].legend(loc="upper right", fontsize=8)
        axes[0,1].grid(True, alpha=0.25)

        # Individual weight lines
        for col, c, lbl in zip(cols, colors_w, labels_w):
            axes[1,0].plot(df["episode"], df[col], color=c, lw=1.5, label=lbl)
        axes[1,0].set_title("Individual Reward Weight Trajectories")
        axes[1,0].set_ylabel("Weight"); axes[1,0].legend(fontsize=8)
        axes[1,0].grid(True, alpha=0.25)

        # Performance vs difficulty
        ax = axes[1,1]
        sc = ax.scatter(df["difficulty"], df["mean_reward"],
                        c=df["episode"], cmap="viridis", s=20, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="Episode")
        ax.set_xlabel("Difficulty Score"); ax.set_ylabel("Mean Reward")
        ax.set_title("Difficulty vs Mean Reward"); ax.grid(True, alpha=0.25)

        fig.suptitle("HARS System Dashboard — Phase 12-C / Phase 16", fontsize=14)
        plt.tight_layout()
        _save(fig, "hars_dashboard")

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------
    def _forecast_dashboard(self):
        path = PROCESSED_DIR / FORECAST_DATASET_FILENAME
        if not path.exists(): return
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        test = df.tail(454).dropna(subset=["yhat"])

        fig, axes = plt.subplots(2, 2, figsize=(16, 9))

        axes[0,0].plot(range(len(test)), test["cpu_util_percent"].values,
                       color="#1f77b4", lw=1.0, label="Actual", alpha=0.9)
        axes[0,0].plot(range(len(test)), test["yhat"].values,
                       color="#e74c3c", lw=1.0, label="Forecast", alpha=0.9)
        axes[0,0].fill_between(range(len(test)),
                               test["yhat_lower"].values, test["yhat_upper"].values,
                               alpha=0.15, color="#e74c3c", label="95% CI")
        axes[0,0].set_title("Actual vs Prophet Forecast (Test Period)")
        axes[0,0].set_xlabel("Step"); axes[0,0].set_ylabel("CPU Util (%)")
        axes[0,0].legend(fontsize=9); axes[0,0].grid(True, alpha=0.25)

        residuals = test["cpu_util_percent"] - test["yhat"]
        axes[0,1].plot(range(len(residuals)), residuals.values,
                       color="#2ecc71", lw=0.7, alpha=0.8)
        axes[0,1].axhline(0, color="black", ls="--", lw=0.8)
        axes[0,1].set_title("Forecast Residuals")
        axes[0,1].set_xlabel("Step"); axes[0,1].set_ylabel("Residual (%CPU)")
        axes[0,1].grid(True, alpha=0.25)

        abs_err = residuals.abs()
        axes[1,0].hist(abs_err.values, bins=40, color="#9b59b6",
                       edgecolor="white", alpha=0.85, density=True)
        axes[1,0].axvline(abs_err.mean(), color="#e74c3c", ls="--", lw=1.5,
                          label=f"μ={abs_err.mean():.2f}")
        axes[1,0].set_title("Prediction Error Distribution")
        axes[1,0].set_xlabel("Absolute Error"); axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.25)

        rolling_err = abs_err.rolling(12, min_periods=1).mean()
        axes[1,1].plot(range(len(rolling_err)), rolling_err.values,
                       color="#e74c3c", lw=1.5)
        axes[1,1].set_title("Rolling MAE (12-step = 1 hour)")
        axes[1,1].set_xlabel("Step"); axes[1,1].set_ylabel("Rolling MAE")
        axes[1,1].grid(True, alpha=0.25)

        fig.suptitle("Prophet Forecast Evaluation Dashboard", fontsize=14)
        plt.tight_layout()
        _save(fig, "forecast_dashboard")

    # ------------------------------------------------------------------
    # CloudSim
    # ------------------------------------------------------------------
    def _cloudsim_dashboard(self):
        path = CLOUDSIM_RESULTS_DIR / CLOUDSIM_METRICS_FILENAME
        if not path.exists(): return
        df = pd.read_csv(path)
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        axes[0,0].plot(df["step"], df["cpu_util"], color="#1f77b4", lw=0.9, alpha=0.85)
        axes[0,0].axhline(85, color="#e74c3c", ls="--", lw=0.8, label="SLA 85%")
        axes[0,0].set_title("CPU Utilization"); axes[0,0].set_ylabel("CPU Util (%)")
        axes[0,0].legend(fontsize=9); axes[0,0].grid(True, alpha=0.25)

        axes[0,1].plot(df["step"], df["memory_util"], color="#9b59b6", lw=0.9)
        axes[0,1].set_title("Memory Utilization"); axes[0,1].set_ylabel("Mem Util (%)")
        axes[0,1].grid(True, alpha=0.25)

        axes[1,0].step(df["step"], df["vm_count"], color="#2ecc71", lw=1.5, where="post")
        axes[1,0].set_title("VM Count Over Time"); axes[1,0].set_ylabel("VMs")
        axes[1,0].grid(True, alpha=0.25)

        axes[1,1].plot(df["step"], df["queue_length"], color="#f39c12", lw=0.9)
        axes[1,1].set_title("Queue Length"); axes[1,1].set_ylabel("Queue")
        axes[1,1].grid(True, alpha=0.25)

        axes[2,0].plot(df["step"], df["cumulative_energy"], color="#e74c3c", lw=1.5)
        ax2r = axes[2,0].twinx()
        ax2r.plot(df["step"], df["cumulative_cost"], color="#f39c12", lw=1.5, ls="--")
        axes[2,0].set_ylabel("Cumulative Energy (kWh)", color="#e74c3c")
        ax2r.set_ylabel("Cumulative Cost (USD)", color="#f39c12")
        axes[2,0].set_title("Energy & Cost Accumulation"); axes[2,0].grid(True, alpha=0.25)

        axes[2,1].plot(df["step"], df["response_time_ms"], color="#3498db", lw=0.9, alpha=0.85)
        axes[2,1].axhline(df["response_time_ms"].mean(), color="#e74c3c",
                          ls="--", lw=1.2, label=f"Mean={df['response_time_ms'].mean():.1f}ms")
        axes[2,1].set_title("Response Time"); axes[2,1].set_ylabel("RT (ms)")
        axes[2,1].legend(fontsize=9); axes[2,1].grid(True, alpha=0.25)

        for ax in axes.flat:
            ax.set_xlabel("Step (5-min intervals)")
        fig.suptitle("CloudSim Plus Digital Twin — Infrastructure Metrics Dashboard",
                     fontsize=14)
        plt.tight_layout()
        _save(fig, "cloudsim_dashboard")

    # ------------------------------------------------------------------
    # Comparison (M9 / Ablation)
    # ------------------------------------------------------------------
    def _comparison_dashboard(self):
        p1 = EXPERIMENT_RESULTS_DIR / EXPERIMENT_METRICS_FILENAME
        p2 = RESULTS_ABLATION / "ablation_results.csv"
        path = p1 if p1.exists() else (p2 if p2.exists() else None)
        if path is None: return
        df = pd.read_csv(path)
        agg = df.groupby("strategy")[["episode_reward","sla_violation_rate",
                                       "estimated_cost","estimated_energy_kwh",
                                       "mean_vm_count","scale_events"]].mean()
        colors = [STRATEGY_COLORS.get(s,"#7f8c8d") for s in agg.index]

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        specs = [
            ("episode_reward","Episode Reward","Reward"),
            ("sla_violation_rate","SLA Violation Rate","Rate"),
            ("estimated_cost","Estimated Cost (USD)","USD"),
            ("estimated_energy_kwh","Energy Consumption (kWh)","kWh"),
            ("mean_vm_count","Mean VM Count","VMs"),
            ("scale_events","Scale Events","Count"),
        ]
        for ax, (col, title, ylabel) in zip(axes.flat, specs):
            bars = ax.bar(range(len(agg)), agg[col].values, color=colors,
                          edgecolor="white", alpha=0.9)
            ax.set_title(title); ax.set_ylabel(ylabel)
            ax.set_xticks(range(len(agg)))
            ax.set_xticklabels(agg.index, rotation=20, ha="right", fontsize=8)
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
            ax.grid(axis="y", alpha=0.25)

        fig.suptitle("Strategy Comparison Dashboard — All Metrics", fontsize=14)
        plt.tight_layout()
        _save(fig, "comparison_dashboard")

    # ------------------------------------------------------------------
    # Statistical
    # ------------------------------------------------------------------
    def _statistical_dashboard(self):
        path = RESULTS_STATS / "statistical_raw.csv"
        if not path.exists(): return
        df = pd.read_csv(path)
        strategies = df["strategy"].unique().tolist()
        metrics = ["episode_reward","sla_violation_rate","estimated_cost","estimated_energy_kwh"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        COLORS = {"static":"#95a5a6","dqn":"#e74c3c","prophet_dqn_hars":"#f39c12"}
        for ax, metric in zip(axes.flat, metrics):
            data = [df[df["strategy"]==s][metric].values for s in strategies]
            bp = ax.boxplot(data, patch_artist=True)
            for patch, s in zip(bp["boxes"], strategies):
                patch.set_facecolor(COLORS.get(s,"#7f8c8d")); patch.set_alpha(0.75)
            ax.set_xticks(range(1, len(strategies)+1))
            ax.set_xticklabels(strategies, rotation=15, ha="right", fontsize=9)
            ax.set_title(metric.replace("_"," ").title()); ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"Statistical Validation — Boxplots ({len(df['seed'].unique())} seeds)",
                     fontsize=13)
        plt.tight_layout()
        _save(fig, "statistical_dashboard")
