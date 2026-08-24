"""
milestone_07_policy_replay/pipeline.py
========================================
Milestone 7 CLI — replays the exported policy through the environment,
logs every action, and produces analysis plots.

Usage:
    python -m milestone_07_policy_replay.pipeline
    python -m milestone_07_policy_replay.pipeline --episodes 3
    python -m milestone_07_policy_replay.pipeline --seed 42 --no-plots
    python -m milestone_07_policy_replay.pipeline --policy-dir data/models
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logger import get_logger
from config.settings import (
    POLICY_EXPORT_DIR,
    PROCESSED_DIR,
    REPLAY_LOG_FILENAME,
    REPLAY_PLOTS_DIR,
    REPORTS_DIR,
)
from milestone_07_policy_replay.replay_engine import PolicyReplayEngine

logger = get_logger(__name__)


class ReplayPipeline:
    """
    Runs the policy replay engine for one or more episodes,
    logs results, and generates summary plots.

    Parameters
    ----------
    policy_dir : Path
        Directory containing exported_policy.pt and policy_metadata.json.
    data_path : Path, optional
        Dataset CSV for the environment. Defaults to forecast_dataset.csv.
    n_episodes : int
        Number of replay episodes to run.
    generate_plots : bool
        Whether to generate replay visualisation plots.
    """

    def __init__(
        self,
        policy_dir: Path = POLICY_EXPORT_DIR,
        data_path: Path = None,
        n_episodes: int = 1,
        generate_plots: bool = True,
    ) -> None:
        self.policy_dir   = Path(policy_dir)
        self.data_path    = data_path
        self.n_episodes   = n_episodes
        self.generate_plots = generate_plots

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Replay the exported policy for n_episodes and return the action log.

        Returns
        -------
        pd.DataFrame — combined step-level action log across all episodes.
        """
        logger.info("=" * 60)
        logger.info("MILESTONE 7 — POLICY REPLAY ENGINE STARTED")
        logger.info("  Policy dir : %s", self.policy_dir)
        logger.info("  Episodes   : %d", self.n_episodes)
        logger.info("=" * 60)

        # Confirm exported policy exists
        policy_file = self.policy_dir / "exported_policy.pt"
        if not policy_file.exists():
            logger.error(
                "Exported policy not found: %s\n"
                "Run Milestone 6 first:  python -m milestone_06_policy_export.pipeline",
                policy_file,
            )
            sys.exit(1)

        engine = PolicyReplayEngine(
            policy_dir=self.policy_dir,
            data_path=self.data_path,
            log_dir=REPORTS_DIR,
            random_start=False,
        )

        all_logs = []
        episode_summaries = []

        for ep in range(self.n_episodes):
            logger.info("--- Replaying episode %d / %d ---", ep + 1, self.n_episodes)
            df = engine.replay(seed=ep)
            df["episode"] = ep
            all_logs.append(df)

            summary = {
                "episode":           ep,
                "steps":             len(df),
                "total_reward":      round(df["reward"].sum(), 4),
                "mean_reward":       round(df["reward"].mean(), 4),
                "sla_violations":    int(df["sla_violated"].sum()),
                "sla_rate":          round(df["sla_violated"].mean(), 4),
                "scale_up":          int((df["action"] == 0).sum()),
                "scale_down":        int((df["action"] == 1).sum()),
                "no_op":             int((df["action"] == 2).sum()),
                "mean_cpu_util":     round(df["cpu_util"].mean(), 4),
                "mean_vm_count":     round(df["vm_count"].mean(), 4),
                "mean_queue_length": round(df["queue_length"].mean(), 4),
            }
            episode_summaries.append(summary)
            self._log_episode_summary(summary)

        combined_log = pd.concat(all_logs, ignore_index=True)

        # Save combined log
        log_path = REPORTS_DIR / REPLAY_LOG_FILENAME
        combined_log.to_csv(log_path, index=False)
        logger.info("Combined replay log saved: %s", log_path)

        # Aggregate summary
        self._log_aggregate_summary(episode_summaries)

        # Plots
        if self.generate_plots:
            self._generate_plots(combined_log)

        logger.info("=" * 60)
        logger.info("MILESTONE 7 COMPLETE")
        logger.info("=" * 60)
        return combined_log

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, log_df: pd.DataFrame) -> bool:
        """Validate the replay log for correctness."""
        logger.info("Running M7 validation checks...")
        passed = True

        # Check 1: log has data
        if log_df.empty:
            logger.error("FAIL — Replay log is empty.")
            return False
        logger.info("PASS — Log has %d rows.", len(log_df))

        # Check 2: all actions are valid (0, 1, 2)
        invalid_actions = ~log_df["action"].isin([0, 1, 2])
        if invalid_actions.any():
            logger.error("FAIL — %d invalid action values.", invalid_actions.sum())
            passed = False
        else:
            logger.info("PASS — All actions valid (0=up, 1=down, 2=noop).")

        # Check 3: VM count bounded
        from config.settings import ENV_MIN_VMS, ENV_MAX_VMS
        oob = (log_df["vm_count"] < ENV_MIN_VMS) | (log_df["vm_count"] > ENV_MAX_VMS)
        if oob.any():
            logger.error("FAIL — %d VM count values out of [%d, %d].", oob.sum(), ENV_MIN_VMS, ENV_MAX_VMS)
            passed = False
        else:
            logger.info("PASS — VM counts in [%d, %d].", ENV_MIN_VMS, ENV_MAX_VMS)

        # Check 4: rewards are finite
        if not np.isfinite(log_df["reward"].values).all():
            logger.error("FAIL — Non-finite reward values found.")
            passed = False
        else:
            logger.info("PASS — All rewards are finite.")

        # Check 5: log file exists on disk
        log_path = REPORTS_DIR / REPLAY_LOG_FILENAME
        if log_path.exists():
            logger.info("PASS — Log file saved: %s", log_path)
        else:
            logger.error("FAIL — Log file not found on disk.")
            passed = False

        result = "ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"
        logger.info("M7 Validation: %s", result)
        return passed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_episode_summary(s: dict) -> None:
        logger.info(
            "  Steps: %d | Reward: %+.4f | SLA violations: %d (%.1f%%) | "
            "↑%d ↓%d ·%d | AvgCPU: %.1f%% | AvgVMs: %.1f",
            s["steps"], s["total_reward"],
            s["sla_violations"], s["sla_rate"] * 100,
            s["scale_up"], s["scale_down"], s["no_op"],
            s["mean_cpu_util"], s["mean_vm_count"],
        )

    @staticmethod
    def _log_aggregate_summary(summaries: list) -> None:
        if len(summaries) <= 1:
            return
        rewards = [s["total_reward"] for s in summaries]
        sla_rates = [s["sla_rate"] for s in summaries]
        logger.info(
            "AGGREGATE (%d episodes) | "
            "Reward: mean=%+.4f std=%.4f | "
            "SLA rate: mean=%.3f",
            len(summaries),
            np.mean(rewards), np.std(rewards),
            np.mean(sla_rates),
        )

    def _generate_plots(self, log_df: pd.DataFrame) -> None:
        """Generate replay-specific plots using ResultsPlotter."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            REPLAY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

            # Use the first episode for time-series plots
            ep0 = log_df[log_df["episode"] == 0] if "episode" in log_df.columns else log_df

            # Plot 1: CPU utilization + VM count over time
            fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

            axes[0].plot(ep0["step"], ep0["cpu_util"], color="#1f77b4", linewidth=1.0)
            axes[0].axhline(85, color="#e74c3c", linestyle="--", linewidth=0.8,
                            alpha=0.7, label="SLA threshold (85%)")
            axes[0].set_ylabel("CPU Util (%)")
            axes[0].set_title("Policy Replay — Episode 0")
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.3)

            axes[1].step(ep0["step"], ep0["vm_count"], color="#2ecc71",
                         linewidth=1.2, where="post")
            axes[1].set_ylabel("VM Count")
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(ep0["step"], ep0["queue_length"], color="#f39c12",
                         linewidth=0.8, alpha=0.8)
            axes[2].set_ylabel("Queue Length")
            axes[2].set_xlabel("Step")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(REPLAY_PLOTS_DIR / "replay_overview.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Plot 2: Action distribution pie chart
            action_counts = log_df["action_name"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))
            colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
            ax.pie(action_counts.values, labels=action_counts.index,
                   autopct="%1.1f%%", startangle=90, colors=colors)
            ax.set_title("Action Distribution (Policy Replay)")
            fig.savefig(REPLAY_PLOTS_DIR / "action_distribution.png", dpi=150,
                        bbox_inches="tight")
            plt.close(fig)

            # Plot 3: Reward over steps
            fig, ax = plt.subplots(figsize=(14, 3))
            ax.plot(ep0["step"], ep0["reward"], color="#9b59b6", linewidth=0.7, alpha=0.8)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.4)
            ax.set_title("Step Reward — Policy Replay Episode 0")
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(REPLAY_PLOTS_DIR / "replay_reward.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info("Replay plots saved to: %s", REPLAY_PLOTS_DIR)

        except Exception as exc:
            logger.warning("Plot generation failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Milestone 7: Policy Replay Engine"
    )
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=POLICY_EXPORT_DIR,
        help=f"Directory with exported_policy.pt (default: {POLICY_EXPORT_DIR})",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to forecast_dataset.csv (default: auto-detect)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of replay episodes to run (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for first episode (default: 0)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    pipeline = ReplayPipeline(
        policy_dir=args.policy_dir,
        data_path=args.data_path,
        n_episodes=args.episodes,
        generate_plots=not args.no_plots,
    )
    log_df = pipeline.run()
    pipeline.validate(log_df)
