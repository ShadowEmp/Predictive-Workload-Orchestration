"""
milestone_05_hars/pipeline.py
================================
DQN + HARS training pipeline (Milestone 5).

HARS wraps the standard DQN training loop:
- After every episode, HARSScheduler computes a difficulty score and
  generates new reward weights.
- The updated weights are pushed to CloudEnvironment via
  env.update_reward_weights() — no environment recreation needed.
- All HARS metrics (difficulty, weights, performance) are logged to
  data/reports/hars_log.csv and plotted at the end.

Usage:
    python -m milestone_05_hars.pipeline
    python -m milestone_05_hars.pipeline --episodes 200
    python -m milestone_05_hars.pipeline --episodes 50 --no-plots
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logger import get_logger
from config.settings import (
    DQN_BEST_MODEL_FILENAME,
    DQN_EPSILON_DECAY_STEPS,
    DQN_LOG_FREQ,
    DQN_MODEL_DIR,
    DQN_MODEL_FILENAME,
    DQN_SAVE_FREQ,
    DQN_TOTAL_EPISODES,
    DQN_TRAINING_LOG_FILENAME,
    ENV_OBS_DIM,
    HARS_LOG_FILENAME,
    HARS_PLOTS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
)
from milestone_03_environment.action_space import N_ACTIONS
from milestone_03_environment.cloud_environment import CloudEnvironment
from milestone_04_dqn.agent import DQNAgent
from milestone_05_hars.scheduler import HARSScheduler

logger = get_logger(__name__)


class HARSPipeline:
    """
    DQN + HARS joint training pipeline.

    Parameters
    ----------
    data_path : Path, optional
        Dataset path. Defaults to forecast_dataset.csv.
    total_episodes : int
        Number of training episodes.
    device : str
        Torch device ('cpu' or 'cuda').
    generate_plots : bool
        Whether to generate HARS visualisation plots at the end.
    """

    def __init__(
        self,
        data_path: Path = None,
        total_episodes: int = DQN_TOTAL_EPISODES,
        device: str = "cpu",
        generate_plots: bool = True,
    ) -> None:
        self.data_path = data_path
        self.total_episodes = total_episodes
        self.device = device
        self.generate_plots = generate_plots

        DQN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Run DQN training with HARS active.

        Returns
        -------
        dict with keys: training_log, hars_log, best_reward
        """
        start = time.time()

        logger.info("=" * 60)
        logger.info("MILESTONE 5 — DQN + HARS TRAINING STARTED")
        logger.info("  Episodes : %d", self.total_episodes)
        logger.info("  Device   : %s", self.device)
        logger.info("=" * 60)

        # Build environment and agent
        env = CloudEnvironment(data_path=self.data_path)
        agent = DQNAgent(
            obs_dim=ENV_OBS_DIM,
            n_actions=N_ACTIONS,
            device=self.device,
            epsilon_decay_steps=max(
                DQN_EPSILON_DECAY_STEPS,
                self.total_episodes * 100,
            ),
        )

        # HARS scheduler wraps the environment
        hars = HARSScheduler(env=env, log_dir=REPORTS_DIR)

        training_log = []
        best_reward = -np.inf

        for episode in range(1, self.total_episodes + 1):

            # --- Run one episode ---
            obs, _ = env.reset(seed=episode)
            ep_reward = 0.0
            ep_losses = []

            while True:
                action = agent.select_action(obs, training=True)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                loss = agent.step(obs, action, reward, next_obs, done)
                if loss is not None:
                    ep_losses.append(loss)
                ep_reward += reward
                obs = next_obs
                if done:
                    break

            summary = env.episode_summary()
            mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

            # --- HARS update (after every episode) ---
            new_weights = hars.on_episode_end(
                episode=episode,
                episode_reward=ep_reward,
                sla_violation_rate=summary["sla_violation_rate"],
                scale_events=summary["scale_events"],
                steps=summary["steps_completed"],
                mean_loss=mean_loss,
            )

            # --- Log ---
            difficulty = hars.difficulty_history[-1] if hars.difficulty_history else 0.0
            row = {
                "episode":         episode,
                "episode_reward":  round(ep_reward, 4),
                "mean_loss":       round(mean_loss, 6),
                "epsilon":         round(agent.epsilon, 4),
                "difficulty":      round(difficulty, 4),
                "w_util":          round(new_weights.get("util", 0), 4),
                "w_sla":           round(new_weights.get("sla", 0), 4),
                "w_cost":          round(new_weights.get("cost", 0), 4),
                "w_scale":         round(new_weights.get("scale", 0), 4),
                "sla_violations":  summary["sla_violations"],
                "scale_events":    summary["scale_events"],
            }
            training_log.append(row)

            # --- Console progress ---
            if episode % DQN_LOG_FREQ == 0:
                recent = training_log[-DQN_LOG_FREQ:]
                avg_r = np.mean([r["episode_reward"] for r in recent])
                avg_l = np.mean([r["mean_loss"] for r in recent])
                logger.info(
                    "Ep %4d/%d | AvgReward: %+.4f | AvgLoss: %.6f | "
                    "Eps: %.3f | Difficulty: %.3f | w_sla: %.3f",
                    episode, self.total_episodes,
                    avg_r, avg_l, agent.epsilon, difficulty,
                    new_weights.get("sla", 0),
                )

            # --- Checkpointing ---
            if episode % DQN_SAVE_FREQ == 0:
                agent.save(DQN_MODEL_DIR / DQN_MODEL_FILENAME)

            if ep_reward > best_reward:
                best_reward = ep_reward
                agent.save(DQN_MODEL_DIR / DQN_BEST_MODEL_FILENAME)

        # --- Save logs ---
        self._save_training_log(training_log)
        hars.save_log()

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info("MILESTONE 5 COMPLETE in %.1fs | Best reward: %.4f", elapsed, best_reward)
        logger.info("=" * 60)

        # --- Plots ---
        if self.generate_plots:
            self._generate_plots(training_log, hars)

        return {
            "training_log": training_log,
            "hars_log": hars._log_rows,
            "best_reward": best_reward,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, result: dict) -> bool:
        """Run post-training validation checks."""
        logger.info("Running M5 validation checks...")
        passed = True
        log = result["training_log"]
        hars_log = result["hars_log"]

        # Check 1: training log has expected entries
        if len(log) != self.total_episodes:
            logger.error("FAIL — Expected %d log rows, got %d.", self.total_episodes, len(log))
            passed = False
        else:
            logger.info("PASS — Training log has %d rows.", len(log))

        # Check 2: reward is generally increasing (last 20% avg > first 20% avg)
        n = len(log)
        if n >= 20:
            first_avg = np.mean([r["episode_reward"] for r in log[:n // 5]])
            last_avg  = np.mean([r["episode_reward"] for r in log[-n // 5:]])
            if last_avg > first_avg:
                logger.info("PASS — Reward improved: %.4f → %.4f", first_avg, last_avg)
            else:
                logger.warning(
                    "WARNING — Reward did not clearly improve: %.4f → %.4f "
                    "(may need more episodes).", first_avg, last_avg,
                )

        # Check 3: difficulty history populated
        if hars_log:
            logger.info("PASS — HARS log has %d entries.", len(hars_log))
        else:
            logger.error("FAIL — HARS log is empty.")
            passed = False

        # Check 4: weight evolution — sla weight should vary
        if hars_log and len(hars_log) > 10:
            sla_weights = [r["w_sla"] for r in hars_log]
            if max(sla_weights) - min(sla_weights) > 0.001:
                logger.info("PASS — SLA weight adapted over training (range: %.4f).",
                            max(sla_weights) - min(sla_weights))
            else:
                logger.warning("WARNING — SLA weight barely moved. HARS may not be adapting.")

        # Check 5: best model file exists
        best_model = DQN_MODEL_DIR / DQN_BEST_MODEL_FILENAME
        if best_model.exists():
            logger.info("PASS — Best model saved: %s", best_model)
        else:
            logger.error("FAIL — Best model not found.")
            passed = False

        result_str = "ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"
        logger.info("M5 Validation result: %s", result_str)
        return passed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_training_log(self, log: list) -> None:
        if not log:
            return
        path = REPORTS_DIR / DQN_TRAINING_LOG_FILENAME
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=log[0].keys())
            writer.writeheader()
            writer.writerows(log)
        logger.info("Training log saved: %s", path)

    def _generate_plots(self, training_log: list, hars: HARSScheduler) -> None:
        """Generate all M5 plots using ResultsPlotter."""
        try:
            import pandas as pd
            from milestone_10_visualization.plotter import ResultsPlotter

            HARS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            plotter = ResultsPlotter(plots_dir=HARS_PLOTS_DIR)

            train_df = pd.DataFrame(training_log)
            plotter.plot_training_reward(train_df)
            plotter.plot_training_loss(train_df)
            plotter.plot_epsilon_curve(train_df)

            if hars._log_rows:
                hars_df = pd.DataFrame(hars._log_rows)
                plotter.plot_difficulty_score(hars_df)
                plotter.plot_reward_weights(hars_df)

            logger.info("HARS plots saved to: %s", HARS_PLOTS_DIR)
        except Exception as exc:
            logger.warning("Plot generation failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Milestone 5: DQN + HARS Training Pipeline"
    )
    parser.add_argument(
        "--episodes", type=int, default=DQN_TOTAL_EPISODES,
        help=f"Number of training episodes (default: {DQN_TOTAL_EPISODES})"
    )
    parser.add_argument(
        "--data-path", type=Path, default=None,
        help="Path to forecast_dataset.csv (default: data/processed/forecast_dataset.csv)"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Torch device (default: cpu)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation at the end"
    )
    args = parser.parse_args()

    pipeline = HARSPipeline(
        data_path=args.data_path,
        total_episodes=args.episodes,
        device=args.device,
        generate_plots=not args.no_plots,
    )
    result = pipeline.run()
    pipeline.validate(result)
