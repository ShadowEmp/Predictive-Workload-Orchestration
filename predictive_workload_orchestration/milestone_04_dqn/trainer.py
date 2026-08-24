"""
milestone_04_dqn/trainer.py
=============================
DQN training loop — runs episodes, logs metrics, saves checkpoints.
"""

import csv
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config.logger import get_logger
from config.settings import (
    DQN_BEST_MODEL_FILENAME,
    DQN_LOG_FREQ,
    DQN_MODEL_DIR,
    DQN_MODEL_FILENAME,
    DQN_PLOTS_DIR,
    DQN_SAVE_FREQ,
    DQN_TOTAL_EPISODES,
    DQN_TRAINING_LOG_FILENAME,
    ENV_OBS_DIM,
    REPORTS_DIR,
)
from milestone_03_environment.action_space import N_ACTIONS
from milestone_03_environment.cloud_environment import CloudEnvironment
from milestone_04_dqn.agent import DQNAgent

logger = get_logger(__name__)


class DQNTrainer:
    """
    Manages the DQN training loop.

    Parameters
    ----------
    env : CloudEnvironment
        The training environment.
    agent : DQNAgent
        The DQN agent.
    total_episodes : int
        Total number of training episodes.
    model_dir : Path
        Directory for saving model checkpoints.
    log_dir : Path
        Directory for saving training logs.
    """

    def __init__(
        self,
        env: CloudEnvironment,
        agent: DQNAgent,
        total_episodes: int = DQN_TOTAL_EPISODES,
        model_dir: Path = DQN_MODEL_DIR,
        log_dir: Path = REPORTS_DIR,
    ) -> None:
        self.env = env
        self.agent = agent
        self.total_episodes = total_episodes
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._best_reward = -np.inf
        self._log_rows = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self) -> list:
        """
        Run the full training loop.

        Returns
        -------
        list of episode log dicts.
        """
        logger.info("=" * 60)
        logger.info("DQN TRAINING STARTED — %d episodes", self.total_episodes)
        logger.info("=" * 60)
        start_time = time.time()

        for episode in range(1, self.total_episodes + 1):
            log = self._run_episode(episode)
            self._log_rows.append(log)

            if episode % DQN_LOG_FREQ == 0:
                self._log_progress(episode)

            if episode % DQN_SAVE_FREQ == 0:
                self.agent.save(self.model_dir / DQN_MODEL_FILENAME)

            if log["episode_reward"] > self._best_reward:
                self._best_reward = log["episode_reward"]
                self.agent.save(self.model_dir / DQN_BEST_MODEL_FILENAME)

        self._save_log()
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("DQN TRAINING COMPLETE in %.1fs | Best reward: %.4f",
                    elapsed, self._best_reward)
        logger.info("=" * 60)
        return self._log_rows

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_episode(self, episode: int) -> dict:
        obs, _ = self.env.reset(seed=episode)
        episode_reward = 0.0
        episode_loss = []

        while True:
            action = self.agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            loss = self.agent.step(obs, action, reward, next_obs, done)
            if loss is not None:
                episode_loss.append(loss)
            episode_reward += reward
            obs = next_obs
            if done:
                break

        summary = self.env.episode_summary()
        return {
            "episode": episode,
            "episode_reward": episode_reward,
            "mean_loss": float(np.mean(episode_loss)) if episode_loss else 0.0,
            "epsilon": self.agent.epsilon,
            "sla_violations": summary["sla_violations"],
            "scale_events": summary["scale_events"],
            "steps": summary["steps_completed"],
        }

    def _log_progress(self, episode: int) -> None:
        recent = self._log_rows[-DQN_LOG_FREQ:]
        mean_reward = np.mean([r["episode_reward"] for r in recent])
        mean_loss   = np.mean([r["mean_loss"] for r in recent])
        eps         = self._log_rows[-1]["epsilon"]
        logger.info(
            "Ep %4d/%d | AvgReward: %+.4f | AvgLoss: %.6f | Epsilon: %.3f",
            episode, self.total_episodes, mean_reward, mean_loss, eps,
        )

    def _save_log(self) -> None:
        path = self.log_dir / DQN_TRAINING_LOG_FILENAME
        if not self._log_rows:
            return
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._log_rows[0].keys())
            writer.writeheader()
            writer.writerows(self._log_rows)
        logger.info("Training log saved: %s", path)
