"""
milestone_04_dqn/pipeline.py
==============================
Milestone 4 pipeline — sets up environment, agent, and runs training.

Usage:
    python3 -m milestone_04_dqn.pipeline
    python3 -m milestone_04_dqn.pipeline --episodes 50 --data-path data/processed/forecast_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logger import get_logger
from config.settings import (
    DQN_MODEL_DIR,
    DQN_TOTAL_EPISODES,
    ENV_OBS_DIM,
    PROCESSED_DIR,
)
from milestone_03_environment.action_space import N_ACTIONS
from milestone_03_environment.cloud_environment import CloudEnvironment
from milestone_04_dqn.agent import DQNAgent
from milestone_04_dqn.trainer import DQNTrainer

logger = get_logger(__name__)


class DQNPipeline:
    """Orchestrates DQN training pipeline."""

    def __init__(
        self,
        data_path: Path = None,
        total_episodes: int = DQN_TOTAL_EPISODES,
        device: str = "cpu",
    ) -> None:
        self.data_path = data_path
        self.total_episodes = total_episodes
        self.device = device

    def run(self) -> list:
        env = CloudEnvironment(data_path=self.data_path)
        agent = DQNAgent(
            obs_dim=ENV_OBS_DIM,
            n_actions=N_ACTIONS,
            device=self.device,
        )
        trainer = DQNTrainer(
            env=env,
            agent=agent,
            total_episodes=self.total_episodes,
        )
        return trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=DQN_TOTAL_EPISODES)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    DQNPipeline(
        data_path=args.data_path,
        total_episodes=args.episodes,
        device=args.device,
    ).run()
