"""
milestone_09_experiments/runner.py
=====================================
ExperimentRunner: benchmarks all 6 strategies across multiple episodes
and collects comprehensive metrics.
"""

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from config.logger import get_logger
from config.settings import (
    EXPERIMENT_METRICS_FILENAME,
    EXPERIMENT_RESULTS_DIR,
    EXPERIMENT_STRATEGIES,
)
from milestone_03_environment.cloud_environment import CloudEnvironment

logger = get_logger(__name__)

EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ExperimentRunner:
    """
    Runs benchmark episodes for each strategy and collects metrics.

    Parameters
    ----------
    env_factory : callable
        Returns a fresh CloudEnvironment when called.
    strategies : dict
        Mapping of strategy_name → policy callable.
    n_episodes : int
        Number of evaluation episodes per strategy.
    results_dir : Path
        Where to save experiment_results.csv.
    """

    def __init__(
        self,
        env_factory: Callable,
        strategies: Dict[str, Callable],
        n_episodes: int = 5,
        results_dir: Path = EXPERIMENT_RESULTS_DIR,
    ) -> None:
        self.env_factory = env_factory
        self.strategies  = strategies
        self.n_episodes  = n_episodes
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        """
        Run all strategies and return a metrics DataFrame.

        Returns
        -------
        pd.DataFrame with one row per (strategy, episode).
        """
        all_rows = []

        for strategy_name, policy in self.strategies.items():
            logger.info("Running strategy: %s (%d episodes)", strategy_name, self.n_episodes)
            for ep in range(self.n_episodes):
                row = self._run_episode(strategy_name, policy, seed=ep)
                all_rows.append(row)
                logger.info(
                    "  %s ep%d: reward=%.4f | SLA=%.3f | cost=%.4f | scale=%d",
                    strategy_name, ep,
                    row["episode_reward"], row["sla_violation_rate"],
                    row["estimated_cost"], row["scale_events"],
                )

        df = pd.DataFrame(all_rows)
        out_path = self.results_dir / EXPERIMENT_METRICS_FILENAME
        df.to_csv(out_path, index=False)
        logger.info("Experiment results saved: %s", out_path)
        return df

    def _run_episode(
        self,
        strategy_name: str,
        policy: Callable,
        seed: int,
    ) -> dict:
        env = self.env_factory()
        obs, _ = env.reset(seed=seed)

        step_rewards, cpu_utils, vm_counts = [], [], []

        while True:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step_rewards.append(reward)
            cpu_utils.append(info.get("cpu_util", 0.0))
            vm_counts.append(info["vm_count"])
            if terminated or truncated:
                break

        summary = env.episode_summary()
        n_steps = max(len(step_rewards), 1)

        # Estimate cost: $0.05/VM/hour × 5 min per step
        estimated_cost = sum(v * 0.05 * (5 / 60) for v in vm_counts)

        # Energy: simplified linear model
        estimated_energy = sum(
            v * (50 + 200 * (c / 100)) * (5 / 60) / 1000
            for v, c in zip(vm_counts, cpu_utils)
        )

        return {
            "strategy":          strategy_name,
            "episode":           seed,
            "episode_reward":    round(sum(step_rewards), 4),
            "mean_cpu_util":     round(float(np.mean(cpu_utils)), 4),
            "mean_vm_count":     round(float(np.mean(vm_counts)), 4),
            "sla_violation_rate": round(summary["sla_violation_rate"], 4),
            "sla_violations":    summary["sla_violations"],
            "scale_events":      summary["scale_events"],
            "scaling_frequency": round(summary["scale_events"] / n_steps, 4),
            "estimated_cost":    round(estimated_cost, 4),
            "estimated_energy_kwh": round(estimated_energy, 6),
            "final_queue_length": round(summary["final_queue_length"], 4),
            "steps":             summary["steps_completed"],
        }
