"""
milestone_05_hars/scheduler.py
================================
HARSScheduler: the top-level HARS orchestrator.

Called once per episode (after episode completes):
1. PerformanceMonitor records episode metrics.
2. DifficultyScoreEngine computes current difficulty.
3. RewardWeightGenerator produces new weights.
4. Weights are pushed to the CloudEnvironment via update_reward_weights().

Also writes a per-episode HARS log (CSV) for analysis and plotting.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

from config.logger import get_logger
from config.settings import HARS_LOG_FILENAME, REPORTS_DIR
from milestone_05_hars.difficulty_engine import DifficultyScoreEngine
from milestone_05_hars.performance_monitor import PerformanceMonitor
from milestone_05_hars.reward_weight_generator import RewardWeightGenerator

logger = get_logger(__name__)


class HARSScheduler:
    """
    Hybrid Adaptive Reward Scheduler.

    Parameters
    ----------
    env : CloudEnvironment
        The training environment (receives updated weights).
    log_dir : Path
        Directory for HARS CSV log.
    """

    def __init__(self, env, log_dir: Path = REPORTS_DIR) -> None:
        self.env = env
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.monitor  = PerformanceMonitor()
        self.difficulty_engine = DifficultyScoreEngine()
        self.weight_generator  = RewardWeightGenerator()

        self._log_rows: List[dict] = []

    def on_episode_end(
        self,
        episode: int,
        episode_reward: float,
        sla_violation_rate: float,
        scale_events: int,
        steps: int,
        mean_loss: float = 0.0,
    ) -> Dict[str, float]:
        """
        Called at the end of each training episode.

        Returns
        -------
        dict
            New reward weights applied to the environment.
        """
        # 1. Record
        self.monitor.record(
            episode_reward=episode_reward,
            sla_violation_rate=sla_violation_rate,
            scale_events=scale_events,
            steps=steps,
            mean_loss=mean_loss,
        )

        # 2. Compute difficulty
        stats = self.monitor.get_stats()
        difficulty = self.difficulty_engine.compute(stats)

        # 3. Generate new weights
        new_weights = self.weight_generator.generate(difficulty)

        # 4. Push to environment
        self.env.update_reward_weights(new_weights)

        # 5. Log
        row = {
            "episode":    episode,
            "difficulty": round(difficulty, 6),
            **{f"w_{k}": round(v, 6) for k, v in new_weights.items()},
            "mean_reward": round(stats["mean_reward"], 4),
            "mean_sla_rate": round(stats["mean_sla_rate"], 4),
        }
        self._log_rows.append(row)
        return new_weights

    def save_log(self) -> None:
        """Write the HARS episode log to CSV."""
        if not self._log_rows:
            return
        path = self.log_dir / HARS_LOG_FILENAME
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._log_rows[0].keys())
            writer.writeheader()
            writer.writerows(self._log_rows)
        logger.info("HARS log saved: %s", path)

    @property
    def difficulty_history(self) -> list:
        return self.difficulty_engine.history

    @property
    def weight_history(self) -> list:
        return self.weight_generator.history
