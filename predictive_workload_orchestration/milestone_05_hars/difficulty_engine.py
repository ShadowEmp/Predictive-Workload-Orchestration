"""
milestone_05_hars/difficulty_engine.py
========================================
DifficultyScoreEngine: computes a scalar difficulty score ∈ [0, 1]
that quantifies how hard the current training phase is for the agent.

Difficulty is a weighted combination of:
  - SLA violation rate (high = agent struggling with SLA)
  - Reward improvement rate (low or negative = learning stalled)
  - Loss trend (high loss = network not converging)

EMA smoothing prevents rapid oscillation in the difficulty signal.
"""

import numpy as np

from config.logger import get_logger
from config.settings import HARS_DIFFICULTY_SMOOTHING

logger = get_logger(__name__)


class DifficultyScoreEngine:
    """
    Computes a smoothed difficulty score from performance statistics.

    Difficulty = 0 → agent performing well (easy phase)
    Difficulty = 1 → agent struggling (hard phase)

    The score drives HARS weight adaptation:
    - High difficulty → increase SLA weight (penalise violations more)
    - Low difficulty  → balance weights toward cost and stability

    Parameters
    ----------
    smoothing : float
        EMA coefficient α (0 = no smoothing, 1 = never updates).
    """

    def __init__(self, smoothing: float = HARS_DIFFICULTY_SMOOTHING) -> None:
        self.smoothing = smoothing
        self._difficulty: float = 0.5   # start at mid-range
        self._prev_reward: float = None
        self._history: list = []

    def compute(self, stats: dict) -> float:
        """
        Compute and update the difficulty score.

        Parameters
        ----------
        stats : dict
            Output of PerformanceMonitor.get_stats().

        Returns
        -------
        float
            Current smoothed difficulty score ∈ [0, 1].
        """
        sla_rate      = stats.get("mean_sla_rate", 0.5)
        mean_reward   = stats.get("mean_reward", 0.0)
        mean_loss     = stats.get("mean_loss", 1.0)

        # SLA component: normalise to [0, 1]
        sla_component = float(np.clip(sla_rate, 0.0, 1.0))

        # Reward stagnation component: how flat is the reward?
        if self._prev_reward is None:
            reward_stagnation = 0.5
        else:
            delta = mean_reward - self._prev_reward
            # Positive delta → improving → low difficulty contribution
            reward_stagnation = float(np.clip(0.5 - delta / 2.0, 0.0, 1.0))

        # Loss component: clip normalised loss to [0, 1]
        loss_component = float(np.clip(mean_loss / 0.5, 0.0, 1.0))

        # Combine (equal weights)
        raw_difficulty = (sla_component * 0.5 + reward_stagnation * 0.3 + loss_component * 0.2)
        raw_difficulty = float(np.clip(raw_difficulty, 0.0, 1.0))

        # EMA smoothing
        self._difficulty = (
            self.smoothing * self._difficulty
            + (1 - self.smoothing) * raw_difficulty
        )
        self._prev_reward = mean_reward
        self._history.append(self._difficulty)

        logger.debug(
            "Difficulty: %.4f (SLA=%.3f, stagnation=%.3f, loss=%.3f)",
            self._difficulty, sla_component, reward_stagnation, loss_component,
        )
        return self._difficulty

    @property
    def history(self) -> list:
        return list(self._history)

    @property
    def current(self) -> float:
        return self._difficulty
