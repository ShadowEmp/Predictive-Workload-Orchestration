"""
milestone_05_hars/reward_weight_generator.py
==============================================
RewardWeightGenerator: maps difficulty score → reward component weights.

High difficulty (agent struggling):
  → increase SLA weight (penalise violations more urgently)
  → decrease cost weight (don't penalise over-provisioning during struggle)
  → maintain utilization weight

Low difficulty (agent performing well):
  → shift weight toward cost and stability
  → reduce SLA emphasis

Weights are adapted incrementally (max change per episode = ADAPT_RATE)
to prevent destabilising jumps in the reward signal.
"""

from typing import Dict

import numpy as np

from config.logger import get_logger
from config.settings import (
    ENV_REWARD_WEIGHTS,
    HARS_WEIGHT_ADAPT_RATE,
    HARS_WEIGHT_MAX,
    HARS_WEIGHT_MIN,
)

logger = get_logger(__name__)


class RewardWeightGenerator:
    """
    Generates adapted reward weights from the difficulty score.

    Parameters
    ----------
    initial_weights : dict
        Starting weights (defaults to ENV_REWARD_WEIGHTS).
    adapt_rate : float
        Maximum weight change per episode.
    w_min : float
        Minimum weight for any single component.
    w_max : float
        Maximum weight for any single component.
    """

    def __init__(
        self,
        initial_weights: Dict[str, float] = None,
        adapt_rate: float = HARS_WEIGHT_ADAPT_RATE,
        w_min: float = HARS_WEIGHT_MIN,
        w_max: float = HARS_WEIGHT_MAX,
    ) -> None:
        self.weights = dict(initial_weights or ENV_REWARD_WEIGHTS)
        self.adapt_rate = adapt_rate
        self.w_min = w_min
        self.w_max = w_max
        self._history: list = []

    def generate(self, difficulty: float) -> Dict[str, float]:
        """
        Compute new weights from the current difficulty score.

        Strategy:
        - difficulty > 0.5: shift weight from cost → sla
        - difficulty < 0.5: shift weight from sla → cost
        - Scale component adapts to keep total = 1.0

        Parameters
        ----------
        difficulty : float
            Current difficulty score ∈ [0, 1].

        Returns
        -------
        dict
            New weight values (sum ≈ 1.0).
        """
        target = self._target_weights(difficulty)
        new_weights = {}

        for key in self.weights:
            delta = target[key] - self.weights[key]
            # Clip change to adapt_rate
            delta = float(np.clip(delta, -self.adapt_rate, self.adapt_rate))
            new_w = float(np.clip(self.weights[key] + delta, self.w_min, self.w_max))
            new_weights[key] = new_w

        # Renormalise so weights sum to 1.0
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        self.weights = new_weights
        self._history.append(dict(new_weights))

        logger.debug(
            "HARS weights (difficulty=%.3f): %s",
            difficulty, {k: f"{v:.3f}" for k, v in new_weights.items()},
        )
        return new_weights

    @property
    def history(self) -> list:
        return list(self._history)

    # ------------------------------------------------------------------

    @staticmethod
    def _target_weights(difficulty: float) -> Dict[str, float]:
        """
        Define target weights as a linear function of difficulty.

        difficulty=0.0 (easy) → more cost + scale focus
        difficulty=1.0 (hard) → more sla focus
        """
        # SLA weight: 0.3 (easy) → 0.6 (hard)
        w_sla  = 0.30 + 0.30 * difficulty
        # Cost weight: 0.25 (easy) → 0.10 (hard)
        w_cost = 0.25 - 0.15 * difficulty
        # Scale weight: 0.10 (easy) → 0.05 (hard)
        w_scale = 0.10 - 0.05 * difficulty
        # Util gets the remainder
        w_util  = 1.0 - w_sla - w_cost - w_scale

        return {
            "util":  float(np.clip(w_util,  0.05, 0.70)),
            "sla":   float(np.clip(w_sla,   0.05, 0.70)),
            "cost":  float(np.clip(w_cost,  0.05, 0.70)),
            "scale": float(np.clip(w_scale, 0.05, 0.70)),
        }
