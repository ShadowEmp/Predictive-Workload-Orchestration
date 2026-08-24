"""
milestone_05_hars/performance_monitor.py
==========================================
PerformanceMonitor: tracks episode-level metrics over a rolling window.

Tracks: episode reward, SLA violation rate, scaling frequency, loss.
Computes rolling means used by the DifficultyScoreEngine.
"""

from collections import deque
from typing import Dict, Optional

import numpy as np

from config.logger import get_logger
from config.settings import HARS_PERF_WINDOW

logger = get_logger(__name__)


class PerformanceMonitor:
    """
    Rolling-window performance tracker for HARS.

    Parameters
    ----------
    window : int
        Number of recent episodes to average over.
    """

    def __init__(self, window: int = HARS_PERF_WINDOW) -> None:
        self.window = window
        self._rewards     = deque(maxlen=window)
        self._sla_rates   = deque(maxlen=window)
        self._scale_rates = deque(maxlen=window)
        self._losses      = deque(maxlen=window)

    def record(
        self,
        episode_reward: float,
        sla_violation_rate: float,
        scale_events: int,
        steps: int,
        mean_loss: float = 0.0,
    ) -> None:
        """Record metrics from one episode."""
        self._rewards.append(episode_reward)
        self._sla_rates.append(sla_violation_rate)
        self._scale_rates.append(scale_events / max(steps, 1))
        self._losses.append(mean_loss)

    def get_stats(self) -> Dict[str, float]:
        """Return current rolling statistics."""
        if not self._rewards:
            return {
                "mean_reward": 0.0, "std_reward": 0.0,
                "mean_sla_rate": 0.0, "mean_scale_rate": 0.0,
                "mean_loss": 0.0, "n_episodes": 0,
            }
        return {
            "mean_reward":     float(np.mean(self._rewards)),
            "std_reward":      float(np.std(self._rewards)),
            "mean_sla_rate":   float(np.mean(self._sla_rates)),
            "mean_scale_rate": float(np.mean(self._scale_rates)),
            "mean_loss":       float(np.mean(self._losses)),
            "n_episodes":      len(self._rewards),
        }

    @property
    def is_ready(self) -> bool:
        """True once the window is fully populated."""
        return len(self._rewards) >= self.window
