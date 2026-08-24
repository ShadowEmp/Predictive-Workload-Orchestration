"""
milestone_03_environment/state_space.py
=========================================
StateBuilder: constructs and normalises the observation vector at each time step.

Responsibilities:
1. Extract the configured feature columns from the current dataset row.
2. Normalise each feature to [0, 1] using pre-defined bounds.
3. Append runtime-computed features (vm_count_norm, queue_length_norm).
4. Return a float32 numpy array ready for the neural network.

Design notes:
- Normalisation uses fixed bounds from settings.py rather than data-driven
  min/max scaling. This prevents train/test distribution shift: the agent
  trained on data where max CPU = 92% would fail if during replay it sees 95%.
  Fixed bounds (0–100 for %) are always valid regardless of which episode slice
  is being played.
- If a feature column is missing from the dataset (e.g. 'yhat' when running
  without M2), we substitute the fallback value (0.5 = neutral mid-range).
  This keeps the environment runnable in isolation.
- Clip-then-normalise: values are clipped to [low, high] before scaling to
  prevent normalised values outside [0, 1] from confusing the neural network.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.logger import get_logger
from config.settings import (
    ENV_CPU_PER_VM,
    ENV_FEATURE_BOUNDS,
    ENV_MAX_QUEUE,
    ENV_MAX_VMS,
    ENV_OBS_DIM,
    ENV_OBS_FEATURES,
)

logger = get_logger(__name__)

# Fallback value used when a feature column is absent from the dataset
_MISSING_FEATURE_FALLBACK = 0.5


class StateBuilder:
    """
    Constructs and normalises the environment observation vector.

    Parameters
    ----------
    feature_bounds : dict
        Mapping of feature name → (min, max) for normalisation.
    obs_features : list[str]
        Ordered list of feature column names to extract from the dataset row.
    obs_dim : int
        Total observation dimension (len(obs_features) + 2 runtime features).
    """

    def __init__(
        self,
        feature_bounds: Optional[Dict[str, tuple]] = None,
        obs_features: Optional[list] = None,
        obs_dim: int = ENV_OBS_DIM,
    ) -> None:
        self.feature_bounds = feature_bounds or ENV_FEATURE_BOUNDS
        self.obs_features = obs_features or ENV_OBS_FEATURES
        self.obs_dim = obs_dim

        # Warn once if yhat is absent (M2 not run)
        self._warned_missing: set = set()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(
        self,
        row: pd.Series,
        vm_count: int,
        queue_length: float,
    ) -> np.ndarray:
        """
        Build the normalised observation vector for one time step.

        Parameters
        ----------
        row : pd.Series
            One row of forecast_dataset.csv (indexed by feature name).
        vm_count : int
            Current number of running VMs.
        queue_length : float
            Current request queue length (0 = no backlog).

        Returns
        -------
        np.ndarray
            float32 array of shape (obs_dim,), values in [0.0, 1.0].
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # Fill dataset features
        for i, feature in enumerate(self.obs_features):
            value = self._extract_feature(row, feature)
            obs[i] = self._normalise(feature, value)

        # Append runtime-computed features
        n_dataset_features = len(self.obs_features)
        obs[n_dataset_features]     = self._normalise_vm_count(vm_count)
        obs[n_dataset_features + 1] = self._normalise_queue(queue_length)

        return obs

    def get_feature_names(self) -> list:
        """Return ordered list of all observation feature names."""
        return self.obs_features + ["vm_count_norm", "queue_length_norm"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_feature(self, row: pd.Series, feature: str) -> float:
        """Extract a feature value from the row, with fallback for missing columns."""
        if feature not in row.index or pd.isna(row[feature]):
            if feature not in self._warned_missing:
                logger.warning(
                    "Feature '%s' missing or NaN in dataset row. "
                    "Using fallback value %.2f. "
                    "(Run M2 pipeline to generate forecast_dataset.csv with 'yhat'.)",
                    feature, _MISSING_FEATURE_FALLBACK,
                )
                self._warned_missing.add(feature)
            # Fallback: return mid-range of the feature's bound
            low, high = self.feature_bounds.get(feature, (0.0, 1.0))
            return (low + high) / 2.0
        return float(row[feature])

    def _normalise(self, feature: str, value: float) -> float:
        """
        Clip value to [low, high] then scale to [0, 1].

        Formula: (value - low) / (high - low)
        Special case: if low == high, return 0.5 (degenerate bound).
        """
        low, high = self.feature_bounds.get(feature, (0.0, 1.0))
        if high == low:
            return 0.5
        clipped = float(np.clip(value, low, high))
        return (clipped - low) / (high - low)

    @staticmethod
    def _normalise_vm_count(vm_count: int) -> float:
        """Normalise VM count to [0, 1] using ENV_MAX_VMS as denominator."""
        return float(np.clip(vm_count / ENV_MAX_VMS, 0.0, 1.0))

    @staticmethod
    def _normalise_queue(queue_length: float) -> float:
        """Normalise queue length to [0, 1] using ENV_MAX_QUEUE as denominator."""
        return float(np.clip(queue_length / ENV_MAX_QUEUE, 0.0, 1.0))
