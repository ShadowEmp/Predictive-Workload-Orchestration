"""
milestone_03_environment/cloud_environment.py
===============================================
CloudEnvironment: the Gymnasium RL environment for cloud auto-scaling.

This is the central class of Milestone 3.  It implements the full
Gymnasium interface and ties together StateBuilder, ActionExecutor,
and RewardCalculator.

MDP Summary:
    State  : 13-dim float32 vector (11 dataset features + vm_count + queue)
    Actions: Discrete(3) — SCALE_UP, SCALE_DOWN, NO_OP
    Reward : Composite fixed-weight reward (HARS replaces weights in M5)
    Episode: 288 steps (24 hours at 5-min resolution)

Gymnasium compliance:
    - reset(seed, options) → (obs, info)         ✓
    - step(action)         → (obs, reward, terminated, truncated, info)  ✓
    - observation_space    → Box(obs_dim,)        ✓
    - action_space         → Discrete(3)          ✓
    - render(mode)         → None (text to logger) ✓
    - close()              → None                 ✓
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from config.logger import get_logger
from config.settings import (
    ENV_CPU_PER_VM,
    ENV_EPISODE_LENGTH,
    ENV_INITIAL_VMS,
    ENV_MAX_QUEUE,
    ENV_MAX_VMS,
    ENV_MIN_VMS,
    ENV_OBS_DIM,
    ENV_RANDOM_START,
    ENV_REWARD_WEIGHTS,
    ENV_SCALING_COST,
    ENV_SLA_THRESHOLD,
    ENV_TARGET_CPU_UTIL,
    FORECAST_DATASET_FILENAME,
    PROCESSED_DATASET_FILENAME,
    PROCESSED_DIR,
    TARGET_COLUMN,
)
from milestone_03_environment.action_space import ActionExecutor, N_ACTIONS
from milestone_03_environment.reward import RewardCalculator
from milestone_03_environment.state_space import StateBuilder

logger = get_logger(__name__)


class CloudEnvironment(gym.Env):
    """
    Cloud auto-scaling environment compatible with Gymnasium API.

    The environment simulates a cloud cluster that must be kept at target
    utilization by scaling VMs up or down.  Workload data comes from
    forecast_dataset.csv (M2 output), falling back to processed_dataset.csv.

    Parameters
    ----------
    data_path : Path, optional
        Path to the dataset CSV.  Defaults to forecast_dataset.csv in
        PROCESSED_DIR; falls back to processed_dataset.csv if not found.
    episode_length : int
        Number of steps per episode.
    random_start : bool
        If True, randomise episode start index on each reset().
    min_vms : int
        Hard lower bound on VM count.
    max_vms : int
        Hard upper bound on VM count.
    initial_vms : int
        VM count at episode start.
    reward_weights : dict, optional
        Override default reward weights (used by HARS in M5).
    render_mode : str, optional
        Only 'human' (text log) is supported.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        data_path: Optional[Path] = None,
        episode_length: int = ENV_EPISODE_LENGTH,
        random_start: bool = ENV_RANDOM_START,
        min_vms: int = ENV_MIN_VMS,
        max_vms: int = ENV_MAX_VMS,
        initial_vms: int = ENV_INITIAL_VMS,
        reward_weights: Optional[Dict[str, float]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.episode_length = episode_length
        self.random_start = random_start
        self.min_vms = min_vms
        self.max_vms = max_vms
        self.initial_vms = initial_vms
        self.render_mode = render_mode

        # Load dataset
        self._data: pd.DataFrame = self._load_dataset(data_path)
        self._n_rows = len(self._data)

        # Sub-modules
        self._state_builder = StateBuilder()
        self._action_executor = ActionExecutor(min_vms=min_vms, max_vms=max_vms)
        self._reward_calc = RewardCalculator(
            weights=reward_weights or dict(ENV_REWARD_WEIGHTS)
        )

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(ENV_OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Episode state (initialised in reset())
        self._step_idx: int = 0         # position within episode
        self._data_idx: int = 0         # absolute row index in dataset
        self._start_idx: int = 0        # episode start row
        self._vm_count: int = initial_vms
        self._queue_length: float = 0.0
        self._episode_reward: float = 0.0
        self._episode_sla_violations: int = 0
        self._episode_scale_events: int = 0
        self._np_random: Optional[np.random.Generator] = None

        logger.info(
            "CloudEnvironment initialised | dataset: %d rows | "
            "episode_length: %d | obs_dim: %d | actions: %d",
            self._n_rows, episode_length, ENV_OBS_DIM, N_ACTIONS,
        )

    # ------------------------------------------------------------------
    # Gymnasium API: reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to the start of a new episode.

        Parameters
        ----------
        seed : int, optional
            RNG seed for reproducibility.
        options : dict, optional
            Not used; kept for Gymnasium compliance.

        Returns
        -------
        observation : np.ndarray
        info : dict
        """
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()

        # Choose start index
        max_start = max(0, self._n_rows - self.episode_length - 1)
        if self.random_start and max_start > 0:
            self._start_idx = int(self._np_random.integers(0, max_start))
        else:
            self._start_idx = 0

        # Reset episode state
        self._step_idx = 0
        self._data_idx = self._start_idx
        self._vm_count = self.initial_vms
        self._queue_length = 0.0
        self._episode_reward = 0.0
        self._episode_sla_violations = 0
        self._episode_scale_events = 0
        self._action_executor.reset()

        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_text()

        return obs, info

    # ------------------------------------------------------------------
    # Gymnasium API: step
    # ------------------------------------------------------------------

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Parameters
        ----------
        action : int
            Discrete action index (0=SCALE_UP, 1=SCALE_DOWN, 2=NO_OP).

        Returns
        -------
        observation : np.ndarray
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        # 1. Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 2. Execute scaling action
        self._vm_count = self._action_executor.execute(action, self._vm_count)

        # 3. Get current workload from dataset
        row = self._data.iloc[self._data_idx]
        cpu_util = self._get_cpu_util(row)

        # 4. Compute effective capacity and queue dynamics
        capacity = self._vm_count * ENV_CPU_PER_VM         # total CPU capacity %
        demand = cpu_util                                   # current demand %
        overflow = max(0.0, demand - capacity)             # unserved demand
        self._queue_length = min(
            self._queue_length + overflow, ENV_MAX_QUEUE
        )
        # Queue drains when capacity exceeds demand
        drain = max(0.0, capacity - demand)
        self._queue_length = max(0.0, self._queue_length - drain * 0.5)

        # 5. Compute effective CPU utilization (capped at 100%)
        effective_cpu = min(100.0, demand / max(capacity, 1.0) * 100.0)

        # 6. Compute reward
        reward_dict = self._reward_calc.compute(
            cpu_util=effective_cpu,
            vm_count=self._vm_count,
            is_scaling_action=self._action_executor.is_scaling_action(),
        )
        reward = reward_dict["total"]

        # 7. Track episode statistics
        self._episode_reward += reward
        if effective_cpu > ENV_SLA_THRESHOLD:
            self._episode_sla_violations += 1
        if self._action_executor.is_scaling_action():
            self._episode_scale_events += 1

        # 8. Advance time
        self._step_idx += 1
        self._data_idx = min(self._start_idx + self._step_idx, self._n_rows - 1)

        # 9. Termination conditions
        terminated = self._step_idx >= self.episode_length
        # Truncate if queue is maxed out AND we still have capacity headroom
        # (indicates a runaway divergence — stop the episode early)
        truncated = (
            self._queue_length >= ENV_MAX_QUEUE
            and self._vm_count >= self.max_vms
        )

        # 10. Build next observation
        obs = self._get_observation()
        info = self._get_info(reward_dict=reward_dict, cpu_util=effective_cpu)

        if self.render_mode == "human":
            self._render_text(reward_dict=reward_dict, cpu_util=effective_cpu)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Gymnasium API: render and close
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Render current state to the logger (human mode only)."""
        if self.render_mode == "human":
            self._render_text()

    def close(self) -> None:
        """Clean up resources (no external resources to release here)."""
        logger.debug("CloudEnvironment closed.")

    # ------------------------------------------------------------------
    # Reward weight update (called by HARS in Milestone 5)
    # ------------------------------------------------------------------

    def update_reward_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update the reward function weights without recreating the environment.

        Called by the HARS module to dynamically adjust the reward signal
        during training.

        Parameters
        ----------
        new_weights : dict
            Keys: 'util', 'sla', 'cost', 'scale'. Values: float weights.
        """
        self._reward_calc.update_weights(new_weights)

    # ------------------------------------------------------------------
    # Episode summary
    # ------------------------------------------------------------------

    def episode_summary(self) -> Dict[str, Any]:
        """Return a summary dict of the completed episode."""
        return {
            "episode_reward":       self._episode_reward,
            "sla_violations":       self._episode_sla_violations,
            "sla_violation_rate":   self._episode_sla_violations / max(self._step_idx, 1),
            "scale_events":         self._episode_scale_events,
            "final_vm_count":       self._vm_count,
            "final_queue_length":   self._queue_length,
            "steps_completed":      self._step_idx,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dataset(self, data_path: Optional[Path]) -> pd.DataFrame:
        """
        Load the dataset, preferring forecast_dataset.csv over processed_dataset.csv.
        """
        if data_path is not None:
            path = Path(data_path)
        else:
            forecast_path = PROCESSED_DIR / FORECAST_DATASET_FILENAME
            processed_path = PROCESSED_DIR / PROCESSED_DATASET_FILENAME
            if forecast_path.exists():
                path = forecast_path
                logger.info("Using forecast_dataset.csv (includes Prophet yhat).")
            elif processed_path.exists():
                path = processed_path
                logger.warning(
                    "forecast_dataset.csv not found. Falling back to "
                    "processed_dataset.csv (no yhat column — run M2 pipeline)."
                )
            else:
                raise FileNotFoundError(
                    "Neither forecast_dataset.csv nor processed_dataset.csv found "
                    f"in {PROCESSED_DIR}. Run M1 and M2 pipelines first."
                )

        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        logger.info("Dataset loaded: %d rows × %d cols from %s", *df.shape, path.name)
        return df

    def _get_cpu_util(self, row: pd.Series) -> float:
        """Extract CPU utilization from the current row."""
        if TARGET_COLUMN in row.index and not pd.isna(row[TARGET_COLUMN]):
            return float(row[TARGET_COLUMN])
        return 50.0  # safe default if column is missing

    def _get_observation(self) -> np.ndarray:
        """Build the observation vector for the current state."""
        row = self._data.iloc[self._data_idx]
        return self._state_builder.build(
            row=row,
            vm_count=self._vm_count,
            queue_length=self._queue_length,
        )

    def _get_info(
        self,
        reward_dict: Optional[Dict[str, float]] = None,
        cpu_util: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build the info dictionary for this step."""
        info: Dict[str, Any] = {
            "step":          self._step_idx,
            "data_idx":      self._data_idx,
            "vm_count":      self._vm_count,
            "queue_length":  self._queue_length,
            "episode_reward": self._episode_reward,
        }
        if reward_dict is not None:
            info["reward_components"] = reward_dict
        if cpu_util is not None:
            info["cpu_util"] = cpu_util
        return info

    def _render_text(
        self,
        reward_dict: Optional[Dict[str, float]] = None,
        cpu_util: Optional[float] = None,
    ) -> None:
        """Log a human-readable one-line summary of the current state."""
        logger.info(
            "Step %3d | VMs: %2d | CPU: %5.1f%% | Queue: %5.1f | "
            "Reward: %+.4f | Action: %-10s",
            self._step_idx,
            self._vm_count,
            cpu_util or 0.0,
            self._queue_length,
            reward_dict["total"] if reward_dict else 0.0,
            self._action_executor.last_action_name,
        )
