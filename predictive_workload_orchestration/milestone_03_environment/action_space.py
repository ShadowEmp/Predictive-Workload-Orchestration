"""
milestone_03_environment/action_space.py
==========================================
ActionExecutor: translates a discrete action index into a VM count update.

Action definitions:
    0 = SCALE_UP   → vm_count += 1
    1 = SCALE_DOWN → vm_count -= 1
    2 = NO_OP      → vm_count unchanged

Design notes:
- VM count is always clamped to [min_vms, max_vms] after the action.
  The environment never violates the VM fleet boundaries regardless of action.
- We track the last action to compute the scaling penalty in the reward function
  without coupling ActionExecutor to RewardCalculator.
- Action names and deltas are defined in a class-level dict so they can be
  iterated by the DQN's epsilon-greedy exploration and by the HARS module.
"""

from typing import Dict, Tuple

import numpy as np

from config.logger import get_logger
from config.settings import ENV_MAX_VMS, ENV_MIN_VMS

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Action catalogue: index → (name, vm_delta)
# ---------------------------------------------------------------------------
ACTION_CATALOGUE: Dict[int, Tuple[str, int]] = {
    0: ("SCALE_UP",   +1),
    1: ("SCALE_DOWN", -1),
    2: ("NO_OP",       0),
}

N_ACTIONS = len(ACTION_CATALOGUE)


class ActionExecutor:
    """
    Translates discrete action indices into VM fleet updates.

    Parameters
    ----------
    min_vms : int
        Minimum allowed VM count (hard lower bound).
    max_vms : int
        Maximum allowed VM count (hard upper bound).
    """

    def __init__(
        self,
        min_vms: int = ENV_MIN_VMS,
        max_vms: int = ENV_MAX_VMS,
    ) -> None:
        self.min_vms = min_vms
        self.max_vms = max_vms
        self._last_action: int = 2          # NO_OP as default
        self._last_action_name: str = "NO_OP"
        self._scale_events: int = 0         # cumulative scaling events this episode

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, action: int, current_vm_count: int) -> int:
        """
        Apply the action to the current VM count.

        Parameters
        ----------
        action : int
            Action index (0=SCALE_UP, 1=SCALE_DOWN, 2=NO_OP).
        current_vm_count : int
            VM count before the action.

        Returns
        -------
        int
            New VM count after clamping to [min_vms, max_vms].
        """
        if action not in ACTION_CATALOGUE:
            raise ValueError(
                f"Invalid action {action}. Must be one of {list(ACTION_CATALOGUE.keys())}."
            )

        name, delta = ACTION_CATALOGUE[action]
        new_count = int(np.clip(current_vm_count + delta, self.min_vms, self.max_vms))

        self._last_action = action
        self._last_action_name = name

        if delta != 0:
            self._scale_events += 1
            logger.debug(
                "Action: %s | VM count: %d → %d",
                name, current_vm_count, new_count,
            )

        return new_count

    def is_scaling_action(self) -> bool:
        """Return True if the last executed action changed the VM count."""
        _, delta = ACTION_CATALOGUE[self._last_action]
        return delta != 0

    def reset(self) -> None:
        """Reset per-episode tracking state."""
        self._last_action = 2
        self._last_action_name = "NO_OP"
        self._scale_events = 0

    @property
    def last_action(self) -> int:
        return self._last_action

    @property
    def last_action_name(self) -> str:
        return self._last_action_name

    @property
    def scale_events(self) -> int:
        """Total number of scaling actions taken this episode."""
        return self._scale_events

    @staticmethod
    def action_name(action: int) -> str:
        """Return the human-readable name for an action index."""
        return ACTION_CATALOGUE.get(action, ("UNKNOWN", 0))[0]

    @staticmethod
    def n_actions() -> int:
        """Return the total number of available actions."""
        return N_ACTIONS
