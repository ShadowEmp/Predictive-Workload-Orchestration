"""
milestone_03_environment/reward.py
=====================================
RewardCalculator: computes the composite reward at each environment step.

Reward formula:
    R(t) = w_util  × R_util(t)
         + w_sla   × R_sla(t)
         + w_cost  × R_cost(t)
         + w_scale × R_scale(t)

Component definitions:
    R_util  = 1 - |cpu_util - target_util| / 100
              ∈ [0, 1], peaks at 1.0 when cpu_util == target_util

    R_sla   = -max(0, cpu_util - sla_threshold) / (100 - sla_threshold)
              ∈ [-1, 0], 0 when under threshold, -1 at cpu_util = 100%

    R_cost  = -(vm_count - vm_required) / max_vms   if vm_count > vm_required
            = 0                                      otherwise
              ∈ [-1, 0], penalises over-provisioning only

    R_scale = -scaling_cost   if action is scale-up or scale-down
            = 0               if no-op
              ∈ [-scaling_cost, 0]

Design notes:
- All components are normalised to approximately [-1, 1] before weighting
  so no single component dominates due to scale differences.
- vm_required = ceil(cpu_util / target_util_per_vm), where
  target_util_per_vm = ENV_TARGET_CPU_UTIL / ENV_CPU_PER_VM.
  This is the minimum number of VMs needed to keep each VM at target utilization.
- This fixed-weight reward is intentionally simple. HARS (Milestone 5) replaces
  the weights with dynamically computed values based on training progress.
  The component functions themselves do NOT change — only the weights change.
- The RewardCalculator is stateless per step: it does not maintain episode
  history. The CloudEnvironment accumulates episode totals.
"""

import math
from typing import Dict, Optional

import numpy as np

from config.logger import get_logger
from config.settings import (
    ENV_CPU_PER_VM,
    ENV_MAX_VMS,
    ENV_REWARD_WEIGHTS,
    ENV_SCALING_COST,
    ENV_SLA_THRESHOLD,
    ENV_TARGET_CPU_UTIL,
)

logger = get_logger(__name__)


class RewardCalculator:
    """
    Computes the step reward from the current environment state.

    Parameters
    ----------
    weights : dict
        Reward component weights: {"util", "sla", "cost", "scale"}.
        Defaults to ENV_REWARD_WEIGHTS from settings.
    target_cpu_util : float
        Ideal cluster CPU utilization percentage.
    sla_threshold : float
        CPU utilization above which SLA is considered violated.
    cpu_per_vm : float
        CPU capacity handled by one VM (as a % of total cluster).
    max_vms : int
        Maximum VM count (denominator for cost normalisation).
    scaling_cost : float
        Fixed penalty per scaling action.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        target_cpu_util: float = ENV_TARGET_CPU_UTIL,
        sla_threshold: float = ENV_SLA_THRESHOLD,
        cpu_per_vm: float = ENV_CPU_PER_VM,
        max_vms: int = ENV_MAX_VMS,
        scaling_cost: float = ENV_SCALING_COST,
    ) -> None:
        self.weights = weights if weights is not None else dict(ENV_REWARD_WEIGHTS)
        self.target_cpu_util = target_cpu_util
        self.sla_threshold = sla_threshold
        self.cpu_per_vm = cpu_per_vm
        self.max_vms = max_vms
        self.scaling_cost = scaling_cost

        # Derived constant: VMs needed per 1% CPU utilization
        self._vm_per_pct = 1.0 / cpu_per_vm

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        cpu_util: float,
        vm_count: int,
        is_scaling_action: bool,
    ) -> Dict[str, float]:
        """
        Compute the composite reward and all component values.

        Parameters
        ----------
        cpu_util : float
            Current cluster CPU utilization percentage [0, 100].
        vm_count : int
            Current number of running VMs.
        is_scaling_action : bool
            True if the last action was SCALE_UP or SCALE_DOWN.

        Returns
        -------
        dict with keys:
            "total"  — weighted sum of all components
            "util"   — utilization efficiency component
            "sla"    — SLA penalty component
            "cost"   — cost (over-provisioning) component
            "scale"  — scaling churn penalty component
        """
        r_util  = self._util_reward(cpu_util)
        r_sla   = self._sla_reward(cpu_util)
        r_cost  = self._cost_reward(cpu_util, vm_count)
        r_scale = self._scale_reward(is_scaling_action)

        total = (
            self.weights["util"]  * r_util
            + self.weights["sla"]  * r_sla
            + self.weights["cost"] * r_cost
            + self.weights["scale"] * r_scale
        )

        return {
            "total":  float(total),
            "util":   float(r_util),
            "sla":    float(r_sla),
            "cost":   float(r_cost),
            "scale":  float(r_scale),
        }

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update reward component weights (called by HARS in Milestone 5).

        Parameters
        ----------
        new_weights : dict
            New weight values for each component.
            Values should sum to approximately 1.0 for interpretability,
            but this is not enforced (HARS may use non-normalised weights).
        """
        self.weights.update(new_weights)
        logger.debug("Reward weights updated: %s", self.weights)

    # ------------------------------------------------------------------
    # Reward components
    # ------------------------------------------------------------------

    def _util_reward(self, cpu_util: float) -> float:
        """
        Utilization efficiency reward.

        Measures how close the cluster is to the target utilization.
        Peak reward (1.0) at cpu_util == target_util.
        Linear decay to 0.0 at the extremes (0% or 100% CPU).
        """
        return 1.0 - abs(cpu_util - self.target_cpu_util) / 100.0

    def _sla_reward(self, cpu_util: float) -> float:
        """
        SLA violation penalty.

        Zero when cpu_util <= sla_threshold.
        Linearly negative as cpu_util exceeds the threshold.
        Minimum -1.0 at cpu_util = 100%.
        """
        breach = max(0.0, cpu_util - self.sla_threshold)
        denominator = max(100.0 - self.sla_threshold, 1e-9)
        return -(breach / denominator)

    def _cost_reward(self, cpu_util: float, vm_count: int) -> float:
        """
        Over-provisioning cost penalty.

        Computes the minimum VMs required to serve current demand at target
        utilization.  Penalises only over-provisioning (excess VMs beyond
        what is needed).

        Under-provisioning is already penalised by the SLA component.
        """
        vm_required = math.ceil(cpu_util * self._vm_per_pct)
        vm_required = max(vm_required, 1)   # always need at least 1 VM
        excess = max(0, vm_count - vm_required)
        return -(excess / self.max_vms)

    def _scale_reward(self, is_scaling_action: bool) -> float:
        """
        Scaling churn penalty.

        Applies a small fixed penalty for any scaling action to discourage
        excessive oscillation (rapid scale-up / scale-down cycles).
        """
        return -self.scaling_cost if is_scaling_action else 0.0
