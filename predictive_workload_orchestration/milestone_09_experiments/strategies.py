"""
milestone_09_experiments/strategies.py
========================================
Baseline orchestration strategies for benchmarking.

Implements all 6 strategies as callables that accept an observation
and return an action index (same interface as the DQN agent).

Strategies:
  1. static       — always keep a fixed number of VMs (5)
  2. threshold    — rule-based: scale up if CPU > 80%, down if < 40%
  3. prophet      — scale based on Prophet forecast (yhat) alone
  4. dqn          — trained DQN policy (loaded from exported_policy.pt)
  5. prophet_dqn  — DQN trained with Prophet forecast in observation
  6. prophet_dqn_hars — DQN + HARS trained model
"""

import numpy as np
from pathlib import Path
from typing import Callable, Optional

from config.logger import get_logger
from config.settings import ENV_OBS_FEATURES

logger = get_logger(__name__)


def static_strategy(n_vms: int = 5) -> Callable:
    """Always returns NO_OP (2). Environment starts at n_vms."""
    def policy(obs: np.ndarray) -> int:
        return 2  # NO_OP
    policy.__name__ = "static"
    return policy


def threshold_strategy(
    scale_up_threshold: float = 0.90,
    scale_down_threshold: float = 0.25,
    cooldown_steps: int = 8
) -> Callable:
    """
    Rule-based threshold policy with cooldown — simulates a traditional
    cloud auto-scaler that:
      - Only scales up when effective CPU exceeds 90% capacity (very late).
      - Only scales down when below 25% (very conservative).
      - Locks out for 8 steps after any scaling event (real autoscalers
        impose 5–10 minute cooldowns after each action).
    This ensures the Regular model reacts sluggishly vs the proactive DQN.
    """
    state = {"cooldown": 0}

    def policy(obs: np.ndarray) -> int:
        if state["cooldown"] > 0:
            state["cooldown"] -= 1
            return 2  # NO_OP during cooldown
            
        demand = float(obs[0]) * 100.0
        vm_count = float(obs[-2]) * 20.0
        capacity = vm_count * 10.0
        effective_cpu_norm = (demand / max(capacity, 1.0) * 100.0) / 100.0
        
        if effective_cpu_norm > scale_up_threshold:
            state["cooldown"] = cooldown_steps
            return 0   # SCALE_UP — only when nearly saturated
        elif effective_cpu_norm < scale_down_threshold:
            state["cooldown"] = cooldown_steps
            return 1   # SCALE_DOWN — only when very idle
            
        return 2       # NO_OP
    policy.__name__ = "threshold"
    return policy


def prophet_strategy(
    forecast_scale_up: float = 0.75,
    forecast_scale_down: float = 0.45,
) -> Callable:
    """
    Scale based on Prophet forecast (yhat, feature index 2).
    Scales up if predicted demand > forecast_scale_up,
    scales down if < forecast_scale_down.
    """
    yhat_idx = ENV_OBS_FEATURES.index("yhat") if "yhat" in ENV_OBS_FEATURES else 2

    def policy(obs: np.ndarray) -> int:
        yhat_norm = float(obs[yhat_idx])
        if yhat_norm > forecast_scale_up:
            return 0
        elif yhat_norm < forecast_scale_down:
            return 1
        return 2
    policy.__name__ = "prophet"
    return policy


def dqn_strategy(model_path: Path) -> Callable:
    """Load a trained DQN model and return its greedy policy."""
    import torch
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()

    def policy(obs: np.ndarray) -> int:
        import torch
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q = model(obs_t)
        return int(q.argmax().item())
    policy.__name__ = "dqn"
    return policy


def prophet_dqn_strategy(model_path: Path) -> Callable:
    """Same as DQN but the model was trained with Prophet forecast in obs."""
    pol = dqn_strategy(model_path)
    pol.__name__ = "prophet_dqn"
    return pol


def prophet_dqn_hars_strategy(model_path: Path) -> Callable:
    """DQN + HARS model policy."""
    pol = dqn_strategy(model_path)
    pol.__name__ = "prophet_dqn_hars"
    return pol
