"""
milestone_04_dqn/network.py
=============================
DQN neural network — feed-forward Q-network.

Architecture:
    Input  → Linear(obs_dim, 128) → ReLU
           → Linear(128, 128)     → ReLU
           → Linear(128, n_actions)

The same architecture is used for both the online network and the
target network.  The target network weights are periodically hard-copied
from the online network.
"""

from typing import List

import torch
import torch.nn as nn

from config.logger import get_logger
from config.settings import DQN_HIDDEN_LAYERS, ENV_OBS_DIM
from milestone_03_environment.action_space import N_ACTIONS

logger = get_logger(__name__)


class DQNNetwork(nn.Module):
    """
    Feed-forward Q-network.

    Parameters
    ----------
    obs_dim : int
        Dimension of the observation (input) vector.
    n_actions : int
        Number of discrete actions (output neurons).
    hidden_layers : list[int]
        Number of neurons in each hidden layer.
    """

    def __init__(
        self,
        obs_dim: int = ENV_OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden_layers: List[int] = None,
    ) -> None:
        super().__init__()
        hidden_layers = hidden_layers or DQN_HIDDEN_LAYERS

        layers = []
        in_dim = obs_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))

        self.net = nn.Sequential(*layers)
        self._init_weights()

        logger.info(
            "DQNNetwork: obs_dim=%d → hidden=%s → n_actions=%d | params=%d",
            obs_dim, hidden_layers, n_actions,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns Q-values for all actions."""
        return self.net(x)

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
