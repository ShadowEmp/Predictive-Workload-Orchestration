"""
milestone_04_dqn/agent.py
==========================
DQNAgent: wraps online network, target network, replay buffer, and
epsilon-greedy exploration into a single training agent.

Algorithm:
    For each step:
        1. Select action with epsilon-greedy policy
        2. Execute action in environment
        3. Store (s, a, r, s', done) in replay buffer
        4. If buffer ready: sample batch, compute TD target, update online network
        5. Every TARGET_UPDATE_FREQ steps: copy online → target network
        6. Decay epsilon linearly
"""

import copy
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config.logger import get_logger
from config.settings import (
    DQN_BATCH_SIZE,
    DQN_EPSILON_DECAY_STEPS,
    DQN_EPSILON_END,
    DQN_EPSILON_START,
    DQN_GAMMA,
    DQN_LEARNING_RATE,
    DQN_MIN_BUFFER_SIZE,
    DQN_TARGET_UPDATE_FREQ,
)
from milestone_04_dqn.network import DQNNetwork
from milestone_04_dqn.replay_buffer import ReplayBuffer

logger = get_logger(__name__)


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.

    Parameters
    ----------
    obs_dim : int
        Observation space dimension.
    n_actions : int
        Number of discrete actions.
    device : str
        Torch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        device: str = "cpu",
        learning_rate: float = DQN_LEARNING_RATE,
        gamma: float = DQN_GAMMA,
        epsilon_start: float = DQN_EPSILON_START,
        epsilon_end: float = DQN_EPSILON_END,
        epsilon_decay_steps: int = DQN_EPSILON_DECAY_STEPS,
        target_update_freq: int = DQN_TARGET_UPDATE_FREQ,
        buffer_size: int = 50_000,
        batch_size: int = DQN_BATCH_SIZE,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(device)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size

        # Networks
        self.online_net = DQNNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = copy.deepcopy(self.online_net).to(self.device)
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=buffer_size)

        # Counters
        self.total_steps: int = 0
        self.train_steps: int = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        training : bool
            If False, always act greedily (inference mode).
        """
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax().item())

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """
        Store transition and potentially perform one training update.

        Returns the loss value if a training step was performed, else None.
        """
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
        self._decay_epsilon()

        if len(self.buffer) < DQN_MIN_BUFFER_SIZE:
            return None

        loss = self._train_step()

        if self.total_steps % self.target_update_freq == 0:
            self._update_target()

        return loss

    def _train_step(self) -> float:
        """Sample a batch and perform one gradient update."""
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.online_net(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values (Bellman equation)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_steps += 1
        return float(loss.item())

    def _update_target(self) -> None:
        """Hard copy: online network → target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _decay_epsilon(self) -> None:
        """Linear epsilon decay from epsilon_start to epsilon_end."""
        fraction = min(1.0, self.total_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save online network weights and training state."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "train_steps": self.train_steps,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
        }, path)
        logger.info("DQN model saved: %s", path)

    def load(self, path: Path) -> None:
        """Load network weights and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.total_steps = checkpoint["total_steps"]
        self.train_steps = checkpoint["train_steps"]
        logger.info("DQN model loaded: %s", path)
