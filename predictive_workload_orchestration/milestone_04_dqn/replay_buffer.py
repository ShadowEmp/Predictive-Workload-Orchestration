"""
milestone_04_dqn/replay_buffer.py
===================================
Experience replay buffer for DQN training.

Stores (state, action, reward, next_state, done) tuples and supports
uniform random batch sampling.  Uses a circular buffer to cap memory.
"""

from collections import deque
from typing import Tuple

import numpy as np

from config.logger import get_logger
from config.settings import DQN_BATCH_SIZE, DQN_BUFFER_SIZE

logger = get_logger(__name__)


class ReplayBuffer:
    """
    Circular experience replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = DQN_BUFFER_SIZE) -> None:
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition."""
        self._buffer.append((
            state.astype(np.float32),
            int(action),
            float(reward),
            next_state.astype(np.float32),
            bool(done),
        ))

    def sample(
        self, batch_size: int = DQN_BATCH_SIZE
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions.

        Returns
        -------
        states, actions, rewards, next_states, dones — all as numpy arrays.
        """
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Buffer has {len(self._buffer)} transitions, "
                f"need at least {batch_size}."
            )
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]

        states      = np.array([t[0] for t in batch], dtype=np.float32)
        actions     = np.array([t[1] for t in batch], dtype=np.int64)
        rewards     = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones       = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True when buffer has enough transitions to start training."""
        from config.settings import DQN_MIN_BUFFER_SIZE
        return len(self._buffer) >= DQN_MIN_BUFFER_SIZE
