"""
milestone_06_policy_export/exporter.py
========================================
PolicyExporter: serialises a trained DQN agent's online network to disk.

Exports:
- exported_policy.pt   — TorchScript traced model (portable, no source needed)
- policy_metadata.json — obs_dim, n_actions, training info, export timestamp
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from config.logger import get_logger
from config.settings import (
    POLICY_EXPORT_DIR,
    POLICY_EXPORT_FILENAME,
    POLICY_METADATA_FILENAME,
    ENV_OBS_DIM,
)
from milestone_03_environment.action_space import N_ACTIONS
from milestone_04_dqn.network import DQNNetwork

logger = get_logger(__name__)


class PolicyExporter:
    """
    Exports a trained DQN policy for deployment/replay use.

    Parameters
    ----------
    export_dir : Path
        Directory where the exported policy and metadata are written.
    """

    def __init__(self, export_dir: Path = POLICY_EXPORT_DIR) -> None:
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        network: DQNNetwork,
        obs_dim: int = ENV_OBS_DIM,
        n_actions: int = N_ACTIONS,
        training_episodes: Optional[int] = None,
        best_reward: Optional[float] = None,
    ) -> Path:
        """
        Export the policy network as TorchScript + JSON metadata.

        Parameters
        ----------
        network : DQNNetwork
            Trained online Q-network.
        obs_dim : int
            Observation dimension.
        n_actions : int
            Number of actions.
        training_episodes : int, optional
            Number of training episodes (for metadata).
        best_reward : float, optional
            Best episode reward achieved during training.

        Returns
        -------
        Path to the exported .pt file.
        """
        network.eval()
        dummy_input = torch.zeros(1, obs_dim)
        traced = torch.jit.trace(network, dummy_input)
        policy_path = self.export_dir / POLICY_EXPORT_FILENAME
        torch.jit.save(traced, policy_path)
        logger.info("Policy exported (TorchScript): %s", policy_path)

        metadata = {
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "training_episodes": training_episodes,
            "best_reward": best_reward,
            "policy_file": POLICY_EXPORT_FILENAME,
            "action_map": {"0": "SCALE_UP", "1": "SCALE_DOWN", "2": "NO_OP"},
        }
        meta_path = self.export_dir / POLICY_METADATA_FILENAME
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("Policy metadata written: %s", meta_path)

        return policy_path

    @staticmethod
    def load(export_dir: Path = POLICY_EXPORT_DIR) -> tuple:
        """
        Load an exported policy.

        Returns
        -------
        (traced_model, metadata_dict)
        """
        policy_path = Path(export_dir) / POLICY_EXPORT_FILENAME
        meta_path   = Path(export_dir) / POLICY_METADATA_FILENAME

        model = torch.jit.load(policy_path, map_location="cpu")
        model.eval()

        with open(meta_path) as fh:
            metadata = json.load(fh)

        logger.info("Policy loaded from: %s", policy_path)
        return model, metadata

    @staticmethod
    def predict(model, observation: "np.ndarray") -> int:  # noqa: F821
        """
        Run inference on a single observation.

        Parameters
        ----------
        model : TorchScript model
        observation : np.ndarray of shape (obs_dim,)

        Returns
        -------
        int — action index
        """
        import numpy as np
        obs_t = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            q_values = model(obs_t)
        return int(q_values.argmax().item())
