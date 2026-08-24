"""
milestone_07_policy_replay/replay_engine.py
=============================================
PolicyReplayEngine: replays a trained policy through the environment,
logging every action and collecting step-level metrics.

Usage:
    engine = PolicyReplayEngine(policy_dir, data_path)
    results = engine.replay()
    # results is a DataFrame of step-level metrics
"""

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.logger import get_logger
from config.settings import (
    POLICY_EXPORT_DIR,
    PROCESSED_DIR,
    REPLAY_LOG_FILENAME,
    REPORTS_DIR,
)
from milestone_03_environment.cloud_environment import CloudEnvironment
from milestone_06_policy_export.exporter import PolicyExporter

logger = get_logger(__name__)


class PolicyReplayEngine:
    """
    Replays a trained policy through the CloudEnvironment.

    Parameters
    ----------
    policy_dir : Path
        Directory containing exported_policy.pt and policy_metadata.json.
    data_path : Path, optional
        Dataset path for the environment.
    log_dir : Path
        Where to save the replay action log CSV.
    random_start : bool
        If False, always start from the beginning of the dataset.
    """

    def __init__(
        self,
        policy_dir: Path = POLICY_EXPORT_DIR,
        data_path: Optional[Path] = None,
        log_dir: Path = REPORTS_DIR,
        random_start: bool = False,
    ) -> None:
        self.policy_dir  = Path(policy_dir)
        self.data_path   = data_path
        self.log_dir     = Path(log_dir)
        self.random_start = random_start
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def replay(self, seed: int = 0) -> pd.DataFrame:
        """
        Run one full episode with the trained policy.

        Returns
        -------
        pd.DataFrame
            Step-level action log with columns:
            step, action, action_name, cpu_util, vm_count,
            queue_length, reward, sla_violated
        """
        logger.info("Loading policy from: %s", self.policy_dir)
        model, metadata = PolicyExporter.load(self.policy_dir)

        env = CloudEnvironment(
            data_path=self.data_path,
            random_start=self.random_start,
        )
        obs, _ = env.reset(seed=seed)

        rows = []
        step = 0

        while True:
            action = PolicyExporter.predict(model, obs)
            obs, reward, terminated, truncated, info = env.step(action)

            from milestone_03_environment.action_space import ActionExecutor
            action_name = ActionExecutor.action_name(action)

            rows.append({
                "step":         step,
                "action":       action,
                "action_name":  action_name,
                "cpu_util":     round(info.get("cpu_util", 0.0), 4),
                "vm_count":     info["vm_count"],
                "queue_length": round(info["queue_length"], 4),
                "reward":       round(reward, 6),
                "sla_violated": int(info.get("cpu_util", 0.0) > 85.0),
            })
            step += 1

            if terminated or truncated:
                break

        log_path = self.log_dir / REPLAY_LOG_FILENAME
        df = pd.DataFrame(rows)
        df.to_csv(log_path, index=False)

        summary = env.episode_summary()
        logger.info(
            "Replay complete: %d steps | reward=%.4f | SLA violations=%d | scale_events=%d",
            step, summary["episode_reward"],
            summary["sla_violations"], summary["scale_events"],
        )
        logger.info("Replay action log saved: %s", log_path)
        return df
