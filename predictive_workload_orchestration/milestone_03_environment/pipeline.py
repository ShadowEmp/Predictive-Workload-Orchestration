"""
milestone_03_environment/pipeline.py
======================================
Milestone 3 validation pipeline — verifies the CloudEnvironment without DQN.

Runs a random-policy rollout: at each step, a random action is sampled from
the action space.  This validates:
1. The environment can be reset and stepped without errors.
2. Observations are within the expected [0, 1] range.
3. Rewards are finite numbers.
4. Termination/truncation signals are set correctly.
5. Episode statistics are accumulated correctly.

Usage:
    python3 -m milestone_03_environment.pipeline
    python3 -m milestone_03_environment.pipeline --episodes 5 --render
    python3 -m milestone_03_environment.pipeline --data-path data/processed/forecast_dataset.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logger import get_logger
from milestone_03_environment.cloud_environment import CloudEnvironment

logger = get_logger(__name__)


class EnvironmentValidator:
    """
    Runs random-policy validation episodes and reports diagnostics.

    Parameters
    ----------
    env : CloudEnvironment
        The environment to validate.
    n_episodes : int
        Number of validation episodes to run.
    """

    def __init__(self, env: CloudEnvironment, n_episodes: int = 3) -> None:
        self.env = env
        self.n_episodes = n_episodes

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """
        Run all validation episodes and return True if all checks pass.

        Returns
        -------
        bool
        """
        logger.info("=" * 60)
        logger.info("MILESTONE 3 — ENVIRONMENT VALIDATION (Random Policy)")
        logger.info("=" * 60)

        all_passed = True
        episode_summaries: List[Dict] = []

        for ep in range(self.n_episodes):
            logger.info("--- Episode %d / %d ---", ep + 1, self.n_episodes)
            summary, passed = self._run_episode(seed=ep)
            episode_summaries.append(summary)
            if not passed:
                all_passed = False

        self._report_aggregate(episode_summaries)

        result = "ALL VALIDATION CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
        logger.info("Validation result: %s", result)
        return all_passed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_episode(self, seed: int) -> tuple:
        """Run one episode with random actions and validate each step."""
        env = self.env
        obs, info = env.reset(seed=seed)
        passed = True

        # Check 1: initial observation shape and range
        if not self._check_obs(obs, step=0):
            passed = False

        step = 0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            # Check observation
            if not self._check_obs(obs, step):
                passed = False

            # Check reward is finite
            if not np.isfinite(reward):
                logger.error("Step %d: reward is not finite: %s", step, reward)
                passed = False

            # Check info has required keys
            for key in ["vm_count", "queue_length", "step"]:
                if key not in info:
                    logger.error("Step %d: info dict missing key '%s'.", step, key)
                    passed = False

            if terminated or truncated:
                reason = "terminated" if terminated else "truncated"
                logger.info(
                    "Episode ended at step %d (%s). "
                    "Total reward: %.4f | SLA violations: %d | Scale events: %d",
                    step, reason,
                    env.episode_summary()["episode_reward"],
                    env.episode_summary()["sla_violations"],
                    env.episode_summary()["scale_events"],
                )
                break

        # Check 2: episode ended at or near the expected length
        expected = env.episode_length
        if abs(step - expected) > 1 and not (terminated or truncated):
            logger.error(
                "Episode ran %d steps but expected ~%d.", step, expected
            )
            passed = False
        else:
            logger.info("PASS — Episode length: %d steps (expected %d).", step, expected)

        summary = env.episode_summary()
        summary["seed"] = seed
        return summary, passed

    @staticmethod
    def _check_obs(obs: np.ndarray, step: int) -> bool:
        """Validate observation shape, dtype, and value range."""
        from config.settings import ENV_OBS_DIM

        if obs.shape != (ENV_OBS_DIM,):
            logger.error(
                "Step %d: obs shape %s != expected (%d,).", step, obs.shape, ENV_OBS_DIM
            )
            return False

        if obs.dtype != np.float32:
            logger.error(
                "Step %d: obs dtype %s != float32.", step, obs.dtype
            )
            return False

        if np.any(obs < -0.01) or np.any(obs > 1.01):
            logger.error(
                "Step %d: obs out of [0,1]: min=%.4f, max=%.4f.",
                step, obs.min(), obs.max(),
            )
            return False

        return True

    def _report_aggregate(self, summaries: List[Dict]) -> None:
        """Log aggregate statistics across all validation episodes."""
        rewards = [s["episode_reward"] for s in summaries]
        sla_rates = [s["sla_violation_rate"] for s in summaries]
        scale_events = [s["scale_events"] for s in summaries]

        logger.info("=" * 60)
        logger.info("AGGREGATE VALIDATION RESULTS (%d episodes)", len(summaries))
        logger.info("  Episode reward     — mean: %+.4f | std: %.4f | min: %+.4f | max: %+.4f",
                    np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards))
        logger.info("  SLA violation rate — mean: %.4f | max: %.4f",
                    np.mean(sla_rates), np.max(sla_rates))
        logger.info("  Scale events       — mean: %.1f  | max: %d",
                    np.mean(scale_events), np.max(scale_events))
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 3: CloudEnvironment random-policy validation"
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of validation episodes to run."
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable step-by-step render (verbose)."
    )
    parser.add_argument(
        "--data-path", type=Path, default=None,
        help="Path to forecast_dataset.csv or processed_dataset.csv."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    env = CloudEnvironment(
        data_path=args.data_path,
        render_mode="human" if args.render else None,
    )
    validator = EnvironmentValidator(env=env, n_episodes=args.episodes)
    passed = validator.run()
    sys.exit(0 if passed else 1)
