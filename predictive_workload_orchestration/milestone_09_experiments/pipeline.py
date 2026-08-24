"""
milestone_09_experiments/pipeline.py
=======================================
Experiment pipeline — builds strategy set and runs benchmarks.

Usage:
    python3 -m milestone_09_experiments.pipeline
    python3 -m milestone_09_experiments.pipeline --episodes 3 --data-path data/processed/forecast_dataset.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logger import get_logger
from config.settings import (
    DQN_MODEL_DIR,
    DQN_BEST_MODEL_FILENAME,
    POLICY_EXPORT_DIR,
    POLICY_EXPORT_FILENAME,
)
from milestone_03_environment.cloud_environment import CloudEnvironment
from milestone_09_experiments.runner import ExperimentRunner
from milestone_09_experiments.strategies import (
    static_strategy,
    threshold_strategy,
    prophet_strategy,
    dqn_strategy,
    prophet_dqn_strategy,
    prophet_dqn_hars_strategy,
)

logger = get_logger(__name__)


def build_strategies(policy_dir: Path) -> dict:
    """Build the strategy dictionary, gracefully skipping missing models."""
    strategies = {
        "static":    static_strategy(),
        "threshold": threshold_strategy(),
        "prophet":   prophet_strategy(),
    }

    dqn_path = policy_dir / POLICY_EXPORT_FILENAME
    if dqn_path.exists():
        strategies["dqn"]             = dqn_strategy(dqn_path)
        strategies["prophet_dqn"]     = prophet_dqn_strategy(dqn_path)
        strategies["prophet_dqn_hars"] = prophet_dqn_hars_strategy(dqn_path)
    else:
        logger.warning(
            "DQN exported policy not found at %s. "
            "Skipping dqn / prophet_dqn / prophet_dqn_hars strategies.",
            dqn_path,
        )

    return strategies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--policy-dir", type=Path, default=POLICY_EXPORT_DIR)
    args = parser.parse_args()

    def env_factory():
        return CloudEnvironment(data_path=args.data_path)

    strategies = build_strategies(args.policy_dir)
    runner = ExperimentRunner(
        env_factory=env_factory,
        strategies=strategies,
        n_episodes=args.episodes,
    )
    df = runner.run()
    print("\n=== EXPERIMENT RESULTS ===")
    print(df.groupby("strategy")[
        ["episode_reward", "sla_violation_rate", "estimated_cost",
         "scale_events", "estimated_energy_kwh"]
    ].mean().round(4).to_string())
