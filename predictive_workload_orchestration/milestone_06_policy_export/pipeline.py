"""
milestone_06_policy_export/pipeline.py
========================================
Milestone 6 CLI — exports the trained DQN policy to TorchScript + metadata.

Loads the best DQN model from data/models/dqn_best_model.pt, traces it to
a portable TorchScript file, and writes a JSON metadata sidecar.

Usage:
    python -m milestone_06_policy_export.pipeline
    python -m milestone_06_policy_export.pipeline --model-path data/models/dqn_best_model.pt
    python -m milestone_06_policy_export.pipeline --verify
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logger import get_logger
from config.settings import (
    DQN_BEST_MODEL_FILENAME,
    DQN_MODEL_DIR,
    ENV_OBS_DIM,
    POLICY_EXPORT_DIR,
)
from milestone_03_environment.action_space import N_ACTIONS, ACTION_CATALOGUE
from milestone_04_dqn.agent import DQNAgent
from milestone_04_dqn.network import DQNNetwork
from milestone_06_policy_export.exporter import PolicyExporter

logger = get_logger(__name__)


class PolicyExportPipeline:
    """
    Loads a trained DQN checkpoint and exports it as a portable TorchScript model.

    Parameters
    ----------
    model_path : Path
        Path to the .pt checkpoint file (default: data/models/dqn_best_model.pt).
    export_dir : Path
        Directory to write exported_policy.pt and policy_metadata.json.
    verify : bool
        If True, reload the exported model and compare actions to the original
        for 100 random observations to confirm reproducibility.
    """

    def __init__(
        self,
        model_path: Path = None,
        export_dir: Path = POLICY_EXPORT_DIR,
        verify: bool = True,
    ) -> None:
        self.model_path = Path(model_path or DQN_MODEL_DIR / DQN_BEST_MODEL_FILENAME)
        self.export_dir = Path(export_dir)
        self.verify = verify

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """
        Run the export pipeline.

        Returns
        -------
        bool — True if export and (optional) verification both pass.
        """
        logger.info("=" * 60)
        logger.info("MILESTONE 6 — POLICY EXPORT PIPELINE STARTED")
        logger.info("=" * 60)

        # --- Step 1: Load checkpoint ---
        if not self.model_path.exists():
            logger.error(
                "Model checkpoint not found: %s\n"
                "Run Milestone 4 or 5 first to train the DQN.",
                self.model_path,
            )
            return False

        logger.info("[1/3] Loading checkpoint: %s", self.model_path)
        agent = DQNAgent(obs_dim=ENV_OBS_DIM, n_actions=N_ACTIONS, device="cpu")
        agent.load(self.model_path)
        logger.info(
            "      Checkpoint loaded | steps=%d | epsilon=%.4f",
            agent.total_steps, agent.epsilon,
        )

        # --- Step 2: Export ---
        logger.info("[2/3] Exporting TorchScript policy to: %s", self.export_dir)
        exporter = PolicyExporter(export_dir=self.export_dir)
        policy_path = exporter.export(
            network=agent.online_net,
            obs_dim=ENV_OBS_DIM,
            n_actions=N_ACTIONS,
            training_episodes=agent.total_steps,
        )
        logger.info("      Exported: %s", policy_path)

        # --- Step 3: Verify (optional) ---
        passed = True
        if self.verify:
            logger.info("[3/3] Verifying exported policy reproduces training actions...")
            passed = self._verify(agent.online_net, policy_path)
        else:
            logger.info("[3/3] Verification skipped.")

        result = "PASSED" if passed else "FAILED"
        logger.info("=" * 60)
        logger.info("MILESTONE 6 COMPLETE — Export verification: %s", result)
        logger.info("  TorchScript : %s", policy_path)
        logger.info(
            "  Metadata    : %s",
            self.export_dir / "policy_metadata.json",
        )
        logger.info("=" * 60)
        return passed

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify(self, original_net: DQNNetwork, policy_path: Path) -> bool:
        """
        Load the exported TorchScript model and compare its greedy actions
        to the original network on 100 random observations.

        A mismatch here indicates a TorchScript tracing bug.
        """
        import torch

        ts_model, metadata = PolicyExporter.load(self.export_dir)
        logger.info(
            "Loaded exported policy: obs_dim=%d  n_actions=%d",
            metadata["obs_dim"], metadata["n_actions"],
        )

        rng = np.random.default_rng(42)
        mismatches = 0
        n_tests = 100

        original_net.eval()
        for i in range(n_tests):
            obs = rng.random(ENV_OBS_DIM).astype(np.float32)

            # Original network action
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                orig_action = int(original_net(obs_t).argmax().item())

            # Exported TorchScript action
            ts_action = PolicyExporter.predict(ts_model, obs)

            if orig_action != ts_action:
                mismatches += 1
                logger.warning(
                    "  Mismatch at obs %d: original=%s (%d)  exported=%s (%d)",
                    i,
                    ACTION_CATALOGUE[orig_action][0], orig_action,
                    ACTION_CATALOGUE[ts_action][0], ts_action,
                )

        if mismatches == 0:
            logger.info(
                "PASS — Exported policy matches original on all %d test observations.",
                n_tests,
            )
            return True
        else:
            logger.error(
                "FAIL — %d / %d observations produced different actions.",
                mismatches, n_tests,
            )
            return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Milestone 6: Export trained DQN policy to TorchScript"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            f"Path to DQN checkpoint (default: "
            f"{DQN_MODEL_DIR / DQN_BEST_MODEL_FILENAME})"
        ),
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=POLICY_EXPORT_DIR,
        help=f"Directory for exported files (default: {POLICY_EXPORT_DIR})",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip action-reproducibility verification after export.",
    )
    args = parser.parse_args()

    pipeline = PolicyExportPipeline(
        model_path=args.model_path,
        export_dir=args.export_dir,
        verify=not args.no_verify,
    )
    success = pipeline.run()
    sys.exit(0 if success else 1)
