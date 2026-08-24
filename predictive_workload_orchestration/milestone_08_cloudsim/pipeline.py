"""
milestone_08_cloudsim/pipeline.py
====================================
Milestone 8 CLI — CloudSim Plus digital twin simulation.

Takes the M7 policy replay action log as input, generates a simulation
configuration, runs the CloudSim Plus JAR (or falls back to the Python
stub when Java is not available), and writes cloudsim_metrics.csv.

Execution order:
  1. Load replay_action_log.csv from M7
  2. Generate simulation_config.json
  3. Run simulation (JAR if available, Python stub otherwise)
  4. Collect and display metrics summary
  5. Validate outputs

Usage:
    python -m milestone_08_cloudsim.pipeline
    python -m milestone_08_cloudsim.pipeline --jar path/to/cloudsim_runner.jar
    python -m milestone_08_cloudsim.pipeline --replay-log data/reports/replay_action_log.csv
    python -m milestone_08_cloudsim.pipeline --no-plots
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logger import get_logger
from config.settings import (
    CLOUDSIM_JAVA_DIR,
    CLOUDSIM_METRICS_FILENAME,
    CLOUDSIM_RESULTS_DIR,
    REPORTS_DIR,
)
from milestone_08_cloudsim.simulation_config import CloudSimConfig

logger = get_logger(__name__)


class CloudSimPipeline:
    """
    Orchestrates the CloudSim Plus digital twin simulation for Milestone 8.

    Parameters
    ----------
    replay_log_path : Path, optional
        Path to replay_action_log.csv from M7.
        Defaults to data/reports/replay_action_log.csv.
    jar_path : Path, optional
        Path to cloudsim_runner.jar. If None, auto-detects in java/.
        Falls back to Python stub if JAR not found.
    results_dir : Path
        Where to write simulation outputs.
    generate_plots : bool
        Whether to generate CloudSim result plots at the end.
    """

    def __init__(
        self,
        replay_log_path: Path = None,
        jar_path: Path = None,
        results_dir: Path = CLOUDSIM_RESULTS_DIR,
        generate_plots: bool = True,
    ) -> None:
        self.replay_log_path = Path(
            replay_log_path or REPORTS_DIR / "replay_action_log.csv"
        )
        self.jar_path = (
            Path(jar_path)
            if jar_path
            else CLOUDSIM_JAVA_DIR / "cloudsim_runner.jar"
        )
        self.results_dir = Path(results_dir)
        self.generate_plots = generate_plots

        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the full CloudSim simulation pipeline.

        Returns
        -------
        pd.DataFrame — simulation metrics table.
        """
        logger.info("=" * 60)
        logger.info("MILESTONE 8 — CLOUDSIM PLUS DIGITAL TWIN STARTED")
        logger.info("=" * 60)

        # Step 1: Load M7 replay log
        logger.info("[1/4] Loading replay action log: %s", self.replay_log_path)
        action_log = self._load_replay_log()
        logger.info("      Loaded %d steps from replay log.", len(action_log))

        # Step 2: Build simulation configuration
        logger.info("[2/4] Generating simulation_config.json...")
        sim = CloudSimConfig(
            action_log=action_log,
            results_dir=self.results_dir,
        )
        config_path = sim.generate_config()
        logger.info("      Config written: %s", config_path)

        # Step 3: Run simulation
        logger.info("[3/4] Running simulation...")
        if self.jar_path.exists():
            logger.info("      JAR found — running CloudSim Plus Java simulation.")
        else:
            logger.info(
                "      JAR not found at %s.", self.jar_path
            )
            logger.info(
                "      Using Python simulation stub "
                "(energy/cost/response-time approximation)."
            )
            logger.info(
                "      To use real CloudSim Plus: place cloudsim_runner.jar "
                "in %s and rerun.", CLOUDSIM_JAVA_DIR
            )

        metrics_df = sim.run_simulation(jar_path=self.jar_path)
        logger.info(
            "      Simulation complete: %d steps × %d metrics.",
            *metrics_df.shape,
        )

        # Step 4: Display summary
        logger.info("[4/4] Simulation summary:")
        self._log_metrics_summary(metrics_df)

        # Generate plots
        if self.generate_plots:
            self._generate_plots(metrics_df)

        logger.info("=" * 60)
        logger.info("MILESTONE 8 COMPLETE")
        logger.info("  Results: %s", self.results_dir / CLOUDSIM_METRICS_FILENAME)
        logger.info("=" * 60)

        return metrics_df

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, metrics_df: pd.DataFrame) -> bool:
        """Post-run validation checks."""
        logger.info("Running M8 validation checks...")
        passed = True

        # Check 1: metrics file exists
        metrics_path = self.results_dir / CLOUDSIM_METRICS_FILENAME
        if metrics_path.exists():
            logger.info("PASS — Metrics file saved: %s", metrics_path)
        else:
            logger.error("FAIL — Metrics file not found.")
            passed = False

        # Check 2: DataFrame not empty
        if metrics_df.empty:
            logger.error("FAIL — Metrics DataFrame is empty.")
            passed = False
        else:
            logger.info("PASS — Metrics DataFrame has %d rows.", len(metrics_df))

        # Check 3: Required columns present
        required = ["step", "vm_count", "cpu_util", "energy_kwh",
                    "cost_usd", "response_time_ms"]
        missing = [c for c in required if c not in metrics_df.columns]
        if missing:
            logger.error("FAIL — Missing metric columns: %s", missing)
            passed = False
        else:
            logger.info("PASS — All required metric columns present.")

        # Check 4: Energy and cost are non-negative
        if (metrics_df["energy_kwh"] < 0).any():
            logger.error("FAIL — Negative energy values found.")
            passed = False
        else:
            logger.info("PASS — Energy values are non-negative.")

        if (metrics_df["cost_usd"] < 0).any():
            logger.error("FAIL — Negative cost values found.")
            passed = False
        else:
            logger.info("PASS — Cost values are non-negative.")

        # Check 5: VM count bounded
        from config.settings import ENV_MIN_VMS, ENV_MAX_VMS
        oob = (
            (metrics_df["vm_count"] < ENV_MIN_VMS) |
            (metrics_df["vm_count"] > ENV_MAX_VMS)
        )
        if oob.any():
            logger.error("FAIL — %d VM count values out of bounds.", oob.sum())
            passed = False
        else:
            logger.info("PASS — VM counts in [%d, %d].", ENV_MIN_VMS, ENV_MAX_VMS)

        result = "ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"
        logger.info("M8 Validation: %s", result)
        return passed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_replay_log(self) -> pd.DataFrame:
        """Load the M7 replay action log, with a clear error if missing."""
        if not self.replay_log_path.exists():
            logger.error(
                "Replay action log not found: %s\n"
                "Run Milestone 7 first: "
                "python -m milestone_07_policy_replay.pipeline",
                self.replay_log_path,
            )
            sys.exit(1)
        return pd.read_csv(self.replay_log_path)

    @staticmethod
    def _log_metrics_summary(df: pd.DataFrame) -> None:
        """Log a human-readable summary of key simulation metrics."""
        total_energy = df["energy_kwh"].sum()
        total_cost   = df["cost_usd"].sum()
        mean_cpu     = df["cpu_util"].mean()
        mean_rt      = df["response_time_ms"].mean()
        sla_count    = df["sla_violated"].sum() if "sla_violated" in df.columns else 0
        sla_rate     = sla_count / len(df) * 100 if len(df) > 0 else 0

        logger.info("  ┌─────────────────────────────────────────┐")
        logger.info("  │  CloudSim Plus Simulation Results        │")
        logger.info("  ├─────────────────────────────────────────┤")
        logger.info("  │  Total energy consumed : %8.4f kWh   │", total_energy)
        logger.info("  │  Total cost            : $%7.4f       │", total_cost)
        logger.info("  │  Mean CPU utilization  : %7.2f %%      │", mean_cpu)
        logger.info("  │  Mean response time    : %7.2f ms      │", mean_rt)
        logger.info("  │  SLA violations        : %4d / %d (%.1f%%) │",
                    sla_count, len(df), sla_rate)
        logger.info("  │  Cumulative energy     : %8.4f kWh   │",
                    df["cumulative_energy"].iloc[-1] if "cumulative_energy" in df.columns
                    else total_energy)
        logger.info("  │  Cumulative cost       : $%7.4f       │",
                    df["cumulative_cost"].iloc[-1] if "cumulative_cost" in df.columns
                    else total_cost)
        logger.info("  └─────────────────────────────────────────┘")

    def _generate_plots(self, metrics_df: pd.DataFrame) -> None:
        """Generate CloudSim result visualisations."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plots_dir = self.results_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            steps = metrics_df["step"]

            # Plot 1: Energy and Cost over time
            fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

            axes[0].plot(steps, metrics_df["cumulative_energy"],
                         color="#e74c3c", linewidth=1.5, label="Cumulative Energy (kWh)")
            axes[0].set_ylabel("Energy (kWh)")
            axes[0].set_title("CloudSim Plus — Energy and Cost")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(steps, metrics_df["cumulative_cost"],
                         color="#f39c12", linewidth=1.5, label="Cumulative Cost (USD)")
            axes[1].set_ylabel("Cost (USD)")
            axes[1].set_xlabel("Step")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(plots_dir / "energy_cost.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Plot 2: CPU utilisation + VM count + response time
            fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

            axes[0].plot(steps, metrics_df["cpu_util"],
                         color="#1f77b4", linewidth=0.9, alpha=0.85)
            axes[0].axhline(85, color="#e74c3c", linestyle="--",
                            linewidth=0.8, alpha=0.7, label="SLA 85%")
            axes[0].set_ylabel("CPU Util (%)")
            axes[0].set_title("CloudSim Plus — Infrastructure Metrics")
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.3)

            axes[1].step(steps, metrics_df["vm_count"],
                         color="#2ecc71", linewidth=1.2, where="post")
            axes[1].set_ylabel("VM Count")
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(steps, metrics_df["response_time_ms"],
                         color="#9b59b6", linewidth=0.8, alpha=0.85)
            axes[2].set_ylabel("Response Time (ms)")
            axes[2].set_xlabel("Step")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(plots_dir / "infrastructure_metrics.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Plot 3: Power consumption over time
            if "power_watts" in metrics_df.columns:
                fig, ax = plt.subplots(figsize=(14, 4))
                ax.fill_between(steps, metrics_df["power_watts"],
                                alpha=0.4, color="#e74c3c")
                ax.plot(steps, metrics_df["power_watts"],
                        color="#e74c3c", linewidth=0.8)
                ax.set_title("CloudSim Plus — Power Consumption")
                ax.set_xlabel("Step")
                ax.set_ylabel("Power (Watts)")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                fig.savefig(plots_dir / "power_consumption.png",
                            dpi=150, bbox_inches="tight")
                plt.close(fig)

            logger.info("CloudSim plots saved to: %s", plots_dir)

        except Exception as exc:
            logger.warning("Plot generation failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Milestone 8: CloudSim Plus Digital Twin Simulation"
    )
    parser.add_argument(
        "--replay-log",
        type=Path,
        default=None,
        help=(
            "Path to replay_action_log.csv from M7 "
            "(default: data/reports/replay_action_log.csv)"
        ),
    )
    parser.add_argument(
        "--jar",
        type=Path,
        default=None,
        help=(
            "Path to cloudsim_runner.jar "
            "(falls back to Python stub if not provided)"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=CLOUDSIM_RESULTS_DIR,
        help=f"Directory for simulation outputs (default: {CLOUDSIM_RESULTS_DIR})",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    args = parser.parse_args()

    pipeline = CloudSimPipeline(
        replay_log_path=args.replay_log,
        jar_path=args.jar,
        results_dir=args.results_dir,
        generate_plots=not args.no_plots,
    )
    metrics = pipeline.run()
    pipeline.validate(metrics)
