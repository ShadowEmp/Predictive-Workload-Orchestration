"""
milestone_08_cloudsim/simulation_config.py
============================================
CloudSimConfig: builds the JSON configuration file consumed by
the Java CloudSim Plus simulation.

CloudSim Plus requires a Java runtime.  This Python module:
1. Generates simulation_config.json from settings + replay action log.
2. Invokes the CloudSim Plus JAR (if available).
3. Parses the output metrics CSV.
4. Falls back to a Python-based simulation stub if Java is unavailable.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from config.logger import get_logger
from config.settings import (
    CLOUDSIM_HOST_BW_MBPS,
    CLOUDSIM_HOST_PES,
    CLOUDSIM_HOST_RAM_MB,
    CLOUDSIM_HOST_STORAGE_MB,
    CLOUDSIM_JAVA_DIR,
    CLOUDSIM_METRICS_FILENAME,
    CLOUDSIM_N_HOSTS,
    CLOUDSIM_RESULTS_DIR,
    CLOUDSIM_VM_TYPES,
    ENV_CPU_PER_VM,
)

logger = get_logger(__name__)

CLOUDSIM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class CloudSimConfig:
    """
    Generates CloudSim Plus simulation configuration.

    Parameters
    ----------
    action_log : pd.DataFrame
        Replay action log from PolicyReplayEngine.
    results_dir : Path
        Where to write simulation output.
    """

    def __init__(
        self,
        action_log: pd.DataFrame,
        results_dir: Path = CLOUDSIM_RESULTS_DIR,
    ) -> None:
        self.action_log  = action_log
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def generate_config(self) -> Path:
        """Write simulation_config.json and return its path."""
        config = {
            "datacenter": {
                "n_hosts":     CLOUDSIM_N_HOSTS,
                "host_pes":    CLOUDSIM_HOST_PES,
                "host_ram_mb": CLOUDSIM_HOST_RAM_MB,
                "host_bw":     CLOUDSIM_HOST_BW_MBPS,
                "host_storage": CLOUDSIM_HOST_STORAGE_MB,
            },
            "vm_types": CLOUDSIM_VM_TYPES,
            "scaling_actions": self.action_log[["step", "action_name", "vm_count"]].to_dict("records"),
            "workload": self.action_log[["step", "cpu_util"]].to_dict("records"),
        }
        config_path = self.results_dir / "simulation_config.json"
        with open(config_path, "w") as fh:
            json.dump(config, fh, indent=2)
        logger.info("CloudSim config written: %s", config_path)
        return config_path

    def run_simulation(self, jar_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Run CloudSim Plus simulation.

        If the JAR is available, invokes it. Otherwise, runs the Python stub.

        Returns
        -------
        pd.DataFrame — simulation metrics.
        """
        config_path = self.generate_config()
        jar = jar_path or (CLOUDSIM_JAVA_DIR / "cloudsim_runner.jar")

        if jar.exists():
            return self._run_java(jar, config_path)
        else:
            logger.warning(
                "CloudSim Plus JAR not found at %s. "
                "Running Python simulation stub.", jar,
            )
            return self._run_python_stub()

    def _run_java(self, jar_path: Path, config_path: Path) -> pd.DataFrame:
        """Invoke CloudSim Plus JAR via subprocess."""
        cmd = [
            "java", "-jar", str(jar_path),
            "--config", str(config_path),
            "--output", str(self.results_dir / CLOUDSIM_METRICS_FILENAME),
        ]
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"CloudSim Plus failed:\n{result.stderr}")
        return pd.read_csv(self.results_dir / CLOUDSIM_METRICS_FILENAME)

    def _run_python_stub(self) -> pd.DataFrame:
        """
        Python simulation stub — approximates CloudSim Plus metrics
        from the action log without running Java.

        This stub is used for:
        - Development and testing when Java is unavailable
        - CI/CD environments
        """
        log = self.action_log
        n_steps = len(log)
        rng = np.random.default_rng(42)

        rows = []
        cumulative_cost = 0.0
        cumulative_energy = 0.0

        for _, row in log.iterrows():
            vm_count   = max(1, int(row["vm_count"]))
            cpu_util   = float(row["cpu_util"])
            sla_violated = int(row.get("sla_violated", 0))

            # Energy model: power = idle + (max-idle) × utilization
            power_watts = vm_count * (50 + 200 * cpu_util / 100)
            energy_kwh  = power_watts * (5 / 60) / 1000  # 5-min step

            # Cost model: $0.05 per VM per hour
            cost = vm_count * 0.05 * (5 / 60)

            # Response time model: increases with queue
            queue = float(row.get("queue_length", 0.0))
            response_time = 100 + queue * 10 + rng.normal(0, 5)

            cumulative_cost   += cost
            cumulative_energy += energy_kwh

            rows.append({
                "step":             int(row["step"]),
                "vm_count":         vm_count,
                "cpu_util":         round(cpu_util, 4),
                "memory_util":      round(rng.uniform(40, 70), 4),
                "power_watts":      round(power_watts, 2),
                "energy_kwh":       round(energy_kwh, 6),
                "cost_usd":         round(cost, 6),
                "cumulative_cost":  round(cumulative_cost, 4),
                "cumulative_energy": round(cumulative_energy, 6),
                "response_time_ms": round(max(0, response_time), 2),
                "queue_length":     round(queue, 4),
                "sla_violated":     sla_violated,
            })

        df = pd.DataFrame(rows)
        out_path = self.results_dir / CLOUDSIM_METRICS_FILENAME
        df.to_csv(out_path, index=False)
        logger.info("Python stub simulation complete. Metrics saved: %s", out_path)
        return df
