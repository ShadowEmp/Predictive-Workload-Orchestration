import asyncio
import uuid
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from cloudverse.backend.events.bus import event_bus
from cloudverse.backend.models.events import CloudVerseEvent

# Import from existing project
import sys
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from milestone_03_environment.cloud_environment import CloudEnvironment
from milestone_09_experiments.strategies import dqn_strategy
from config.settings import PROJECT_ROOT

NUM_HOSTS = 5  # 5 sector/host districts in the city

class SimulationBridge:
    def __init__(self, simulation_id: str = None):
        self.simulation_id = simulation_id or str(uuid.uuid4())
        self.env = None
        self.policy = None
        self.is_running = False
        self.is_paused = False
        self.speed_multiplier = 1.0
        self.base_step_delay = 1.0  # seconds per simulation step at 1x

        self.step_idx = 0
        self.prev_vm_count = 0
        self.prev_queue_length = 0

        # --- Challenge State ---
        self.failed_host: int = -1          # -1 = no failure; 0-4 = host index
        self.flash_sale_active: bool = False
        self.flash_sale_multiplier: float = 1.0
        self.memory_pressure_steps_left: int = 0

    # ------------------------------------------------------------------
    # Challenge control methods (called by FastAPI routes)
    # ------------------------------------------------------------------

    def inject_host_failure(self) -> dict:
        """Take a random host offline."""
        self.failed_host = random.randint(0, NUM_HOSTS - 1)
        return {"failed_host": self.failed_host}

    def inject_flash_sale(self, multiplier: float = 4.0) -> dict:
        """Toggle flash sale traffic spike."""
        self.flash_sale_active = not getattr(self, "flash_sale_active", False)
        self.flash_sale_multiplier = multiplier if self.flash_sale_active else 1.0
        return {"multiplier": self.flash_sale_multiplier, "active": self.flash_sale_active}

    def inject_memory_pressure(self, duration_steps: int = 15) -> dict:
        """Apply memory pressure for a number of steps."""
        self.memory_pressure_steps_left = duration_steps
        return {"duration_steps": duration_steps}

    def clear_host_failure(self):
        self.failed_host = -1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_host_traffic(self, cpu_util: float, queue_length: float, vm_count: int) -> dict:
        """
        Derive per-host request rates from simulation telemetry.

        Request-traffic model (research-consistent):
          - queue_length  = unprocessed requests waiting → direct proxy for inbound request volume
          - cpu_util      = active processing load         → proxy for request service rate
          - request_rate  = queue_length * 10 + cpu_util * 2
          (combined, this gives a 0–500+ range that maps naturally to car density)

        The 5 hosts share load based on their active VM proportion + small random variance.
        """
        base_rate = float(queue_length) * 10.0 + float(cpu_util) * 2.0

        # Flash-sale spike multiplier
        if getattr(self, "flash_sale_active", False):
            base_rate *= self.flash_sale_multiplier

        # Memory pressure: effectively slows processing → queue backs up → rate appears higher
        # but we'll flag it separately so the frontend can also slow cars down
        if self.memory_pressure_steps_left > 0:
            base_rate *= 1.5  # queue builds faster under memory pressure

        # Distribute across NUM_HOSTS with per-host variance
        # Hosts with more VMs get proportionally more traffic
        vms_per_host = max(1, vm_count) / NUM_HOSTS
        host_weights = []
        for h in range(NUM_HOSTS):
            w = 1.0 + (random.gauss(0, 0.15))  # ±15% natural variance
            # Failed host gets zero weight
            if h == self.failed_host:
                w = 0.0
            host_weights.append(max(0.0, w))

        total_weight = sum(host_weights) or 1.0
        host_traffic = [base_rate * (w / total_weight) for w in host_weights]

        # Normalise to 0–1 for the frontend (max_expected_rate = 500)
        MAX_RATE = 500.0
        host_traffic_norm = [min(1.0, r / MAX_RATE) for r in host_traffic]
        request_rate_norm = min(1.0, base_rate / MAX_RATE)

        return {
            "request_rate": round(request_rate_norm, 4),
            "request_rate_raw": round(base_rate, 2),
            "host_traffic": [round(v, 4) for v in host_traffic_norm],
            "failed_host": self.failed_host,
            "flash_sale_active": getattr(self, "flash_sale_active", False),
            "flash_sale_steps_left": 9999 if getattr(self, "flash_sale_active", False) else 0,
            "memory_pressure_active": self.memory_pressure_steps_left > 0,
            "memory_pressure_steps_left": self.memory_pressure_steps_left,
        }

    # ------------------------------------------------------------------

    def initialize(self):
        # Load environment and policy
        dataset_path = PROJECT_ROOT / "data" / "processed" / "forecast_dataset.csv"
        df = pd.read_csv(dataset_path)

        # Instantiate CloudEnvironment
        self.env = CloudEnvironment(
            data_path=dataset_path,
            episode_length=288,  # typically 24h at 5min steps
            min_vms=1,
            max_vms=20,
            initial_vms=5
        )

        policy_path = PROJECT_ROOT / "data" / "models" / "exported_policy.pt"
        self.policy = dqn_strategy(policy_path)
        self.prev_vm_count = 5

    async def _emit(self, event_type: str, payload: dict, severity: str = "info"):
        event = CloudVerseEvent(
            simulation_id=self.simulation_id,
            simulation_time=self.step_idx,
            event_type=event_type,
            source="cloudsim",
            severity=severity,
            payload=payload
        )
        await event_bus.publish(event)

    async def start(self):
        if not self.env:
            self.initialize()

        self.is_running = True
        self.is_paused = False

        obs, _ = self.env.reset(seed=42)
        await self._emit("SIMULATION_STARTED", {"message": "Simulation initialized"})

        while self.is_running:
            if self.is_paused:
                await asyncio.sleep(0.1)
                continue

            # Execute step
            action = self.policy(obs)

            # For logging Q-values if we could extract them, but the TorchScript policy only returns argmax.
            # We'll emit DQN_DECISION
            action_name = "SCALE_UP" if action == 0 else "SCALE_DOWN" if action == 1 else "NO_OP"
            await self._emit("DQN_DECISION", {"action": action, "action_name": action_name})

            obs, reward, terminated, truncated, info = self.env.step(action)
            self.step_idx += 1

            # Tick down challenge countdowns

            if self.memory_pressure_steps_left > 0:
                self.memory_pressure_steps_left -= 1
                if self.memory_pressure_steps_left == 0:
                    await self._emit("MEMORY_PRESSURE_ENDED", {"message": "Memory pressure resolved"})

            cpu_util = info.get("cpu_util", 0)
            current_vms = info.get("vm_count", self.prev_vm_count)
            current_queue = info.get("queue_length", 0)

            # Compute per-host traffic
            traffic_payload = self._compute_host_traffic(cpu_util, current_queue, current_vms)

            # Emit WORKLOAD_UPDATED with full traffic data
            await self._emit("WORKLOAD_UPDATED", {
                "cpu_util": cpu_util,
                "vm_count": current_vms,
                **traffic_payload,
            })

            # Check VM Count changes
            if current_vms > self.prev_vm_count:
                for i in range(current_vms - self.prev_vm_count):
                    await self._emit("VM_CREATED", {"vm_count": self.prev_vm_count + i + 1})
            elif current_vms < self.prev_vm_count:
                for i in range(self.prev_vm_count - current_vms):
                    await self._emit("VM_DELETED", {"vm_count": self.prev_vm_count - i - 1})
            self.prev_vm_count = current_vms

            # Queue changes
            if current_queue > self.prev_queue_length:
                await self._emit("QUEUE_INCREASED", {"queue_length": current_queue}, severity="warning")
            elif current_queue < self.prev_queue_length:
                await self._emit("QUEUE_DECREASED", {"queue_length": current_queue})
            self.prev_queue_length = current_queue

            if terminated or truncated:
                await self._emit("SIMULATION_COMPLETED", {"summary": self.env.episode_summary()})
                self.is_running = False
                break

            await asyncio.sleep(self.base_step_delay / self.speed_multiplier)

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        self.is_running = False

    def set_speed(self, speed: float):
        self.speed_multiplier = max(0.1, float(speed))
