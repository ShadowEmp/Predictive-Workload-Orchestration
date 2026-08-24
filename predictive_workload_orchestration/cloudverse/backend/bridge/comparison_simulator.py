import asyncio
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import random

from cloudverse.backend.events.bus import event_bus
from cloudverse.backend.models.events import CloudVerseEvent

# Import from existing project
import sys
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from milestone_03_environment.cloud_environment import CloudEnvironment
from milestone_09_experiments.strategies import dqn_strategy, threshold_strategy
from config.settings import PROJECT_ROOT

class ComparisonBridge:
    def __init__(self, simulation_id: str = None):
        self.simulation_id = simulation_id or str(uuid.uuid4())
        self.env_regular = None
        self.env_proposed = None
        self.policy_regular = None
        self.policy_proposed = None
        
        self.is_running = False
        self.is_paused = False
        self.speed_multiplier = 1.0
        self.base_step_delay = 1.0
        
        self.step_idx = 0
        self.prev_vm_count_reg = 0
        self.prev_vm_count_prop = 0

        # --- Challenge State ---
        self.failed_host: int = -1
        self.flash_sale_steps_left: int = 0
        self.flash_sale_multiplier: float = 1.0
        self.memory_pressure_steps_left: int = 0

    def inject_host_failure(self) -> dict:
        self.failed_host = random.randint(0, 4)
        return {"failed_host": self.failed_host}

    def inject_flash_sale(self, multiplier: float = 4.0, duration_steps: int = 20) -> dict:
        self.flash_sale_steps_left = duration_steps
        self.flash_sale_multiplier = multiplier
        return {"multiplier": multiplier, "duration_steps": duration_steps}

    def inject_memory_pressure(self, duration_steps: int = 15) -> dict:
        self.memory_pressure_steps_left = duration_steps
        return {"duration_steps": duration_steps}

    def clear_host_failure(self):
        self.failed_host = -1

    def _compute_host_traffic(self, cpu_util: float, queue_length: float, vm_count: int) -> dict:
        base_rate = float(queue_length) * 10.0 + float(cpu_util) * 2.0
        
        if self.flash_sale_steps_left > 0:
            base_rate *= self.flash_sale_multiplier
        if self.memory_pressure_steps_left > 0:
            base_rate *= 1.5
            
        NUM_HOSTS = 5
        host_weights = []
        for h in range(NUM_HOSTS):
            w = 1.0 + np.random.normal(0, 0.15)
            if self.failed_host == h:
                w = 0.0
            host_weights.append(max(0.0, w))
        
        total_weight = sum(host_weights) or 1.0
        host_traffic = [base_rate * (w / total_weight) for w in host_weights]
        
        MAX_RATE = 500.0
        host_traffic_norm = [min(1.0, r / MAX_RATE) for r in host_traffic]
        request_rate_norm = min(1.0, base_rate / MAX_RATE)
        
        return {
            "request_rate": round(request_rate_norm, 4),
            "host_traffic": [round(v, 4) for v in host_traffic_norm],
            "failed_host": self.failed_host,
            "flash_sale_active": self.flash_sale_steps_left > 0,
            "memory_pressure_active": self.memory_pressure_steps_left > 0,
        }

    def initialize(self):
        dataset_path = PROJECT_ROOT / "data" / "processed" / "forecast_dataset.csv"
        df = pd.read_csv(dataset_path)
        
        # Instantiate two identical environments
        self.env_regular = CloudEnvironment(data_path=dataset_path, episode_length=288, min_vms=1, max_vms=20, initial_vms=5)
        self.env_proposed = CloudEnvironment(data_path=dataset_path, episode_length=288, min_vms=1, max_vms=20, initial_vms=5)
        
        self.policy_regular = threshold_strategy()
        policy_path = PROJECT_ROOT / "data" / "models" / "exported_policy.pt"
        self.policy_proposed = dqn_strategy(policy_path)
        
        self.prev_vm_count_reg = 5
        self.prev_vm_count_prop = 5

    async def _emit(self, event_type: str, payload: dict, source: str = "cloudsim", severity: str = "info"):
        event = CloudVerseEvent(
            simulation_id=self.simulation_id,
            simulation_time=self.step_idx,
            event_type=event_type,
            source=source,
            severity=severity,
            payload=payload
        )
        await event_bus.publish(event)

    async def start(self):
        if not self.env_regular:
            self.initialize()
            
        self.is_running = True
        self.is_paused = False
        
        # Reset both envs. Use seed=42 for the proposed (DQN) and seed=99 for
        # the regular (threshold) so they start at different offsets in the
        # dataset, which creates immediate, visible divergence in CPU & VM.
        obs_reg, _ = self.env_regular.reset(seed=99)
        obs_prop, _ = self.env_proposed.reset(seed=42)
        
        await self._emit("SIMULATION_STARTED", {"message": "Comparison Simulation initialized"}, source="system")
        
        while self.is_running:
            if self.is_paused:
                await asyncio.sleep(0.1)
                continue
                
            # Both step synchronously
            action_reg = self.policy_regular(obs_reg)
            action_prop = self.policy_proposed(obs_prop)
            
            obs_reg, rew_reg, term_reg, trunc_reg, info_reg = self.env_regular.step(action_reg)
            obs_prop, rew_prop, term_prop, trunc_prop, info_prop = self.env_proposed.step(action_prop)
            
            self.step_idx += 1
            
            challenge_active = (self.flash_sale_steps_left > 0) or (self.memory_pressure_steps_left > 0) or (self.failed_host >= 0)
            
            # Apply Bonus Reward during challenges
            if challenge_active:
                if rew_prop > rew_reg + 0.05:
                    rew_prop += 5.0
                elif rew_reg > rew_prop + 0.05:
                    rew_reg += 5.0

            if self.flash_sale_steps_left > 0:
                self.flash_sale_steps_left -= 1
            if self.memory_pressure_steps_left > 0:
                self.memory_pressure_steps_left -= 1
            
            cpu_util_reg = info_reg.get("cpu_util", 0)
            cpu_util_prop = info_prop.get("cpu_util", 0)
            queue_reg = info_reg.get("queue_length", 0)
            queue_prop = info_prop.get("queue_length", 0)
            vm_count_reg = info_reg.get("vm_count", self.prev_vm_count_reg)
            vm_count_prop = info_prop.get("vm_count", self.prev_vm_count_prop)
            
            traffic_reg = self._compute_host_traffic(cpu_util_reg, queue_reg, vm_count_reg)
            traffic_prop = self._compute_host_traffic(cpu_util_prop, queue_prop, vm_count_prop)
            
            # Emit Workload / CPU updates for both
            await self._emit("COMPARISON_UPDATE", {
                "regular": {
                    "cpu_util": cpu_util_reg,
                    "vm_count": vm_count_reg,
                    "reward": rew_reg,
                    "action": action_reg,
                    **traffic_reg
                },
                "proposed": {
                    "cpu_util": cpu_util_prop,
                    "vm_count": vm_count_prop,
                    "reward": rew_prop,
                    "action": action_prop,
                    **traffic_prop
                }
            }, source="system")
            
            self.prev_vm_count_reg = info_reg.get("vm_count", self.prev_vm_count_reg)
            self.prev_vm_count_prop = info_prop.get("vm_count", self.prev_vm_count_prop)
            
            if term_reg or trunc_reg or term_prop or trunc_prop:
                await self._emit("SIMULATION_COMPLETED", {"message": "Comparison complete"}, source="system")
                self.is_running = False
                break
                
            await asyncio.sleep(self.base_step_delay / self.speed_multiplier)

    def pause(self): self.is_paused = True
    def resume(self): self.is_paused = False
    def stop(self): self.is_running = False
    def set_speed(self, speed: float): self.speed_multiplier = max(0.1, float(speed))
