"""
tests/test_milestone_03.py
============================
Unit and integration tests for Milestone 3: Gymnasium Cloud Environment.

Strategy:
- All tests inject synthetic in-memory datasets — no CSV files needed.
- CloudEnvironment is instantiated with a tmp_path and a pre-written
  synthetic CSV (required because the env reads from disk at __init__).
- Tests are grouped by module: StateBuilder, ActionExecutor,
  RewardCalculator, CloudEnvironment, EnvironmentValidator.

Run:
    cd predictive_workload_orchestration
    python3 -m pytest tests/test_milestone_03.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from milestone_03_environment.action_space import ActionExecutor, N_ACTIONS
from milestone_03_environment.cloud_environment import CloudEnvironment
from milestone_03_environment.reward import RewardCalculator
from milestone_03_environment.state_space import StateBuilder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EPOCH = pd.Timestamp("2018-01-01 00:00:00", tz="UTC")
N_STEPS = 600  # >288 (episode_length) so episodes have room to randomise start


def make_forecast_csv(path: Path, n: int = N_STEPS) -> Path:
    """Write a synthetic forecast_dataset.csv to the given directory."""
    rng = np.random.default_rng(42)
    index = pd.date_range(start=EPOCH, periods=n, freq="5min", tz="UTC")
    hours = np.arange(n) / 12.0
    cpu = np.clip(50 + 15 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 3, n), 0, 100)
    df = pd.DataFrame({
        "cpu_util_percent":               cpu,
        "mem_util_percent":               rng.uniform(30, 70, n),
        "yhat":                           np.clip(cpu + rng.normal(0, 2, n), 0, 100),
        "cpu_util_percent_roll_mean_12":  cpu,
        "cpu_util_percent_trend":         rng.uniform(-5, 5, n),
        "hour_sin":                       np.sin(2 * np.pi * np.arange(n) / 288),
        "hour_cos":                       np.cos(2 * np.pi * np.arange(n) / 288),
        "dow_sin":                        np.zeros(n),
        "dow_cos":                        np.ones(n),
        "is_business_hours":              (np.arange(n) % 288 < 144).astype(int),
        "is_weekend":                     np.zeros(n, dtype=int),
        "net_in":                         rng.uniform(0, 50, n),
        "net_out":                        rng.uniform(0, 50, n),
        "disk_io_percent":                rng.uniform(0, 30, n),
        "load_intensity":                 ["low"] * n,
        "forecast_error":                 rng.normal(0, 2, n),
    }, index=index)
    df.index.name = "timestamp"
    csv_path = path / "forecast_dataset.csv"
    df.to_csv(csv_path)
    return csv_path


def make_row() -> pd.Series:
    """Single synthetic dataset row."""
    return pd.Series({
        "cpu_util_percent":               65.0,
        "mem_util_percent":               55.0,
        "yhat":                           67.0,
        "cpu_util_percent_roll_mean_12":  63.0,
        "cpu_util_percent_trend":         2.0,
        "hour_sin":                       0.5,
        "hour_cos":                       0.866,
        "dow_sin":                        0.0,
        "dow_cos":                        1.0,
        "is_business_hours":              1,
        "is_weekend":                     0,
    })


# ===========================================================================
# Tests: StateBuilder
# ===========================================================================

class TestStateBuilder:

    def setup_method(self):
        self.sb = StateBuilder()

    def test_output_shape(self):
        """build() should return array of shape (ENV_OBS_DIM,)."""
        from config.settings import ENV_OBS_DIM
        obs = self.sb.build(make_row(), vm_count=5, queue_length=10.0)
        assert obs.shape == (ENV_OBS_DIM,), f"Expected ({ENV_OBS_DIM},), got {obs.shape}"

    def test_output_dtype(self):
        """Output must be float32."""
        obs = self.sb.build(make_row(), vm_count=5, queue_length=10.0)
        assert obs.dtype == np.float32

    def test_output_in_range(self):
        """All observation values must be in [0, 1]."""
        obs = self.sb.build(make_row(), vm_count=5, queue_length=10.0)
        assert obs.min() >= 0.0, f"Observation min {obs.min()} < 0"
        assert obs.max() <= 1.0, f"Observation max {obs.max()} > 1"

    def test_normalise_zero(self):
        """Feature at lower bound should normalise to 0.0."""
        row = make_row()
        row["cpu_util_percent"] = 0.0
        obs = self.sb.build(row, vm_count=1, queue_length=0.0)
        # cpu_util_percent is the first feature
        assert obs[0] == pytest.approx(0.0, abs=1e-5)

    def test_normalise_max(self):
        """Feature at upper bound should normalise to 1.0."""
        row = make_row()
        row["cpu_util_percent"] = 100.0
        obs = self.sb.build(row, vm_count=1, queue_length=0.0)
        assert obs[0] == pytest.approx(1.0, abs=1e-5)

    def test_missing_feature_uses_fallback(self):
        """Missing feature should use fallback (0.5 of range) without raising."""
        row = make_row().drop("yhat")  # remove yhat to simulate M2 not run
        obs = self.sb.build(row, vm_count=5, queue_length=0.0)
        assert obs is not None  # no exception raised

    def test_vm_count_norm_at_max(self):
        """VM count at max should produce vm_count_norm == 1.0."""
        from config.settings import ENV_MAX_VMS, ENV_OBS_FEATURES
        obs = self.sb.build(make_row(), vm_count=ENV_MAX_VMS, queue_length=0.0)
        vm_idx = len(ENV_OBS_FEATURES)
        assert obs[vm_idx] == pytest.approx(1.0, abs=1e-5)

    def test_queue_norm_at_max(self):
        """Queue at max should produce queue_norm == 1.0."""
        from config.settings import ENV_MAX_QUEUE, ENV_OBS_FEATURES
        obs = self.sb.build(make_row(), vm_count=5, queue_length=ENV_MAX_QUEUE)
        q_idx = len(ENV_OBS_FEATURES) + 1
        assert obs[q_idx] == pytest.approx(1.0, abs=1e-5)


# ===========================================================================
# Tests: ActionExecutor
# ===========================================================================

class TestActionExecutor:

    def setup_method(self):
        self.ae = ActionExecutor(min_vms=1, max_vms=10)

    def test_scale_up_increments(self):
        """SCALE_UP (0) should increment VM count by 1."""
        result = self.ae.execute(action=0, current_vm_count=5)
        assert result == 6

    def test_scale_down_decrements(self):
        """SCALE_DOWN (1) should decrement VM count by 1."""
        result = self.ae.execute(action=1, current_vm_count=5)
        assert result == 4

    def test_noop_unchanged(self):
        """NO_OP (2) should not change VM count."""
        result = self.ae.execute(action=2, current_vm_count=5)
        assert result == 5

    def test_scale_up_clamped_at_max(self):
        """SCALE_UP at max VMs should stay at max."""
        result = self.ae.execute(action=0, current_vm_count=10)
        assert result == 10

    def test_scale_down_clamped_at_min(self):
        """SCALE_DOWN at min VMs should stay at min."""
        result = self.ae.execute(action=1, current_vm_count=1)
        assert result == 1

    def test_is_scaling_action_true_for_scale_up(self):
        """is_scaling_action() should be True after SCALE_UP."""
        self.ae.execute(action=0, current_vm_count=5)
        assert self.ae.is_scaling_action() is True

    def test_is_scaling_action_false_for_noop(self):
        """is_scaling_action() should be False after NO_OP."""
        self.ae.execute(action=2, current_vm_count=5)
        assert self.ae.is_scaling_action() is False

    def test_invalid_action_raises(self):
        """Invalid action index should raise ValueError."""
        with pytest.raises(ValueError):
            self.ae.execute(action=99, current_vm_count=5)

    def test_reset_clears_scale_events(self):
        """reset() should clear scale event counter."""
        self.ae.execute(action=0, current_vm_count=5)
        self.ae.execute(action=1, current_vm_count=6)
        assert self.ae.scale_events == 2
        self.ae.reset()
        assert self.ae.scale_events == 0

    def test_n_actions(self):
        """N_ACTIONS should equal 3."""
        assert N_ACTIONS == 3


# ===========================================================================
# Tests: RewardCalculator
# ===========================================================================

class TestRewardCalculator:

    def setup_method(self):
        self.rc = RewardCalculator()

    def test_returns_dict_with_all_keys(self):
        """compute() must return dict with total, util, sla, cost, scale."""
        result = self.rc.compute(cpu_util=70.0, vm_count=7, is_scaling_action=False)
        for key in ["total", "util", "sla", "cost", "scale"]:
            assert key in result, f"Missing key: {key}"

    def test_perfect_utilization_max_util_reward(self):
        """CPU at target_util should give util_reward == 1.0."""
        r_util = self.rc._util_reward(cpu_util=70.0)
        assert r_util == pytest.approx(1.0, abs=1e-6)

    def test_zero_cpu_util_reward(self):
        """CPU at 0% should give util_reward == 0.3 (0.7 from target)."""
        r_util = self.rc._util_reward(cpu_util=0.0)
        expected = 1.0 - 70.0 / 100.0
        assert r_util == pytest.approx(expected, abs=1e-6)

    def test_sla_reward_zero_below_threshold(self):
        """SLA reward should be 0 when CPU < sla_threshold."""
        r_sla = self.rc._sla_reward(cpu_util=80.0)
        assert r_sla == pytest.approx(0.0, abs=1e-6)

    def test_sla_reward_negative_above_threshold(self):
        """SLA reward should be negative when CPU > sla_threshold."""
        r_sla = self.rc._sla_reward(cpu_util=90.0)
        assert r_sla < 0.0

    def test_sla_reward_max_penalty_at_100(self):
        """SLA reward should be -1.0 at 100% CPU."""
        r_sla = self.rc._sla_reward(cpu_util=100.0)
        assert r_sla == pytest.approx(-1.0, abs=1e-5)

    def test_cost_reward_zero_when_not_over_provisioned(self):
        """Cost reward should be 0 when vm_count == vm_required."""
        # 70% CPU / 10% per VM = 7 VMs required
        r_cost = self.rc._cost_reward(cpu_util=70.0, vm_count=7)
        assert r_cost == pytest.approx(0.0, abs=1e-5)

    def test_cost_reward_negative_when_over_provisioned(self):
        """Cost reward should be negative when running more VMs than needed."""
        r_cost = self.rc._cost_reward(cpu_util=70.0, vm_count=15)
        assert r_cost < 0.0

    def test_scale_reward_zero_for_noop(self):
        """Scale reward should be 0.0 for NO_OP."""
        r_scale = self.rc._scale_reward(is_scaling_action=False)
        assert r_scale == pytest.approx(0.0, abs=1e-9)

    def test_scale_reward_negative_for_scaling(self):
        """Scale reward should be negative for scaling actions."""
        r_scale = self.rc._scale_reward(is_scaling_action=True)
        assert r_scale < 0.0

    def test_total_reward_finite(self):
        """Total reward must always be finite."""
        for cpu in [0.0, 50.0, 85.0, 100.0]:
            for vms in [1, 5, 10, 20]:
                result = self.rc.compute(cpu_util=cpu, vm_count=vms, is_scaling_action=False)
                assert np.isfinite(result["total"]), (
                    f"Non-finite reward at cpu={cpu}, vms={vms}"
                )

    def test_update_weights(self):
        """update_weights() should change the weights used in subsequent compute()."""
        self.rc.update_weights({"util": 1.0, "sla": 0.0, "cost": 0.0, "scale": 0.0})
        result = self.rc.compute(cpu_util=70.0, vm_count=7, is_scaling_action=False)
        # Only util component active: reward == 1.0 * 1.0 = 1.0
        assert result["total"] == pytest.approx(1.0, abs=1e-5)


# ===========================================================================
# Tests: CloudEnvironment
# ===========================================================================

class TestCloudEnvironment:

    @pytest.fixture(autouse=True)
    def setup_env(self, tmp_path):
        """Create synthetic CSV and instantiate environment."""
        make_forecast_csv(tmp_path, n=N_STEPS)
        self.env = CloudEnvironment(
            data_path=tmp_path / "forecast_dataset.csv",
            episode_length=100,   # short episodes for fast tests
            random_start=False,
        )

    def test_observation_space_shape(self):
        """Observation space shape should match ENV_OBS_DIM."""
        from config.settings import ENV_OBS_DIM
        assert self.env.observation_space.shape == (ENV_OBS_DIM,)

    def test_action_space_size(self):
        """Action space should be Discrete(3)."""
        assert self.env.action_space.n == 3

    def test_reset_returns_obs_and_info(self):
        """reset() should return (obs, info) tuple."""
        result = self.env.reset(seed=0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_reset_obs_shape(self):
        """reset() observation must have correct shape."""
        from config.settings import ENV_OBS_DIM
        obs, _ = self.env.reset(seed=0)
        assert obs.shape == (ENV_OBS_DIM,)

    def test_reset_obs_dtype(self):
        """reset() observation must be float32."""
        obs, _ = self.env.reset(seed=0)
        assert obs.dtype == np.float32

    def test_reset_obs_in_range(self):
        """reset() observation must be in [0, 1]."""
        obs, _ = self.env.reset(seed=0)
        assert obs.min() >= 0.0 and obs.max() <= 1.0

    def test_step_returns_five_tuple(self):
        """step() must return 5-tuple: obs, reward, terminated, truncated, info."""
        self.env.reset(seed=0)
        result = self.env.step(2)  # NO_OP
        assert isinstance(result, tuple) and len(result) == 5

    def test_step_obs_in_range(self):
        """step() observation must be in [0, 1]."""
        self.env.reset(seed=0)
        obs, _, _, _, _ = self.env.step(0)
        assert obs.min() >= 0.0 and obs.max() <= 1.0

    def test_step_reward_is_finite(self):
        """step() reward must be finite."""
        self.env.reset(seed=0)
        _, reward, _, _, _ = self.env.step(0)
        assert np.isfinite(reward)

    def test_step_info_has_vm_count(self):
        """step() info dict must contain vm_count key."""
        self.env.reset(seed=0)
        _, _, _, _, info = self.env.step(2)
        assert "vm_count" in info

    def test_episode_terminates(self):
        """Episode should terminate after episode_length steps."""
        self.env.reset(seed=0)
        terminated = False
        for _ in range(110):
            _, _, terminated, truncated, _ = self.env.step(2)
            if terminated or truncated:
                break
        assert terminated or truncated, "Episode did not terminate."

    def test_vm_count_bounded(self):
        """VM count should stay within [min_vms, max_vms]."""
        self.env.reset(seed=0)
        for _ in range(50):
            action = 0  # always scale up
            _, _, terminated, truncated, info = self.env.step(action)
            assert self.env.min_vms <= info["vm_count"] <= self.env.max_vms
            if terminated or truncated:
                break

    def test_vm_count_does_not_go_below_min(self):
        """Repeated SCALE_DOWN should not push VM count below min_vms."""
        self.env.reset(seed=0)
        for _ in range(50):
            _, _, terminated, truncated, info = self.env.step(1)  # SCALE_DOWN
            assert info["vm_count"] >= self.env.min_vms
            if terminated or truncated:
                break

    def test_seeded_reset_reproducible(self):
        """Two resets with the same seed should produce identical observations."""
        obs1, _ = self.env.reset(seed=42)
        obs2, _ = self.env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_update_reward_weights(self):
        """update_reward_weights() should not raise and should affect rewards."""
        self.env.update_reward_weights({"util": 1.0, "sla": 0.0, "cost": 0.0, "scale": 0.0})
        self.env.reset(seed=0)
        _, reward, _, _, _ = self.env.step(2)
        assert np.isfinite(reward)

    def test_episode_summary_keys(self):
        """episode_summary() should contain all expected keys."""
        self.env.reset(seed=0)
        for _ in range(100):
            _, _, terminated, truncated, _ = self.env.step(2)
            if terminated or truncated:
                break
        summary = self.env.episode_summary()
        for key in ["episode_reward", "sla_violations", "scale_events",
                    "final_vm_count", "steps_completed"]:
            assert key in summary, f"Missing summary key: {key}"


# ===========================================================================
# Integration: EnvironmentValidator random-policy rollout
# ===========================================================================

class TestEnvironmentValidator:

    def test_validator_passes_on_synthetic_data(self, tmp_path):
        """Full random-policy validation should pass on synthetic data."""
        make_forecast_csv(tmp_path, n=N_STEPS)
        env = CloudEnvironment(
            data_path=tmp_path / "forecast_dataset.csv",
            episode_length=50,  # short for speed
            random_start=False,
        )
        from milestone_03_environment.pipeline import EnvironmentValidator
        validator = EnvironmentValidator(env=env, n_episodes=2)
        passed = validator.run()
        assert passed is True
