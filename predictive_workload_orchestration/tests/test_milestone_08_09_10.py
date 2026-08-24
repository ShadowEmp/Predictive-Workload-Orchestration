"""
tests/test_milestone_08_09_10.py
==================================
Unit tests for Milestones 8 (CloudSim), 9 (Experiments), 10 (Visualization).

All tests use synthetic in-memory data only.

Run:
    python3 -m pytest tests/test_milestone_08_09_10.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Shared fixture: synthetic action log (output of PolicyReplayEngine)
# ---------------------------------------------------------------------------

def make_action_log(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "step":         range(n),
        "action":       rng.integers(0, 3, n),
        "action_name":  ["NO_OP"] * n,
        "cpu_util":     rng.uniform(40, 90, n),
        "vm_count":     rng.integers(3, 12, n),
        "queue_length": rng.uniform(0, 20, n),
        "reward":       rng.standard_normal(n),
        "sla_violated": rng.integers(0, 2, n),
    })


def make_forecast_csv(path: Path, n: int = 400) -> Path:
    rng = np.random.default_rng(42)
    index = pd.date_range("2018-01-01", periods=n, freq="5min", tz="UTC")
    cpu = np.clip(50 + 15 * np.sin(np.arange(n) / 12) + rng.normal(0, 3, n), 0, 100)
    df = pd.DataFrame({
        "cpu_util_percent": cpu,
        "mem_util_percent": rng.uniform(30, 70, n),
        "yhat": np.clip(cpu + rng.normal(0, 2, n), 0, 100),
        "cpu_util_percent_roll_mean_12": cpu,
        "cpu_util_percent_trend": rng.uniform(-5, 5, n),
        "hour_sin": np.sin(np.arange(n) / 12),
        "hour_cos": np.cos(np.arange(n) / 12),
        "dow_sin": np.zeros(n), "dow_cos": np.ones(n),
        "is_business_hours": np.ones(n, dtype=int),
        "is_weekend": np.zeros(n, dtype=int),
        "net_in": rng.uniform(0, 50, n),
        "net_out": rng.uniform(0, 50, n),
        "disk_io_percent": rng.uniform(0, 30, n),
        "load_intensity": ["low"] * n,
        "forecast_error": rng.normal(0, 2, n),
    }, index=index)
    df.index.name = "timestamp"
    p = path / "forecast_dataset.csv"
    df.to_csv(p)
    return p


# ===========================================================================
# Tests: CloudSimConfig (Python stub)
# ===========================================================================

class TestCloudSimConfig:

    def test_generate_config_creates_json(self, tmp_path):
        """generate_config() should write simulation_config.json."""
        from milestone_08_cloudsim.simulation_config import CloudSimConfig
        log = make_action_log()
        sim = CloudSimConfig(action_log=log, results_dir=tmp_path)
        path = sim.generate_config()
        assert path.exists()
        assert path.name == "simulation_config.json"

    def test_config_json_structure(self, tmp_path):
        """simulation_config.json must contain 'datacenter', 'vm_types', 'workload'."""
        import json
        from milestone_08_cloudsim.simulation_config import CloudSimConfig
        log = make_action_log()
        path = CloudSimConfig(action_log=log, results_dir=tmp_path).generate_config()
        with open(path) as fh:
            data = json.load(fh)
        for key in ["datacenter", "vm_types", "workload", "scaling_actions"]:
            assert key in data, f"Missing key: {key}"

    def test_python_stub_returns_dataframe(self, tmp_path):
        """_run_python_stub() should return a non-empty DataFrame."""
        from milestone_08_cloudsim.simulation_config import CloudSimConfig
        log = make_action_log()
        df = CloudSimConfig(action_log=log, results_dir=tmp_path)._run_python_stub()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(log)

    def test_python_stub_has_required_columns(self, tmp_path):
        """Stub output must have energy, cost, response_time columns."""
        from milestone_08_cloudsim.simulation_config import CloudSimConfig
        log = make_action_log()
        df = CloudSimConfig(action_log=log, results_dir=tmp_path)._run_python_stub()
        for col in ["energy_kwh", "cost_usd", "response_time_ms", "cpu_util", "vm_count"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_stub_saves_csv(self, tmp_path):
        """Python stub should write cloudsim_metrics.csv."""
        from milestone_08_cloudsim.simulation_config import CloudSimConfig
        from config.settings import CLOUDSIM_METRICS_FILENAME
        log = make_action_log()
        CloudSimConfig(action_log=log, results_dir=tmp_path)._run_python_stub()
        assert (tmp_path / CLOUDSIM_METRICS_FILENAME).exists()

    def test_run_simulation_uses_stub_when_no_jar(self, tmp_path):
        """run_simulation() should fall back to Python stub when JAR absent."""
        from milestone_08_cloudsim.simulation_config import CloudSimConfig
        log = make_action_log()
        df = CloudSimConfig(action_log=log, results_dir=tmp_path).run_simulation(
            jar_path=tmp_path / "nonexistent.jar"
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_energy_is_nonnegative(self, tmp_path):
        """All energy_kwh values must be >= 0."""
        from milestone_08_cloudsim.simulation_config import CloudSimConfig
        log = make_action_log()
        df = CloudSimConfig(action_log=log, results_dir=tmp_path)._run_python_stub()
        assert (df["energy_kwh"] >= 0).all()

    def test_cost_is_nonnegative(self, tmp_path):
        """All cost_usd values must be >= 0."""
        from milestone_08_cloudsim.simulation_config import CloudSimConfig
        log = make_action_log()
        df = CloudSimConfig(action_log=log, results_dir=tmp_path)._run_python_stub()
        assert (df["cost_usd"] >= 0).all()


# ===========================================================================
# Tests: ExperimentRunner
# ===========================================================================

class TestExperimentRunner:

    def test_runner_returns_dataframe(self, tmp_path):
        """run() should return a DataFrame with one row per (strategy, episode)."""
        from milestone_03_environment.cloud_environment import CloudEnvironment
        from milestone_09_experiments.runner import ExperimentRunner
        from milestone_09_experiments.strategies import static_strategy, threshold_strategy

        make_forecast_csv(tmp_path)

        def env_factory():
            return CloudEnvironment(
                data_path=tmp_path / "forecast_dataset.csv",
                episode_length=30,
                random_start=False,
            )

        strategies = {
            "static":    static_strategy(),
            "threshold": threshold_strategy(),
        }
        runner = ExperimentRunner(
            env_factory=env_factory,
            strategies=strategies,
            n_episodes=2,
            results_dir=tmp_path,
        )
        df = runner.run()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 2 strategies × 2 episodes

    def test_runner_has_required_columns(self, tmp_path):
        """Results DataFrame must have strategy, episode_reward, sla_violation_rate."""
        from milestone_03_environment.cloud_environment import CloudEnvironment
        from milestone_09_experiments.runner import ExperimentRunner
        from milestone_09_experiments.strategies import static_strategy

        make_forecast_csv(tmp_path)

        def env_factory():
            return CloudEnvironment(
                data_path=tmp_path / "forecast_dataset.csv",
                episode_length=20, random_start=False,
            )

        df = ExperimentRunner(
            env_factory=env_factory,
            strategies={"static": static_strategy()},
            n_episodes=1,
            results_dir=tmp_path,
        ).run()

        for col in ["strategy", "episode_reward", "sla_violation_rate",
                    "estimated_cost", "scale_events"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_static_has_zero_scale_events(self, tmp_path):
        """Static strategy should produce 0 scale events (always NO_OP)."""
        from milestone_03_environment.cloud_environment import CloudEnvironment
        from milestone_09_experiments.runner import ExperimentRunner
        from milestone_09_experiments.strategies import static_strategy

        make_forecast_csv(tmp_path)

        def env_factory():
            return CloudEnvironment(
                data_path=tmp_path / "forecast_dataset.csv",
                episode_length=20, random_start=False,
            )

        df = ExperimentRunner(
            env_factory=env_factory,
            strategies={"static": static_strategy()},
            n_episodes=1,
            results_dir=tmp_path,
        ).run()

        assert df["scale_events"].sum() == 0

    def test_saves_results_csv(self, tmp_path):
        """run() should write experiment_results.csv."""
        from milestone_03_environment.cloud_environment import CloudEnvironment
        from milestone_09_experiments.runner import ExperimentRunner
        from milestone_09_experiments.strategies import static_strategy
        from config.settings import EXPERIMENT_METRICS_FILENAME

        make_forecast_csv(tmp_path)

        def env_factory():
            return CloudEnvironment(
                data_path=tmp_path / "forecast_dataset.csv",
                episode_length=20, random_start=False,
            )

        ExperimentRunner(
            env_factory=env_factory,
            strategies={"static": static_strategy()},
            n_episodes=1,
            results_dir=tmp_path,
        ).run()

        assert (tmp_path / EXPERIMENT_METRICS_FILENAME).exists()

    def test_strategies_module_functions(self):
        """All strategy factories should return callables."""
        from milestone_09_experiments.strategies import (
            static_strategy, threshold_strategy, prophet_strategy
        )
        obs = np.random.rand(13).astype(np.float32)
        for factory in [static_strategy, threshold_strategy, prophet_strategy]:
            pol = factory()
            action = pol(obs)
            assert 0 <= action <= 2, f"{factory.__name__} returned invalid action {action}"


# ===========================================================================
# Tests: ResultsPlotter
# ===========================================================================

class TestResultsPlotter:

    def _make_results_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        for strategy in ["static", "threshold", "dqn"]:
            for ep in range(3):
                rows.append({
                    "strategy":          strategy,
                    "episode":           ep,
                    "episode_reward":    float(rng.uniform(-5, 5)),
                    "mean_cpu_util":     float(rng.uniform(40, 90)),
                    "mean_vm_count":     float(rng.uniform(3, 12)),
                    "sla_violation_rate": float(rng.uniform(0, 0.5)),
                    "scale_events":      int(rng.integers(0, 50)),
                    "estimated_cost":    float(rng.uniform(0, 2)),
                    "estimated_energy_kwh": float(rng.uniform(0, 10)),
                    "final_queue_length": float(rng.uniform(0, 20)),
                    "steps":             288,
                    "sla_violations":    int(rng.integers(0, 100)),
                    "scaling_frequency": float(rng.uniform(0, 0.3)),
                })
        return pd.DataFrame(rows)

    def _make_training_log(self) -> pd.DataFrame:
        n = 50
        return pd.DataFrame({
            "episode":        range(1, n + 1),
            "episode_reward": np.cumsum(np.random.randn(n)),
            "mean_loss":      np.abs(np.random.randn(n)) * 0.1,
            "epsilon":        np.linspace(1.0, 0.05, n),
        })

    def _make_hars_log(self) -> pd.DataFrame:
        n = 50
        return pd.DataFrame({
            "episode":    range(1, n + 1),
            "difficulty": np.clip(np.cumsum(np.random.randn(n) * 0.02) + 0.5, 0, 1),
            "w_util":     np.linspace(0.4, 0.3, n),
            "w_sla":      np.linspace(0.4, 0.5, n),
            "w_cost":     np.linspace(0.15, 0.1, n),
            "w_scale":    np.linspace(0.05, 0.1, n),
        })

    def test_plot_training_reward(self, tmp_path):
        from milestone_10_visualization.plotter import ResultsPlotter
        p = ResultsPlotter(plots_dir=tmp_path)
        p.plot_training_reward(self._make_training_log())
        assert (tmp_path / "training_reward.png").exists()

    def test_plot_training_loss(self, tmp_path):
        from milestone_10_visualization.plotter import ResultsPlotter
        p = ResultsPlotter(plots_dir=tmp_path)
        p.plot_training_loss(self._make_training_log())
        assert (tmp_path / "training_loss.png").exists()

    def test_plot_epsilon_curve(self, tmp_path):
        from milestone_10_visualization.plotter import ResultsPlotter
        p = ResultsPlotter(plots_dir=tmp_path)
        p.plot_epsilon_curve(self._make_training_log())
        assert (tmp_path / "epsilon_curve.png").exists()

    def test_plot_difficulty_score(self, tmp_path):
        from milestone_10_visualization.plotter import ResultsPlotter
        p = ResultsPlotter(plots_dir=tmp_path)
        p.plot_difficulty_score(self._make_hars_log())
        assert (tmp_path / "difficulty_score.png").exists()

    def test_plot_reward_weights(self, tmp_path):
        from milestone_10_visualization.plotter import ResultsPlotter
        p = ResultsPlotter(plots_dir=tmp_path)
        p.plot_reward_weights(self._make_hars_log())
        assert (tmp_path / "reward_weights.png").exists()

    def test_plot_all_strategy_comparisons(self, tmp_path):
        from milestone_10_visualization.plotter import ResultsPlotter
        p = ResultsPlotter(plots_dir=tmp_path)
        p.plot_all_strategy_comparisons(self._make_results_df())
        # At least SLA and cost plots should exist
        assert (tmp_path / "sla_violations.png").exists()
        assert (tmp_path / "cost_comparison.png").exists()

    def test_plot_radar_chart(self, tmp_path):
        from milestone_10_visualization.plotter import ResultsPlotter
        p = ResultsPlotter(plots_dir=tmp_path)
        p.plot_radar_chart(self._make_results_df())
        assert (tmp_path / "strategy_radar.png").exists()

    def test_plot_forecast_accuracy(self, tmp_path):
        from milestone_10_visualization.plotter import ResultsPlotter
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
            "cpu_util_percent": rng.uniform(30, 90, n),
            "yhat":             rng.uniform(30, 90, n),
            "yhat_lower":       rng.uniform(25, 85, n),
            "yhat_upper":       rng.uniform(35, 95, n),
        })
        p = ResultsPlotter(plots_dir=tmp_path)
        p.plot_forecast_accuracy(df, n_samples=100)
        assert (tmp_path / "forecast_accuracy.png").exists()

    def test_plot_response_time(self, tmp_path):
        from milestone_10_visualization.plotter import ResultsPlotter
        df = pd.DataFrame({"response_time_ms": np.random.uniform(80, 300, 100)})
        p = ResultsPlotter(plots_dir=tmp_path)
        p.plot_response_time(df)
        assert (tmp_path / "response_time.png").exists()
