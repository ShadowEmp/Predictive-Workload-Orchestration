"""
tests/test_milestone_06_07.py
==============================
Unit tests for Milestone 6 (Policy Export) and Milestone 7 (Policy Replay).

Run:
    python3 -m pytest tests/test_milestone_06_07.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from milestone_04_dqn.network import DQNNetwork
from milestone_06_policy_export.exporter import PolicyExporter

OBS_DIM = 13
N_ACTIONS = 3


# ===========================================================================
# Tests: PolicyExporter
# ===========================================================================

class TestPolicyExporter:

    def test_export_creates_pt_file(self, tmp_path):
        """export() should write exported_policy.pt."""
        net = DQNNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_layers=[32])
        exporter = PolicyExporter(export_dir=tmp_path)
        exporter.export(net, obs_dim=OBS_DIM, n_actions=N_ACTIONS)
        from config.settings import POLICY_EXPORT_FILENAME
        assert (tmp_path / POLICY_EXPORT_FILENAME).exists()

    def test_export_creates_metadata_json(self, tmp_path):
        """export() should write policy_metadata.json."""
        net = DQNNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_layers=[32])
        exporter = PolicyExporter(export_dir=tmp_path)
        exporter.export(net, obs_dim=OBS_DIM, n_actions=N_ACTIONS,
                        training_episodes=100, best_reward=5.0)
        from config.settings import POLICY_METADATA_FILENAME
        assert (tmp_path / POLICY_METADATA_FILENAME).exists()

    def test_metadata_contents(self, tmp_path):
        """Metadata JSON should contain obs_dim and n_actions."""
        import json
        net = DQNNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_layers=[32])
        PolicyExporter(export_dir=tmp_path).export(net, obs_dim=OBS_DIM, n_actions=N_ACTIONS)
        from config.settings import POLICY_METADATA_FILENAME
        with open(tmp_path / POLICY_METADATA_FILENAME) as fh:
            meta = json.load(fh)
        assert meta["obs_dim"] == OBS_DIM
        assert meta["n_actions"] == N_ACTIONS

    def test_load_returns_model_and_metadata(self, tmp_path):
        """load() should return (model, dict)."""
        net = DQNNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_layers=[32])
        PolicyExporter(export_dir=tmp_path).export(net, obs_dim=OBS_DIM, n_actions=N_ACTIONS)
        model, meta = PolicyExporter.load(tmp_path)
        assert model is not None
        assert isinstance(meta, dict)

    def test_predict_returns_valid_action(self, tmp_path):
        """predict() should return an action in [0, N_ACTIONS)."""
        net = DQNNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_layers=[32])
        PolicyExporter(export_dir=tmp_path).export(net, obs_dim=OBS_DIM, n_actions=N_ACTIONS)
        model, _ = PolicyExporter.load(tmp_path)
        obs = np.random.rand(OBS_DIM).astype(np.float32)
        action = PolicyExporter.predict(model, obs)
        assert 0 <= action < N_ACTIONS

    def test_predict_is_deterministic(self, tmp_path):
        """Same observation should always produce same action."""
        net = DQNNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_layers=[32])
        PolicyExporter(export_dir=tmp_path).export(net, obs_dim=OBS_DIM, n_actions=N_ACTIONS)
        model, _ = PolicyExporter.load(tmp_path)
        obs = np.ones(OBS_DIM, dtype=np.float32) * 0.5
        a1 = PolicyExporter.predict(model, obs)
        a2 = PolicyExporter.predict(model, obs)
        assert a1 == a2

    def test_exported_model_reproduces_training_actions(self, tmp_path):
        """Exported TorchScript model should match original network output."""
        net = DQNNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_layers=[32])
        net.eval()
        PolicyExporter(export_dir=tmp_path).export(net, obs_dim=OBS_DIM, n_actions=N_ACTIONS)
        ts_model, _ = PolicyExporter.load(tmp_path)

        obs = torch.randn(1, OBS_DIM)
        with torch.no_grad():
            orig_action = int(net(obs).argmax().item())
            ts_action   = PolicyExporter.predict(ts_model, obs.numpy()[0])
        assert orig_action == ts_action


# ===========================================================================
# Tests: PolicyReplayEngine
# ===========================================================================

def make_forecast_csv_m6(path: Path, n: int = 400) -> Path:
    rng = np.random.default_rng(7)
    index = pd.date_range("2018-01-01", periods=n, freq="5min", tz="UTC")
    cpu = np.clip(50 + 15 * np.sin(np.arange(n) / 12 * np.pi / 12) + rng.normal(0, 3, n), 0, 100)
    df = pd.DataFrame({
        "cpu_util_percent": cpu,
        "mem_util_percent": rng.uniform(30, 70, n),
        "yhat": np.clip(cpu + rng.normal(0, 2, n), 0, 100),
        "cpu_util_percent_roll_mean_12": cpu,
        "cpu_util_percent_trend": rng.uniform(-5, 5, n),
        "hour_sin": np.sin(np.arange(n) / 12),
        "hour_cos": np.cos(np.arange(n) / 12),
        "dow_sin": np.zeros(n),
        "dow_cos": np.ones(n),
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


class TestPolicyReplayEngine:

    def test_replay_produces_dataframe(self, tmp_path):
        """replay() should return a non-empty DataFrame."""
        # Export a random policy first
        net = DQNNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_layers=[32])
        PolicyExporter(export_dir=tmp_path).export(net, obs_dim=OBS_DIM, n_actions=N_ACTIONS)

        csv_path = make_forecast_csv_m6(tmp_path)

        from milestone_07_policy_replay.replay_engine import PolicyReplayEngine
        engine = PolicyReplayEngine(
            policy_dir=tmp_path,
            data_path=csv_path,
            log_dir=tmp_path,
            random_start=False,
        )
        df = engine.replay(seed=0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_replay_has_required_columns(self, tmp_path):
        """Replay log must contain step, action, vm_count, reward."""
        net = DQNNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_layers=[32])
        PolicyExporter(export_dir=tmp_path).export(net, obs_dim=OBS_DIM, n_actions=N_ACTIONS)
        csv_path = make_forecast_csv_m6(tmp_path)

        from milestone_07_policy_replay.replay_engine import PolicyReplayEngine
        df = PolicyReplayEngine(
            policy_dir=tmp_path, data_path=csv_path,
            log_dir=tmp_path, random_start=False,
        ).replay(seed=0)

        for col in ["step", "action", "action_name", "vm_count", "reward"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_replay_actions_valid(self, tmp_path):
        """All actions in replay log must be in {0, 1, 2}."""
        net = DQNNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_layers=[32])
        PolicyExporter(export_dir=tmp_path).export(net, obs_dim=OBS_DIM, n_actions=N_ACTIONS)
        csv_path = make_forecast_csv_m6(tmp_path)

        from milestone_07_policy_replay.replay_engine import PolicyReplayEngine
        df = PolicyReplayEngine(
            policy_dir=tmp_path, data_path=csv_path,
            log_dir=tmp_path, random_start=False,
        ).replay(seed=0)

        assert set(df["action"].unique()).issubset({0, 1, 2})

    def test_replay_saves_csv(self, tmp_path):
        """replay() should write replay_action_log.csv."""
        net = DQNNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_layers=[32])
        PolicyExporter(export_dir=tmp_path).export(net, obs_dim=OBS_DIM, n_actions=N_ACTIONS)
        csv_path = make_forecast_csv_m6(tmp_path)

        from milestone_07_policy_replay.replay_engine import PolicyReplayEngine
        from config.settings import REPLAY_LOG_FILENAME
        PolicyReplayEngine(
            policy_dir=tmp_path, data_path=csv_path,
            log_dir=tmp_path, random_start=False,
        ).replay(seed=0)

        assert (tmp_path / REPLAY_LOG_FILENAME).exists()
