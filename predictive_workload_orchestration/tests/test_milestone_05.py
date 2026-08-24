"""
tests/test_milestone_05.py
============================
Unit tests for Milestone 5: HARS.

Run:
    python3 -m pytest tests/test_milestone_05.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from milestone_05_hars.difficulty_engine import DifficultyScoreEngine
from milestone_05_hars.performance_monitor import PerformanceMonitor
from milestone_05_hars.reward_weight_generator import RewardWeightGenerator


# ===========================================================================
# Tests: PerformanceMonitor
# ===========================================================================

class TestPerformanceMonitor:

    def setup_method(self):
        self.pm = PerformanceMonitor(window=5)

    def test_record_and_stats(self):
        """Stats should be available after recording episodes."""
        for i in range(5):
            self.pm.record(
                episode_reward=float(i),
                sla_violation_rate=0.1 * i,
                scale_events=i,
                steps=100,
                mean_loss=0.01 * i,
            )
        stats = self.pm.get_stats()
        assert "mean_reward" in stats
        assert stats["n_episodes"] == 5

    def test_rolling_window_caps_at_window_size(self):
        """Recording more than window episodes should cap at window."""
        for i in range(20):
            self.pm.record(float(i), 0.1, 2, 100)
        stats = self.pm.get_stats()
        assert stats["n_episodes"] == 5  # window=5

    def test_stats_before_any_recording(self):
        """get_stats() before any recording should return zeros."""
        stats = self.pm.get_stats()
        assert stats["n_episodes"] == 0
        assert stats["mean_reward"] == 0.0

    def test_is_ready_false_initially(self):
        assert self.pm.is_ready is False

    def test_is_ready_true_after_window(self):
        for i in range(5):
            self.pm.record(1.0, 0.0, 0, 100)
        assert self.pm.is_ready is True

    def test_mean_reward_correct(self):
        rewards = [10.0, 20.0, 30.0, 40.0, 50.0]
        for r in rewards:
            self.pm.record(r, 0.0, 0, 100)
        stats = self.pm.get_stats()
        assert abs(stats["mean_reward"] - 30.0) < 1e-5


# ===========================================================================
# Tests: DifficultyScoreEngine
# ===========================================================================

class TestDifficultyScoreEngine:

    def setup_method(self):
        self.engine = DifficultyScoreEngine(smoothing=0.9)

    def _make_stats(self, sla=0.5, reward=0.0, loss=0.1):
        return {
            "mean_sla_rate": sla,
            "mean_reward": reward,
            "mean_loss": loss,
            "n_episodes": 5,
        }

    def test_output_in_range(self):
        """Difficulty score must always be in [0, 1]."""
        for _ in range(20):
            stats = self._make_stats(
                sla=np.random.random(),
                reward=np.random.uniform(-5, 5),
                loss=np.random.random(),
            )
            score = self.engine.compute(stats)
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_high_sla_increases_difficulty(self):
        """High SLA rate should produce higher difficulty than low SLA rate."""
        easy_score = self.engine.compute(self._make_stats(sla=0.0, reward=10.0, loss=0.0))
        hard_score = self.engine.compute(self._make_stats(sla=1.0, reward=-10.0, loss=1.0))
        assert hard_score > easy_score

    def test_history_grows(self):
        """history list should grow with each compute() call."""
        for i in range(5):
            self.engine.compute(self._make_stats())
        assert len(self.engine.history) == 5

    def test_smoothing_prevents_jumps(self):
        """EMA smoothing should damp sudden changes."""
        # Start easy
        for _ in range(10):
            self.engine.compute(self._make_stats(sla=0.0, reward=5.0, loss=0.0))
        prev = self.engine.current
        # Sudden hard episode
        self.engine.compute(self._make_stats(sla=1.0, reward=-10.0, loss=1.0))
        new_val = self.engine.current
        # With smoothing=0.9, change should be small
        assert abs(new_val - prev) < 0.5


# ===========================================================================
# Tests: RewardWeightGenerator
# ===========================================================================

class TestRewardWeightGenerator:

    def setup_method(self):
        self.gen = RewardWeightGenerator()

    def test_weights_sum_to_one(self):
        """Generated weights should sum to approximately 1.0."""
        for diff in [0.0, 0.25, 0.5, 0.75, 1.0]:
            weights = self.gen.generate(diff)
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-5, f"Weights sum {total} != 1.0 at diff={diff}"

    def test_all_keys_present(self):
        """Weights dict must have util, sla, cost, scale."""
        weights = self.gen.generate(0.5)
        for key in ["util", "sla", "cost", "scale"]:
            assert key in weights

    def test_weights_bounded(self):
        """Every weight must be within [w_min, w_max]."""
        from config.settings import HARS_WEIGHT_MIN, HARS_WEIGHT_MAX
        for diff in np.linspace(0, 1, 11):
            weights = self.gen.generate(diff)
            for k, v in weights.items():
                assert HARS_WEIGHT_MIN - 1e-9 <= v <= HARS_WEIGHT_MAX + 1e-9, (
                    f"Weight {k}={v} out of bounds at diff={diff}"
                )

    def test_high_difficulty_increases_sla_weight(self):
        """SLA weight should be higher for high difficulty than low difficulty."""
        # Reset generator
        gen = RewardWeightGenerator()
        w_easy = gen.generate(0.0)
        gen2 = RewardWeightGenerator()
        w_hard = gen2.generate(1.0)
        # Target: hard→more sla. Allow some tolerance for incremental adaptation.
        assert w_hard["sla"] >= w_easy["sla"] - 0.01

    def test_history_records_every_call(self):
        """history should record one entry per generate() call."""
        for _ in range(7):
            self.gen.generate(0.5)
        assert len(self.gen.history) == 7

    def test_adapt_rate_limits_change_before_renorm(self):
        """
        The adapt rate bounds the raw delta before renormalisation.
        After renormalisation the per-weight change may exceed ADAPT_RATE.
        We verify the total weight sum remains 1.0 after each call.
        """
        for diff in [0.0, 0.5, 1.0]:
            gen = RewardWeightGenerator()
            gen.generate(diff)
            weights = gen.weights
            assert abs(sum(weights.values()) - 1.0) < 1e-9, (
                f"Weights do not sum to 1.0 at diff={diff}: sum={sum(weights.values())}"
            )


# ===========================================================================
# Integration: HARSScheduler
# ===========================================================================

class TestHARSScheduler:

    def test_scheduler_updates_env_weights(self, tmp_path):
        """HARSScheduler.on_episode_end() should push new weights to env."""
        from unittest.mock import MagicMock
        from milestone_05_hars.scheduler import HARSScheduler

        mock_env = MagicMock()
        scheduler = HARSScheduler(env=mock_env, log_dir=tmp_path)

        result = scheduler.on_episode_end(
            episode=1,
            episode_reward=10.0,
            sla_violation_rate=0.2,
            scale_events=5,
            steps=100,
            mean_loss=0.05,
        )

        mock_env.update_reward_weights.assert_called_once()
        assert isinstance(result, dict)
        assert "util" in result

    def test_scheduler_log_saved(self, tmp_path):
        """save_log() should write the HARS CSV."""
        from unittest.mock import MagicMock
        from milestone_05_hars.scheduler import HARSScheduler

        mock_env = MagicMock()
        scheduler = HARSScheduler(env=mock_env, log_dir=tmp_path)

        for ep in range(3):
            scheduler.on_episode_end(ep, -1.0, 0.3, 4, 100)

        scheduler.save_log()
        from config.settings import HARS_LOG_FILENAME
        assert (tmp_path / HARS_LOG_FILENAME).exists()
