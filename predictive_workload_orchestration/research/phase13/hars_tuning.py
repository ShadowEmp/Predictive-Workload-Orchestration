"""
research/phase13/hars_tuning.py — Phase 13 Step 3: HARS hyperparameter tuning.
Freezes Prophet and DQN backbone. Varies HARS parameters only.
"""
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config.logger import get_logger
from config.settings import (
    HARS_TUNE_GRID, TUNING_EPISODES, ENV_OBS_DIM,
    RESULTS_TUNING, DQN_BEST_MODEL_FILENAME, DQN_MODEL_DIR,
)
from milestone_03_environment.action_space import N_ACTIONS
from milestone_03_environment.cloud_environment import CloudEnvironment
from milestone_04_dqn.agent import DQNAgent
from milestone_05_hars.difficulty_engine import DifficultyScoreEngine
from milestone_05_hars.performance_monitor import PerformanceMonitor
from milestone_05_hars.reward_weight_generator import RewardWeightGenerator
from milestone_05_hars.scheduler import HARSScheduler
from research.experiment_tracker import log_experiment

logger = get_logger(__name__)
PLOTS_DIR = RESULTS_TUNING / "hars_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _run_hars_config(cfg: dict, n_episodes: int):
    env = CloudEnvironment()
    agent = DQNAgent(obs_dim=ENV_OBS_DIM, n_actions=N_ACTIONS, device="cpu")
    try:
        agent.load(DQN_MODEL_DIR / DQN_BEST_MODEL_FILENAME)
    except Exception:
        pass  # use random init if no checkpoint

    # Build custom HARS with config params
    from milestone_05_hars.performance_monitor import PerformanceMonitor
    from milestone_05_hars.difficulty_engine import DifficultyScoreEngine
    from milestone_05_hars.reward_weight_generator import RewardWeightGenerator

    monitor  = PerformanceMonitor(window=cfg["perf_window"])
    diff_eng = DifficultyScoreEngine(smoothing=cfg["smoothing"])
    wt_gen   = RewardWeightGenerator(adapt_rate=cfg["adapt_rate"],
                                     w_min=cfg["w_min"], w_max=cfg["w_max"])

    rewards, difficulties, sla_rates = [], [], []
    t0 = time.time()
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=ep)
        ep_r = 0.0; ep_loss = []
        while True:
            a = agent.select_action(obs, training=True)
            nobs, r, term, trunc, _ = env.step(a)
            loss = agent.step(obs, a, r, nobs, term or trunc)
            if loss is not None: ep_loss.append(loss)
            ep_r += r; obs = nobs
            if term or trunc: break

        s = env.episode_summary()
        monitor.record(ep_r, s["sla_violation_rate"], s["scale_events"],
                       s["steps_completed"], float(np.mean(ep_loss)) if ep_loss else 0)
        stats = monitor.get_stats()
        diff  = diff_eng.compute(stats)
        wts   = wt_gen.generate(diff)
        env.update_reward_weights(wts)
        rewards.append(ep_r)
        difficulties.append(diff)
        sla_rates.append(s["sla_violation_rate"])

    return {
        "mean_reward":     float(np.mean(rewards)),
        "std_reward":      float(np.std(rewards)),
        "mean_sla_rate":   float(np.mean(sla_rates)),
        "mean_difficulty": float(np.mean(difficulties)),
        "reward_stability": float(1 / (1 + np.std(rewards[-20:]))),
        "train_time_s":    round(time.time() - t0, 2),
    }


class HARSTuner:

    def run(self):
        logger.info("=== Phase 13-3: HARS Hyperparameter Tuning ===")
        rows = []
        for i, cfg in enumerate(HARS_TUNE_GRID):
            logger.info("Config %d/%d: %s", i+1, len(HARS_TUNE_GRID), cfg)
            res = _run_hars_config(cfg, TUNING_EPISODES)
            row = {**cfg, "config_id": i+1, **res}
            rows.append(row)
            log_experiment(phase="13-HARS", model="HARS", dataset="Alibaba-v2018",
                           hyperparameters=cfg, training_time_s=res["train_time_s"],
                           reward=res["mean_reward"], sla_rate=res["mean_sla_rate"],
                           difficulty_score=res["mean_difficulty"],
                           notes=f"HARS grid config {i+1}")

        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_TUNING / "hars_tuning_results.csv", index=False)
        self._plot_comparison(df)
        self._write_report(df)
        return df

    def _plot_comparison(self, df):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        x = range(len(df))
        for ax, col, clr, title in zip(
            axes,
            ["mean_reward","mean_sla_rate","reward_stability"],
            ["#2ecc71","#e74c3c","#f39c12"],
            ["Mean Reward","Mean SLA Rate","Reward Stability"],
        ):
            bars = ax.bar(x, df[col], color=clr, edgecolor="white", alpha=0.85)
            ax.set_title(f"HARS Tuning — {title}")
            ax.set_xticks(x); ax.set_xticklabels([f"C{i+1}" for i in x])
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "hars_tuning_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _write_report(self, df):
        best = df.loc[df["mean_reward"].idxmax()]
        rows_md = "\n".join(
            f"| C{int(r['config_id'])} | {r['perf_window']} | {r['smoothing']} | "
            f"{r['adapt_rate']} | {r['w_min']}/{r['w_max']} | "
            f"{r['mean_reward']:.4f} | {r['mean_sla_rate']*100:.2f}% |"
            for _, r in df.iterrows()
        )
        report = f"""# HARS Hyperparameter Tuning Report

**Phase:** 13 — Step 3  
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  
**Episodes per config:** {TUNING_EPISODES}

## Grid Results

| Config | PerfWin | Smooth | Adapt | W Min/Max | Reward | SLA |
|--------|---------|--------|-------|-----------|--------|-----|
{rows_md}

## Best Configuration
- perf_window: {int(best['perf_window'])}
- smoothing: {best['smoothing']}
- adapt_rate: {best['adapt_rate']}
- w_min/w_max: {best['w_min']} / {best['w_max']}
- Mean reward: {best['mean_reward']:.4f}
- SLA rate: {best['mean_sla_rate']*100:.2f}%
"""
        (RESULTS_TUNING / "hars_tuning_report.md").write_text(report)
        logger.info("hars_tuning_report.md written.")
