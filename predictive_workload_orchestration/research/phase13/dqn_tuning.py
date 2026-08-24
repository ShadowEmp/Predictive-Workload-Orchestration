"""
research/phase13/dqn_tuning.py — Phase 13 Step 2: DQN hyperparameter tuning.
Trains DQN for TUNING_EPISODES per config, evaluates, saves best model.
"""
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config.logger import get_logger
from config.settings import (
    DQN_TUNE_GRID, TUNING_EPISODES, ENV_OBS_DIM,
    RESULTS_TUNING, RESULTS_MODELS,
)
from milestone_03_environment.action_space import N_ACTIONS
from milestone_03_environment.cloud_environment import CloudEnvironment
from milestone_04_dqn.agent import DQNAgent
from research.experiment_tracker import log_experiment

logger = get_logger(__name__)
PLOTS_DIR = RESULTS_TUNING / "dqn_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _train_and_eval(cfg: dict, n_episodes: int, n_eval: int = 3):
    env = CloudEnvironment()
    agent = DQNAgent(
        obs_dim=ENV_OBS_DIM, n_actions=N_ACTIONS, device="cpu",
        learning_rate=cfg["lr"], gamma=cfg["gamma"],
        batch_size=cfg["batch"], target_update_freq=cfg["target_freq"],
        epsilon_decay_steps=cfg["eps_decay"],
    )
    rewards = []
    t0 = time.time()
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=ep)
        ep_r = 0.0
        while True:
            action = agent.select_action(obs, training=True)
            next_obs, r, term, trunc, _ = env.step(action)
            agent.step(obs, action, r, next_obs, term or trunc)
            ep_r += r; obs = next_obs
            if term or trunc: break
        rewards.append(ep_r)
    train_time = time.time() - t0

    # Eval
    eval_rewards, sla_rates = [], []
    for seed in range(n_eval):
        env2 = CloudEnvironment()
        obs, _ = env2.reset(seed=seed + 100)
        er = 0.0
        while True:
            a = agent.select_action(obs, training=False)
            obs, r, term, trunc, _ = env2.step(a)
            er += r
            if term or trunc: break
        eval_rewards.append(er)
        sla_rates.append(env2.episode_summary()["sla_violation_rate"])

    # Convergence: first ep where rolling-20 exceeds 80% of final avg
    rolling = pd.Series(rewards).rolling(10, min_periods=1).mean().values
    final_avg = float(np.mean(rewards[-10:]))
    thresh = 0.8 * final_avg if final_avg > 0 else 0
    conv = next((i for i, v in enumerate(rolling) if v >= thresh), n_episodes)

    return {
        "mean_reward":    float(np.mean(eval_rewards)),
        "std_reward":     float(np.std(eval_rewards)),
        "mean_sla_rate":  float(np.mean(sla_rates)),
        "convergence_ep": int(conv),
        "train_time_s":   round(train_time, 2),
        "agent":          agent,
        "rewards_hist":   rewards,
    }


class DQNTuner:

    def run(self):
        logger.info("=== Phase 13-2: DQN Hyperparameter Tuning ===")
        rows = []
        best_reward = -np.inf
        best_agent = None

        for i, cfg in enumerate(DQN_TUNE_GRID):
            logger.info("Config %d/%d: %s", i+1, len(DQN_TUNE_GRID), cfg)
            res = _train_and_eval(cfg, TUNING_EPISODES)
            row = {**cfg, "config_id": i+1,
                   "mean_reward": res["mean_reward"],
                   "std_reward":  res["std_reward"],
                   "mean_sla_rate": res["mean_sla_rate"],
                   "convergence_ep": res["convergence_ep"],
                   "train_time_s": res["train_time_s"]}
            rows.append(row)
            log_experiment(phase="13-DQN", model="DQN", dataset="Alibaba-v2018",
                           hyperparameters=cfg, training_time_s=res["train_time_s"],
                           reward=res["mean_reward"], sla_rate=res["mean_sla_rate"],
                           convergence_episode=res["convergence_ep"],
                           notes=f"Grid search config {i+1}")
            if res["mean_reward"] > best_reward:
                best_reward = res["mean_reward"]
                best_agent = res["agent"]
                best_cfg = cfg

        results_df = pd.DataFrame(rows)
        results_df.to_csv(RESULTS_TUNING / "dqn_tuning_results.csv", index=False)

        if best_agent is not None:
            best_agent.save(RESULTS_MODELS / "best_dqn_model.pt")
            logger.info("Best DQN saved. Reward=%.4f  Config=%s", best_reward, best_cfg)

        self._plot_comparison(results_df)
        self._write_report(results_df)
        return results_df

    def _plot_comparison(self, df):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        x = range(len(df))
        for ax, col, clr, title in zip(
            axes,
            ["mean_reward","mean_sla_rate","convergence_ep"],
            ["#2ecc71","#e74c3c","#3498db"],
            ["Mean Reward","Mean SLA Rate","Convergence Episode"],
        ):
            bars = ax.bar(x, df[col], color=clr, edgecolor="white", alpha=0.85)
            ax.set_title(f"DQN Tuning — {title}")
            ax.set_xlabel("Config"); ax.set_ylabel(title)
            ax.set_xticks(x); ax.set_xticklabels([f"C{i+1}" for i in x])
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            best = df[col].idxmax() if col == "mean_reward" else df[col].idxmin()
            bars[best].set_edgecolor("black"); bars[best].set_linewidth(2)
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "dqn_tuning_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _write_report(self, df):
        best = df.loc[df["mean_reward"].idxmax()]
        rows_md = "\n".join(
            f"| C{r['config_id']} | {r['lr']} | {r['gamma']} | {r['batch']} | "
            f"{r['target_freq']} | {r['eps_decay']} | {r['mean_reward']:.4f} | "
            f"{r['mean_sla_rate']*100:.2f}% | {r['convergence_ep']} |"
            for _, r in df.iterrows()
        )
        report = f"""# DQN Hyperparameter Tuning Report

**Phase:** 13 — Step 2  
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  
**Episodes per config:** {TUNING_EPISODES}  
**Grid size:** {len(df)} configurations

## Grid Results

| Config | LR | Gamma | Batch | TgtFreq | EpsDecay | Reward | SLA | ConvEp |
|--------|-----|-------|-------|---------|---------|--------|-----|--------|
{rows_md}

## Best Configuration
- **Config:** C{int(best['config_id'])}
- **Learning rate:** {best['lr']}
- **Gamma:** {best['gamma']}
- **Batch size:** {int(best['batch'])}
- **Target update freq:** {int(best['target_freq'])}
- **Epsilon decay steps:** {int(best['eps_decay'])}
- **Mean reward:** {best['mean_reward']:.4f}
- **SLA rate:** {best['mean_sla_rate']*100:.2f}%
- **Convergence episode:** {int(best['convergence_ep'])}

## Recommendation
Use the best configuration above for the full training run.
Best model saved to `results/models/best_dqn_model.pt`.
"""
        (RESULTS_TUNING / "dqn_tuning_report.md").write_text(report)
        logger.info("dqn_tuning_report.md written.")
