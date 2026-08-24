# System Architecture

## Predictive Workload Orchestration

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTIVE WORKLOAD ORCHESTRATION             │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌───────────────────────────┐   │
│  │ M1       │   │ M2       │   │ M3  CloudEnvironment       │   │
│  │ Dataset  │──▶│ Prophet  │──▶│     (Gymnasium)            │   │
│  │ Pipeline │   │ Forecast │   │     obs_space: Box(13,)    │   │
│  └──────────┘   └──────────┘   │     act_space: Discrete(3) │   │
│                                └───────────┬───────────────┘   │
│                                            │ reset/step         │
│                                ┌───────────▼───────────────┐   │
│                                │ M4  DQN Agent              │   │
│                                │     ReplayBuffer           │   │
│                                │     OnlineNet ──▶ TargetNet│   │
│                                └───────────┬───────────────┘   │
│                                            │ after each episode  │
│                                ┌───────────▼───────────────┐   │
│                                │ M5  HARS Scheduler         │   │
│                                │     PerfMonitor            │   │
│                                │     DifficultyEngine       │   │
│                                │     WeightGenerator        │   │
│                                └───────────┬───────────────┘   │
│                                            │ update weights      │
│                                            ▼                    │
│              ┌────────────────────────────────────────────┐    │
│              │              RewardCalculator               │    │
│              │   R = w_util×Ru + w_sla×Rs + w_cost×Rc     │    │
│              └────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │ M6       │   │ M7       │   │ M8       │   │ M9       │   │
│  │ Policy   │──▶│ Replay   │──▶│CloudSim  │──▶│Experiments│  │
│  │ Export   │   │ Engine   │   │Plus Twin │   │ Runner   │   │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   │
│                                                       │          │
│                              ┌────────────────────────▼──────┐ │
│                              │   M10 Visualization / M11 Docs │ │
│                              └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

| Module | Class | Responsibility |
|--------|-------|----------------|
| M1 | `DatasetPipeline` | Download, clean, aggregate, feature-engineer Alibaba trace |
| M1 | `RawDataLoader` | Schema-validated CSV loading, timestamp parsing |
| M1 | `DataCleaner` | Dedup, clip, ffill+bfill |
| M1 | `WorkloadAggregator` | Per-machine → cluster-level 5-min time series |
| M1 | `FeatureEngineer` | Rolling stats, lags, cyclic time, trend, load intensity |
| M1 | `EDAReport` | 8 EDA plots + feature documentation |
| M2 | `ProphetTrainer` | Train Facebook Prophet on cpu_util_percent |
| M2 | `ForecastEvaluator` | MAE/RMSE/MAPE + cross-validation + 5 plots |
| M2 | `ForecastMerger` | Merge yhat/yhat_lower/yhat_upper into processed_dataset |
| M3 | `CloudEnvironment` | Gymnasium RL environment: obs/act spaces, step, reset |
| M3 | `StateBuilder` | 13-dim normalised observation vector |
| M3 | `ActionExecutor` | Discrete(3) → VM delta with clamping |
| M3 | `RewardCalculator` | 4-component composite reward with weight update hook |
| M4 | `DQNAgent` | Epsilon-greedy, replay buffer, online+target networks |
| M4 | `DQNNetwork` | Feed-forward Q-network (PyTorch) |
| M4 | `ReplayBuffer` | Circular experience buffer with uniform sampling |
| M4 | `DQNTrainer` | Training loop, logging, checkpointing |
| M5 | `HARSScheduler` | End-of-episode HARS orchestration |
| M5 | `PerformanceMonitor` | Rolling-window episode metric tracking |
| M5 | `DifficultyScoreEngine` | EMA difficulty score from SLA/reward/loss |
| M5 | `RewardWeightGenerator` | Maps difficulty → adapted reward weights |
| M6 | `PolicyExporter` | TorchScript export + JSON metadata |
| M7 | `PolicyReplayEngine` | Replay policy, log every action, save CSV |
| M8 | `CloudSimConfig` | Generate JSON config, run Java JAR or Python stub |
| M9 | `ExperimentRunner` | Benchmark all 6 strategies, collect metrics |
| M10 | `ResultsPlotter` | 15 publication-quality figures |

---

## Data Flow

```
data/raw/machine_usage.csv
        │
        ▼  [M1 DatasetPipeline]
data/processed/processed_dataset.csv  (25 columns, 5-min intervals)
        │
        ▼  [M2 ForecastingPipeline]
data/processed/forecast_dataset.csv   (+ yhat, yhat_lower, yhat_upper, forecast_error)
data/processed/prophet_model.pkl
        │
        ▼  [M3 CloudEnvironment]
        │   reads forecast_dataset.csv row-by-row during episodes
        │
        ▼  [M4 DQNTrainer]
data/models/dqn_best_model.pt
data/reports/dqn_training_log.csv
        │
        ▼  [M5 HARSScheduler]  (runs concurrently with M4)
data/reports/hars_log.csv
        │
        ▼  [M6 PolicyExporter]
data/models/exported_policy.pt
data/models/policy_metadata.json
        │
        ▼  [M7 PolicyReplayEngine]
data/reports/replay_action_log.csv
        │
        ▼  [M8 CloudSimConfig]
data/cloudsim_results/cloudsim_metrics.csv
        │
        ▼  [M9 ExperimentRunner]
data/experiment_results/experiment_results.csv
        │
        ▼  [M10 ResultsPlotter]
data/reports/final_plots/*.png   (15 figures)
```
