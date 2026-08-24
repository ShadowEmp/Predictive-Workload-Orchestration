# Project Files Reference

**Predictive Workload Orchestration** — Complete file-by-file explanation.

> Every file in this project explained: what it does, what it contains,
> what it produces, and how it connects to the rest of the system.

---

## Table of Contents

1. [Root Level Files](#1-root-level-files)
2. [config/](#2-config)
3. [data/](#3-data)
4. [milestone_01_dataset/](#4-milestone_01_dataset)
5. [milestone_02_forecasting/](#5-milestone_02_forecasting)
6. [milestone_03_environment/](#6-milestone_03_environment)
7. [milestone_04_dqn/](#7-milestone_04_dqn)
8. [milestone_05_hars/](#8-milestone_05_hars)
9. [milestone_06_policy_export/](#9-milestone_06_policy_export)
10. [milestone_07_policy_replay/](#10-milestone_07_policy_replay)
11. [milestone_08_cloudsim/](#11-milestone_08_cloudsim)
12. [milestone_09_experiments/](#12-milestone_09_experiments)
13. [milestone_10_visualization/](#13-milestone_10_visualization)
14. [milestone_11_docs/](#14-milestone_11_docs)
15. [research/](#15-research)
16. [tests/](#16-tests)
17. [results/](#17-results)

---

## 1. Root Level Files

### `README.md`
Project overview and quickstart guide. Lists all 11 milestones with their
status, explains the dataset source, and gives the exact commands to run
each pipeline step.

### `PROJECT_FILES.md` ← this file
Complete reference explaining every file in the project.

### `PROJECT_JOURNAL.md`
Development diary maintained throughout the project. Contains one entry
per milestone recording: objectives achieved, files created, dependencies
added, key design decisions, challenges encountered, and solutions applied.
Useful source material when writing the final project report.

### `CHANGELOG.md`
Records every significant code change — additions, fixes, and refactors —
with the affected module, root cause, and what was changed. Follows the
format `[Date] | Type | Module | Description`.

### `requirements.txt`
Pinned Python dependency list for the entire project. Includes numpy,
pandas, matplotlib, prophet, gymnasium, torch, scipy, scikit-learn, and
testing libraries. Install with `pip install -r requirements.txt`.

### `pytest.ini`
pytest configuration. Registers the custom `slow` mark so that tests
requiring real Prophet training can be skipped during fast CI runs using
`pytest -m "not slow"`.


## 2. config/

### `config/settings.py`
**The single source of truth for the entire project.**

Every path, constant, and hyperparameter is defined here. No other file
hardcodes any value — everything imports from this module.

| Section | What it controls |
|---------|-----------------|
| Root paths | `PROJECT_ROOT`, `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `REPORTS_DIR`, `LOGS_DIR` — auto-created at import time |
| Dataset URLs | Alibaba OSS download links and filename map for all 6 raw CSV files |
| Column schemas | `MACHINE_USAGE_COLUMNS`, `MACHINE_META_COLUMNS` — positional column names since Alibaba CSVs have no header row |
| M1 aggregation | `AGGREGATION_FREQ = "5min"`, which columns to aggregate, per-column aggregation functions, missing value strategy |
| M1 features | `ROLLING_WINDOWS = [12, 36]` (1h, 3h), `LAG_STEPS = [1, 3, 6, 12]`, load intensity percentile thresholds |
| M2 Prophet | `changepoint_prior_scale`, `seasonality_prior_scale`, Fourier orders (daily=10, weekly=5), CV window sizes |
| M3 environment | Episode length (288 steps = 24h), VM fleet (1–20 VMs), reward weights, observation feature list (11 columns), normalisation bounds per feature |
| M4 DQN | Hidden layers `[128, 128]`, learning rate `1e-3`, gamma `0.99`, buffer size 50k, batch 64, epsilon schedule, target update frequency |
| M5 HARS | Performance window (20 episodes), EMA smoothing (0.9), weight adapt rate (0.05), weight bounds (0.05–0.70) |
| M6–M7 | Export directory, filenames for `.pt` and `.json`, replay log filename |
| M8 CloudSim | Datacenter: 10 hosts × 8 PEs × 16 GB RAM; 3 VM types (small/medium/large) |
| M9 Experiments | Strategy name list, results directory |
| M10 Visualization | DPI (150), figure size presets |
| Research 12–17 | Results directory tree (baseline/tuning/ablation/statistics/visualizations/reports/logs/models), tuning grids for Prophet/DQN/HARS, 5 random seeds for statistical validation, export formats (PNG/PDF/SVG) |

### `config/logger.py`
Factory function `get_logger(name)` returning a named Python logger.
Writes simultaneously to stdout and `logs/pipeline.log`.
Called as `logger = get_logger(__name__)` in every module so log lines
identify their source. Prevents duplicate handlers if called multiple times.

### `config/__init__.py`
Package initializer. Re-exports everything from `settings.py` so other
modules can do `from config.settings import X` or `from config import X`.

---

## 3. data/

### `data/raw/`
Original Alibaba Cluster Trace 2018 files — never modified by the pipeline.

| File | Size | Description |
|------|------|-------------|
| `machine_usage.csv` | 8.4 GB | Per-machine CPU, memory, network, disk utilization sampled every 30–300 seconds. 246,934,820 rows across 4,023 machines over 8 days. **Primary input to M1.** |
| `machine_meta.csv` | 540 KB | Static machine metadata: CPU count, memory size, failure domain labels, operational status |
| `container_meta.csv` | 18 MB | Container-to-machine assignment, application group (app_du), CPU/memory request and limit values |
| `container_usage.csv` | 9.8 GB | Per-container CPU, memory, CPI, network, disk utilization |
| `batch_task.csv` | 766 MB | Batch job task records: task name, job name, start/end times, CPU and memory requests |
| `batch_instance.csv` | 5.7 GB | Per-instance batch job records: actual CPU/memory average and peak usage, sequence numbers |

All 6 files have no header row — columns are assigned positionally using the schema
constants in `config/settings.py`. Timestamps are seconds offset from 2018-01-01 UTC.

### `data/processed/`

| File | Produced by | Description |
|------|------------|-------------|
| `processed_dataset.csv` | M1 pipeline | 2,268 rows × 25 columns. Cluster-level 5-minute time series with all engineered features. Index = UTC timestamps. **Primary input to M2 and M3.** |
| `forecast_dataset.csv` | M2 pipeline | 2,268 rows × 30 columns. `processed_dataset.csv` extended with Prophet forecast columns: `yhat`, `yhat_lower`, `yhat_upper`, `forecast_error`, `forecast_abs_error`. **Input to M3 through M9.** |
| `prophet_model.pkl` | M2 trainer | Serialized trained Prophet model (Python pickle). Reloaded for evaluation, cross-validation, and Phase 12 baseline reports. |

### `data/models/`

| File | Description |
|------|-------------|
| `dqn_model.pt` | DQN checkpoint saved every 50 episodes during training (latest weights) |
| `dqn_best_model.pt` | Best-performing DQN checkpoint by episode reward — used for policy export and evaluation |
| `exported_policy.pt` | TorchScript traced model. Portable — runs without DQN source code. Used by M7 replay and M9 experiments. |
| `policy_metadata.json` | Sidecar metadata: `obs_dim=13`, `n_actions=3`, action name map, training episode count, export timestamp |

### `data/reports/`

| File / Directory | Produced by | Description |
|------------------|------------|-------------|
| `eda_report.txt` | M1 EDA | Structured text: dataset shape, time range, per-column statistics, missing value report, timestamp continuity check, load intensity distribution |
| `FEATURE_DOCUMENTATION.md` | M1 EDA | Markdown feature catalogue — every column in `processed_dataset.csv` with name, type, and description |
| `forecast_accuracy_report.txt` | M2 evaluator | MAE, RMSE, MAPE (hold-out), plus CV metrics when `--skip-cv` is not set |
| `dqn_training_log.csv` | M5 pipeline | Per-episode training log: reward, loss, epsilon, difficulty score, w_util, w_sla, w_cost, w_scale, SLA violations, scale events |
| `hars_log.csv` | M5 HARS scheduler | Per-episode HARS state: difficulty score, all 4 reward weights, rolling mean reward, rolling SLA rate |
| `replay_action_log.csv` | M7 pipeline | 288-step policy replay log: step, action index, action name, cpu_util, vm_count, queue_length, reward, sla_violated |
| `eda_plots/` | M1 EDA | 8 plots: CPU time series overview, distribution histogram + boxplot, daily seasonality, weekly seasonality, rolling features overlay, autocorrelation, load intensity pie chart, feature correlation heatmap |
| `forecast_plots/` | M2 evaluator | 4 plots: actual vs forecast, full Prophet forecast overview, residual time series + histogram, component decomposition (trend, daily, weekly) |
| `hars_plots/` | M5 pipeline | 5 plots: training reward curve, training loss (log scale), epsilon decay, HARS difficulty score, reward weight stacked area |
| `replay_plots/` | M7 pipeline | 3 plots: CPU/VM/queue overview, action distribution pie, per-step reward |
| `final_plots/` | M10 pipeline | 15 publication-quality plots — see M10 section |

### `data/cloudsim_results/`

| File | Description |
|------|-------------|
| `simulation_config.json` | JSON config passed to the CloudSim Plus Java simulator: datacenter spec, VM types, per-step scaling actions and workload from the replay log |
| `cloudsim_metrics.csv` | 288-row output: vm_count, cpu_util, memory_util, power_watts, energy_kwh, cost_usd, cumulative_cost, cumulative_energy, response_time_ms, queue_length, sla_violated |

### `data/experiment_results/`

| File | Description |
|------|-------------|
| `experiment_results.csv` | Benchmark results for all 6 strategies across multiple episodes: episode_reward, mean_cpu_util, mean_vm_count, sla_violation_rate, scale_events, estimated_cost, estimated_energy_kwh, final_queue_length |

---

## 4. milestone_01_dataset/

**Purpose:** Download, clean, aggregate, and feature-engineer the raw Alibaba
cluster trace into a ready-to-use time series dataset.

**Run:** `python -m milestone_01_dataset.pipeline`

### `milestone_01_dataset/downloader.py`
**Class:** `DatasetDownloader`

Downloads all 6 Alibaba dataset `.tar.gz` files from the official OSS server.
- Idempotent — skips files that already exist locally
- Extracts `.tar.gz` archives and returns the path to the first `.csv` inside
- Writes `data/raw/download_manifest.json` with MD5 checksums for verification
- Uses `urllib.request.urlretrieve` with a progress hook that logs % complete

### `milestone_01_dataset/loader.py`
**Class:** `RawDataLoader`

Loads raw CSV files from `data/raw/` and validates their schema.
- Assigns column names positionally (Alibaba CSVs have no header row)
- Validates that all expected columns are present — raises `ValueError` on mismatch
- Converts integer Unix-second timestamps (offset from 2018-01-01 UTC) to `pd.Timestamp`
- Coerces all metric columns to `float`, turning non-numeric values to `NaN`

### `milestone_01_dataset/cleaner.py`
**Class:** `DataCleaner`

Cleans raw per-machine data before aggregation. Operates on individual machine
time series independently to prevent cross-machine contamination.
- Removes duplicate `(machine_id, timestamp)` pairs
- Clips `cpu_util_percent`, `mem_util_percent`, `disk_io_percent` to `[0, 100]`
- Clips `net_in`, `net_out`, `mem_gps`, `mkpi` to `[0, ∞)`
- Flags machines with more than 12 consecutive NaN rows (possible outages)
- Applies forward-fill then backward-fill **per machine**
- Raises `ValueError` if any column still has NaN after filling (loud failure)

### `milestone_01_dataset/aggregator.py`
**Class:** `WorkloadAggregator`

Converts per-machine cleaned data into a single cluster-level 5-minute time series.

Two-stage process:
1. Resample each machine's data to 5-min frequency independently
2. Average across all machines at each time step → cluster-level signal

Also:
- Detects gaps in the resulting DatetimeIndex (logged as warnings)
- `_enforce_continuous_index()` reindexes to a perfectly regular range and
  fills any synthetic gap rows with ffill+bfill to guarantee a gapless output

### `milestone_01_dataset/feature_engineer.py`
**Class:** `FeatureEngineer`

Adds 5 groups of engineered features to the aggregated time series:

| Group | Features | Purpose |
|-------|----------|---------|
| Rolling statistics | `cpu_util_percent_roll_mean_12/36`, `cpu_util_percent_roll_std_12/36`, `mem_util_percent_roll_mean_12/36` | Trend smoothing and volatility awareness |
| Lag features | `cpu_util_percent_lag_1/3/6/12` | Temporal memory in RL state vector |
| Cyclic time | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_business_hours`, `is_weekend` | Daily and weekly seasonality encoding |
| Trend | `cpu_util_percent_trend` (first diff), `cpu_util_percent_trend_smooth` | Direction of change signal |
| Load intensity | `load_intensity` (categorical), `load_intensity_code` (0–3) | Percentile-based bucketing |

Removes warm-up rows (first 36 steps) where rolling/lag features are undefined.

### `milestone_01_dataset/eda.py`
**Class:** `EDAReport`

Generates the exploratory data analysis report and plots.

Outputs:
- `data/reports/eda_report.txt` — structured text with 6 sections: overview, target statistics, all column statistics, missing value report, load intensity distribution, timestamp continuity check
- `data/reports/FEATURE_DOCUMENTATION.md` — markdown feature catalogue
- 8 PNG plots saved to `data/reports/eda_plots/`

Uses `matplotlib.use("Agg")` for headless server compatibility.
Never calls `plt.show()` — all plots saved directly to disk.

### `milestone_01_dataset/pipeline.py`
**Class:** `DatasetPipeline`

Orchestrates the entire M1 workflow. Wires all sub-modules together in order:
downloader → loader → cleaner → aggregator → feature_engineer → EDA → save.

CLI flags:
- `--download` — trigger dataset download before processing
- `--nrows N` — limit rows loaded (useful for quick testing)
- `--skip-eda` — skip plot generation for faster iteration

`validate(df)` method runs 5 post-run checks and prints PASS/FAIL for each:
no missing values, continuous timestamps, row count ≥ 1000, required columns present, CPU in [0, 100].

### `milestone_01_dataset/__init__.py`
Package initializer. Exposes `DatasetPipeline` as the public API.

---

## 5. milestone_02_forecasting/

**Purpose:** Train a Facebook Prophet model on the cluster CPU time series,
evaluate its accuracy, and merge forecast columns into the processed dataset.

**Run:** `python -m milestone_02_forecasting.pipeline --skip-cv`

### `milestone_02_forecasting/trainer.py`
**Class:** `ProphetTrainer`

Loads `processed_dataset.csv`, prepares a Prophet-format `{ds, y}` DataFrame,
trains the model, and generates a forecast DataFrame.

Key details:
- Strips UTC timezone from the index before passing to Prophet (Prophet does not support timezone-aware `ds`)
- Performs a **temporal** train/test split (80/20) — no shuffling, time order preserved
- Adds two custom seasonalities instead of using Prophet defaults:
  - `daily` with `fourier_order=10` — captures sharp business-hours transitions
  - `weekly` with `fourier_order=5` — captures weekday vs weekend patterns
- Yearly seasonality is explicitly disabled (only 8 days of data)
- Generates forecast covering training period + test period + 288 future steps
- Saves the trained model as `data/processed/prophet_model.pkl`

### `milestone_02_forecasting/evaluator.py`
**Class:** `ForecastEvaluator`

Evaluates the trained Prophet model on the held-out test set.

Metrics computed:
- **MAE** — mean absolute error in %CPU
- **RMSE** — root mean squared error (penalises large spikes more than MAE)
- **MAPE** — mean absolute percentage error (guards against division by zero when actual = 0)
- **R²** — coefficient of determination

Optional cross-validation using Prophet's built-in rolling-origin CV engine
(`cross_validation()` from `prophet.diagnostics`). Produces `cv_mae`, `cv_rmse`,
`cv_mape` metrics across multiple cutpoints.

Generates 4 plots: actual vs forecast, full forecast overview, residuals
(time series + histogram), and component decomposition (trend + seasonalities).

### `milestone_02_forecasting/merger.py`
**Class:** `ForecastMerger`

Merges Prophet forecast columns (`yhat`, `yhat_lower`, `yhat_upper`) back
into `processed_dataset.csv` using a left-join on timestamp.

Important fix: `pd.DataFrame.merge()` resets the DatetimeIndex to a
RangeIndex. The merger preserves the original index before merging and
restores it after, ensuring `forecast_dataset.csv` has correct timestamps.

Also adds:
- `forecast_error = cpu_util_percent − yhat`
- `forecast_abs_error = |forecast_error|`

Saves result as `data/processed/forecast_dataset.csv`.

### `milestone_02_forecasting/pipeline.py`
**Class:** `ForecastingPipeline`

Orchestrates M2: trainer → evaluator → merger → save.

CLI: `--skip-cv` skips the slow cross-validation step.
`validate(df)` checks: yhat column present, values in plausible range,
forecast_error present, coverage ≥ 50%.

### `milestone_02_forecasting/__init__.py`
Exposes `ForecastingPipeline` as the public API.

---

## 6. milestone_03_environment/

**Purpose:** Implement a Gymnasium-compatible RL environment that simulates
a cloud cluster where a DQN agent learns to auto-scale VMs.

**Run:** `python -m milestone_03_environment.pipeline` (random-policy validation)

### `milestone_03_environment/state_space.py`
**Class:** `StateBuilder`

Constructs and normalises the 13-dimensional float32 observation vector at
each time step.

- Extracts 11 feature columns from the current `forecast_dataset.csv` row
- Normalises each feature to `[0, 1]` using **fixed bounds** from `settings.py`
  (not data-driven min/max — prevents distribution shift between train and inference)
- Appends 2 runtime-computed features: `vm_count_norm` and `queue_length_norm`
- Handles missing features gracefully (e.g. `yhat` absent when M2 not run)
  by substituting the mid-range fallback value with a one-time warning

The 11 dataset features used:
`cpu_util_percent`, `mem_util_percent`, `yhat`, `cpu_util_percent_roll_mean_12`,
`cpu_util_percent_trend`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`,
`is_business_hours`, `is_weekend`

### `milestone_03_environment/action_space.py`
**Class:** `ActionExecutor`

Translates discrete action indices into VM fleet changes.

| Action | Name | VM delta |
|--------|------|----------|
| 0 | SCALE_UP | +1 VM |
| 1 | SCALE_DOWN | −1 VM |
| 2 | NO_OP | 0 |

VM count is always clamped to `[min_vms, max_vms]` after every action.
Tracks `scale_events` and `last_action_name` per episode.
`reset()` clears per-episode counters.

### `milestone_03_environment/reward.py`
**Class:** `RewardCalculator`

Computes the composite reward at each step:

```
R = w_util × R_util + w_sla × R_sla + w_cost × R_cost + w_scale × R_scale
```

| Component | Formula | Range |
|-----------|---------|-------|
| `R_util` | `1 - |cpu_util - 70| / 100` | [0, 1] |
| `R_sla` | `−max(0, cpu_util − 85) / 15` | [−1, 0] |
| `R_cost` | `−max(0, vm_count − vm_required) / 20` | [−1, 0] |
| `R_scale` | `−0.01` if scaling, else `0` | {−0.01, 0} |

Default weights: `util=0.40, sla=0.40, cost=0.15, scale=0.05`

`update_weights(dict)` — called by HARS to change weights without
recreating the environment.

### `milestone_03_environment/cloud_environment.py`
**Class:** `CloudEnvironment(gymnasium.Env)`

The central Gymnasium environment. Implements the full Gymnasium API.

**State:** 13-dim float32 observation vector from `StateBuilder`
**Actions:** `Discrete(3)` — SCALE_UP / SCALE_DOWN / NO_OP
**Episode:** 288 steps (24 hours at 5-min resolution)
**Dataset:** Reads `forecast_dataset.csv` row-by-row per step

Queue dynamics:
- When demand > capacity: `queue += demand − capacity`
- When capacity > demand: `queue -= 0.5 × (capacity − demand)`
- Queue capped at `ENV_MAX_QUEUE = 100`

Termination: `terminated=True` at step 288.
Truncation: `truncated=True` if queue is maxed AND VMs at ceiling.

`update_reward_weights(dict)` — HARS integration hook. Updates reward
component weights in-place without recreating the environment.

`episode_summary()` — returns dict with episode_reward, sla_violations,
sla_violation_rate, scale_events, final_vm_count, final_queue_length, steps_completed.

### `milestone_03_environment/pipeline.py`
**Class:** `EnvironmentValidator`

Runs N random-policy episodes and validates each step:
- Observation shape `(13,)` and dtype `float32`
- All observation values in `[0, 1]`
- Reward is finite
- Info dict contains required keys
- Episode terminates at the correct step count

Logs aggregate statistics across all episodes.

CLI: `--episodes N`, `--render`, `--data-path`

### `milestone_03_environment/__init__.py`
Exposes `CloudEnvironment` as the public API.

---

## 7. milestone_04_dqn/

**Purpose:** Implement and train a standard Deep Q-Network agent on the
cloud auto-scaling environment using a fixed reward function.

**Run:** `python -m milestone_04_dqn.pipeline --episodes 200`

### `milestone_04_dqn/network.py`
**Class:** `DQNNetwork(nn.Module)`

Feed-forward Q-network built with PyTorch.

Architecture:
```
Input(13) → Linear(13→128) → ReLU → Linear(128→128) → ReLU → Linear(128→3)
```

- Two hidden layers of 128 neurons each (configurable via `DQN_HIDDEN_LAYERS`)
- Xavier uniform weight initialisation, zero bias initialisation
- Total parameters: 18,691
- `forward(x)` returns Q-values for all 3 actions simultaneously

The same class is used for both the online network and the target network.

### `milestone_04_dqn/replay_buffer.py`
**Class:** `ReplayBuffer`

Circular experience replay buffer. Stores `(state, action, reward, next_state, done)` tuples.

- Implemented as a `collections.deque` with `maxlen=capacity` — old experiences
  are automatically discarded when full
- `push()` stores one transition
- `sample(batch_size)` returns a random batch as 5 numpy arrays
- `is_ready` property — True once the buffer contains ≥ `DQN_MIN_BUFFER_SIZE` transitions

### `milestone_04_dqn/agent.py`
**Class:** `DQNAgent`

Wraps the online network, target network, replay buffer, and exploration
policy into a single training agent.

Training algorithm:
1. **Epsilon-greedy selection** — random action with probability ε, else argmax Q
2. **Store transition** in replay buffer
3. **Sample batch** and compute Bellman target: `y = r + γ × max Q'(s')`
4. **Gradient update** on `MSE(Q(s,a), y)` with gradient clipping at `max_norm=10`
5. **Hard target copy** every `TARGET_UPDATE_FREQ` steps
6. **Linear epsilon decay** from 1.0 to 0.05 over `EPSILON_DECAY_STEPS`

`save(path)` / `load(path)` — saves/restores weights, optimizer state,
epsilon, and step counters in a single `.pt` checkpoint.

### `milestone_04_dqn/trainer.py`
**Class:** `DQNTrainer`

Manages the training loop. For each episode:
- Resets environment with `seed=episode` for reproducibility
- Runs until `terminated` or `truncated`
- Calls `agent.step()` which handles buffer push + optional training
- Saves checkpoint every `DQN_SAVE_FREQ` episodes
- Saves best model whenever episode reward improves
- Writes `data/reports/dqn_training_log.csv` at the end

### `milestone_04_dqn/pipeline.py`
**Class:** `DQNPipeline`

CLI orchestrator. Creates environment and agent, then runs `DQNTrainer.train()`.
Flags: `--episodes N`, `--data-path`, `--device cpu/cuda`

### `milestone_04_dqn/__init__.py`
Exposes `DQNAgent` and `DQNPipeline`.

---

## 8. milestone_05_hars/

**Purpose:** Replace the fixed reward weights with a dynamic curriculum
that adapts based on how well the agent is performing.

**Run:** `python -m milestone_05_hars.pipeline --episodes 200 --device cuda`

### `milestone_05_hars/performance_monitor.py`
**Class:** `PerformanceMonitor`

Tracks episode-level metrics over a rolling window of 20 episodes.

Maintains deques for: episode reward, SLA violation rate, scaling frequency,
training loss. `get_stats()` returns rolling mean and standard deviation
for each metric. `is_ready` is True once the window is fully populated.

### `milestone_05_hars/difficulty_engine.py`
**Class:** `DifficultyScoreEngine`

Computes a scalar difficulty score ∈ [0, 1] from three signals:

| Component | Weight | Meaning |
|-----------|--------|---------|
| SLA violation rate | 50% | How often the agent exceeds the 85% CPU threshold |
| Reward stagnation | 30% | Whether reward is still improving |
| Training loss | 20% | Whether the neural network is still converging |

The raw score is smoothed with exponential moving average (α=0.9) to
prevent rapid oscillation. High difficulty → agent is struggling.
Low difficulty → agent is performing well.

### `milestone_05_hars/reward_weight_generator.py`
**Class:** `RewardWeightGenerator`

Maps the difficulty score to new reward component weights.

Target weight schedule:
- `w_sla`: 0.30 (easy) → 0.60 (hard) — more SLA emphasis when struggling
- `w_cost`: 0.25 (easy) → 0.10 (hard) — less cost penalty when struggling
- `w_scale`: 0.10 (easy) → 0.05 (hard)
- `w_util`: remainder

Changes are incremental — maximum `ADAPT_RATE=0.05` per episode — preventing
destabilising jumps. After clipping, weights are renormalised to sum to 1.0.

### `milestone_05_hars/scheduler.py`
**Class:** `HARSScheduler`

Top-level HARS orchestrator. Called once at the end of each training episode.

Workflow per episode:
1. `PerformanceMonitor.record()` — store episode metrics
2. `DifficultyScoreEngine.compute()` — compute difficulty
3. `RewardWeightGenerator.generate()` — produce new weights
4. `env.update_reward_weights()` — push to environment (no restart needed)
5. Append row to internal log

`save_log()` writes `data/reports/hars_log.csv`.

### `milestone_05_hars/pipeline.py`
**Class:** `HARSPipeline`

DQN + HARS joint training loop. Wraps the standard DQN episode with a
`HARSScheduler.on_episode_end()` call after each episode.

At the end of training:
- Saves `dqn_training_log.csv` (enriched with difficulty and weight columns)
- Saves `hars_log.csv`
- Generates 5 plots via `ResultsPlotter`

CLI: `--episodes N`, `--device cpu/cuda`, `--no-plots`

`validate(result)` checks: correct number of log rows, reward improving,
HARS log populated, SLA weight varying, best model saved.

### `milestone_05_hars/__init__.py`
Exposes `HARSScheduler` and `HARSPipeline`.

---

## 9. milestone_06_policy_export/

**Purpose:** Serialize the trained DQN policy to a portable format that
runs without any DQN source code.

**Run:** `python -m milestone_06_policy_export.pipeline`

### `milestone_06_policy_export/exporter.py`
**Class:** `PolicyExporter`

Exports and loads the trained policy using PyTorch TorchScript.

`export(network)`:
- Sets the network to eval mode
- Traces it with `torch.jit.trace()` using a dummy input tensor
- Saves the traced model to `data/models/exported_policy.pt`
- Writes `data/models/policy_metadata.json` with: `obs_dim=13`, `n_actions=3`,
  action name map `{0: SCALE_UP, 1: SCALE_DOWN, 2: NO_OP}`, export timestamp,
  training episode count, best reward achieved

`load(dir)` — static method. Loads `exported_policy.pt` and `policy_metadata.json`.
Returns `(traced_model, metadata_dict)`.

`predict(model, observation)` — static method. Runs inference on a single
`float32` numpy array of shape `(13,)`. Returns the greedy action index.

### `milestone_06_policy_export/pipeline.py`
**Class:** `PolicyExportPipeline`

Orchestrates export in 3 steps:
1. Load `dqn_best_model.pt` checkpoint into a `DQNAgent`
2. Export via `PolicyExporter.export()`
3. Verify — reload the TorchScript model and compare its greedy actions
   to the original network on 100 random observations

Verification confirms the export is lossless (all 100 observations produce
identical actions). Exits with code 1 if any mismatch is found.

CLI: `--model-path`, `--export-dir`, `--no-verify`

### `milestone_06_policy_export/__init__.py`
Exposes `PolicyExporter` and `PolicyExportPipeline`.

---

## 10. milestone_07_policy_replay/

**Purpose:** Replay the exported policy through the environment for one or
more full episodes, logging every action and producing analysis plots.

**Run:** `python -m milestone_07_policy_replay.pipeline`

### `milestone_07_policy_replay/replay_engine.py`
**Class:** `PolicyReplayEngine`

Core replay logic. Loads `exported_policy.pt` via `PolicyExporter.load()`,
creates a `CloudEnvironment`, and runs a complete episode using greedy
action selection (no exploration).

Logs each step: step index, action index, action name, cpu_util, vm_count,
queue_length, reward, sla_violated.

Saves the log to `data/reports/replay_action_log.csv` — this file is the
primary input to M8 CloudSim simulation.

### `milestone_07_policy_replay/pipeline.py`
**Class:** `ReplayPipeline`

Runs N replay episodes, combines logs, and generates 3 plots:
- `replay_overview.png` — CPU utilization, VM count, queue length over time
- `action_distribution.png` — pie chart of SCALE_UP / SCALE_DOWN / NO_OP split
- `replay_reward.png` — per-step reward curve

`validate(log_df)` checks: non-empty log, all actions in {0,1,2}, VM count
bounded, all rewards finite, CSV file saved.

CLI: `--episodes N`, `--seed N`, `--policy-dir`, `--data-path`, `--no-plots`

### `milestone_07_policy_replay/__init__.py`
Exposes `PolicyReplayEngine` and `ReplayPipeline`.

---

## 11. milestone_08_cloudsim/

**Purpose:** Simulate the DQN policy in a realistic cloud infrastructure
model using CloudSim Plus, collecting hardware-level metrics.

**Run:** `python -m milestone_08_cloudsim.pipeline`

### `milestone_08_cloudsim/simulation_config.py`
**Class:** `CloudSimConfig`

Generates `simulation_config.json` from the M7 replay log and runs the
simulation.

`generate_config()` — builds a JSON file containing:
- Datacenter spec: 10 hosts, 8 PEs per host, 16 GB RAM, 10 Gbps bandwidth
- VM type definitions (small/medium/large)
- Per-step scaling actions from the replay log
- Per-step CPU workload from the replay log

`run_simulation(jar_path)`:
- If `cloudsim_runner.jar` exists: invokes it via `subprocess` (full Java simulation)
- If not found: falls back to the **Python simulation stub**

Python stub computes per-step:
- **Energy**: `power = idle + (max−idle) × cpu_util/100` → energy in kWh
- **Cost**: `$0.05 × vm_count × (5 min / 60)`
- **Response time**: `100ms + queue × 10ms + Gaussian noise`

Saves results to `data/cloudsim_results/cloudsim_metrics.csv`.

### `milestone_08_cloudsim/java/CloudSimRunner.java`
Java skeleton for the real CloudSim Plus integration.

Documents the full simulation outline (commented out, requires
`cloudsimplus-7.3.0.jar` on classpath):
- Creates `DatacenterSimple` with 10 heterogeneous hosts using `PowerModelHostSimple`
- Creates VMs according to the scaling action log
- Creates Cloudlets from the CPU workload trace
- Runs simulation and writes metrics CSV

Currently prints a message and exits. The Python stub provides equivalent
output for all development and research purposes.

### `milestone_08_cloudsim/pipeline.py`
**Class:** `CloudSimPipeline`

Orchestrates M8:
1. Load `replay_action_log.csv` from M7
2. Generate `simulation_config.json`
3. Run simulation (JAR or Python stub)
4. Log a formatted summary table
5. Generate 3 plots: energy+cost accumulation, infrastructure overview, power consumption

`validate(metrics_df)` checks: CSV saved, non-empty DataFrame, required
columns present, energy/cost non-negative, VM count bounded.

CLI: `--replay-log`, `--jar`, `--results-dir`, `--no-plots`

### `milestone_08_cloudsim/__init__.py`
Exposes `CloudSimConfig` and `CloudSimPipeline`.

---

## 12. milestone_09_experiments/

**Purpose:** Benchmark all 6 orchestration strategies on identical episodes
and produce a comprehensive comparison table.

**Run:** `python -m milestone_09_experiments.pipeline --episodes 5`

### `milestone_09_experiments/strategies.py`
Contains 6 strategy factory functions. Each returns a callable `policy(obs) → action`.

| Function | Strategy | Logic |
|----------|----------|-------|
| `static_strategy()` | Static | Always returns NO_OP (2). Fixed at initial 5 VMs. |
| `threshold_strategy()` | Threshold | SCALE_UP if `cpu_norm > 0.80`, SCALE_DOWN if `< 0.40`, else NO_OP |
| `prophet_strategy()` | Prophet | Uses `yhat` (index 2 in obs): SCALE_UP if `yhat_norm > 0.75`, SCALE_DOWN if `< 0.45` |
| `dqn_strategy(path)` | DQN | Loads `exported_policy.pt`, runs greedy inference |
| `prophet_dqn_strategy(path)` | Prophet + DQN | Same as DQN (Prophet already in obs vector) |
| `prophet_dqn_hars_strategy(path)` | Full system | Same model, trained with HARS weights |

All 3 DQN-based strategies gracefully fall back to `static_strategy()` if
the exported model file does not exist.

### `milestone_09_experiments/runner.py`
**Class:** `ExperimentRunner`

Runs N episodes per strategy and collects 12 metrics per episode:
`episode_reward`, `mean_cpu_util`, `mean_vm_count`, `sla_violation_rate`,
`sla_violations`, `scale_events`, `scaling_frequency`, `estimated_cost`,
`estimated_energy_kwh`, `final_queue_length`, `steps`.

Saves all results to `data/experiment_results/experiment_results.csv`.

### `milestone_09_experiments/pipeline.py`
CLI entry point. Calls `build_strategies()` to load all 6 policies, then
runs `ExperimentRunner`. Prints a summary table grouped by strategy.

CLI: `--episodes N`, `--data-path`, `--policy-dir`

### `milestone_09_experiments/__init__.py`
Exposes `ExperimentRunner`.

---

## 13. milestone_10_visualization/

**Purpose:** Generate all final publication-quality figures from the
outputs of every previous milestone.

**Run:** `python -m milestone_10_visualization.pipeline`

### `milestone_10_visualization/plotter.py`
**Class:** `ResultsPlotter`

Central plotting class. All plots use `matplotlib.use("Agg")` for headless
compatibility and are saved directly to disk — `plt.show()` is never called.

| Method | Input | Output file |
|--------|-------|-------------|
| `plot_training_reward(log_df)` | DQN training log | `training_reward.png` — raw reward + 20-ep rolling mean |
| `plot_training_loss(log_df)` | DQN training log | `training_loss.png` — loss on log scale |
| `plot_epsilon_curve(log_df)` | DQN training log | `epsilon_curve.png` — epsilon decay over episodes |
| `plot_difficulty_score(hars_df)` | HARS log | `difficulty_score.png` — EMA-smoothed difficulty with fill |
| `plot_reward_weights(hars_df)` | HARS log | `reward_weights.png` — stacked area chart of all 4 weights |
| `plot_all_strategy_comparisons(results_df)` | Experiment results | 7 bar charts: reward, SLA, cost, energy, VM count, queue, scaling |
| `plot_radar_chart(results_df)` | Experiment results | `strategy_radar.png` — normalised spider chart for all strategies |
| `plot_forecast_accuracy(forecast_df)` | forecast_dataset.csv | `forecast_accuracy.png` — actual vs yhat with 95% CI band |
| `plot_response_time(cloudsim_df)` | CloudSim metrics | `response_time.png` — RT time series + histogram |

### `milestone_10_visualization/pipeline.py`
**Class:** `VisualizationPipeline`

Orchestrates M10. Loads all result files, calls the relevant plotter
methods, and handles missing files gracefully (logs a warning and skips
rather than crashing).

Sections:
- Training (M4/M5) — requires `dqn_training_log.csv` and `hars_log.csv`
- Forecast (M2) — requires `forecast_dataset.csv`
- Strategy comparison (M9) — requires `experiment_results.csv`
- CloudSim (M8) — requires `cloudsim_metrics.csv`

CLI: `--no-training` skips the training/HARS plots if M4/M5 not run.

### `milestone_10_visualization/__init__.py`
Exposes `ResultsPlotter` and `VisualizationPipeline`.

---

## 14. milestone_11_docs/

**Purpose:** Architecture, algorithms, class diagrams, sequence diagrams,
and complete module reference — all in Markdown.

### `milestone_11_docs/architecture.md`
System architecture documentation. Contains:
- Full ASCII block diagram of the end-to-end system showing data flow
  from Alibaba dataset through Prophet → DQN+HARS → Policy Export →
  CloudSim → Experiments → Visualization
- Module responsibilities table (every class, its module, and its function)
- Data flow table showing which file each pipeline step reads and writes

### `milestone_11_docs/class_diagrams.md`
ASCII class diagrams for all major milestone pairs:
- M1: DatasetPipeline and its 5 sub-modules
- M3: CloudEnvironment composed of StateBuilder, ActionExecutor, RewardCalculator
- M4+M5: DQNAgent (online/target networks, ReplayBuffer) alongside HARSScheduler
- M6+M7: PolicyExporter and PolicyReplayEngine

### `milestone_11_docs/algorithms.md`
Pseudocode for all 5 core algorithms:
1. **DQN Training Loop** — epsilon-greedy selection, Bellman update, target copy, epsilon decay
2. **HARS Update** — difficulty computation from SLA/stagnation/loss, incremental weight adaptation
3. **Prophet Forecasting** — trend + seasonality decomposition, Fourier series formulas
4. **Composite Reward Function** — formulas for all 4 components with normalisation
5. **State Normalisation** — clip-then-scale formula with degenerate bound handling

### `milestone_11_docs/sequence_diagrams.md`
Three ASCII sequence diagrams:
1. **Training Loop** — DQNTrainer ↔ DQNAgent ↔ CloudEnvironment ↔ HARSScheduler per episode
2. **Policy Export and Replay** — User → PolicyExporter → DQNAgent → PolicyReplayEngine → CloudEnvironment
3. **Full End-to-End** — all milestones M1→M2→M4+M5→M6→M7→M8→M9 in sequence

### `milestone_11_docs/module_reference.md`
Complete module reference table for every milestone package:
- Module name, class name, key public methods, CLI command
- Covers M1 through M10 including the research phase
- Lists all test files and their coverage

### `milestone_11_docs/__init__.py`
Empty package initializer.

---

## 15. research/

**Purpose:** Complete research and validation phase — baseline evaluation,
hyperparameter tuning, ablation study, statistical validation,
publication-quality visualization, and documentation generation.

**Run:** `python -m research.pipeline` (all phases)

### `research/experiment_tracker.py`
**Function:** `log_experiment(...)`

Global experiment logger. Every experiment — tuning config, evaluation run,
ablation episode, statistical seed — appends one row to `results/experiments.csv`.

Never overwrites existing rows (append-only). Thread-safe via a file-level lock.
Columns: experiment_id (MD5 hash of phase+model+hyperparameters), date, phase,
model, dataset, hyperparameters (JSON), training_time_s, eval_time_s, reward,
mae, rmse, mape, r2, cpu_util, queue_len, energy_kwh, cost_usd, sla_rate,
response_time_ms, convergence_episode, difficulty_score, notes.

### `research/pipeline.py`
**Function:** `main()`

Master pipeline that runs Phase 12 through Phase 17 in strict order.

CLI:
- `--phases 12 13 14 15 16 17` — select which phases to run (default: all)
- `--fast` — skip slow Prophet and HARS tuning in Phase 13 (DQN tuning only)

Catches and logs exceptions per phase so a failure in one phase does not
stop the remaining phases. Prints a final manifest of all output files.

---

### `research/phase12/` — Baseline Evaluation

#### `phase12/prophet_eval.py` — `ProphetEvaluator`
Loads `forecast_dataset.csv` and evaluates Prophet on the 20% test set.
Computes MAE, RMSE, MAPE, R² using `sklearn.metrics`.
Generates 4 plots: actual vs forecast, residuals, error distribution, rolling MAE.
Writes `results/baseline/prophet_evaluation_report.md` and `prophet_metrics.csv`.

#### `phase12/dqn_eval.py` — `DQNEvaluator`
Loads `dqn_training_log.csv` for training metrics (best reward, convergence episode,
final loss). Loads `dqn_best_model.pt` and runs 5 greedy evaluation episodes for
policy metrics (mean reward, SLA rate, scaling frequency, energy, cost).
Generates 2 plots: training curves (reward/loss/epsilon/SLA/scale), policy metrics bar chart.
Writes `results/baseline/dqn_evaluation_report.md` and `dqn_metrics.csv`.

#### `phase12/hars_eval.py` — `HARSEvaluator`
Loads `hars_log.csv` and evaluates 5 HARS properties:
mean/final difficulty score, mean/final SLA weight, reward improvement,
weight stability index.
Generates 4 plots: difficulty evolution, reward weight trajectories,
performance score, weight distribution histograms.
Writes `results/baseline/hars_evaluation_report.md` and `hars_metrics.csv`.

#### `phase12/cloudsim_eval.py` — `CloudSimEvaluator`
Loads `cloudsim_metrics.csv` and computes 14 infrastructure metrics:
mean/std CPU, memory, power, energy, cost, response time (mean + 95th percentile),
SLA violations, queue length.
Generates 4 plots: infrastructure overview, energy/cost accumulation,
SLA rolling rate, response time distribution.
Writes `results/baseline/cloudsim_baseline_report.md` and `cloudsim_metrics_summary.csv`.

---

### `research/phase13/` — Hyperparameter Tuning

#### `phase13/prophet_tuning.py` — `ProphetTuner`
Grid search over 6 Prophet configurations varying `changepoint_prior_scale`
(0.001–0.50), `seasonality_prior_scale` (1.0–15.0), and `seasonality_mode`
(additive / multiplicative). Each config is trained and evaluated in ~0.1 seconds.
Best config identified by lowest MAE. Saves `results/models/best_prophet_model.pkl`.
Writes `results/tuning/prophet_tuning_results.csv` and `prophet_tuning_report.md`.

#### `phase13/dqn_tuning.py` — `DQNTuner`
Grid search over 6 DQN configurations varying: learning rate (5e-4 / 1e-3),
gamma (0.95 / 0.99), batch size (64 / 128), target update frequency (200 / 500),
epsilon decay steps (5,000 / 10,000). Each config trains for `TUNING_EPISODES=20`
episodes then evaluates for 3 episodes.
Best config selected by mean evaluation reward. Saves `results/models/best_dqn_model.pt`.
Writes `results/tuning/dqn_tuning_results.csv` and `dqn_tuning_report.md`.

#### `phase13/hars_tuning.py` — `HARSTuner`
Grid search over 6 HARS configurations varying: performance window (10/20/30),
EMA smoothing (0.7/0.8/0.9), adapt rate (0.03/0.05/0.10), weight bounds.
Each config trains a DQN+HARS setup from the pretrained checkpoint for 20 episodes.
Writes `results/tuning/hars_tuning_results.csv` and `hars_tuning_report.md`.

---

### `research/phase14/` — Ablation Study

#### `phase14/ablation.py` — `AblationStudy`
Runs 5 episodes per strategy for all 6 strategies in the ladder:
Static → Threshold → Prophet → DQN → Prophet+DQN → Prophet+DQN+HARS.
Collects 12 metrics per episode. Saves `results/ablation/ablation_results.csv`
and `ablation_summary.csv` (mean ± std per strategy).
Generates 3 plots: grouped bar charts, radar chart, SLA vs cost scatter.
Writes `results/ablation/ablation_report.md`.

---

### `research/phase15/` — Statistical Validation

#### `phase15/statistical_validation.py` — `StatisticalValidator`
Runs each of 3 strategies (static, dqn, prophet_dqn_hars) with 5 random seeds
`[0, 42, 123, 999, 2024]`. Computes for each metric: mean, median, std, min,
max, 95% confidence interval (Student's t-distribution via `scipy.stats`).
Performs Welch's t-test (DQN vs static, HARS vs static) to assess statistical
significance.
Generates 2 plots: boxplots for 4 metrics, 95% CI bar chart for episode reward.
Writes `results/statistics/statistical_raw.csv`, `statistical_summary.csv`,
`statistical_validation_report.md`.

---

### `research/phase16/` — Publication Visualization

#### `phase16/visualization.py` — `PublicationVisualizer`
Generates 6 comprehensive dashboard figures, each saved in **PNG, PDF, and SVG**.

| Dashboard | Contents |
|-----------|----------|
| `training_dashboard` | Reward curve, loss (log), epsilon decay, SLA violations, scaling activity |
| `hars_dashboard` | Difficulty score, stacked weight area chart, individual weight lines, difficulty vs reward scatter |
| `forecast_dashboard` | Actual vs forecast, residuals, error distribution, rolling MAE |
| `cloudsim_dashboard` | CPU, memory, VM count, queue, energy+cost (dual axis), response time |
| `comparison_dashboard` | 6 bar charts comparing all strategies on all metrics |
| `statistical_dashboard` | Boxplots across 5 seeds for 4 metrics |

Uses publication-quality `rcParams`: DejaVu Sans font, spines removed on
top/right, consistent 150 DPI across all formats.

---

### `research/phase17/` — Documentation

#### `phase17/documentation.py` — `DocumentationWriter`
Generates 10 final markdown documents in `results/reports/`:

| File | Contents |
|------|----------|
| `experimental_setup.md` | Hardware spec, library versions, dataset stats, all hyperparameter tables |
| `evaluation_report.md` | Summary of Phase 12 findings for all 4 modules |
| `hyperparameter_selection.md` | Tuning methodology, search grids, best configs, rationale |
| `ablation_study.md` | Strategy ladder results, component contribution analysis |
| `statistical_analysis.md` | CI tables, t-test results, reproducibility assessment |
| `cloudsim_results.md` | Infrastructure metrics, energy/cost table, discussion |
| `results_summary.md` | All strategies compared in one table, key findings |
| `project_conclusion.md` | Problem addressed, proposed solution, contributions, limitations |
| `future_work.md` | 5 proposed extensions with architecture diagrams, challenges, ethical considerations |
| `industrial_applicability.md` | 7-stage deployment roadmap from research prototype to enterprise platform |

### `research/__init__.py`
Empty package initializer.

---

## 16. tests/

**Purpose:** Unit and integration tests for all milestones. All tests use
synthetic in-memory data — no real dataset, no network calls, no slow training.

**Run:** `python -m pytest tests/ -m "not slow" -v`
**With slow tests:** `python -m pytest tests/ -v`

### `tests/test_milestone_01.py`
**23 tests** across 4 classes.

| Class | Tests | What is verified |
|-------|-------|-----------------|
| `TestDataCleaner` | 5 | No NaN after cleaning, out-of-range clipping, duplicate removal, ffill for partial NaN |
| `TestWorkloadAggregator` | 5 | Returns DataFrame, DatetimeIndex, continuous 5-min intervals, CPU in [0,100], no NaN |
| `TestFeatureEngineer` | 11 | All 5 feature groups present, valid values, no NaN after warm-up drop, sin/cos in [-1,1], is_weekend binary |
| `TestDatasetPipelineIntegration` | 2 | Full pipeline produces CSV; validation passes (uses monkeypatch to inject synthetic data) |

### `tests/test_milestone_02.py`
**24 tests** (+ 1 slow integration test skipped by default).

| Class | Tests | What is verified |
|-------|-------|-----------------|
| `TestProphetTrainer` | 9 | Prophet DataFrame has only {ds, y}, timezone stripped, no NaN, temporal split ordering, no overlap, future df length and 5-min freq, save/load round-trip |
| `TestForecastEvaluator` | 8 | Alignment returns DataFrame with correct columns, metric keys present, MAE ≥ 0, RMSE ≥ MAE, perfect forecast gives MAE=0, zero-division guard, report file created |
| `TestForecastMerger` | 7 | Returns DataFrame, yhat column added, forecast_error added, row count preserved, original columns retained, error computation correct, CSV saved |
| `TestForecastingPipelineIntegration` | 1 slow | Full pipeline on synthetic data produces valid output and passes validation |

### `tests/test_milestone_03.py`
**47 tests** across 5 classes.

| Class | Tests | What is verified |
|-------|-------|-----------------|
| `TestStateBuilder` | 8 | Shape (13,), dtype float32, all values in [0,1], lower/upper bound normalisation, missing feature fallback, vm_count/queue normalisation at max |
| `TestActionExecutor` | 10 | All 3 actions, clamping at min/max, is_scaling_action flag, invalid action raises ValueError, reset clears counters, N_ACTIONS=3 |
| `TestRewardCalculator` | 12 | Dict has all keys, util reward peaks at target, SLA reward zero below threshold, SLA reward -1 at 100%, cost zero when not over-provisioned, scale reward negative for scaling, total reward always finite, weight update changes output |
| `TestCloudEnvironment` | 16 | Spaces shape and size, reset returns correct types, obs in [0,1] and float32, step returns 5-tuple, reward finite, info has vm_count, episode terminates, VM count bounded, never below min, seeded reset reproducible, weight update hook, episode_summary keys |
| `TestEnvironmentValidator` | 1 | Random-policy rollout passes all checks on synthetic data |

### `tests/test_milestone_04.py`
**18 tests** across 3 classes.

| Class | Tests | What is verified |
|-------|-------|-----------------|
| `TestDQNNetwork` | 5 | Output shape (batch, 3), all Q-values finite, batch size 1 works, parameter count > 0, bias initialized to zero |
| `TestReplayBuffer` | 7 | Push increments length, capacity capped at maxlen, sample returns correct shapes, raises ValueError when too small, states are float32, is_ready=False when empty |
| `TestDQNAgent` | 6 | Action in [0, N_ACTIONS), greedy action deterministic, epsilon decays after steps, epsilon lower-bounded, step returns None before buffer ready, save/load restores epsilon and step count, target copy matches online weights |

### `tests/test_milestone_05.py`
**17 tests** across 4 classes.

| Class | Tests | What is verified |
|-------|-------|-----------------|
| `TestPerformanceMonitor` | 6 | Stats after recording, window capping, stats before recording returns zeros, is_ready False/True, mean reward correct |
| `TestDifficultyScoreEngine` | 4 | Output in [0,1] always, high SLA increases difficulty, history grows, smoothing damps sudden changes |
| `TestRewardWeightGenerator` | 6 | Weights sum to 1.0, all keys present, bounded within [w_min, w_max], high difficulty increases SLA weight, history records every call, normalisation preserves sum |
| `TestHARSScheduler` | 1 | `on_episode_end()` calls `env.update_reward_weights()` and returns a dict; `save_log()` creates CSV |

### `tests/test_milestone_06_07.py`
**15 tests** across 2 classes.

| Class | Tests | What is verified |
|-------|-------|-----------------|
| `TestPolicyExporter` | 7 | Export creates .pt file, export creates metadata JSON, JSON contains obs_dim and n_actions, load returns model and dict, predict returns valid action [0,2], predict is deterministic, exported model produces same actions as original network |
| `TestPolicyReplayEngine` | 4 | Replay returns DataFrame, has required columns, all actions valid {0,1,2}, saves CSV to disk |

### `tests/test_milestone_08_09_10.py`
**28 tests** across 3 classes.

| Class | Tests | What is verified |
|-------|-------|-----------------|
| `TestCloudSimConfig` | 8 | `generate_config()` creates JSON, JSON has required top-level keys, stub returns DataFrame of correct length, stub has all required metric columns, stub saves CSV, `run_simulation()` uses stub when JAR absent, energy ≥ 0, cost ≥ 0 |
| `TestExperimentRunner` | 5 | `run()` returns DataFrame, has required columns, static strategy has 0 scale events, saves `experiment_results.csv`, all 3 baseline strategy factories return valid callables |
| `TestResultsPlotter` | 9 | Each plot method creates the expected PNG file: training_reward, training_loss, epsilon_curve, difficulty_score, reward_weights, strategy_comparisons (at least SLA + cost), radar chart, forecast_accuracy, response_time |

### `tests/__init__.py`
Empty package initializer.

---

## 17. results/

All research phase outputs. Nothing here is source code — everything is
generated by `python -m research.pipeline`.

### `results/experiments.csv`
Master experiment tracker. One row per experiment run across all phases.
Contains 22 columns including experiment_id, date, phase, model, dataset,
hyperparameters, timing, and all evaluation metrics. Append-only — previous
runs are never overwritten, making it a complete audit trail.

### `results/baseline/`
Phase 12 baseline evaluation outputs.

| File | Description |
|------|-------------|
| `prophet_evaluation_report.md` | MAE, RMSE, MAPE, R² with config table, analysis, and limitations |
| `prophet_metrics.csv` | Single-row metrics CSV for programmatic comparison |
| `dqn_evaluation_report.md` | Training metrics (best reward, convergence episode) + policy metrics (SLA rate, scaling frequency) |
| `dqn_metrics.csv` | Merged training and policy metrics |
| `hars_evaluation_report.md` | Difficulty score stats, SLA weight evolution, weight stability index |
| `hars_metrics.csv` | Single-row HARS metrics |
| `cloudsim_baseline_report.md` | 14 infrastructure metrics with analysis and formatted summary table |
| `cloudsim_metrics_summary.csv` | Single-row CloudSim metrics |
| `prophet_plots/` | 4 PNG plots from Phase 12-A |
| `dqn_plots/` | 2 PNG plots from Phase 12-B |
| `hars_plots/` | 4 PNG plots from Phase 12-C |
| `cloudsim_plots/` | 4 PNG plots from Phase 12-D |

### `results/tuning/`
Phase 13 hyperparameter tuning outputs.

| File | Description |
|------|-------------|
| `prophet_tuning_results.csv` | 6-row grid search results: all config parameters + MAE/RMSE/MAPE/R² |
| `prophet_tuning_report.md` | Grid results table, best config, recommendation |
| `dqn_tuning_results.csv` | 6-row grid search: all config parameters + reward/SLA/convergence |
| `dqn_tuning_report.md` | Grid results table, best config, recommendation |
| `hars_tuning_results.csv` | 6-row grid search: HARS parameters + reward/SLA/difficulty/stability |
| `hars_tuning_report.md` | Grid results table, best HARS configuration |
| `prophet_plots/` | Bar chart comparing MAE/RMSE/MAPE across 6 Prophet configs |
| `dqn_plots/` | Bar chart comparing reward/SLA/convergence across 6 DQN configs |
| `hars_plots/` | Bar chart comparing reward/SLA/stability across 6 HARS configs |

### `results/ablation/`
Phase 14 ablation study outputs.

| File | Description |
|------|-------------|
| `ablation_results.csv` | Raw per-episode results for all 6 strategies × 5 episodes |
| `ablation_summary.csv` | Mean ± std per strategy across all metrics |
| `ablation_report.md` | Strategy ladder description, results table, component contribution analysis |
| `plots/ablation_bars.png` | 4-panel bar chart: reward, SLA, cost, energy per strategy |
| `plots/ablation_radar.png` | Normalised radar chart comparing all 6 strategies |
| `plots/sla_cost_tradeoff.png` | Scatter plot: SLA rate vs estimated cost per strategy |

### `results/statistics/`
Phase 15 statistical validation outputs.

| File | Description |
|------|-------------|
| `statistical_raw.csv` | Raw results for 3 strategies × 5 seeds |
| `statistical_summary.csv` | Mean, median, std, min, max, 95% CI per (strategy, metric) |
| `statistical_validation_report.md` | CI tables, Welch's t-test p-values, reproducibility assessment |
| `plots/boxplots.png` | Boxplots for 4 metrics across strategies and seeds |
| `plots/confidence_intervals.png` | 95% CI bars for episode reward per strategy |

### `results/visualizations/`
Phase 16 publication-quality figures — each in 3 formats.

| Base name | Contents |
|-----------|----------|
| `training_dashboard` | 5-panel training metrics dashboard |
| `hars_dashboard` | 4-panel HARS system behaviour dashboard |
| `forecast_dashboard` | 4-panel Prophet forecast evaluation dashboard |
| `cloudsim_dashboard` | 6-panel CloudSim infrastructure metrics dashboard |
| `comparison_dashboard` | 6-panel strategy comparison bar charts |
| `statistical_dashboard` | Boxplot dashboard across seeds |

Each saved as `.png` (150 DPI), `.pdf` (vector), and `.svg` (vector, transparent).

### `results/reports/`
Phase 17 final documentation — 10 markdown files suitable for direct
inclusion in the VTU project report.

| File | Use in report |
|------|--------------|
| `experimental_setup.md` | Chapter: Experimental Setup |
| `evaluation_report.md` | Chapter: System Evaluation |
| `hyperparameter_selection.md` | Chapter: Hyperparameter Selection |
| `ablation_study.md` | Chapter: Ablation Study |
| `statistical_analysis.md` | Chapter: Statistical Validation |
| `cloudsim_results.md` | Chapter: CloudSim Digital Twin Results |
| `results_summary.md` | Chapter: Results and Discussion |
| `project_conclusion.md` | Chapter: Conclusion |
| `future_work.md` | Chapter: Future Work |
| `industrial_applicability.md` | Chapter: Industrial Applicability |

### `results/models/`
Best models saved by the research tuning phase.

| File | Description |
|------|-------------|
| `best_prophet_model.pkl` | Prophet model with best MAE from Phase 13 grid search |
| `best_dqn_model.pt` | DQN model with best evaluation reward from Phase 13 grid search |

### `results/logs/`
Runtime logs from research phase processes.

| File | Description |
|------|-------------|
| `hars_tuning.log` | Full stdout output from the HARS tuning background process |

---

## Quick Reference: CLI Commands

Every pipeline step is run as a Python module from the project root:

```bash
cd predictive_workload_orchestration
source venv/bin/activate

# Milestones
python -m milestone_01_dataset.pipeline              # Dataset preparation
python -m milestone_02_forecasting.pipeline --skip-cv # Prophet forecasting
python -m milestone_03_environment.pipeline           # Env validation (random policy)
python -m milestone_04_dqn.pipeline --episodes 200    # Baseline DQN training
python -m milestone_05_hars.pipeline --episodes 200 --device cuda  # DQN + HARS
python -m milestone_06_policy_export.pipeline         # Export trained policy
python -m milestone_07_policy_replay.pipeline         # Replay exported policy
python -m milestone_08_cloudsim.pipeline              # CloudSim simulation
python -m milestone_09_experiments.pipeline --episodes 5  # Benchmark all strategies
python -m milestone_10_visualization.pipeline         # Generate all plots

# Research phases
python -m research.pipeline                           # All phases 12–17
python -m research.pipeline --phases 12               # Baseline evaluation only
python -m research.pipeline --phases 13 --fast        # DQN tuning only
python -m research.pipeline --phases 14 15            # Ablation + stats
python -m research.pipeline --phases 16 17            # Plots + docs

# Tests
python -m pytest tests/ -m "not slow" -v              # All fast tests (165 tests)
python -m pytest tests/ -v                            # Including slow integration tests
```

---

## Quick Reference: Key Output Files

| File | What it is |
|------|------------|
| `data/processed/processed_dataset.csv` | 2,268 rows × 25 features — M1 deliverable |
| `data/processed/forecast_dataset.csv` | 2,268 rows × 30 cols — M2 deliverable (includes yhat) |
| `data/processed/prophet_model.pkl` | Trained Prophet model |
| `data/models/dqn_best_model.pt` | Best DQN checkpoint |
| `data/models/exported_policy.pt` | TorchScript policy (no source code needed) |
| `data/models/policy_metadata.json` | Policy metadata (obs_dim, n_actions, action map) |
| `data/reports/dqn_training_log.csv` | Per-episode training: reward, loss, epsilon, HARS weights |
| `data/reports/hars_log.csv` | Per-episode HARS: difficulty, weights, performance |
| `data/reports/replay_action_log.csv` | 288-step policy replay log — input to CloudSim |
| `data/cloudsim_results/cloudsim_metrics.csv` | Infrastructure simulation metrics |
| `data/experiment_results/experiment_results.csv` | All 6 strategies benchmarked |
| `results/experiments.csv` | Master experiment tracker (all research runs) |
| `results/visualizations/*.pdf` | 6 publication dashboards in PDF format |
| `results/reports/project_conclusion.md` | Final project conclusion for thesis |

---

## Data Flow Summary

```
data/raw/machine_usage.csv  (246M rows)
    │
    ▼  milestone_01_dataset
data/processed/processed_dataset.csv  (2,268 rows × 25 features)
    │
    ▼  milestone_02_forecasting
data/processed/forecast_dataset.csv   (2,268 rows × 30 cols, adds yhat)
data/processed/prophet_model.pkl
    │
    ├──▶  milestone_03_environment  (reads forecast_dataset.csv row-by-row)
    │         │
    │         ▼  milestone_04/05 (DQN + HARS training)
    │     data/models/dqn_best_model.pt
    │     data/reports/dqn_training_log.csv
    │     data/reports/hars_log.csv
    │         │
    │         ▼  milestone_06 (policy export)
    │     data/models/exported_policy.pt
    │     data/models/policy_metadata.json
    │         │
    │         ▼  milestone_07 (policy replay)
    │     data/reports/replay_action_log.csv
    │         │
    │         ▼  milestone_08 (CloudSim)
    │     data/cloudsim_results/cloudsim_metrics.csv
    │
    └──▶  milestone_09 (experiments — all 6 strategies)
              │
              ▼
    data/experiment_results/experiment_results.csv
              │
              ▼  milestone_10 / research phases 12–17
    results/  (74 output files: CSV, MD, PNG, PDF, SVG)
```

---

*Generated for Predictive Workload Orchestration — VTU Final Year Project*
*Python 3.12 · PyTorch 2.11 · Prophet 1.1.5 · Gymnasium 0.29.1 · NumPy 1.26.4*
