# Module Reference

## config/

### `settings.py`
Central configuration. All paths, constants, and hyperparameters. Import from here — never hardcode values.

Key sections:
- Root paths (`PROJECT_ROOT`, `DATA_DIR`, `PROCESSED_DIR`, ...)
- Dataset schema constants
- M1 aggregation and feature engineering parameters
- M2 Prophet hyperparameters
- M3 environment constants (obs/action/reward)
- M4 DQN hyperparameters
- M5 HARS parameters
- M6–M7 export/replay paths
- M8 CloudSim infrastructure
- M9 experiment strategy names
- M10 visualization settings

### `logger.py`
`get_logger(name)` — returns named logger writing to stdout + `logs/pipeline.log`.

---

## milestone_01_dataset/

| Module | Class | Key methods |
|--------|-------|-------------|
| `downloader.py` | `DatasetDownloader` | `download_all()`, `download_file()` |
| `loader.py` | `RawDataLoader` | `load_machine_usage()`, `load_machine_meta()` |
| `cleaner.py` | `DataCleaner` | `clean(df)`, `get_missing_summary(df)` |
| `aggregator.py` | `WorkloadAggregator` | `aggregate(df)` |
| `feature_engineer.py` | `FeatureEngineer` | `engineer(df)` |
| `eda.py` | `EDAReport` | `run()` |
| `pipeline.py` | `DatasetPipeline` | `run()`, `validate(df)` |

**CLI:** `python3 -m milestone_01_dataset.pipeline [--download] [--nrows N] [--skip-eda]`

---

## milestone_02_forecasting/

| Module | Class | Key methods |
|--------|-------|-------------|
| `trainer.py` | `ProphetTrainer` | `run()` → `(model, forecast_df, test_df)` |
| `evaluator.py` | `ForecastEvaluator` | `run(run_cv)` → `metrics_dict` |
| `merger.py` | `ForecastMerger` | `merge(processed_df, forecast_df)` |
| `pipeline.py` | `ForecastingPipeline` | `run()`, `validate(df)` |

**CLI:** `python3 -m milestone_02_forecasting.pipeline [--skip-cv]`

---

## milestone_03_environment/

| Module | Class | Key methods |
|--------|-------|-------------|
| `state_space.py` | `StateBuilder` | `build(row, vm_count, queue_length)` → `np.ndarray` |
| `action_space.py` | `ActionExecutor` | `execute(action, vm_count)` → `int` |
| `reward.py` | `RewardCalculator` | `compute(cpu_util, vm_count, is_scaling)` → `dict`, `update_weights(w)` |
| `cloud_environment.py` | `CloudEnvironment` | `reset(seed)`, `step(action)`, `update_reward_weights(w)`, `episode_summary()` |
| `pipeline.py` | `EnvironmentValidator` | `run()` → `bool` |

**CLI:** `python3 -m milestone_03_environment.pipeline [--episodes N] [--render]`

---

## milestone_04_dqn/

| Module | Class | Key methods |
|--------|-------|-------------|
| `network.py` | `DQNNetwork` | `forward(x)` → Q-values |
| `replay_buffer.py` | `ReplayBuffer` | `push(...)`, `sample(batch_size)`, `is_ready` |
| `agent.py` | `DQNAgent` | `select_action(obs, training)`, `step(...)`, `save(path)`, `load(path)` |
| `trainer.py` | `DQNTrainer` | `train()` → `log_rows` |
| `pipeline.py` | `DQNPipeline` | `run()` |

**CLI:** `python3 -m milestone_04_dqn.pipeline [--episodes N] [--device cpu]`

---

## milestone_05_hars/

| Module | Class | Key methods |
|--------|-------|-------------|
| `performance_monitor.py` | `PerformanceMonitor` | `record(...)`, `get_stats()`, `is_ready` |
| `difficulty_engine.py` | `DifficultyScoreEngine` | `compute(stats)` → `float` |
| `reward_weight_generator.py` | `RewardWeightGenerator` | `generate(difficulty)` → `dict` |
| `scheduler.py` | `HARSScheduler` | `on_episode_end(...)` → `dict`, `save_log()` |

---

## milestone_06_policy_export/

| Module | Class | Key methods |
|--------|-------|-------------|
| `exporter.py` | `PolicyExporter` | `export(network)` → `Path`, `load(dir)` → `(model, meta)`, `predict(model, obs)` → `int` |

---

## milestone_07_policy_replay/

| Module | Class | Key methods |
|--------|-------|-------------|
| `replay_engine.py` | `PolicyReplayEngine` | `replay(seed)` → `pd.DataFrame` |

---

## milestone_08_cloudsim/

| Module | Class | Key methods |
|--------|-------|-------------|
| `simulation_config.py` | `CloudSimConfig` | `generate_config()`, `run_simulation(jar_path)` → `pd.DataFrame` |

**Java:** `milestone_08_cloudsim/java/CloudSimRunner.java` — skeleton for CloudSim Plus JAR.

---

## milestone_09_experiments/

| Module | Class/Function | Key interface |
|--------|---------------|---------------|
| `strategies.py` | `static_strategy()`, `threshold_strategy()`, `prophet_strategy()`, `dqn_strategy(path)`, ... | Returns callable `(obs) → action` |
| `runner.py` | `ExperimentRunner` | `run()` → `pd.DataFrame` |
| `pipeline.py` | `build_strategies(policy_dir)` | Returns strategy dict |

**CLI:** `python3 -m milestone_09_experiments.pipeline [--episodes N]`

---

## milestone_10_visualization/

| Module | Class | Key methods |
|--------|-------|-------------|
| `plotter.py` | `ResultsPlotter` | `plot_training_reward(log)`, `plot_training_loss(log)`, `plot_epsilon_curve(log)`, `plot_difficulty_score(hars)`, `plot_reward_weights(hars)`, `plot_all_strategy_comparisons(results)`, `plot_radar_chart(results)`, `plot_forecast_accuracy(df)`, `plot_response_time(df)` |

---

## tests/

| File | Coverage |
|------|----------|
| `test_milestone_01.py` | M1: 23 tests |
| `test_milestone_02.py` | M2: 24 tests + 1 slow |
| `test_milestone_03.py` | M3: 47 tests |
| `test_milestone_04.py` | M4: 18 tests |
| `test_milestone_05.py` | M5: 17 tests |
| `test_milestone_06_07.py` | M6+M7: 15 tests |
| `test_milestone_08_09_10.py` | M8+M9+M10: 28 tests |

**Total: 172 tests (165 fast, 1 slow)**

Run all fast tests: `python3 -m pytest tests/ -m "not slow" -v`
