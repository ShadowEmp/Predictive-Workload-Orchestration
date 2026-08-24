# Changelog — Predictive Workload Orchestration

All significant architectural and implementation changes are recorded here.
Format: `[Date] | Type | Module | Description`

Types: `ADD` | `FIX` | `CHANGE` | `REMOVE` | `REFACTOR`

---

## [2026-07-28] — Milestone 1: Dataset Preparation

### ADD | Project scaffold
- Created full project directory structure with 11 milestone packages, config, data, tests, logs, notebooks directories.
- All milestone directories initialised as Python packages with `__init__.py`.

### ADD | `config/settings.py`
- Central configuration file introduced.
- Defines: PROJECT_ROOT, all data directory paths, dataset file list, column schemas, aggregation parameters, feature engineering parameters, output filenames, logging config.
- Design principle: no hardcoded values anywhere in implementation files.

### ADD | `config/logger.py`
- Centralised logging factory using Python's standard `logging` module.
- Dual output: stdout (console) and `logs/pipeline.log` (file).
- Uses `get_logger(__name__)` pattern for per-module named loggers.
- Prevents duplicate handlers on repeated calls.

### ADD | `milestone_01_dataset/downloader.py`
- `DatasetDownloader` class — idempotent download of Alibaba Cluster Trace 2018 files.
- Skips download if file already exists (idempotent).
- Extracts `.tar.gz` archives and returns path to first extracted CSV.
- Writes `data/raw/download_manifest.json` with MD5 checksums.

### ADD | `milestone_01_dataset/loader.py`
- `RawDataLoader` class — loads `machine_usage.csv` and `machine_meta.csv`.
- Validates column schema (Alibaba CSVs have no header row).
- Converts integer Unix-second offsets to `pd.Timestamp` anchored to 2018-01-01 UTC.
- Coerces all metric columns to float, turning non-numeric values to NaN.

### ADD | `milestone_01_dataset/cleaner.py`
- `DataCleaner` class — cleaning in 5 steps: dedup, clip bounds, flag excessive missing, fill, assert clean.
- Clips `cpu_util_percent`, `mem_util_percent`, `disk_io_percent` to [0, 100].
- Clips `mem_gps`, `mkpi`, `net_in`, `net_out` to [0, ∞).
- Applies ffill then bfill **per machine** independently (prevents cross-machine leakage).
- Raises `ValueError` if any column remains NaN after fill (machine fully missing — loud failure).

### ADD | `milestone_01_dataset/aggregator.py`
- `WorkloadAggregator` class — per-machine resample → cluster-level mean.
- Two-stage aggregation: resample each machine independently, then average across machines.
- Gap detection on resulting DatetimeIndex.
- `_enforce_continuous_index` reindexes to perfect regular range and fills synthetic rows.

### ADD | `milestone_01_dataset/feature_engineer.py`
- `FeatureEngineer` class — 5 feature groups: rolling stats, lag features, cyclic time, trend, load intensity.
- Rolling windows: 12 steps (1h) and 36 steps (3h).
- Lag steps: 1, 3, 6, 12 (5min to 1h).
- Cyclic encoding: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_business_hours`, `is_weekend`.
- Trend: `cpu_util_percent_trend` (first diff), `cpu_util_percent_trend_smooth` (diff of rolling mean).
- Load intensity: categorical `load_intensity` + numeric `load_intensity_code` using percentile thresholds.
- Warm-up row removal after all features are computed.

### ADD | `milestone_01_dataset/eda.py`
- `EDAReport` class — generates text EDA report + 8 plots + feature documentation.
- Uses `matplotlib.use("Agg")` for headless compatibility.
- Plots: CPU overview, distribution, daily seasonality, weekly seasonality, rolling features, autocorrelation, load intensity pie, correlation heatmap.
- Writes `data/reports/eda_report.txt` (machine-readable structured sections).
- Writes `data/reports/FEATURE_DOCUMENTATION.md` (full feature catalogue).

### ADD | `milestone_01_dataset/pipeline.py`
- `DatasetPipeline` orchestrator — wires all M1 sub-modules together.
- CLI flags: `--download`, `--nrows`, `--skip-eda`.
- `validate()` method checks: no missing values, continuous timestamps, row count ≥ 1000, required features present, CPU in [0, 100].

### ADD | `tests/test_milestone_01.py`
- 23 unit and integration tests across 4 test classes.
- `TestDataCleaner`: 5 tests.
- `TestWorkloadAggregator`: 5 tests.
- `TestFeatureEngineer`: 11 tests.
- `TestDatasetPipelineIntegration`: 2 integration tests using `monkeypatch`.
- All tests use synthetic in-memory data (no real dataset required to run tests).

### ADD | `requirements.txt`
- Pinned dependencies for full project (all milestones).
- numpy 1.26.4, pandas 2.2.2, matplotlib 3.8.4, prophet 1.1.5, gymnasium 0.29.1, torch 2.3.0.

### ADD | `README.md`
- Project overview, structure, milestone table, quickstart instructions.

---

## Bug Fixes — Milestone 1

### FIX | `milestone_01_dataset/aggregator.py` — Timedelta comparison TypeError
- **Problem**: `pd.tseries.frequencies.to_offset(freq)` returns a `DateOffset` object; comparing it directly with a `Timedelta` via `>` raises `TypeError` in pandas 2.x.
- **Fix**: Replaced `expected_delta.nanos * 1e-9 * 1e9` float arithmetic with `pd.to_timedelta(expected_delta)` to produce a proper `Timedelta` object for comparison.
- **Commit scope**: `aggregator.py:_find_gaps()`

### FIX | `tests/test_milestone_01.py` — `test_ffill_fills_nan` false failure
- **Problem**: Test injected NaN on ALL rows of `machine_000`. With no valid anchor values, ffill+bfill correctly cannot fill — the cleaner raises `ValueError` as designed.
- **Fix**: Changed test to inject NaN on rows 5–20 only (partial range), leaving valid values before and after for the fill to anchor on. This correctly tests the fill behaviour.
- **Commit scope**: `tests/test_milestone_01.py:TestDataCleaner.test_ffill_fills_nan`

---

## Test Results

```
Platform : Python 3.12.3, pytest 8.2.2, pandas 2.2.2, numpy 1.26.4
Date     : 2026-07-28
Result   : 23 passed, 0 failed
Duration : 0.70s
```

---

## [2026-07-28] — Milestone 2: Workload Forecasting

### CHANGE | `config/settings.py` — Added Milestone 2 constants
- Added Prophet configuration block: `PROPHET_TRAIN_FRACTION`, `PROPHET_FORECAST_HORIZON_STEPS`, `PROPHET_DAILY_FOURIER_ORDER`, `PROPHET_WEEKLY_FOURIER_ORDER`, `PROPHET_CV_INITIAL`, `PROPHET_CV_PERIOD`, `PROPHET_CV_HORIZON`, `PROPHET_INTERVAL_WIDTH`, `PROPHET_CHANGEPOINT_PRIOR_SCALE`, `PROPHET_SEASONALITY_PRIOR_SCALE`.
- Added output filenames: `FORECAST_DATASET_FILENAME`, `PROPHET_MODEL_FILENAME`, `FORECAST_ACCURACY_REPORT_FILENAME`.
- Added `FORECAST_PLOTS_DIR` path constant.

### ADD | `milestone_02_forecasting/trainer.py`
- `ProphetTrainer` class — prepares `{ds, y}` DataFrame, strips UTC timezone, performs temporal train/test split, builds Prophet model with custom daily/weekly Fourier orders, fits model, generates forecast DataFrame, saves model as pickle.
- Suppresses Prophet and cmdstanpy verbose logging.
- `load_model()` static method for model reload from disk.

### ADD | `milestone_02_forecasting/evaluator.py`
- `ForecastEvaluator` class — aligns forecast with test actuals via inner join on `ds`, computes MAE/RMSE/MAPE with zero-denominator guard for MAPE.
- Optional cross-validation via `prophet.diagnostics.cross_validation()` with rolling-origin evaluation.
- Generates 5 plots: actual vs forecast, forecast overview (Prophet native), residuals (time-series + histogram), components decomposition, CV metrics vs horizon.
- Writes structured text accuracy report.

### ADD | `milestone_02_forecasting/merger.py`
- `ForecastMerger` class — left-joins Prophet forecast columns into the M1 processed DataFrame by timestamp alignment.
- Strips timezone from index for merge key, restores index after merge.
- Computes `forecast_error = cpu_util_percent - yhat` and `forecast_abs_error`.
- Saves `forecast_dataset.csv`.

### ADD | `milestone_02_forecasting/pipeline.py`
- `ForecastingPipeline` orchestrator — wires trainer → evaluator → merger in order.
- CLI flags: `--skip-cv`, `--data-dir`.
- `validate()` method checks: yhat present, plausible range, forecast_error present, ≥50% coverage.

### ADD | `milestone_02_forecasting/__init__.py`
- Package init, exposes `ForecastingPipeline`.

### ADD | `tests/test_milestone_02.py`
- 24 unit tests across 4 test classes + 1 slow integration test.
- `TestProphetTrainer`: 9 tests — DataFrame preparation, timezone stripping, temporal split, future DataFrame generation, model save/load.
- `TestForecastEvaluator`: 8 tests — alignment, metric keys, MAE/RMSE ordering, perfect forecast, zero-division guard, report file creation.
- `TestForecastMerger`: 7 tests — return type, column presence, row count preservation, original column retention, error computation, CSV output.
- `TestForecastingPipelineIntegration`: 1 slow test (real Prophet training, skipped in normal CI).

### ADD | `pytest.ini`
- Registers `slow` custom mark to avoid `PytestUnknownMarkWarning`.

---

## Bug Fixes — Milestone 2

### FIX | `tests/test_milestone_02.py` — MagicMock not picklable
- **Problem**: `unittest.mock.MagicMock` cannot be serialized by Python's `pickle` module — raises `PicklingError: Can't pickle MagicMock`.
- **Root cause**: `MagicMock` uses `__reduce__` in a way that is intentionally not picklable.
- **Fix**: Replaced `MagicMock` with a plain `dict {"name": "mock_prophet"}` as a stand-in for the model serialization round-trip test. The test now correctly validates that `ProphetTrainer.load_model()` can deserialize any picklable object.
- **Commit scope**: `tests/test_milestone_02.py:TestProphetTrainer.test_save_and_load_model`

---

## Test Results

```
Platform : Python 3.12.3, pytest 8.2.2, pandas 2.2.2, prophet 1.1.5
Date     : 2026-07-28
M2 result: 24 passed, 0 failed, 1 deselected (slow)
Cumulative: 47 passed, 0 failed
Duration : 0.82s (fast tests only)
```

---

## [2026-07-28] — Milestone 3: Gymnasium Cloud Environment

### CHANGE | `config/settings.py` — Added Milestone 3 constants
- Added environment configuration block: `ENV_EPISODE_LENGTH=288`, `ENV_RANDOM_START`, `ENV_MIN_VMS=1`, `ENV_MAX_VMS=20`, `ENV_INITIAL_VMS=5`, `ENV_CPU_PER_VM=10.0`, `ENV_MAX_QUEUE=100.0`.
- Added reward parameters: `ENV_TARGET_CPU_UTIL=70.0`, `ENV_SLA_THRESHOLD=85.0`, `ENV_SCALING_COST=0.01`, `ENV_REWARD_WEIGHTS` (dict with util/sla/cost/scale keys).
- Added `ENV_OBS_FEATURES` list (11 dataset feature names) and `ENV_OBS_DIM=13` (11 + vm_count_norm + queue_length_norm).
- Added `ENV_FEATURE_BOUNDS` dict for fixed-bound normalisation.
- Added `ENV_FALLBACK_DATASET` constant.

### ADD | `milestone_03_environment/state_space.py`
- `StateBuilder` class — extracts 11 feature columns from dataset row, normalises each using fixed bounds from `ENV_FEATURE_BOUNDS`, appends 2 runtime features (vm_count_norm, queue_length_norm), returns float32 numpy array of shape (13,).
- Handles missing features (e.g. `yhat` when M2 not run) with mid-range fallback and one-time warning.
- Clip-before-normalise pattern prevents values outside [0, 1].

### ADD | `milestone_03_environment/action_space.py`
- `ActionExecutor` class — translates discrete action index to VM delta (+1, −1, 0).
- VM count clamped to [min_vms, max_vms] after every action.
- Tracks `scale_events` and `last_action_name` per episode; `reset()` clears episode state.
- `ACTION_CATALOGUE` dict defines all action semantics in one place.

### ADD | `milestone_03_environment/reward.py`
- `RewardCalculator` class — 4-component composite reward.
- `R_util`: peaks at 1.0 when cpu_util == target_util, linear decay to extremes.
- `R_sla`: 0 below threshold, linearly negative above; −1.0 at 100% CPU.
- `R_cost`: 0 if appropriately provisioned, negative for excess VMs only.
- `R_scale`: fixed penalty per scaling action (promotes stability).
- `update_weights(dict)` method — allows HARS (M5) to change weights in-place without recreating the environment.

### ADD | `milestone_03_environment/cloud_environment.py`
- `CloudEnvironment(gymnasium.Env)` — full Gymnasium API implementation.
- `reset(seed, options)` → `(obs, info)` with seeded reproducibility.
- `step(action)` → `(obs, reward, terminated, truncated, info)`.
- Queue dynamics: overflow accumulates at rate `demand − capacity`; drains at `0.5 × (capacity − demand)`.
- `terminated` when `step_idx >= episode_length`; `truncated` when queue at ceiling and VM count at maximum.
- `update_reward_weights()` public hook for HARS integration.
- `episode_summary()` returns dict of episode-level statistics.
- Dataset loading: prefers `forecast_dataset.csv`, falls back to `processed_dataset.csv`.

### ADD | `milestone_03_environment/pipeline.py`
- `EnvironmentValidator` class — runs N random-policy episodes and validates each step.
- Checks: obs shape, obs dtype, obs range [0,1], reward finite, info keys, episode termination.
- `_report_aggregate()` logs mean/std/min/max episode reward, SLA violation rate, scale events.
- CLI: `--episodes`, `--render`, `--data-path`.

### ADD | `milestone_03_environment/__init__.py`
- Package init, exposes `CloudEnvironment`.

### ADD | `tests/test_milestone_03.py`
- 47 tests across 5 test classes, all pass on first run.
- `TestStateBuilder`: 8 tests — shape, dtype, range, normalisation bounds, missing feature fallback, runtime feature normalisation.
- `TestActionExecutor`: 10 tests — all 3 actions, clamping, is_scaling_action, invalid action, reset.
- `TestRewardCalculator`: 12 tests — dict keys, component formulas, total finiteness, weight update.
- `TestCloudEnvironment`: 16 tests — spaces, reset API, step API, termination, VM bounds, reproducibility, reward weight update, episode summary.
- `TestEnvironmentValidator`: 1 integration test — full random-policy rollout passes.

---

## Test Results

```
Platform   : Python 3.12.3, pytest 8.2.2, gymnasium 0.29.1
Date       : 2026-07-28
M3 result  : 47 passed, 0 failed
Cumulative : 94 passed, 0 failed, 1 deselected (slow)
Duration   : 1.02s
```
