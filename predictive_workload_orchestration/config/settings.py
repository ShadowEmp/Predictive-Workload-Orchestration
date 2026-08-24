"""
config/settings.py
==================
Central configuration for the entire Predictive Workload Orchestration project.

All paths, constants, and hyperparameters live here.
Downstream modules import from this file — nothing is hardcoded elsewhere.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist at import time
for _dir in [RAW_DIR, PROCESSED_DIR, REPORTS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset: Alibaba Cluster Trace 2018
# ---------------------------------------------------------------------------
# Official GitHub release page for Alibaba Cluster Trace Program v2018
DATASET_BASE_URL = (
    "https://github.com/alibaba/clusterdata/raw/master/cluster-trace-v2018/data/"
)

# Files we download.  Keys = logical names, values = filenames on the remote.
DATASET_FILES = {
    "machine_usage": "machine_usage.tar.gz",
    "machine_meta": "machine_meta.tar.gz",
    "container_meta": "container_meta.tar.gz",
    "container_usage": "container_usage.tar.gz",
    "batch_task": "batch_task.tar.gz",
    "batch_instance": "batch_instance.tar.gz",
}

# ---------------------------------------------------------------------------
# Column schemas (used for validation in loader.py)
# ---------------------------------------------------------------------------
MACHINE_USAGE_COLUMNS = [
    "machine_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "mem_gps",           # memory bandwidth GB/s
    "mkpi",              # memory access per kilo instructions
    "net_in",            # network inbound MB/s
    "net_out",           # network outbound MB/s
    "disk_io_percent",   # disk I/O utilization
]

MACHINE_META_COLUMNS = [
    "machine_id",
    "time_stamp",
    "failure_domain_1",
    "failure_domain_2",
    "cpu_num",
    "mem_size",
    "status",
]

# ---------------------------------------------------------------------------
# Milestone 1: Dataset Preparation
# ---------------------------------------------------------------------------

# Aggregation
AGGREGATION_INTERVAL_MINUTES = 5          # 5-minute bins
AGGREGATION_FREQ = "5min"                 # pandas resample frequency string

# Primary signals to aggregate (from machine_usage)
AGGREGATE_COLUMNS = [
    "cpu_util_percent",
    "mem_util_percent",
    "net_in",
    "net_out",
    "disk_io_percent",
]

# Aggregation function per column
AGGREGATION_FUNCTIONS = {
    "cpu_util_percent": "mean",
    "mem_util_percent": "mean",
    "net_in": "mean",
    "net_out": "mean",
    "disk_io_percent": "mean",
}

# Missing value strategy
MISSING_VALUE_STRATEGY = "ffill_bfill"    # forward-fill then backward-fill
MAX_CONSECUTIVE_MISSING = 12              # flag if more than 12 consecutive NaNs (1 hour)

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

# Rolling window sizes (in number of 5-min steps)
ROLLING_WINDOWS = [12, 36]   # 12 steps = 1 hour, 36 steps = 3 hours

# Lag steps (in number of 5-min steps)
LAG_STEPS = [1, 3, 6, 12]   # 5min, 15min, 30min, 1 hour

# Load intensity thresholds (percentile-based, set at runtime in feature_engineer.py)
LOAD_INTENSITY_LABELS = ["low", "medium", "high", "critical"]
LOAD_INTENSITY_THRESHOLDS = [25, 50, 75]   # percentile cutoffs

# Primary target column (used by Prophet and DQN)
TARGET_COLUMN = "cpu_util_percent"

# ---------------------------------------------------------------------------
# Output file names
# ---------------------------------------------------------------------------
PROCESSED_DATASET_FILENAME = "processed_dataset.csv"
EDA_REPORT_FILENAME = "eda_report.txt"
EDA_PLOTS_DIR = REPORTS_DIR / "eda_plots"

# ---------------------------------------------------------------------------
# Milestone 2: Workload Forecasting (Prophet)
# ---------------------------------------------------------------------------

# Train/test split (fraction of data used for training)
PROPHET_TRAIN_FRACTION = 0.8

# Number of 5-min steps to forecast ahead (288 = 24 hours)
PROPHET_FORECAST_HORIZON_STEPS = 288

# Fourier order controls seasonality flexibility (higher = more flexible)
# Daily: sharp business-hours transitions need more terms
# Weekly: weekday vs weekend is simpler
PROPHET_DAILY_FOURIER_ORDER = 10
PROPHET_WEEKLY_FOURIER_ORDER = 5

# Cross-validation window sizes (Prophet string format: "N days" or "N hours")
PROPHET_CV_INITIAL = "5 days"    # Minimum training window
PROPHET_CV_PERIOD = "1 day"      # Slide cutoff every 1 day
PROPHET_CV_HORIZON = "1 day"     # Forecast 1 day ahead per cutpoint

# Uncertainty interval width (0.95 = 95% credible interval)
PROPHET_INTERVAL_WIDTH = 0.95

# Prophet changepoint detection sensitivity (0 = rigid, 1 = very flexible)
PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.05

# Seasonality regularisation (lower = smoother seasonality)
PROPHET_SEASONALITY_PRIOR_SCALE = 10.0

# Output file names for Milestone 2
FORECAST_DATASET_FILENAME = "forecast_dataset.csv"
PROPHET_MODEL_FILENAME = "prophet_model.pkl"
FORECAST_ACCURACY_REPORT_FILENAME = "forecast_accuracy_report.txt"
FORECAST_PLOTS_DIR = REPORTS_DIR / "forecast_plots"

# ---------------------------------------------------------------------------
# Milestone 3: Gymnasium Cloud Environment
# ---------------------------------------------------------------------------

# Episode configuration
ENV_EPISODE_LENGTH = 288          # steps per episode (288 × 5 min = 24 hours)
ENV_RANDOM_START = True           # randomise start index on each reset()

# VM fleet configuration
ENV_MIN_VMS = 1                   # minimum VMs always running
ENV_MAX_VMS = 20                  # hard ceiling on VM count
ENV_INITIAL_VMS = 5               # VM count at episode start

# Per-VM capacity: each VM handles this many % of cluster CPU load
# e.g. with 10 VMs and cpu_per_vm=10.0, total capacity = 100% CPU
ENV_CPU_PER_VM = 10.0

# Queue configuration
ENV_MAX_QUEUE = 100.0             # maximum queue length (arbitrary units)

# Reward function: fixed weights (HARS overrides these in Milestone 5)
ENV_TARGET_CPU_UTIL = 70.0        # ideal cluster utilization %
ENV_SLA_THRESHOLD = 85.0          # SLA breach above this CPU %
ENV_SCALING_COST = 0.01           # penalty per scaling action (stability term)

ENV_REWARD_WEIGHTS = {
    "util":  0.40,    # utilization efficiency reward
    "sla":   0.40,    # SLA violation penalty
    "cost":  0.15,    # over-provisioning cost penalty
    "scale": 0.05,    # scaling churn penalty
}

# Observation space: ordered list of feature columns extracted from forecast_dataset.csv
# Plus two synthetic features added at runtime: vm_count_norm, queue_length_norm
ENV_OBS_FEATURES = [
    "cpu_util_percent",                  # current cluster CPU utilization
    "mem_util_percent",                  # current memory utilization
    "yhat",                              # Prophet 1-step-ahead forecast
    "cpu_util_percent_roll_mean_12",     # 1-hour rolling mean
    "cpu_util_percent_trend",            # rate of change
    "hour_sin",                          # cyclic hour encoding
    "hour_cos",
    "dow_sin",                           # cyclic day-of-week encoding
    "dow_cos",
    "is_business_hours",                 # binary business hours flag
    "is_weekend",                        # binary weekend flag
]
# vm_count_norm and queue_length_norm are appended at runtime → total OBS_DIM = 13
ENV_OBS_DIM = len(ENV_OBS_FEATURES) + 2   # +2 for vm_count_norm, queue_length_norm

# Normalisation bounds for each feature (min, max)
# Used by StateBuilder to scale features to [0, 1]
ENV_FEATURE_BOUNDS = {
    "cpu_util_percent":               (0.0,   100.0),
    "mem_util_percent":               (0.0,   100.0),
    "yhat":                           (0.0,   100.0),
    "cpu_util_percent_roll_mean_12":  (0.0,   100.0),
    "cpu_util_percent_trend":         (-50.0,  50.0),
    "hour_sin":                       (-1.0,   1.0),
    "hour_cos":                       (-1.0,   1.0),
    "dow_sin":                        (-1.0,   1.0),
    "dow_cos":                        (-1.0,   1.0),
    "is_business_hours":              (0.0,    1.0),
    "is_weekend":                     (0.0,    1.0),
    "vm_count_norm":                  (0.0,    1.0),
    "queue_length_norm":              (0.0,    1.0),
}

# Fallback dataset if forecast_dataset.csv is not available
ENV_FALLBACK_DATASET = PROCESSED_DATASET_FILENAME

# ---------------------------------------------------------------------------
# Milestone 4: Baseline DQN
# ---------------------------------------------------------------------------

# Network architecture
DQN_HIDDEN_LAYERS = [128, 128]          # neurons per hidden layer
DQN_LEARNING_RATE = 1e-3
DQN_GAMMA = 0.99                        # discount factor

# Replay buffer
DQN_BUFFER_SIZE = 50_000
DQN_BATCH_SIZE = 64
DQN_MIN_BUFFER_SIZE = 1_000             # minimum experiences before training starts

# Exploration (epsilon-greedy)
DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.05
DQN_EPSILON_DECAY_STEPS = 10_000

# Target network
DQN_TARGET_UPDATE_FREQ = 500            # steps between hard target network copies

# Training
DQN_TOTAL_EPISODES = 200
DQN_LOG_FREQ = 10                       # log every N episodes
DQN_SAVE_FREQ = 50                      # save checkpoint every N episodes

# Output files
DQN_MODEL_DIR = PROCESSED_DIR.parent / "models"
DQN_MODEL_FILENAME = "dqn_model.pt"
DQN_BEST_MODEL_FILENAME = "dqn_best_model.pt"
DQN_TRAINING_LOG_FILENAME = "dqn_training_log.csv"
DQN_PLOTS_DIR = REPORTS_DIR / "dqn_plots"

# ---------------------------------------------------------------------------
# Milestone 5: HARS — Hybrid Adaptive Reward Scheduling
# ---------------------------------------------------------------------------

# Performance monitor window
HARS_PERF_WINDOW = 20                   # episodes for rolling performance average

# Difficulty score engine
HARS_DIFFICULTY_SMOOTHING = 0.9        # EMA smoothing factor for difficulty score

# Reward weight adaptation bounds
HARS_WEIGHT_MIN = 0.05
HARS_WEIGHT_MAX = 0.70
HARS_WEIGHT_ADAPT_RATE = 0.05          # max change per episode

# HARS output files
HARS_LOG_FILENAME = "hars_log.csv"
HARS_PLOTS_DIR = REPORTS_DIR / "hars_plots"

# ---------------------------------------------------------------------------
# Milestone 6: Policy Export
# ---------------------------------------------------------------------------
POLICY_EXPORT_DIR = PROCESSED_DIR.parent / "models"
POLICY_EXPORT_FILENAME = "exported_policy.pt"
POLICY_METADATA_FILENAME = "policy_metadata.json"

# ---------------------------------------------------------------------------
# Milestone 7: Policy Replay
# ---------------------------------------------------------------------------
REPLAY_LOG_FILENAME = "replay_action_log.csv"
REPLAY_PLOTS_DIR = REPORTS_DIR / "replay_plots"

# ---------------------------------------------------------------------------
# Milestone 8: CloudSim Plus
# ---------------------------------------------------------------------------
CLOUDSIM_JAVA_DIR = PROJECT_ROOT / "milestone_08_cloudsim" / "java"
CLOUDSIM_RESULTS_DIR = DATA_DIR / "cloudsim_results"
CLOUDSIM_METRICS_FILENAME = "cloudsim_metrics.csv"

# Infrastructure parameters (used in Java and Python)
CLOUDSIM_N_HOSTS = 10
CLOUDSIM_HOST_PES = 8
CLOUDSIM_HOST_RAM_MB = 16384
CLOUDSIM_HOST_BW_MBPS = 10000
CLOUDSIM_HOST_STORAGE_MB = 1_000_000
CLOUDSIM_VM_TYPES = [
    {"name": "small",  "pes": 1, "ram": 1024, "bw": 1000, "mips": 1000},
    {"name": "medium", "pes": 2, "ram": 2048, "bw": 2000, "mips": 2000},
    {"name": "large",  "pes": 4, "ram": 4096, "bw": 4000, "mips": 4000},
]

# ---------------------------------------------------------------------------
# Milestone 9: Experiments
# ---------------------------------------------------------------------------
EXPERIMENT_STRATEGIES = [
    "static",
    "threshold",
    "prophet",
    "dqn",
    "prophet_dqn",
    "prophet_dqn_hars",
]
EXPERIMENT_RESULTS_DIR = DATA_DIR / "experiment_results"
EXPERIMENT_METRICS_FILENAME = "experiment_results.csv"

# ---------------------------------------------------------------------------
# Milestone 10: Visualization
# ---------------------------------------------------------------------------
VIZ_PLOTS_DIR = REPORTS_DIR / "final_plots"
VIZ_DPI = 150
VIZ_FIGSIZE_WIDE = (14, 4)
VIZ_FIGSIZE_SQUARE = (8, 6)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = LOGS_DIR / "pipeline.log"

# ---------------------------------------------------------------------------
# Research Phase 12-17: Evaluation, Tuning, Ablation, Statistics, Docs
# ---------------------------------------------------------------------------

RESULTS_DIR          = PROJECT_ROOT / "results"
RESULTS_BASELINE     = RESULTS_DIR / "baseline"
RESULTS_TUNING       = RESULTS_DIR / "tuning"
RESULTS_ABLATION     = RESULTS_DIR / "ablation"
RESULTS_STATS        = RESULTS_DIR / "statistics"
RESULTS_VIZ          = RESULTS_DIR / "visualizations"
RESULTS_REPORTS      = RESULTS_DIR / "reports"
RESULTS_LOGS         = RESULTS_DIR / "logs"
RESULTS_MODELS       = RESULTS_DIR / "models"

for _d in [
    RESULTS_BASELINE, RESULTS_TUNING, RESULTS_ABLATION,
    RESULTS_STATS, RESULTS_VIZ, RESULTS_REPORTS, RESULTS_LOGS, RESULTS_MODELS,
]:
    _d.mkdir(parents=True, exist_ok=True)

# Experiment tracker
EXPERIMENTS_CSV      = RESULTS_DIR / "experiments.csv"

# Phase 12 output names
P12_PROPHET_REPORT   = RESULTS_BASELINE / "prophet_evaluation_report.md"
P12_DQN_REPORT       = RESULTS_BASELINE / "dqn_evaluation_report.md"
P12_HARS_REPORT      = RESULTS_BASELINE / "hars_evaluation_report.md"
P12_CLOUDSIM_REPORT  = RESULTS_BASELINE / "cloudsim_baseline_report.md"

# Phase 13: Tuning search spaces
PROPHET_TUNE_GRID = [
    {"changepoint_prior_scale": 0.001, "seasonality_prior_scale": 1.0,  "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.01,  "seasonality_prior_scale": 5.0,  "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.05,  "seasonality_prior_scale": 10.0, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.10,  "seasonality_prior_scale": 10.0, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.50,  "seasonality_prior_scale": 15.0, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.05,  "seasonality_prior_scale": 10.0, "seasonality_mode": "multiplicative"},
]

DQN_TUNE_GRID = [
    {"lr": 1e-3,  "gamma": 0.99, "batch": 64,  "target_freq": 500,  "eps_decay": 10_000},
    {"lr": 5e-4,  "gamma": 0.99, "batch": 64,  "target_freq": 500,  "eps_decay": 10_000},
    {"lr": 1e-3,  "gamma": 0.95, "batch": 64,  "target_freq": 500,  "eps_decay": 10_000},
    {"lr": 1e-3,  "gamma": 0.99, "batch": 128, "target_freq": 500,  "eps_decay": 10_000},
    {"lr": 1e-3,  "gamma": 0.99, "batch": 64,  "target_freq": 200,  "eps_decay": 10_000},
    {"lr": 1e-3,  "gamma": 0.99, "batch": 64,  "target_freq": 500,  "eps_decay": 5_000},
]

HARS_TUNE_GRID = [
    {"perf_window": 10, "smoothing": 0.8, "adapt_rate": 0.03, "w_min": 0.05, "w_max": 0.60},
    {"perf_window": 20, "smoothing": 0.9, "adapt_rate": 0.05, "w_min": 0.05, "w_max": 0.70},
    {"perf_window": 30, "smoothing": 0.9, "adapt_rate": 0.05, "w_min": 0.05, "w_max": 0.70},
    {"perf_window": 20, "smoothing": 0.7, "adapt_rate": 0.05, "w_min": 0.05, "w_max": 0.70},
    {"perf_window": 20, "smoothing": 0.9, "adapt_rate": 0.10, "w_min": 0.05, "w_max": 0.70},
    {"perf_window": 20, "smoothing": 0.9, "adapt_rate": 0.05, "w_min": 0.10, "w_max": 0.65},
]

# Phase 13 tuning episodes (keep short — we run many configs)
TUNING_EPISODES = 20  # DQN uses this; Prophet/HARS use fewer internally

# Phase 15: Statistical validation
STAT_SEEDS = [0, 42, 123, 999, 2024]
STAT_EPISODES_PER_SEED = 5

# Phase 16: Plot export formats
VIZ_FORMATS = ["png", "pdf", "svg"]
VIZ_FIGSIZE_FULL = (16, 6)
VIZ_FIGSIZE_HALF = (8, 5)
VIZ_FIGSIZE_SQUARE_LG = (10, 8)
