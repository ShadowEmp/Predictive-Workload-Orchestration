"""
research/experiment_tracker.py
================================
Global experiment tracker — writes every experiment to experiments.csv.
Never overwrites existing rows. Thread-safe via file-level locking.

Columns:
    experiment_id, date, phase, model, dataset, hyperparameters,
    training_time_s, eval_time_s, reward, mae, rmse, mape, r2,
    cpu_util, queue_len, energy_kwh, cost_usd, sla_rate,
    response_time_ms, convergence_episode, difficulty_score, notes
"""

import csv
import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from config.logger import get_logger
from config.settings import EXPERIMENTS_CSV

logger = get_logger(__name__)

_LOCK = threading.Lock()

COLUMNS = [
    "experiment_id", "date", "phase", "model", "dataset",
    "hyperparameters", "training_time_s", "eval_time_s",
    "reward", "mae", "rmse", "mape", "r2",
    "cpu_util", "queue_len", "energy_kwh", "cost_usd", "sla_rate",
    "response_time_ms", "convergence_episode", "difficulty_score", "notes",
]


def _make_id(phase: str, model: str, hyperparameters: dict) -> str:
    raw = f"{phase}_{model}_{json.dumps(hyperparameters, sort_keys=True)}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def log_experiment(
    phase: str,
    model: str,
    dataset: str,
    hyperparameters: Dict[str, Any],
    training_time_s: float = 0.0,
    eval_time_s: float = 0.0,
    reward: float = float("nan"),
    mae: float = float("nan"),
    rmse: float = float("nan"),
    mape: float = float("nan"),
    r2: float = float("nan"),
    cpu_util: float = float("nan"),
    queue_len: float = float("nan"),
    energy_kwh: float = float("nan"),
    cost_usd: float = float("nan"),
    sla_rate: float = float("nan"),
    response_time_ms: float = float("nan"),
    convergence_episode: int = -1,
    difficulty_score: float = float("nan"),
    notes: str = "",
) -> str:
    """Append one experiment row to experiments.csv. Returns experiment_id."""
    exp_id = _make_id(phase, model, hyperparameters)

    row = {
        "experiment_id":      exp_id,
        "date":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase":              phase,
        "model":              model,
        "dataset":            dataset,
        "hyperparameters":    json.dumps(hyperparameters, separators=(",", ":")),
        "training_time_s":    round(training_time_s, 2),
        "eval_time_s":        round(eval_time_s, 2),
        "reward":             _fmt(reward),
        "mae":                _fmt(mae),
        "rmse":               _fmt(rmse),
        "mape":               _fmt(mape),
        "r2":                 _fmt(r2),
        "cpu_util":           _fmt(cpu_util),
        "queue_len":          _fmt(queue_len),
        "energy_kwh":         _fmt(energy_kwh),
        "cost_usd":           _fmt(cost_usd),
        "sla_rate":           _fmt(sla_rate),
        "response_time_ms":   _fmt(response_time_ms),
        "convergence_episode": convergence_episode,
        "difficulty_score":   _fmt(difficulty_score),
        "notes":              notes,
    }

    path = Path(EXPERIMENTS_CSV)
    with _LOCK:
        write_header = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    logger.info("Experiment logged: %s  [%s / %s]", exp_id, phase, model)
    return exp_id


def _fmt(v) -> str:
    try:
        return f"{float(v):.6f}"
    except (TypeError, ValueError):
        return str(v)
