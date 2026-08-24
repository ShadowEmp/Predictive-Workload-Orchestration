# Predictive Workload Orchestration

A research-grade implementation of **Predictive Workload Orchestration** using:
- **Prophet** — time-series forecasting of CPU demand
- **DQN** — deep reinforcement learning for auto-scaling decisions
- **HARS** — Hybrid Adaptive Reward Scheduling
- **CloudSim Plus** — cloud infrastructure digital twin

Developed as a VTU final-year engineering project.

---

## Project Structure

```
predictive_workload_orchestration/
├── config/                   # Central configuration
├── data/                     # Dataset files (raw + processed)
├── milestone_01_dataset/     # Dataset preparation
├── milestone_02_forecasting/ # Prophet forecasting
├── milestone_03_environment/ # Gymnasium RL environment
├── milestone_04_dqn/         # Baseline DQN
├── milestone_05_hars/        # HARS reward scheduling
├── milestone_06_policy_export/
├── milestone_07_policy_replay/
├── milestone_08_cloudsim/    # CloudSim Plus digital twin
├── milestone_09_experiments/ # Benchmarking
├── milestone_10_visualization/
├── milestone_11_docs/
├── tests/                    # Unit tests
├── logs/                     # Runtime logs
└── requirements.txt
```

---

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Dataset Preparation | ✅ Complete |
| 2 | Workload Forecasting | ✅ Complete |
| 3 | Gymnasium Environment | ✅ Complete |
| 4 | Baseline DQN | ✅ Complete |
| 5 | HARS | ✅ Complete |
| 6 | Policy Export | ✅ Complete |
| 7 | Policy Replay | ✅ Complete |
| 8 | CloudSim Plus | ✅ Complete |
| 9 | Experiments | ✅ Complete |
| 10 | Visualization | ✅ Complete |
| 11 | Documentation | ✅ Complete |

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset files in data/raw/ (or use --download)
#    Download manually from: https://github.com/alibaba/clusterdata

# 3. Run Milestone 1 pipeline
cd predictive_workload_orchestration
python -m milestone_01_dataset.pipeline

# 4. Run tests
python -m pytest tests/test_milestone_01.py -v
```

---

## Dataset

**Alibaba Cluster Trace 2018**
- Source: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018
- Primary file: `machine_usage.csv`
- License: See Alibaba's dataset license terms

---

## License

Research and educational use only.
