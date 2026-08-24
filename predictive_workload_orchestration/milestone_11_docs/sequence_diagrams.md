# Sequence Diagrams

## Training Loop Sequence (M4 + M5)

```
DQNTrainer          DQNAgent         CloudEnvironment    HARSScheduler
    │                   │                    │                   │
    │──── reset() ──────────────────────────►│                   │
    │◄─── (obs, info) ──────────────────────│                   │
    │                   │                    │                   │
    │  WHILE NOT done:  │                    │                   │
    │──select_action()─►│                    │                   │
    │◄── action ────────│                    │                   │
    │                   │                    │                   │
    │──────────────── step(action) ─────────►│                   │
    │◄──────── (obs', reward, done, info) ───│                   │
    │                   │                    │                   │
    │──step(s,a,r,s',d)►│                    │                   │
    │  (push to buffer, │                    │                   │
    │   train if ready) │                    │                   │
    │◄── loss ──────────│                    │                   │
    │  END WHILE        │                    │                   │
    │                   │                    │                   │
    │────────────────────────────────────── on_episode_end() ───►│
    │                   │                    │◄── update_weights()│
    │                   │  (weights updated  │                   │
    │                   │  in RewardCalc.)   │                   │
```

## Policy Export and Replay Sequence (M6 + M7)

```
User        PolicyExporter      DQNAgent         PolicyReplayEngine   CloudEnvironment
 │               │                  │                    │                   │
 │──export()────►│                  │                    │                   │
 │               │──trace(network)──►│                    │                   │
 │               │◄── TorchScript ──│                    │                   │
 │               │──save(.pt)        │                    │                   │
 │               │──write metadata   │                    │                   │
 │               │                  │                    │                   │
 │──replay()──────────────────────────────────────────────►                   │
 │               │                  │             load(policy_dir)            │
 │               │                  │             env.reset()──────────────►│
 │               │                  │             WHILE NOT done:            │
 │               │                  │             predict(obs)               │
 │               │                  │             env.step(action)──────────►│
 │               │                  │             log row                    │
 │               │                  │             save CSV                   │
```

## Full System Sequence (End-to-End)

```
                    M1              M2              M3+M4+M5        M6       M7       M8       M9
User/Runner     DataPipeline    ForecastPipeline   DQNTrainer    Exporter  Replay  CloudSim  Experiments
     │               │                │                │             │        │        │          │
     │──run M1───────►│               │                │             │        │        │          │
     │               │──process───────►│               │             │        │        │          │
     │               │  processed_dataset.csv          │             │        │        │          │
     │                               │                │             │        │        │          │
     │──run M2──────────────────────►│                │             │        │        │          │
     │                               │──train Prophet  │             │        │        │          │
     │                               │  forecast_dataset.csv        │        │        │          │
     │                                                │             │        │        │          │
     │──run M4+M5──────────────────────────────────►│             │        │        │          │
     │                                               │──200 ep───   │        │        │          │
     │                                               │  dqn_best.pt │        │        │          │
     │                                                              │        │        │          │
     │──run M6───────────────────────────────────────────────────►│        │        │          │
     │                                               exported_policy.pt     │        │          │
     │                                                                      │        │          │
     │──run M7──────────────────────────────────────────────────────────►│        │          │
     │                                                             replay_action_log.csv        │
     │                                                                               │          │
     │──run M8──────────────────────────────────────────────────────────────────►│          │
     │                                                                  cloudsim_metrics.csv    │
     │                                                                                          │
     │──run M9─────────────────────────────────────────────────────────────────────────────►│
     │                                                                         experiment_results.csv
```
