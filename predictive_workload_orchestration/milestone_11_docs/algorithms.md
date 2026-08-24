# Algorithms and Pseudocode

## Algorithm 1: DQN Training Loop

```
INPUT:  CloudEnvironment env, DQNAgent agent, int total_episodes
OUTPUT: Trained Q-network Q_θ

PROCEDURE DQN_Train:
  Initialize online_net Q_θ with Xavier weights
  Initialize target_net Q_θ' ← Q_θ
  Initialize ReplayBuffer D with capacity N
  Initialize ε ← ε_start

  FOR episode = 1 TO total_episodes:
    obs ← env.reset(seed=episode)
    episode_reward ← 0

    WHILE NOT done:
      // Epsilon-greedy action selection
      IF random() < ε:
        action ← random_action()
      ELSE:
        action ← argmax_a Q_θ(obs, a)

      next_obs, reward, done ← env.step(action)
      D.push(obs, action, reward, next_obs, done)
      episode_reward += reward
      obs ← next_obs

      // Training step (if buffer ready)
      IF |D| ≥ MIN_BUFFER_SIZE:
        (s, a, r, s', d) ← D.sample(batch_size)
        y = r + γ · max_a' Q_θ'(s', a') · (1 - d)     // Bellman target
        loss = MSE(Q_θ(s, a), y)
        θ ← θ - α · ∇_θ loss                           // gradient descent

      // Target network update
      IF total_steps mod TARGET_UPDATE_FREQ == 0:
        Q_θ' ← Q_θ                                      // hard copy

      // Epsilon decay
      ε ← max(ε_end, ε_start - (ε_start - ε_end) · (total_steps / ε_decay_steps))

    HARS.on_episode_end(episode_reward, ...)             // weight adaptation
  END FOR
  RETURN Q_θ
```

---

## Algorithm 2: HARS — Hybrid Adaptive Reward Scheduling

```
INPUT:  PerformanceMonitor pm, DifficultyEngine de, WeightGenerator wg
OUTPUT: Adapted reward weights w = {util, sla, cost, scale}

PROCEDURE HARS_Update(episode_metrics):
  pm.record(episode_metrics)
  stats ← pm.get_stats()   // rolling averages

  // Compute difficulty score (EMA-smoothed)
  sla_component    ← clip(stats.sla_rate, 0, 1)
  stagnation       ← clip(0.5 - Δreward / 2, 0, 1)
  loss_component   ← clip(stats.loss / 0.5, 0, 1)
  raw_d            ← 0.5·sla + 0.3·stagnation + 0.2·loss
  d                ← α·d_prev + (1-α)·raw_d              // EMA

  // Compute target weights from difficulty
  w*_sla   ← 0.30 + 0.30·d
  w*_cost  ← 0.25 - 0.15·d
  w*_scale ← 0.10 - 0.05·d
  w*_util  ← 1.0 - w*_sla - w*_cost - w*_scale

  // Incremental adaptation (max change = ADAPT_RATE per episode)
  FOR each component k ∈ {util, sla, cost, scale}:
    δ ← clip(w*_k - w_k, -ADAPT_RATE, +ADAPT_RATE)
    w_k ← clip(w_k + δ, W_MIN, W_MAX)

  // Renormalise to sum = 1
  w ← w / sum(w)
  env.update_reward_weights(w)
  RETURN w
```

---

## Algorithm 3: Prophet Forecasting

```
INPUT:  Time series y(t) = cpu_util_percent at 5-min intervals
OUTPUT: Forecast ŷ(t+h) for horizon h

Prophet decomposes:
  y(t) = trend(t) + seasonality_daily(t) + seasonality_weekly(t) + ε(t)

Trend (piecewise linear):
  trend(t) = k + a(t)ᵀδ + (m + a(t)ᵀγ)
  where δ are changepoint magnitudes (L1-regularised by changepoint_prior_scale)

Daily seasonality (Fourier series, N=10):
  s_daily(t) = Σ_{n=1}^{10} [a_n·cos(2πnt/1) + b_n·sin(2πnt/1)]

Weekly seasonality (Fourier series, N=5):
  s_weekly(t) = Σ_{n=1}^{5}  [a_n·cos(2πnt/7) + b_n·sin(2πnt/7)]

Training: MAP estimation via L-BFGS (Stan backend)

Prediction for future step t+h:
  ŷ(t+h) = trend(t+h) + s_daily(t+h) + s_weekly(t+h)
```

---

## Algorithm 4: Composite Reward Function

```
INPUT:  cpu_util [%], vm_count, is_scaling_action, weights {w_util, w_sla, w_cost, w_scale}
OUTPUT: scalar reward R

R_util  = 1 - |cpu_util - TARGET_UTIL| / 100        // ∈ [0, 1]

R_sla   = -max(0, cpu_util - SLA_THRESHOLD)          // ∈ [-15, 0] (normalised to [-1, 0])
          / (100 - SLA_THRESHOLD)

vm_req  = ⌈cpu_util / CPU_PER_VM⌉
R_cost  = -max(0, vm_count - vm_req) / MAX_VMS       // ∈ [-1, 0]

R_scale = -SCALING_COST  if is_scaling_action else 0  // ∈ {-0.01, 0}

R       = w_util·R_util + w_sla·R_sla + w_cost·R_cost + w_scale·R_scale
```

---

## Algorithm 5: State Normalisation

```
INPUT:  raw feature value v, bounds (v_min, v_max)
OUTPUT: normalised value v_norm ∈ [0, 1]

v_clipped ← clip(v, v_min, v_max)
v_norm    ← (v_clipped - v_min) / (v_max - v_min)

Special case: if v_min == v_max → v_norm = 0.5
```
