# Class Diagrams

## Milestone 1 — Dataset Pipeline

```
┌──────────────────────────┐
│   DatasetPipeline        │
│──────────────────────────│
│ - downloader             │
│ - loader                 │
│ - cleaner                │
│ - aggregator             │
│ - feature_engineer       │
│──────────────────────────│
│ + run() → DataFrame      │
│ + validate(df) → bool    │
└──────────┬───────────────┘
           │ uses
    ┌──────┴──────────────────────────────┐
    │              │            │          │
┌───▼────┐  ┌──────▼───┐  ┌──▼──────┐  ┌▼──────────────┐
│RawData │  │DataCleaner│  │Workload │  │FeatureEngineer│
│Loader  │  │           │  │Aggreg.  │  │               │
│+ load_ │  │+ clean()  │  │+ aggre- │  │+ engineer()   │
│  machine│  └──────────┘  │  gate() │  └───────────────┘
│  usage()│                └─────────┘
└─────────┘
```

## Milestone 3 — CloudEnvironment

```
┌─────────────────────────────────────┐
│         CloudEnvironment            │
│         (gymnasium.Env)             │
│─────────────────────────────────────│
│ - _state_builder  : StateBuilder    │
│ - _action_executor: ActionExecutor  │
│ - _reward_calc    : RewardCalc.     │
│ - _data           : pd.DataFrame    │
│─────────────────────────────────────│
│ + reset(seed) → (obs, info)         │
│ + step(action) → (obs,r,t,tr,info)  │
│ + update_reward_weights(w)          │
│ + episode_summary() → dict          │
└──────────┬──────────────────────────┘
           │ composes
    ┌──────┴───────────────────┐
    │          │               │
┌───▼────┐ ┌──▼──────┐  ┌────▼────────┐
│State   │ │Action   │  │Reward       │
│Builder │ │Executor │  │Calculator   │
│        │ │         │  │             │
│+build()│ │+execute()│  │+compute()   │
│→np.arr │ │→int     │  │+update_     │
└────────┘ └─────────┘  │  weights()  │
                         └────────────┘
```

## Milestone 4 + 5 — DQN + HARS

```
┌────────────────────────┐       ┌───────────────────────┐
│    DQNAgent            │       │   HARSScheduler       │
│────────────────────────│       │───────────────────────│
│ - online_net: DQNNet   │       │ - monitor             │
│ - target_net: DQNNet   │       │ - difficulty_engine   │
│ - buffer: ReplayBuffer │       │ - weight_generator    │
│────────────────────────│       │───────────────────────│
│ + select_action(obs)   │◄──────│ + on_episode_end()    │
│ + step(s,a,r,s',d)     │       │   calls env.update_   │
│ + save(path)           │       │   reward_weights()    │
│ + load(path)           │       └───────────────────────┘
└────────────────────────┘
          │ uses
┌─────────▼──────────┐     ┌─────────────────┐
│   DQNNetwork       │     │  ReplayBuffer   │
│    (nn.Module)     │     │─────────────────│
│────────────────────│     │ - _buffer:deque │
│ net: Sequential    │     │─────────────────│
│  Linear→ReLU×2     │     │ + push(s,a,r,…) │
│  Linear→Q-values   │     │ + sample(batch) │
│────────────────────│     │ + is_ready:bool │
│ + forward(x)→Q     │     └─────────────────┘
└────────────────────┘
```

## Milestone 6–7 — Export and Replay

```
┌─────────────────────────┐      ┌───────────────────────┐
│    PolicyExporter       │      │  PolicyReplayEngine   │
│─────────────────────────│      │───────────────────────│
│ + export(network) → .pt │      │ + replay(seed) → df  │
│ + load(dir) → (model,   │─────►│   loads model,        │
│     metadata)           │      │   runs episode,       │
│ + predict(model, obs)   │      │   logs every action   │
│   → action              │      └───────────────────────┘
└─────────────────────────┘
```
