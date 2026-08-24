import React, { useState, useEffect } from 'react';
import './HowItWorks.css';

const books = [
  {
    id: 'book-1',
    kicker: 'CHAPTER I',
    title: 'THE REACTIVE BASELINE',
    subtitle: 'Threshold-based auto-scaling',
    coverImage: '/cover_reactive.jpg', // Copied generated image
    color: '#050505',
    content: (
      <>
        <h2 className="detail-title">Reactive Scaling & Environment</h2>
        <p className="detail-description">
          The baseline explains what happens when scaling decisions are made only after the system is already under pressure. We compare static provisioning and threshold-based rules inside the same Gymnasium environment used by the learning agent.
        </p>
        <div className="doc-section">
          <h3 className="doc-label">Gymnasium CloudEnvironment</h3>
          <ul className="doc-steps">
            <li>
              <div className="doc-step-copy">
                <strong>Capacity Model</strong>
                <span>Each VM provides 100% CPU capacity. Effective CPU = (demand / total_capacity) × 100.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Queue Dynamics</strong>
                <span>Overflow demand enters a queue; queue drains when capacity exceeds demand. Episode truncates if queue maxes out.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Reward Function</strong>
                <span>Composite signal = utilization_reward + sla_penalty + cost_penalty + scaling_penalty.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Environment Files</strong>
                <span>The simulation logic is split across cloud_environment.py, state_space.py, action_space.py, and reward.py so each decision can be tested independently.</span>
              </div>
            </li>
          </ul>
        </div>
        <div className="doc-section">
          <h3 className="doc-label">System Flaws</h3>
          <ul className="doc-steps">
            <li>
              <div className="doc-step-copy">
                <strong>Always Late</strong>
                <span>Reactive systems only act after demand has spiked, leading to SLA violations during boot times.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Oscillation</strong>
                <span>Rapid spikes cause thrashing: scale up during overload, then scale down once traffic dips, then repeat on the next burst.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Blind to Future Demand</strong>
                <span>The baseline only sees the current utilization. It cannot use Prophet forecasts, time-of-day patterns, or queue trend signals to prepare early.</span>
              </div>
            </li>
          </ul>
        </div>
        <div className="doc-section">
          <h3 className="doc-label">Research Observations</h3>
          <ul className="doc-steps">
            <li>
              <div className="doc-step-copy">
                <strong>Static Provisioning</strong>
                <span>The ablation summary records a mean reward of 72.93 with a 32.99% SLA violation rate, showing that fixed capacity is stable but misses bursts.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Threshold Policy</strong>
                <span>The threshold strategy lowers cost to about $1.28, but the SLA violation rate rises to 99.65%, making it unsuitable for production-style orchestration.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Why It Matters</strong>
                <span>This chapter sets the control case: any proposed method must reduce violations without simply over-provisioning VMs forever.</span>
              </div>
            </li>
          </ul>
        </div>
      </>
    )
  },
  {
    id: 'book-2',
    kicker: 'CHAPTER II',
    title: 'PREDICTIVE DQN',
    subtitle: 'Reinforcement Learning',
    coverImage: '/cover_dqn.jpg',
    color: '#0a0510',
    content: (
      <>
        <h2 className="detail-title">Deep Q-Network</h2>
        <p className="detail-description">
          Instead of hardcoded rules, the Deep Q-Network learns a scaling policy from repeated episodes. It watches workload state, VM count, queue pressure, and forecast features, then chooses the action with the best long-term reward.
        </p>
        <div className="doc-section">
          <h3 className="doc-label">Data Ingestion</h3>
          <ul className="doc-steps">
            <li>
              <div className="doc-step-copy">
                <strong>Alibaba Cluster Trace 2018</strong>
                <span>The workload source is machine_usage.csv from Alibaba Cluster Trace 2018. The pipeline cleans missing values, aggregates raw records, engineers time features, and stores processed_dataset.csv.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Forecast Merge</strong>
                <span>Prophet output is merged into forecast_dataset.csv so the agent can observe both current utilization and predicted near-future demand.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Replay Inputs</strong>
                <span>Policy replay writes replay_action_log.csv and replay plots, letting the same learned behavior be inspected before CloudSim validation.</span>
              </div>
            </li>
          </ul>
        </div>
        <div className="doc-section">
          <h3 className="doc-label">DQN Architecture</h3>
          <ul className="doc-steps">
            <li>
              <div className="doc-step-copy">
                <strong>State Space & MLP</strong>
                <span>The agent observes normalized telemetry such as CPU utilization, forecast signal, VM count, queue length, and time context. A PyTorch Q-network maps that state to one value per action.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Action Space</strong>
                <span>Discrete(3) actions: SCALE_UP (add 1 VM), SCALE_DOWN (remove 1 VM), NO_OP (no change).</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Training Dynamics</strong>
                <span>The implementation uses epsilon-greedy exploration, replay memory, mini-batch updates, a target network, and loss tracking to stabilize Q-learning.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Model Artifacts</strong>
                <span>Training exports dqn_model.pt, dqn_best_model.pt, and policy_metadata.json for replay, comparison, and CloudVerse visualization.</span>
              </div>
            </li>
          </ul>
        </div>
        <div className="doc-section">
          <h3 className="doc-label">DQN Results</h3>
          <ul className="doc-steps">
            <li>
              <div className="doc-step-copy">
                <strong>Evaluation Reward</strong>
                <span>The baseline DQN metrics report a mean reward of 106.43 across evaluation episodes, with a best training reward of 86.73 and final average reward of 75.57.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>SLA Improvement</strong>
                <span>The results summary shows DQN reducing SLA violations from roughly 37% under static scaling to below 5% in learned-policy evaluation.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Tuning Insight</strong>
                <span>The best DQN tuning runs reached about 107.8 mean reward, with low SLA rates and convergence around 12 episodes in the tuning experiments.</span>
              </div>
            </li>
          </ul>
        </div>
      </>
    )
  },
  {
    id: 'book-3',
    kicker: 'CHAPTER III',
    title: 'PROPHET & H.A.R.S.',
    subtitle: 'Forecasting & Reward Shaping',
    coverImage: '/cover_prophet.jpg',
    color: '#150802',
    content: (
      <>
        <h2 className="detail-title">Foresight & Balance</h2>
        <p className="detail-description">
          A predictive agent is only as good as its forecast signal, reward design, and validation loop. This chapter connects Prophet forecasting, HARS adaptive reward weights, CloudSim validation, and the CloudVerse 3D interface.
        </p>
        <div className="doc-section">
          <h3 className="doc-label">Forecasting</h3>
          <ul className="doc-steps">
            <li>
              <div className="doc-step-copy">
                <strong>Facebook Prophet</strong>
                <span>A decomposable additive time-series model generates yhat CPU forecasts from historical utilization, giving the policy a lookahead signal instead of only current telemetry.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Hold-out Metrics</strong>
                <span>The recorded Prophet report uses 454 hold-out rows and measures MAE 7.7645 %CPU, RMSE 9.3882 %CPU, and MAPE 20.6540%.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Key Lesson</strong>
                <span>Forecast alone was not enough: the ablation study shows Prophet-only scaling had a 99.65% SLA violation rate, so it needs the DQN policy layer.</span>
              </div>
            </li>
          </ul>
        </div>
        <div className="doc-section">
          <h3 className="doc-label">Reward Shaping (HARS)</h3>
          <ul className="doc-steps">
            <li>
              <div className="doc-step-copy">
                <strong>Dynamic Adaptation</strong>
                <span>HARS evaluates episode difficulty after each run. If the episode was "hard" (many violations), it increases the weight of the SLA penalty. If "easy", it increases the cost/energy efficiency penalty.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Tracked Weights</strong>
                <span>The HARS log records utilization, SLA, cost, and scale weights per episode, making the reward policy explainable instead of a fixed black-box score.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Stability Result</strong>
                <span>The HARS summary reports mean difficulty 0.314, final difficulty 0.369, final SLA weight 0.4107, weight stability 0.986, and reward improvement 11.411.</span>
              </div>
            </li>
          </ul>
        </div>
        <div className="doc-section">
          <h3 className="doc-label">CloudSim Validation</h3>
          <ul className="doc-steps">
            <li>
              <div className="doc-step-copy">
                <strong>Digital Twin Metrics</strong>
                <span>CloudSim validation ran 288 steps with mean CPU 65.58%, mean VM count 5.16, total energy 22.43 kWh, and total cost $6.19.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Latency and SLA</strong>
                <span>The CloudSim summary records mean response time 100.33 ms, p95 response time 108.63 ms, four SLA violations, and a 1.39% SLA rate.</span>
              </div>
            </li>
          </ul>
        </div>
        <div className="doc-section">
          <h3 className="doc-label">CloudVerse Visualization</h3>
          <ul className="doc-steps">
            <li>
              <div className="doc-step-copy">
                <strong>Real-Time 3D Telemetry</strong>
                <span>FastAPI backend emits WebSocket events (DQN_DECISION, QUEUE_INCREASED, etc). React Three Fiber renders the city in dual split-screen (Threshold vs DQN).</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Challenge Injection</strong>
                <span>The backend exposes host failure, flash sale, and memory pressure endpoints so the command center can demonstrate policy behavior under stress.</span>
              </div>
            </li>
            <li>
              <div className="doc-step-copy">
                <strong>Analytics Link</strong>
                <span>The same metrics feed the Analytics page, connecting live simulation telemetry with the research reports, plots, and ablation findings.</span>
              </div>
            </li>
          </ul>
        </div>
      </>
    )
  }
];

export default function HowItWorks() {
  const [mode, setMode] = useState<'gallery' | 'detail'>('gallery');
  const [selectedBook, setSelectedBook] = useState<string | null>(null);
  const [hoveredBook, setHoveredBook] = useState<string | null>(null);

  useEffect(() => {
    // Lock body scroll when in detail mode
    if (mode === 'detail') {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [mode]);

  const handleBookClick = (id: string) => {
    if (mode === 'gallery') {
      setSelectedBook(id);
      setMode('detail');
    }
  };

  const handleClose = () => {
    setMode('gallery');
    setSelectedBook(null);
  };

  const activeBookContent = books.find(b => b.id === selectedBook)?.content;

  return (
    <div className={`tech-stage mode-${mode}`} aria-label="CloudVerse Technology Gallery">
      
      <h1 className="hero-word" aria-hidden="true">MODELS</h1>

      <section className="tech-gallery">
        {books.map((book, idx) => {
          const isSelected = selectedBook === book.id;
          const isHovered = hoveredBook === book.id;
          const cardClass = `book-card book-${idx + 1} ${isSelected ? 'selected' : ''} ${isHovered ? 'hovered' : ''}`;

          return (
            <button
              key={book.id}
              className={cardClass}
              type="button"
              onClick={() => handleBookClick(book.id)}
              onMouseEnter={() => mode === 'gallery' && setHoveredBook(book.id)}
              onMouseLeave={() => setHoveredBook(null)}
              style={{
                '--cover-color': book.color,
                '--cover': `url('${book.coverImage}')`
              } as React.CSSProperties}
            >
              <div className="book">
                <div className="book-shadow"></div>
                <div className="book-back"></div>
                <div className="page-block"></div>
                
                <div className="page-fan">
                  <i></i><i></i><i></i><i></i>
                </div>
                
                <div className="front-cover">
                  <div className="cover-copy">
                    <div className="cover-kicker">{book.kicker}</div>
                    <div className="cover-title">{book.title}</div>
                    <div className="cover-subtitle">{book.subtitle}</div>
                    <div className="cover-footer">CLOUDVERSE.AI</div>
                  </div>
                </div>
              </div>
            </button>
          );
        })}
      </section>

      <div className="detail-panel">
        <div className="detail-scroll">
          {activeBookContent}
        </div>
      </div>

      <button 
        className="close-button" 
        onClick={handleClose}
        aria-label="Close detail view"
      >
        +
      </button>
    </div>
  );
}
