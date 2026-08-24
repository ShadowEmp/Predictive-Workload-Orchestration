import { useMemo, useState } from 'react';
import type { CSSProperties } from 'react';
import { X } from 'lucide-react';
import { CircularGallery } from '../components/ui/circular-gallery';
import type { GalleryItem } from '../components/ui/circular-gallery';
import cloudsimCard from '../assets/tech_cards/cloudsim_tech_card_1787581537842.jpg';
import gymnasiumCard from '../assets/tech_cards/gymnasium_tech_card_1787582189909.jpg';
import harsCard from '../assets/tech_cards/hars_tech_card_1787581523481.jpg';
import pandasNumpyCard from '../assets/tech_cards/pandas_numpy_tech_card_1787581995699.jpg';
import prophetCard from '../assets/tech_cards/prophet_tech_card_1787582040125.jpg';
import pyarrowCard from '../assets/tech_cards/pyarrow_tech_card_1787582022526.jpg';
import pythonCard from '../assets/tech_cards/python_tech_card_1787581874187.jpg';
import pytorchCard from '../assets/tech_cards/pytorch_tech_card_1787581508598.jpg';
import './Technology.css';

type TechItem = GalleryItem & {
  accent: string;
  overview: string;
  projectUse: string[];
  modules: string[];
  researchFinding: string;
};

const TECH_ITEMS: TechItem[] = [
  {
    id: 'python',
    name: 'Python',
    role: 'Core Research Runtime',
    image: pythonCard,
    imageAlt: 'Python technology card',
    accent: '#06b6d4',
    overview: 'Python is the main implementation language for the research pipeline, backend bridge, model training, experiments, and report generation.',
    projectUse: [
      'Runs the milestone pipeline from dataset cleaning to forecasting, DQN training, HARS reward adaptation, replay, and visualization output.',
      'Connects the research code to the CloudVerse backend so learned policies can be shown in the command center.',
      'Keeps configuration, logging, experiment tracking, and reproducible outputs in one consistent runtime.',
    ],
    modules: ['milestone_*', 'research/', 'config/', 'cloudverse/backend/'],
    researchFinding: 'The Python pipeline produced the final reports, model checkpoints, processed datasets, replay logs, and CloudSim validation artifacts.',
  },
  {
    id: 'pandas-numpy',
    name: 'Pandas + NumPy',
    role: 'Data Processing',
    image: pandasNumpyCard,
    imageAlt: 'Pandas and NumPy technology card',
    accent: '#10b981',
    overview: 'Pandas and NumPy transform raw Alibaba trace records into clean workload signals that the forecasting and RL systems can consume.',
    projectUse: [
      'Loads and aggregates CPU and memory utilization records from the dataset.',
      'Builds derived features such as utilization trends, rolling windows, and normalized state inputs.',
      'Creates CSV outputs for forecasting, evaluation, ablation, and statistical validation.',
    ],
    modules: ['milestone_01_dataset', 'milestone_02_forecasting', 'research/phase12-17', 'data/processed/'],
    researchFinding: 'These tools make the pipeline repeatable: every chart, model input, and summary table is backed by tabular experiment outputs.',
  },
  {
    id: 'pyarrow',
    name: 'PyArrow',
    role: 'Fast Data I/O',
    image: pyarrowCard,
    imageAlt: 'PyArrow technology card',
    accent: '#22d3ee',
    overview: 'PyArrow supports efficient tabular data movement, especially useful when repeatedly reading and writing large workload traces.',
    projectUse: [
      'Improves the performance of large dataset handling during preprocessing and experiment runs.',
      'Supports scalable CSV/Parquet-style workflows for research-grade reproducibility.',
      'Keeps the data layer fast enough for repeated tuning and ablation cycles.',
    ],
    modules: ['requirements.txt', 'data/raw/', 'data/processed/'],
    researchFinding: 'PyArrow is part of the project dependency stack specifically to avoid slow data-loading loops during trace-heavy experiments.',
  },
  {
    id: 'prophet',
    name: 'Prophet',
    role: 'Workload Forecasting',
    image: prophetCard,
    imageAlt: 'Prophet technology card',
    accent: '#f59e0b',
    overview: 'Prophet forecasts near-future CPU demand, giving the scaling policy a lookahead signal instead of only current utilization.',
    projectUse: [
      'Trains on historical CPU utilization and produces forecast_dataset.csv.',
      'Adds yhat-style demand estimates that are merged into the RL environment state.',
      'Captures daily and weekly seasonality patterns found in cloud workloads.',
    ],
    modules: ['milestone_02_forecasting', 'data/reports/forecast_accuracy_report.txt', 'results/baseline/prophet_metrics.csv'],
    researchFinding: 'The hold-out report records MAE 7.7645 %CPU, RMSE 9.3882 %CPU, and MAPE 20.6540% on 454 test rows.',
  },
  {
    id: 'gymnasium',
    name: 'Gymnasium',
    role: 'RL Environment',
    image: gymnasiumCard,
    imageAlt: 'Gymnasium technology card',
    accent: '#8b5cf6',
    overview: 'Gymnasium provides the reinforcement-learning environment contract: state observation, action execution, reward calculation, and episode termination.',
    projectUse: [
      'Defines the CloudEnvironment where VM scaling actions are tested.',
      'Models queue pressure, SLA violations, utilization, cost, and scaling penalties.',
      'Provides a fair comparison surface for static, threshold, Prophet-only, DQN, and HARS strategies.',
    ],
    modules: ['milestone_03_environment/cloud_environment.py', 'state_space.py', 'action_space.py', 'reward.py'],
    researchFinding: 'The environment exposed why simple threshold logic fails: the threshold ablation reached a 99.65% SLA violation rate despite lower cost.',
  },
  {
    id: 'pytorch',
    name: 'PyTorch',
    role: 'DQN Learning Engine',
    image: pytorchCard,
    imageAlt: 'PyTorch technology card',
    accent: '#ef4444',
    overview: 'PyTorch powers the Deep Q-Network that learns which VM scaling action maximizes long-term reward.',
    projectUse: [
      'Implements the Q-network, replay buffer, target network updates, and training loop.',
      'Exports dqn_model.pt, dqn_best_model.pt, and policy metadata for replay and validation.',
      'Learns when to scale up, scale down, or hold steady based on workload state.',
    ],
    modules: ['milestone_04_dqn/network.py', 'agent.py', 'trainer.py', 'data/models/*.pt'],
    researchFinding: 'The DQN evaluation reports mean reward 106.43 and reduced SLA violations from roughly 37% under static scaling to below 5%.',
  },
  {
    id: 'hars',
    name: 'HARS',
    role: 'Adaptive Reward Scheduling',
    image: harsCard,
    imageAlt: 'HARS technology card',
    accent: '#c9a24a',
    overview: 'HARS dynamically adjusts reward weights so the agent balances SLA protection, utilization, cost, and scaling stability across easy and hard episodes.',
    projectUse: [
      'Tracks performance and difficulty after episodes.',
      'Raises SLA penalty weight when violations are high and shifts toward efficiency when the system is stable.',
      'Makes the reward function inspectable through logged utilization, SLA, cost, and scale weights.',
    ],
    modules: ['milestone_05_hars', 'data/reports/hars_log.csv', 'results/tuning/hars_tuning_results.csv'],
    researchFinding: 'The HARS summary records weight stability 0.986, final SLA weight 0.4107, and reward improvement 11.411.',
  },
  {
    id: 'cloudsim',
    name: 'CloudSim Plus',
    role: 'Digital Twin Validation',
    image: cloudsimCard,
    imageAlt: 'CloudSim Plus technology card',
    accent: '#38bdf8',
    overview: 'CloudSim Plus validates the learned policy in a datacenter-style simulation with hosts, VMs, workloads, power, latency, and SLA behavior.',
    projectUse: [
      'Converts policy replay into simulation_config.json.',
      'Models infrastructure metrics such as VM count, CPU, memory, energy, power, response time, cost, and queue length.',
      'Provides a validation bridge between research metrics and CloudVerse visualization.',
    ],
    modules: ['milestone_08_cloudsim', 'CloudSimRunner.java', 'data/cloudsim_results/cloudsim_metrics.csv'],
    researchFinding: 'The digital twin summary records 288 steps, 22.43 kWh energy, $6.19 cost, 108.63 ms p95 response, and 1.39% SLA rate.',
  },
];

const SUMMARY = [
  { label: 'Card Images', value: '08' },
  { label: 'Research Pipeline', value: '11 phases' },
  { label: 'Primary Model', value: 'DQN' },
  { label: 'Validation', value: 'CloudSim' },
];

export default function Technology() {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const selectedTech = useMemo(
    () => TECH_ITEMS.find((item) => item.id === selectedId) ?? null,
    [selectedId]
  );

  return (
    <main className={`technology-page ${selectedTech ? 'detail-open' : ''}`}>
      <section className="technology-hero-panel">
        <span className="eyebrow">CloudVerse Architecture</span>
        <h1 className="kage-display">Technology Stack</h1>
        <p>
          Scroll to rotate through the technology cards. Click any card image to open how that
          technology works and where it appears in the predictive workload orchestration project.
        </p>

        <div className="tech-summary-grid" aria-label="Technology summary">
          {SUMMARY.map((item) => (
            <div className="summary-card" key={item.label}>
              <strong>{item.value}</strong>
              <span>{item.label}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="technology-gallery-scroll" aria-label="Scrollable circular technology gallery">
        <div className="technology-gallery-sticky">
          <div className="technology-gallery-copy">
            <span className="eyebrow">Scroll Gallery</span>
            <h2 className="kage-display">Core Tools</h2>
          </div>

          <CircularGallery
            items={TECH_ITEMS}
            selectedId={selectedId}
            radius={620}
            autoRotateSpeed={0.018}
            onItemSelect={setSelectedId}
          />

          <div className="technology-scroll-hint">Scroll to rotate · Click a card for details</div>
        </div>
      </section>

      <aside
        className="technology-detail-panel"
        aria-hidden={!selectedTech}
        style={{ '--detail-accent': selectedTech?.accent ?? '#06b6d4' } as CSSProperties}
      >
        {selectedTech && (
          <>
            <button
              className="technology-detail-close"
              type="button"
              aria-label="Close technology details"
              onClick={() => setSelectedId(null)}
            >
              <X size={20} />
            </button>

            <div className="detail-card-image">
              <img src={selectedTech.image} alt={selectedTech.imageAlt} />
            </div>

            <div className="detail-card-copy">
              <span>{selectedTech.role}</span>
              <h2>{selectedTech.name}</h2>
              <p>{selectedTech.overview}</p>

              <div className="doc-section tech-doc-section">
                <h3 className="doc-label">How It Is Used</h3>
                <ul className="doc-steps">
                  {selectedTech.projectUse.map((detail) => (
                    <li key={detail}>
                      <div className="doc-step-copy">
                        <strong>Project Role</strong>
                        <span>{detail}</span>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="tech-module-list">
                <span>Project files</span>
                {selectedTech.modules.map((module) => (
                  <strong key={module}>{module}</strong>
                ))}
              </div>

              <div className="tech-research-finding">
                <span>Research finding</span>
                <p>{selectedTech.researchFinding}</p>
              </div>
            </div>
          </>
        )}
      </aside>
    </main>
  );
}
