import {
  Activity,
  BarChart3,
  Boxes,
  BrainCircuit,
  Cable,
  Cpu,
  Database,
  FlaskConical,
  Gauge,
  GitBranch,
  Layers3,
  LineChart,
  Network,
  RadioTower,
  Server,
  Sparkles,
  TestTube2,
} from 'lucide-react';
import type { ComponentType, CSSProperties } from 'react';
import './Technology.css';

type TechItem = {
  name: string;
  category: string;
  icon: ComponentType<{ size?: number; strokeWidth?: number }>;
  accent: string;
  purpose: string;
  details: string[];
  usedIn: string;
};

const TECHNOLOGIES: TechItem[] = [
  {
    name: 'Python',
    category: 'Core Runtime',
    icon: Cpu,
    accent: '#06b6d4',
    purpose: 'Main research and orchestration language for every milestone pipeline.',
    details: ['Dataset preparation', 'Forecasting, RL training, replay, experiments', 'Backend bridge logic'],
    usedIn: 'milestone_* modules, research/, config/, cloudverse/backend',
  },
  {
    name: 'Pandas + NumPy',
    category: 'Data Processing',
    icon: Database,
    accent: '#10b981',
    purpose: 'Transforms Alibaba Cluster Trace data into clean, aggregated workload signals.',
    details: ['CPU trace cleaning and aggregation', 'Feature engineering for forecasting', 'Experiment metrics tables'],
    usedIn: 'milestone_01_dataset, milestone_02_forecasting, research reports',
  },
  {
    name: 'PyArrow',
    category: 'Data I/O',
    icon: Boxes,
    accent: '#22d3ee',
    purpose: 'Keeps large dataset reads and writes fast during repeatable research runs.',
    details: ['Efficient tabular storage', 'Large trace handling', 'CSV/Parquet workflow support'],
    usedIn: 'requirements.txt data processing stack',
  },
  {
    name: 'Facebook Prophet',
    category: 'Forecasting Model',
    icon: LineChart,
    accent: '#f59e0b',
    purpose: 'Predicts future CPU demand so scaling can happen before workload spikes arrive.',
    details: ['Trend and seasonality modeling', 'Forecast dataset generation', 'Accuracy evaluation with MAE/RMSE/R2'],
    usedIn: 'milestone_02_forecasting, results/baseline/prophet_*',
  },
  {
    name: 'Gymnasium',
    category: 'RL Environment',
    icon: Network,
    accent: '#8b5cf6',
    purpose: 'Provides the environment interface for workload states, scaling actions, and rewards.',
    details: ['CloudEnvironment simulation loop', 'State and action spaces', 'Reward-driven scaling episodes'],
    usedIn: 'milestone_03_environment',
  },
  {
    name: 'PyTorch',
    category: 'Deep Learning',
    icon: BrainCircuit,
    accent: '#ef4444',
    purpose: 'Runs the Deep Q-Network that learns proactive VM scaling policies.',
    details: ['Q-network architecture', 'Experience replay', 'Exported policy checkpoints'],
    usedIn: 'milestone_04_dqn, data/models/*.pt, results/models',
  },
  {
    name: 'HARS',
    category: 'Adaptive Scheduling',
    icon: Sparkles,
    accent: '#c9a24a',
    purpose: 'Custom Hybrid Adaptive Reward Scheduling layer for balancing SLA and energy goals.',
    details: ['Performance monitor', 'Difficulty score engine', 'Dynamic reward weight generator'],
    usedIn: 'milestone_05_hars, research/phase13, research/phase14',
  },
  {
    name: 'CloudSim Plus + Java',
    category: 'Digital Twin',
    icon: Server,
    accent: '#38bdf8',
    purpose: 'Models datacenter hosts, VMs, cloudlets, power draw, and infrastructure metrics.',
    details: ['Discrete-event simulation design', 'CloudSimRunner Java skeleton', 'Python fallback metrics when JAR is unavailable'],
    usedIn: 'milestone_08_cloudsim/java/CloudSimRunner.java',
  },
  {
    name: 'FastAPI + WebSockets',
    category: 'Backend API',
    icon: RadioTower,
    accent: '#14b8a6',
    purpose: 'Streams live simulation events from Python into the CloudVerse command center.',
    details: ['Simulation start/stop controls', 'Challenge injection endpoints', 'Live telemetry broadcast'],
    usedIn: 'cloudverse/backend/main.py',
  },
  {
    name: 'Pydantic',
    category: 'Event Models',
    icon: Cable,
    accent: '#a78bfa',
    purpose: 'Defines typed event payloads shared across the simulation bridge and frontend.',
    details: ['CloudVerseEvent schema', 'Validated WebSocket messages', 'Structured severity and source fields'],
    usedIn: 'cloudverse/backend/models/events.py',
  },
  {
    name: 'React + TypeScript + Vite',
    category: 'Frontend App',
    icon: Layers3,
    accent: '#60a5fa',
    purpose: 'Builds the CloudVerse interface, routed pages, dashboards, and command center controls.',
    details: ['React Router pages', 'Type-safe component code', 'Fast Vite development and production builds'],
    usedIn: 'cloudverse/frontend/src, package.json',
  },
  {
    name: 'Three.js + React Three Fiber',
    category: '3D Visualization',
    icon: GitBranch,
    accent: '#34d399',
    purpose: 'Renders the interactive city-style datacenter view used by CloudVerse.',
    details: ['Host districts and VM instances', 'Traffic and road systems', 'Drei helpers for scene composition'],
    usedIn: 'cloudverse/frontend/src/cloudverse',
  },
  {
    name: 'Recharts + D3',
    category: 'Analytics UI',
    icon: BarChart3,
    accent: '#fb7185',
    purpose: 'Presents benchmark comparisons, live telemetry, policy curves, and workload trends.',
    details: ['Line, area, bar, and pie charts', 'Dashboard metrics panels', 'Research result storytelling'],
    usedIn: 'src/pages/Analytics.tsx, frontend package.json',
  },
  {
    name: 'Matplotlib + Seaborn',
    category: 'Research Plots',
    icon: Gauge,
    accent: '#f97316',
    purpose: 'Generates reproducible figures for experiments, dashboards, and final reports.',
    details: ['Forecast plots', 'DQN/HARS training curves', 'CloudSim and comparison dashboards'],
    usedIn: 'milestone_10_visualization, research/phase16, data/reports',
  },
  {
    name: 'Pytest + MyPy',
    category: 'Quality Tooling',
    icon: TestTube2,
    accent: '#84cc16',
    purpose: 'Keeps milestone pipelines testable and supports typed development in the Python stack.',
    details: ['Milestone unit tests', 'Coverage support', 'Optional static type checking'],
    usedIn: 'tests/, pytest.ini, requirements.txt',
  },
  {
    name: 'Alibaba Cluster Trace 2018',
    category: 'Dataset',
    icon: Activity,
    accent: '#e0231c',
    purpose: 'Real cloud workload source used to train forecasts, replay policies, and benchmark scaling.',
    details: ['machine_usage.csv workload history', 'Processed and forecast datasets', 'Baseline and ablation experiments'],
    usedIn: 'data/raw, data/processed, results/',
  },
  {
    name: 'Experiment Tracking',
    category: 'Research Workflow',
    icon: FlaskConical,
    accent: '#d946ef',
    purpose: 'Records tuning, ablation, statistical validation, and repeatable benchmark outputs.',
    details: ['CSV experiment logs', 'Phase 12-17 research scripts', 'Final report artifact generation'],
    usedIn: 'research/experiment_tracker.py, research/phase*',
  },
];

const SUMMARY = [
  { label: 'Research Milestones', value: '11' },
  { label: 'Technology Cards', value: TECHNOLOGIES.length.toString().padStart(2, '0') },
  { label: 'Frontend Runtime', value: 'React' },
  { label: 'ML Runtime', value: 'PyTorch' },
];

export default function Technology() {
  return (
    <main className="technology-page">
      <section className="technology-hero">
        <div className="tech-hero-copy">
          <span className="eyebrow">CloudVerse Architecture</span>
          <h1 className="kage-display">Technology Stack</h1>
          <p>
            Every framework, model, data tool, simulation engine, and UI library used by the
            predictive workload orchestration project is framed below as a dedicated card.
          </p>
        </div>

        <div className="tech-summary-grid" aria-label="Technology summary">
          {SUMMARY.map((item) => (
            <div className="summary-card" key={item.label}>
              <strong>{item.value}</strong>
              <span>{item.label}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="technology-card-grid" aria-label="Project technologies">
        {TECHNOLOGIES.map((tech, index) => {
          const Icon = tech.icon;

          return (
            <article className="technology-card" key={tech.name} style={{ '--accent': tech.accent } as CSSProperties}>
              <div className="card-index">{String(index + 1).padStart(2, '0')}</div>
              <div className="card-topline">
                <div className="tech-icon" aria-hidden="true">
                  <Icon size={22} strokeWidth={1.8} />
                </div>
                <span>{tech.category}</span>
              </div>

              <h2>{tech.name}</h2>
              <p className="tech-purpose">{tech.purpose}</p>

              <ul className="tech-detail-list">
                {tech.details.map((detail) => (
                  <li key={detail}>{detail}</li>
                ))}
              </ul>

              <div className="used-in">
                <span>Used in</span>
                <strong>{tech.usedIn}</strong>
              </div>
            </article>
          );
        })}
      </section>
    </main>
  );
}
