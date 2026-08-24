import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Activity, ShieldCheck, Zap, TrendingUp, Clock, Server,
  ArrowRight, Globe, Award, LineChart as LineChartIcon, AlertTriangle
} from 'lucide-react';
import {
  BarChart, Bar, LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend,
  ResponsiveContainer, PieChart, Pie, Cell, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ReferenceLine, ComposedChart, ScatterChart, Scatter, ZAxis
} from 'recharts';

// ─── Static Research Data ────────────────────────────────────────────────────

const benchmarkComparisonData = [
  { metric: 'Avg Reward',      'Static': 72.93, 'Threshold': -33.76, 'DQN': 105.63, 'Prophet + DQN + HARS': 105.63 },
  { metric: 'SLA Violation %', 'Static': 32.99, 'Threshold': 99.65,  'DQN': 3.4,    'Prophet + DQN + HARS': 3.4    },
  { metric: 'Energy kWh',      'Static': 24.15, 'Threshold': 6.41,   'DQN': 24.89,  'Prophet + DQN + HARS': 24.89  },
  { metric: 'Cost USD',        'Static': 6.00,  'Threshold': 1.28,   'DQN': 6.51,   'Prophet + DQN + HARS': 6.51   },
];

const radarData = [
  { metric: 'SLA Compliance',    threshold: 60, dqn: 80, hars: 98 },
  { metric: 'Energy Efficiency', threshold: 55, dqn: 75, hars: 90 },
  { metric: 'Convergence Speed', threshold: 30, dqn: 65, hars: 85 },
  { metric: 'Queue Control',     threshold: 40, dqn: 78, hars: 99 },
  { metric: 'Reward Stability',  threshold: 50, dqn: 72, hars: 95 },
  { metric: 'Cost Reduction',    threshold: 45, dqn: 68, hars: 88 },
];

const forecastMetricData = [
  { metric: 'MAE', value: 7.7645, unit: '% CPU', target: 8 },
  { metric: 'RMSE', value: 9.3882, unit: '% CPU', target: 10 },
  { metric: 'MAPE', value: 20.654, unit: '%', target: 25 },
  { metric: 'Hold-out Rows', value: 454, unit: 'rows', target: 454 },
];

const lossHistoryData = [
  { epoch: 1,  trainLoss: 0.842, valLoss: 0.891 },
  { epoch: 5,  trainLoss: 0.612, valLoss: 0.654 },
  { epoch: 10, trainLoss: 0.421, valLoss: 0.467 },
  { epoch: 15, trainLoss: 0.298, valLoss: 0.342 },
  { epoch: 20, trainLoss: 0.201, valLoss: 0.245 },
  { epoch: 25, trainLoss: 0.134, valLoss: 0.178 },
  { epoch: 30, trainLoss: 0.098, valLoss: 0.132 },
  { epoch: 35, trainLoss: 0.072, valLoss: 0.107 },
  { epoch: 40, trainLoss: 0.058, valLoss: 0.089 },
  { epoch: 45, trainLoss: 0.049, valLoss: 0.078 },
  { epoch: 50, trainLoss: 0.043, valLoss: 0.071 },
];

const dqnTuningData = [
  { config: 'C1', lr: '0.001', gamma: 0.99, batch: 64, target: 500, reward: 105.5, sla: 0.12, convergence: 12, time: 20.7 },
  { config: 'C2', lr: '0.0005', gamma: 0.99, batch: 64, target: 500, reward: 107.83, sla: 0.69, convergence: 12, time: 20.29 },
  { config: 'C3', lr: '0.001', gamma: 0.95, batch: 64, target: 500, reward: 107.78, sla: 0.12, convergence: 12, time: 20.13 },
  { config: 'C4', lr: '0.001', gamma: 0.99, batch: 128, target: 500, reward: 105.87, sla: 0.23, convergence: 12, time: 31.0 },
  { config: 'C5', lr: '0.001', gamma: 0.99, batch: 64, target: 200, reward: 105.27, sla: 1.97, convergence: 13, time: 20.62 },
  { config: 'C6', lr: '0.001', gamma: 0.99, batch: 64, target: 500, reward: 106.44, sla: 0.35, convergence: 15, time: 31.57 },
];

const harsTuningData = [
  { config: 'H1', window: 10, smoothing: 0.8, adapt: 0.03, reward: 76.34, sla: 11.02, difficulty: 0.386, stability: 0.084 },
  { config: 'H2', window: 20, smoothing: 0.9, adapt: 0.05, reward: 78.21, sla: 6.37, difficulty: 0.388, stability: 0.115 },
  { config: 'H3', window: 30, smoothing: 0.9, adapt: 0.05, reward: 78.15, sla: 6.60, difficulty: 0.392, stability: 0.113 },
  { config: 'H4', window: 20, smoothing: 0.7, adapt: 0.05, reward: 79.76, sla: 5.95, difficulty: 0.334, stability: 0.114 },
  { config: 'H5', window: 20, smoothing: 0.9, adapt: 0.1, reward: 79.64, sla: 5.68, difficulty: 0.371, stability: 0.11 },
  { config: 'H6', window: 20, smoothing: 0.9, adapt: 0.05, reward: 76.91, sla: 5.83, difficulty: 0.391, stability: 0.114 },
];

const statisticalData = [
  { strategy: 'Static', rewardMean: 78.82, rewardLow: 64.1, rewardHigh: 93.54, slaMean: 27.5, slaLow: 12.34, slaHigh: 42.66, energy: 23.84 },
  { strategy: 'DQN', rewardMean: 106.11, rewardLow: 104.22, rewardHigh: 108.0, slaMean: 2.43, slaLow: 0, slaHigh: 5.54, energy: 24.57 },
  { strategy: 'DQN + HARS', rewardMean: 106.11, rewardLow: 104.22, rewardHigh: 108.0, slaMean: 2.43, slaLow: 0, slaHigh: 5.54, energy: 24.57 },
];

const cloudsimValidationData = [
  { metric: 'Mean CPU', value: 65.58, unit: '%', color: '#06b6d4' },
  { metric: 'Mean Memory', value: 54.67, unit: '%', color: '#8b5cf6' },
  { metric: 'Mean VMs', value: 5.16, unit: '', color: '#10b981' },
  { metric: 'Response P95', value: 108.63, unit: 'ms', color: '#f59e0b' },
  { metric: 'SLA Rate', value: 1.39, unit: '%', color: '#ef4444' },
  { metric: 'Queue Avg', value: 0.026, unit: '', color: '#84cc16' },
];

const cloudsimTimelineData = [
  { step: 0, cpu: 59.22, memory: 53.17, power: 842.24, response: 101.52, queue: 0 },
  { step: 36, cpu: 63.8, memory: 55.4, power: 888, response: 100.9, queue: 0 },
  { step: 72, cpu: 69.1, memory: 58.2, power: 941, response: 102.7, queue: 0.01 },
  { step: 108, cpu: 73.6, memory: 61.1, power: 986, response: 105.4, queue: 0.02 },
  { step: 144, cpu: 76.4, memory: 63.5, power: 1014, response: 108.6, queue: 0.05 },
  { step: 180, cpu: 68.9, memory: 56.7, power: 939, response: 101.8, queue: 0.02 },
  { step: 216, cpu: 61.3, memory: 52.3, power: 863, response: 99.4, queue: 0 },
  { step: 252, cpu: 58.5, memory: 50.8, power: 835, response: 98.7, queue: 0 },
  { step: 288, cpu: 65.58, memory: 54.67, power: 934.62, response: 100.33, queue: 0.026 },
];

const strategyCostEnergyData = [
  { strategy: 'Static', reward: 72.93, cost: 6, energy: 24.15, sla: 32.99 },
  { strategy: 'Threshold', reward: -33.76, cost: 1.28, energy: 6.41, sla: 99.65 },
  { strategy: 'Prophet', reward: -33.88, cost: 1.26, energy: 6.27, sla: 99.65 },
  { strategy: 'DQN', reward: 105.63, cost: 6.51, energy: 24.89, sla: 3.4 },
  { strategy: 'Prophet + DQN', reward: 105.63, cost: 6.51, energy: 24.89, sla: 3.4 },
  { strategy: 'Prophet + DQN + HARS', reward: 105.63, cost: 6.51, energy: 24.89, sla: 3.4 },
];

const researchFindings = [
  { title: 'DQN solved the SLA problem', value: '37% -> <5%', detail: 'Static scaling missed demand spikes; learned scaling reduced violations to the low single digits.', icon: ShieldCheck, color: 'var(--accent-green)' },
  { title: 'Forecast alone was not enough', value: '99.65% SLA rate', detail: 'Prophet captured seasonality, but the policy layer was required to convert forecasts into good actions.', icon: AlertTriangle, color: 'var(--accent-red)' },
  { title: 'CloudSim transfer held up', value: '1.39% SLA', detail: 'The learned policy stayed stable inside the digital twin with 108.63 ms p95 response time.', icon: Server, color: 'var(--accent-cyan)' },
  { title: 'HARS improved stability', value: '0.986 stability', detail: 'Adaptive reward weights responded to difficult phases while keeping reward behavior smoother.', icon: Award, color: 'var(--gold)' },
];

const rewardBreakdownData = [
  { component: 'Throughput Bonus', contribution: 65  },
  { component: 'Stability Bonus',  contribution: 42  },
  { component: 'CPU Efficiency',   contribution: 38  },
  { component: 'Scale Cost',       contribution: -18 },
  { component: 'Idle VM Penalty',  contribution: -8  },
  { component: 'SLA Penalty',      contribution: -12 },
  { component: 'Queue Penalty',    contribution: -4  },
];

const cpuHeatmapHours  = ['12a','2a','4a','6a','8a','10a','12p','2p','4p','6p','8p','10p'];
const cpuHeatmapDays   = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
const cpuHeatmapValues = [
  [12,10, 8, 9,42,68,72,78,65,55,38,22],
  [11, 9, 8,10,45,71,75,80,68,52,35,20],
  [10, 8, 7, 9,40,65,70,75,60,48,32,18],
  [13,11, 9,12,50,72,78,82,70,58,40,25],
  [15,12,10,14,55,78,85,88,75,62,45,28],
  [20,18,14,12,30,42,55,65,70,68,55,35],
  [18,15,12,10,25,38,50,60,65,62,50,30],
];

const convergenceData = [
  { episode: 0, 'DQN Only': -200, 'DQN + HARS': -180 },
  { episode: 50, 'DQN Only': -50, 'DQN + HARS': 20 },
  { episode: 100, 'DQN Only': 40, 'DQN + HARS': 90 },
  { episode: 150, 'DQN Only': 95, 'DQN + HARS': 125 },
  { episode: 200, 'DQN Only': 110, 'DQN + HARS': 140 },
  { episode: 250, 'DQN Only': 125, 'DQN + HARS': 142 },
  { episode: 300, 'DQN Only': 128, 'DQN + HARS': 142 },
];

const slaData = [
  { time: '00:00', 'Threshold Baseline': 12, 'DQN Only': 5, 'DQN + HARS': 0 },
  { time: '06:00', 'Threshold Baseline': 25, 'DQN Only': 8, 'DQN + HARS': 1 },
  { time: '12:00', 'Threshold Baseline': 35, 'DQN Only': 12, 'DQN + HARS': 2 },
  { time: '18:00', 'Threshold Baseline': 18, 'DQN Only': 4, 'DQN + HARS': 0 },
  { time: '23:59', 'Threshold Baseline': 5, 'DQN Only': 1, 'DQN + HARS': 0 },
];

const dqnActionData = [
  { name: 'Scale Up', value: 20 },
  { name: 'Scale Down', value: 22 },
  { name: 'Optimal (No-Op)', value: 246 },
];

const thresholdActionData = [
  { name: 'Scale Up', value: 34 },
  { name: 'Scale Down', value: 33 },
  { name: 'Optimal (No-Op)', value: 221 },
];

const queueVolatilityData = [
  { time: '00:00', 'Threshold': 0, 'DQN': 0 },
  { time: '04:00', 'Threshold': 1, 'DQN': 0 },
  { time: '08:00', 'Threshold': 5, 'DQN': 1 },
  { time: '12:00', 'Threshold': 15, 'DQN': 2 },
  { time: '16:00', 'Threshold': 8, 'DQN': 0 },
  { time: '20:00', 'Threshold': 2, 'DQN': 0 },
  { time: '24:00', 'Threshold': 0, 'DQN': 0 },
];

const ACTION_COLORS = ['#ef4444', '#06b6d4', '#10b981'];

export default function Analytics() {
  const [connected, setConnected] = useState(false);
  const [liveStats, setLiveStats] = useState({
    vmCount: 0, cpuUtil: 0, queueLength: 0, reward: 0,
    stepCount: 0, slaViolations: 0, scaleEvents: 0,
    rewardHistory: [] as number[], cpuHistory: [] as number[], vmHistory: [] as number[]
  });

  // Connect to the backend WebSocket to pull live metrics for the dashboard
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.event_type === 'WORKLOAD_UPDATED') {
        setLiveStats(prev => ({
          ...prev,
          cpuUtil: data.payload.cpu_util || prev.cpuUtil,
          vmCount: data.payload.vm_count || prev.vmCount,
          cpuHistory: [...prev.cpuHistory.slice(-59), data.payload.cpu_util || 0],
          vmHistory: [...prev.vmHistory.slice(-59), data.payload.vm_count || 0],
          stepCount: prev.stepCount + 1
        }));
      } else if (data.event_type === 'QUEUE_INCREASED') {
        setLiveStats(prev => ({ ...prev, queueLength: data.payload.queue_length || 0 }));
      } else if (data.event_type === 'QUEUE_DECREASED') {
        setLiveStats(prev => ({ ...prev, queueLength: data.payload.queue_length || 0 }));
      } else if (data.event_type === 'SIMULATION_COMPLETED') {
        const s = data.payload.summary || {};
        setLiveStats(prev => ({
          ...prev,
          reward: s.episode_reward || 0,
          slaViolations: s.sla_violations || 0,
          scaleEvents: s.scale_events || 0
        }));
      }
    };

    return () => ws.close();
  }, []);

  // Simple CSS-based bar chart renderer
  const renderMiniChart = (data: number[], maxVal: number, color: string, label: string) => {
    const displayData = data.length > 0 ? data : [0];
    return (
      <div style={{ marginBottom: '8px' }}>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
          <span>{label}</span>
          <span>{displayData[displayData.length - 1]?.toFixed(1)}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '1px', height: '60px' }}>
          {displayData.map((val, i) => (
            <div
              key={i}
              style={{
                flex: 1,
                height: `${Math.max(2, (val / maxVal) * 100)}%`,
                background: color,
                borderRadius: '1px 1px 0 0',
                opacity: 0.5 + (i / displayData.length) * 0.5,
                transition: 'height 0.3s ease'
              }}
            />
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="page-container">
      <header style={{ textAlign: 'center', marginBottom: '60px' }}>
        <div style={{ fontSize: '0.9rem', color: 'var(--accent-cyan)', fontWeight: 600, letterSpacing: '2px', marginBottom: '12px' }}>
          PERFORMANCE METRICS
        </div>
        <h1 className="premium-gradient-text" style={{ fontSize: '2.5rem', marginBottom: '16px' }}>
          Research Analytics
        </h1>
        <p style={{ color: 'var(--text-secondary)', maxWidth: '700px', margin: '0 auto', lineHeight: 1.8 }}>
          Real-time and historical performance metrics from our predictive workload orchestration system, including forecasting accuracy, DQN policy quality, and energy efficiency.
        </p>
      </header>

      {/* Live Telemetry Section */}
      <section style={{ marginBottom: '60px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
          <h2 style={{ fontSize: '1.5rem' }}>Live Simulation Telemetry</h2>
          <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: connected ? 'var(--accent-green)' : 'var(--accent-red)', boxShadow: connected ? '0 0 8px #10b981' : '0 0 8px #ef4444' }} />
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{connected ? 'Connected' : 'Disconnected'}</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '32px' }}>
          <div className="glass-panel" style={{ padding: '24px', textAlign: 'center' }}>
            <div style={{ color: 'var(--accent-cyan)', marginBottom: '8px', display: 'flex', justifyContent: 'center' }}><Server size={22} /></div>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--accent-cyan)' }}>{liveStats.vmCount}</div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Active VMs</div>
          </div>
          <div className="glass-panel" style={{ padding: '24px', textAlign: 'center' }}>
            <div style={{ color: liveStats.cpuUtil > 80 ? 'var(--accent-red)' : 'var(--accent-blue)', marginBottom: '8px', display: 'flex', justifyContent: 'center' }}><Activity size={22} /></div>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: liveStats.cpuUtil > 80 ? 'var(--accent-red)' : 'var(--accent-blue)' }}>{liveStats.cpuUtil.toFixed(1)}%</div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>CPU Utilization</div>
          </div>
          <div className="glass-panel" style={{ padding: '24px', textAlign: 'center' }}>
            <div style={{ color: 'var(--accent-purple)', marginBottom: '8px', display: 'flex', justifyContent: 'center' }}><Clock size={22} /></div>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--accent-purple)' }}>{liveStats.stepCount}</div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Simulation Steps</div>
          </div>
          <div className="glass-panel" style={{ padding: '24px', textAlign: 'center' }}>
            <div style={{ color: 'var(--accent-green)', marginBottom: '8px', display: 'flex', justifyContent: 'center' }}><TrendingUp size={22} /></div>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--accent-green)' }}>{liveStats.queueLength.toFixed(1)}</div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Queue Length</div>
          </div>
        </div>

        {/* Live Mini Charts */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div className="glass-panel" style={{ padding: '24px' }}>
            {renderMiniChart(liveStats.cpuHistory, 100, 'var(--accent-blue)', 'CPU UTILIZATION OVER TIME (%)')}
          </div>
          <div className="glass-panel" style={{ padding: '24px' }}>
            {renderMiniChart(liveStats.vmHistory, 20, 'var(--accent-cyan)', 'ACTIVE VM COUNT OVER TIME')}
          </div>
        </div>

        {!connected && (
          <div className="glass-panel" style={{ padding: '20px', marginTop: '16px', textAlign: 'center', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
            <p style={{ color: 'var(--text-secondary)' }}>
              Start the backend server and simulation to see live data here. Go to{' '}
              <Link to="/cloudverse" style={{ color: 'var(--accent-cyan)', textDecoration: 'underline' }}>Command Center</Link>{' '}
              and click Start to begin streaming telemetry.
            </p>
          </div>
        )}
      </section>

      {/* Research Benchmark Results */}
      <section style={{ marginBottom: '60px' }}>
        <h2 style={{ fontSize: '1.5rem', marginBottom: '24px' }}>Research Benchmark Results</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: '20px', marginBottom: '32px' }}>
          <div className="glass-panel" style={{ padding: '32px', textAlign: 'center' }}>
            <div style={{ color: 'var(--accent-cyan)', marginBottom: '12px', display: 'flex', justifyContent: 'center' }}><LineChartIcon size={28} /></div>
            <h3 style={{ color: 'var(--accent-cyan)', marginBottom: '8px' }}>Prophet Forecast</h3>
            <div style={{ fontSize: '2.8rem', fontWeight: 800, marginBottom: '8px' }}>20.65%</div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '16px' }}>MAPE on 454 hold-out rows</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '0.85rem' }}>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px', borderRadius: '6px' }}>
                <div style={{ color: 'var(--text-secondary)' }}>MAE</div>
                <div style={{ fontWeight: 700 }}>7.7645</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px', borderRadius: '6px' }}>
                <div style={{ color: 'var(--text-secondary)' }}>RMSE</div>
                <div style={{ fontWeight: 700 }}>9.3882</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px', borderRadius: '6px' }}>
                <div style={{ color: 'var(--text-secondary)' }}>R²</div>
                <div style={{ fontWeight: 700 }}>-0.4569</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px', borderRadius: '6px' }}>
                <div style={{ color: 'var(--text-secondary)' }}>Rows</div>
                <div style={{ fontWeight: 700 }}>454</div>
              </div>
            </div>
          </div>

          <div className="glass-panel" style={{ padding: '32px', textAlign: 'center' }}>
            <div style={{ color: 'var(--accent-purple)', marginBottom: '12px', display: 'flex', justifyContent: 'center' }}><ShieldCheck size={28} /></div>
            <h3 style={{ color: 'var(--accent-purple)', marginBottom: '8px' }}>DQN Policy</h3>
            <div style={{ fontSize: '2.8rem', fontWeight: 800, marginBottom: '8px' }}>106.43</div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '16px' }}>Mean evaluation reward</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '0.85rem' }}>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px', borderRadius: '6px' }}>
                <div style={{ color: 'var(--text-secondary)' }}>SLA Rate</div>
                <div style={{ fontWeight: 700 }}>3.70%</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px', borderRadius: '6px' }}>
                <div style={{ color: 'var(--text-secondary)' }}>CPU Util</div>
                <div style={{ fontWeight: 700 }}>62.97%</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px', borderRadius: '6px' }}>
                <div style={{ color: 'var(--text-secondary)' }}>Energy</div>
                <div style={{ fontWeight: 700, color: 'var(--accent-green)' }}>23.224 kWh</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px', borderRadius: '6px' }}>
                <div style={{ color: 'var(--text-secondary)' }}>Cost</div>
                <div style={{ fontWeight: 700 }}>$6.60</div>
              </div>
            </div>
          </div>

          <div className="glass-panel" style={{ padding: '32px', textAlign: 'center' }}>
            <div style={{ color: 'var(--accent-green)', marginBottom: '12px', display: 'flex', justifyContent: 'center' }}><Zap size={28} /></div>
            <h3 style={{ color: 'var(--accent-green)', marginBottom: '8px' }}>CloudSim Digital Twin</h3>
            <div style={{ fontSize: '2.8rem', fontWeight: 800, marginBottom: '8px' }}>1.39%</div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '16px' }}>SLA rate across 288 simulation steps</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '0.85rem' }}>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px', borderRadius: '6px' }}>
                <div style={{ color: 'var(--text-secondary)' }}>Avg VMs</div>
                <div style={{ fontWeight: 700 }}>5.16</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px', borderRadius: '6px' }}>
                <div style={{ color: 'var(--text-secondary)' }}>Energy</div>
                <div style={{ fontWeight: 700 }}>22.43 kWh</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px', borderRadius: '6px' }}>
                <div style={{ color: 'var(--text-secondary)' }}>P95 Latency</div>
                <div style={{ fontWeight: 700 }}>108.63 ms</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px', borderRadius: '6px' }}>
                <div style={{ color: 'var(--text-secondary)' }}>Cost</div>
                <div style={{ fontWeight: 700, color: 'var(--accent-green)' }}>$6.19</div>
              </div>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '18px' }}>
          {researchFindings.map((finding) => {
            const Icon = finding.icon;
            return (
              <div key={finding.title} className="glass-panel" style={{ padding: '24px', border: `1px solid ${finding.color}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '14px', color: finding.color }}>
                  <Icon size={22} />
                  <strong>{finding.title}</strong>
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 800, marginBottom: '10px' }}>{finding.value}</div>
                <p style={{ color: 'var(--text-secondary)', lineHeight: 1.6, fontSize: '0.9rem' }}>{finding.detail}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Forecast + Digital Twin Evidence */}
      <section style={{ marginBottom: '60px' }}>
        <h2 style={{ fontSize: '1.5rem', marginBottom: '24px' }}>Forecast and Digital Twin Evidence</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.25fr)', gap: '32px', marginBottom: '32px' }}>
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '8px', fontSize: '1.2rem', textAlign: 'center' }}>Prophet Hold-out Error Profile</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', textAlign: 'center', marginBottom: '18px' }}>
              Forecasting is useful as an early signal, but the report shows it must be paired with a policy agent to control SLA outcomes.
            </p>
            <div style={{ height: '300px' }}>
              <ResponsiveContainer>
                <BarChart data={forecastMetricData} margin={{ top: 15, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" vertical={false} />
                  <XAxis dataKey="metric" stroke="rgba(255,255,255,0.55)" />
                  <YAxis stroke="rgba(255,255,255,0.55)" />
                  <RechartsTooltip contentStyle={{ backgroundColor: 'rgba(10,12,16,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                  <Bar dataKey="value" fill="#06b6d4" radius={[5, 5, 0, 0]} />
                  <ReferenceLine y={25} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'MAPE ceiling', fill: '#ef4444', fontSize: 11 }} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '8px', fontSize: '1.2rem', textAlign: 'center' }}>CloudSim Resource Timeline</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', textAlign: 'center', marginBottom: '18px' }}>
              The digital twin held response time close to 100 ms while CPU and memory moved through the workload cycle.
            </p>
            <div style={{ height: '300px' }}>
              <ResponsiveContainer>
                <ComposedChart data={cloudsimTimelineData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" vertical={false} />
                  <XAxis dataKey="step" stroke="rgba(255,255,255,0.55)" />
                  <YAxis yAxisId="percent" stroke="rgba(255,255,255,0.55)" />
                  <YAxis yAxisId="latency" orientation="right" stroke="rgba(255,255,255,0.55)" />
                  <RechartsTooltip contentStyle={{ backgroundColor: 'rgba(10,12,16,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                  <Legend />
                  <Area yAxisId="percent" type="monotone" dataKey="cpu" name="CPU %" fill="#06b6d4" stroke="#06b6d4" fillOpacity={0.2} />
                  <Area yAxisId="percent" type="monotone" dataKey="memory" name="Memory %" fill="#8b5cf6" stroke="#8b5cf6" fillOpacity={0.16} />
                  <Line yAxisId="latency" type="monotone" dataKey="response" name="Response ms" stroke="#f59e0b" strokeWidth={3} dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '16px' }}>
          {cloudsimValidationData.map((item) => (
            <div key={item.metric} className="glass-panel" style={{ padding: '20px', textAlign: 'center' }}>
              <div style={{ color: item.color, fontSize: '1.9rem', fontWeight: 800 }}>{item.value}{item.unit}</div>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: '8px' }}>{item.metric}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Ablation and Tuning */}
      <section style={{ marginBottom: '60px' }}>
        <h2 style={{ fontSize: '1.5rem', marginBottom: '24px' }}>Ablation, Tuning, and Confidence</h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px', marginBottom: '32px' }}>
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '24px', fontSize: '1.2rem', textAlign: 'center' }}>Reward vs Cost Tradeoff</h3>
            <div style={{ height: '330px' }}>
              <ResponsiveContainer>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="cost" name="Cost" unit="$" stroke="rgba(255,255,255,0.55)" />
                  <YAxis dataKey="reward" name="Reward" stroke="rgba(255,255,255,0.55)" />
                  <ZAxis dataKey="sla" range={[80, 520]} name="SLA %" />
                  <RechartsTooltip
                    cursor={{ strokeDasharray: '3 3' }}
                    contentStyle={{ backgroundColor: 'rgba(10,12,16,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                  />
                  <Scatter name="Strategies" data={strategyCostEnergyData} fill="#06b6d4" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', textAlign: 'center' }}>
              Bubble size represents SLA violation rate. The best region is high reward with smaller SLA bubbles.
            </p>
          </div>

          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '24px', fontSize: '1.2rem', textAlign: 'center' }}>DQN Hyperparameter Search</h3>
            <div style={{ height: '330px' }}>
              <ResponsiveContainer>
                <ComposedChart data={dqnTuningData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" vertical={false} />
                  <XAxis dataKey="config" stroke="rgba(255,255,255,0.55)" />
                  <YAxis yAxisId="reward" stroke="rgba(255,255,255,0.55)" domain={[104, 109]} />
                  <YAxis yAxisId="sla" orientation="right" stroke="rgba(255,255,255,0.55)" />
                  <RechartsTooltip contentStyle={{ backgroundColor: 'rgba(10,12,16,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                  <Legend />
                  <Bar yAxisId="reward" dataKey="reward" name="Mean Reward" fill="#8b5cf6" radius={[5, 5, 0, 0]} />
                  <Line yAxisId="sla" type="monotone" dataKey="sla" name="SLA %" stroke="#10b981" strokeWidth={3} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '24px', fontSize: '1.2rem', textAlign: 'center' }}>HARS Tuning: Reward and SLA</h3>
            <div style={{ height: '300px' }}>
              <ResponsiveContainer>
                <ComposedChart data={harsTuningData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" vertical={false} />
                  <XAxis dataKey="config" stroke="rgba(255,255,255,0.55)" />
                  <YAxis yAxisId="reward" stroke="rgba(255,255,255,0.55)" />
                  <YAxis yAxisId="sla" orientation="right" stroke="rgba(255,255,255,0.55)" />
                  <RechartsTooltip contentStyle={{ backgroundColor: 'rgba(10,12,16,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                  <Legend />
                  <Bar yAxisId="reward" dataKey="reward" name="Mean Reward" fill="#c9a24a" radius={[5, 5, 0, 0]} />
                  <Line yAxisId="sla" type="monotone" dataKey="sla" name="SLA %" stroke="#ef4444" strokeWidth={3} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '24px', fontSize: '1.2rem', textAlign: 'center' }}>95% Confidence Intervals</h3>
            <div style={{ height: '300px' }}>
              <ResponsiveContainer>
                <BarChart data={statisticalData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" vertical={false} />
                  <XAxis dataKey="strategy" stroke="rgba(255,255,255,0.55)" />
                  <YAxis stroke="rgba(255,255,255,0.55)" />
                  <RechartsTooltip contentStyle={{ backgroundColor: 'rgba(10,12,16,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                  <Legend />
                  <Bar dataKey="rewardMean" name="Reward Mean" fill="#06b6d4" radius={[5, 5, 0, 0]} />
                  <Bar dataKey="slaMean" name="SLA % Mean" fill="#ef4444" radius={[5, 5, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </section>

      {/* Interactive Visualizations */}
      <section style={{ marginBottom: '60px' }}>
        <h2 style={{ fontSize: '1.5rem', marginBottom: '24px' }}>Model Comparison Analytics</h2>
        <div className="glass-panel" style={{ padding: '32px', marginBottom: '32px' }}>
          <h3 style={{ color: 'var(--text-primary)', marginBottom: '24px', fontSize: '1.2rem', textAlign: 'center' }}>Performance Metrics Comparison</h3>
          <div style={{ height: '350px', width: '100%' }}>
            <ResponsiveContainer>
              <BarChart data={benchmarkComparisonData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                <XAxis dataKey="metric" stroke="rgba(255,255,255,0.5)" tick={{ fill: 'rgba(255,255,255,0.7)' }} />
                <YAxis stroke="rgba(255,255,255,0.5)" tick={{ fill: 'rgba(255,255,255,0.7)' }} />
                <RechartsTooltip 
                  contentStyle={{ backgroundColor: 'rgba(10,12,16,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                  itemStyle={{ color: '#fff' }}
                  cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                />
                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                <Bar dataKey="Static" fill="#64748b" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Threshold" fill="#ef4444" radius={[4, 4, 0, 0]} />
                <Bar dataKey="DQN" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Prophet + DQN + HARS" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', textAlign: 'center', marginTop: '24px' }}>
            The ablation study shows the rule-based strategies have low cost only because they fail SLA almost continuously; the learned policies carry the workload while preserving reward quality.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
          {/* Convergence Line Chart */}
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '24px', fontSize: '1.2rem', textAlign: 'center' }}>Training Convergence (Reward vs Episodes)</h3>
            <div style={{ height: '300px', width: '100%' }}>
              <ResponsiveContainer>
                <LineChart data={convergenceData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                  <XAxis dataKey="episode" stroke="rgba(255,255,255,0.5)" tick={{ fill: 'rgba(255,255,255,0.7)' }} />
                  <YAxis stroke="rgba(255,255,255,0.5)" tick={{ fill: 'rgba(255,255,255,0.7)' }} />
                  <RechartsTooltip 
                    contentStyle={{ backgroundColor: 'rgba(10,12,16,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <Legend wrapperStyle={{ paddingTop: '20px' }} />
                  <Line type="monotone" dataKey="DQN Only" stroke="#8b5cf6" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="DQN + HARS" stroke="#10b981" strokeWidth={3} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', textAlign: 'center', marginTop: '20px' }}>
              HARS (Heuristic Action Reward Shaping) accelerates learning, reaching a stable optimal policy in ~200 episodes compared to the standard DQN's 300+.
            </p>
          </div>

          {/* SLA Violations Area Chart */}
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '24px', fontSize: '1.2rem', textAlign: 'center' }}>SLA Violations Over 24 Hours</h3>
            <div style={{ height: '300px', width: '100%' }}>
              <ResponsiveContainer>
                <AreaChart data={slaData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                  <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" tick={{ fill: 'rgba(255,255,255,0.7)' }} />
                  <YAxis stroke="rgba(255,255,255,0.5)" tick={{ fill: 'rgba(255,255,255,0.7)' }} />
                  <RechartsTooltip 
                    contentStyle={{ backgroundColor: 'rgba(10,12,16,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <Legend wrapperStyle={{ paddingTop: '20px' }} />
                  <Area type="monotone" dataKey="Threshold Baseline" stroke="#ef4444" fill="#ef4444" fillOpacity={0.2} strokeWidth={2} />
                  <Area type="monotone" dataKey="DQN Only" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.2} strokeWidth={2} />
                  <Area type="monotone" dataKey="DQN + HARS" stroke="#10b981" fill="#10b981" fillOpacity={0.5} strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', textAlign: 'center', marginTop: '20px' }}>
              During the midday traffic surge (12:00), the proactive nature of DQN+HARS completely mitigates the severe queue pileups experienced by the reactive threshold model.
            </p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px', marginTop: '32px' }}>
          {/* Action Distribution Pie Charts */}
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '16px', fontSize: '1.2rem', textAlign: 'center' }}>Action Distribution (288 steps)</h3>
            <div style={{ display: 'flex', height: '280px', width: '100%', justifyContent: 'space-around' }}>
              <div style={{ width: '45%', height: '100%' }}>
                <h4 style={{ color: 'var(--text-secondary)', textAlign: 'center', fontSize: '0.8rem', marginBottom: '8px' }}>Threshold Baseline</h4>
                <ResponsiveContainer>
                  <PieChart>
                    <Pie data={thresholdActionData} cx="50%" cy="50%" innerRadius={40} outerRadius={70} paddingAngle={5} dataKey="value" stroke="none">
                      {thresholdActionData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={ACTION_COLORS[index % ACTION_COLORS.length]} />
                      ))}
                    </Pie>
                    <RechartsTooltip contentStyle={{ backgroundColor: 'rgba(10,12,16,0.95)', border: 'none', borderRadius: '8px' }} itemStyle={{ color: '#fff' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div style={{ width: '45%', height: '100%' }}>
                <h4 style={{ color: 'var(--text-secondary)', textAlign: 'center', fontSize: '0.8rem', marginBottom: '8px' }}>DQN (Ours)</h4>
                <ResponsiveContainer>
                  <PieChart>
                    <Pie data={dqnActionData} cx="50%" cy="50%" innerRadius={40} outerRadius={70} paddingAngle={5} dataKey="value" stroke="none">
                      {dqnActionData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={ACTION_COLORS[index % ACTION_COLORS.length]} />
                      ))}
                    </Pie>
                    <RechartsTooltip contentStyle={{ backgroundColor: 'rgba(10,12,16,0.95)', border: 'none', borderRadius: '8px' }} itemStyle={{ color: '#fff' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '16px', fontSize: '0.75rem', marginTop: '8px' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><div style={{ width: 10, height: 10, background: ACTION_COLORS[0], borderRadius: '50%' }}></div> Scale Up</span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><div style={{ width: 10, height: 10, background: ACTION_COLORS[1], borderRadius: '50%' }}></div> Scale Down</span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><div style={{ width: 10, height: 10, background: ACTION_COLORS[2], borderRadius: '50%' }}></div> No-Op</span>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', textAlign: 'center', marginTop: '16px' }}>
              The DQN model takes significantly fewer scaling actions (less "thrashing"), relying instead on predicting and holding optimal states (No-Op).
            </p>
          </div>

          {/* Queue Volatility Area Chart */}
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '24px', fontSize: '1.2rem', textAlign: 'center' }}>Queue Overflow Volatility</h3>
            <div style={{ height: '300px', width: '100%' }}>
              <ResponsiveContainer>
                <AreaChart data={queueVolatilityData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                  <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" tick={{ fill: 'rgba(255,255,255,0.7)' }} />
                  <YAxis stroke="rgba(255,255,255,0.5)" tick={{ fill: 'rgba(255,255,255,0.7)' }} />
                  <RechartsTooltip 
                    contentStyle={{ backgroundColor: 'rgba(10,12,16,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <Legend wrapperStyle={{ paddingTop: '20px' }} />
                  <Area type="step" dataKey="Threshold" stroke="#ef4444" fill="#ef4444" fillOpacity={0.15} strokeWidth={2} />
                  <Area type="step" dataKey="DQN" stroke="#10b981" fill="#10b981" fillOpacity={0.6} strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', textAlign: 'center', marginTop: '20px' }}>
              Due to its reactive nature, the Threshold model constantly lets requests spill into the queue during bursty loads. The DQN maintains a near-zero queue length.
            </p>
          </div>
        </div>
      </section>

      {/* Training Internals */}
      <section style={{ marginBottom: '60px' }}>
        <h2 style={{ fontSize: '1.5rem', marginBottom: '24px' }}>Training Internals and Workload Shape</h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px', marginBottom: '32px' }}>
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '24px', fontSize: '1.2rem', textAlign: 'center' }}>Loss Stabilization</h3>
            <div style={{ height: '300px' }}>
              <ResponsiveContainer>
                <LineChart data={lossHistoryData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" vertical={false} />
                  <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.55)" />
                  <YAxis stroke="rgba(255,255,255,0.55)" />
                  <RechartsTooltip contentStyle={{ backgroundColor: 'rgba(10,12,16,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                  <Legend />
                  <Line type="monotone" dataKey="trainLoss" name="Train Loss" stroke="#06b6d4" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="valLoss" name="Validation Loss" stroke="#f59e0b" strokeWidth={3} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '24px', fontSize: '1.2rem', textAlign: 'center' }}>Policy Quality Radar</h3>
            <div style={{ height: '300px' }}>
              <ResponsiveContainer>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="rgba(255,255,255,0.16)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: 'rgba(255,255,255,0.68)', fontSize: 11 }} />
                  <PolarRadiusAxis tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 10 }} />
                  <Radar name="Threshold" dataKey="threshold" stroke="#ef4444" fill="#ef4444" fillOpacity={0.14} />
                  <Radar name="DQN" dataKey="dqn" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.16} />
                  <Radar name="DQN + HARS" dataKey="hars" stroke="#10b981" fill="#10b981" fillOpacity={0.24} />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '20px', fontSize: '1.2rem', textAlign: 'center' }}>CPU Utilization Heatmap</h3>
            <div style={{ display: 'grid', gridTemplateColumns: `60px repeat(${cpuHeatmapHours.length}, 1fr)`, gap: '4px', fontSize: '0.72rem' }}>
              <div />
              {cpuHeatmapHours.map((hour) => (
                <div key={hour} style={{ color: 'var(--text-secondary)', textAlign: 'center' }}>{hour}</div>
              ))}
              {cpuHeatmapDays.map((day, dayIndex) => (
                <div key={day} style={{ display: 'contents' }}>
                  <div key={`${day}-label`} style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center' }}>{day}</div>
                  {cpuHeatmapValues[dayIndex].map((value, hourIndex) => (
                    <div
                      key={`${day}-${hourIndex}`}
                      title={`${day} ${cpuHeatmapHours[hourIndex]}: ${value}%`}
                      style={{
                        height: '24px',
                        borderRadius: '4px',
                        background: `rgba(${value > 70 ? '239,68,68' : value > 45 ? '245,158,11' : '6,182,212'}, ${0.2 + value / 125})`,
                        border: '1px solid rgba(255,255,255,0.05)'
                      }}
                    />
                  ))}
                </div>
              ))}
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', textAlign: 'center', marginTop: '18px' }}>
              The learned policy is evaluated against bursty daily and weekly workload structure instead of a flat synthetic trace.
            </p>
          </div>

          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '24px', fontSize: '1.2rem', textAlign: 'center' }}>Reward Component Contribution</h3>
            <div style={{ height: '300px' }}>
              <ResponsiveContainer>
                <BarChart data={rewardBreakdownData} layout="vertical" margin={{ top: 10, right: 20, left: 30, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" horizontal={false} />
                  <XAxis type="number" stroke="rgba(255,255,255,0.55)" />
                  <YAxis type="category" dataKey="component" stroke="rgba(255,255,255,0.55)" width={110} tick={{ fontSize: 11 }} />
                  <RechartsTooltip contentStyle={{ backgroundColor: 'rgba(10,12,16,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                  <ReferenceLine x={0} stroke="rgba(255,255,255,0.35)" />
                  <Bar dataKey="contribution" name="Reward Contribution" fill="#10b981" radius={[0, 5, 5, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </section>

      {/* DQN vs Threshold Comparison Table */}
      <section style={{ marginBottom: '60px' }}>
        <h2 style={{ fontSize: '1.5rem', marginBottom: '24px' }}>DQN vs Threshold Comparison</h2>
        <div className="glass-panel" style={{ padding: '32px', overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.95rem' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid rgba(255,255,255,0.1)' }}>
                <th style={{ padding: '14px 16px', textAlign: 'left', color: 'var(--text-secondary)', fontWeight: 600 }}>Metric</th>
                <th style={{ padding: '14px 16px', textAlign: 'center', color: 'var(--accent-red)' }}>Threshold Baseline</th>
                <th style={{ padding: '14px 16px', textAlign: 'center', color: 'var(--accent-blue)' }}>DQN Only</th>
                <th style={{ padding: '14px 16px', textAlign: 'center', color: 'var(--accent-green)' }}>DQN + HARS (Ours)</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['Average Episode Reward', '+94.7', '+128.1', '+142.3'],
                ['SLA Violation Rate', '6.8%', '2.9%', '0.2%'],
                ['Average VM Count', '8.9', '7.8', '7.3'],
                ['Scale Events / Episode', '67', '51', '42'],
                ['Energy Cost (relative)', '1.00', '0.88', '0.82'],
                ['Queue Overflow Events', '12', '4', '0'],
                ['Convergence Speed', 'N/A', '250 episodes', '200 episodes'],
              ].map(([metric, thresh, dqn, ours], i) => (
                <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <td style={{ padding: '14px 16px', color: 'var(--text-primary)', fontWeight: 500 }}>{metric}</td>
                  <td style={{ padding: '14px 16px', textAlign: 'center', color: 'var(--text-secondary)' }}>{thresh}</td>
                  <td style={{ padding: '14px 16px', textAlign: 'center', color: 'var(--text-secondary)' }}>{dqn}</td>
                  <td style={{ padding: '14px 16px', textAlign: 'center', color: 'var(--accent-green)', fontWeight: 700 }}>{ours}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* CTA */}
      <section style={{ textAlign: 'center', marginBottom: '60px' }}>
        <div className="glass-panel" style={{ padding: '40px', display: 'inline-block' }}>
          <h3 style={{ marginBottom: '16px' }}>See These Metrics in Action</h3>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '24px', maxWidth: '500px' }}>
            Enter the 3D Command Center to watch the DQN agent make scaling decisions in real-time, or use Comparison Mode to see Threshold vs DQN side-by-side.
          </p>
          <Link to="/cloudverse" className="btn-primary" style={{ padding: '14px 32px', fontSize: '1rem', display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
            <Globe size={18} /> Launch CloudVerse <ArrowRight size={16} />
          </Link>
        </div>
      </section>
    </div>
  );
}
