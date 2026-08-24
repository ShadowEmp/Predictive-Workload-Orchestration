import React, { useEffect, useRef, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { Globe, Activity, Cpu, BarChart3, Zap, Server, ArrowRight, ShieldCheck, TrendingUp, Clock, ChevronDown } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area } from 'recharts';
import { motion, useAnimation, useScroll, useTransform } from 'framer-motion';
import type { Variants } from 'framer-motion';
import CustomGlobe from '../components/ui/globe';
import LandingScene from './LandingScene';

const performanceData = [
  { time: '00:00', load: 30, thresholdVMs: 4, dqnVMs: 3 },
  { time: '04:00', load: 25, thresholdVMs: 3, dqnVMs: 3 },
  { time: '08:00', load: 60, thresholdVMs: 6, dqnVMs: 7 },
  { time: '12:00', load: 95, thresholdVMs: 9, dqnVMs: 11 },
  { time: '16:00', load: 70, thresholdVMs: 8, dqnVMs: 7 },
  { time: '20:00', load: 45, thresholdVMs: 5, dqnVMs: 4 },
  { time: '24:00', load: 35, thresholdVMs: 4, dqnVMs: 3 },
];

const energyData = [
  { name: 'Threshold (Baseline)', value: 100 },
  { name: 'DQN (Ours)', value: 81.6 },
];

const COLORS = ['#ef4444', '#8b5cf6', '#06b6d4', '#10b981'];

export default function Overview() {
  // Framer Motion variants for scroll reveals
  const fadeUp: Variants = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: [0.16, 1, 0.3, 1] } }
  };
  
  const staggerContainer: Variants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.15, delayChildren: 0.2 } }
  };

  // Scroll animations for the Globe background
  const { scrollY } = useScroll();
  
  // At scroll 0: Globe is on the right (x: 20vw), fully visible
  // As user scrolls down, it moves, scales up, and fades into the background
  const globeX = useTransform(scrollY, [0, 800, 1600, 2400], ['22vw', '0vw', '-15vw', '15vw']);
  const globeY = useTransform(scrollY, [0, 800, 1600, 2400], ['0vh', '10vh', '20vh', '5vh']);
  const globeScale = useTransform(scrollY, [0, 800, 1600], [1, 1.5, 1.8]);
  const globeOpacity = useTransform(scrollY, [0, 400], [1, 0.35]);
  
  // To allow clicking the globe in the hero section but not blocking text lower down
  const globePointerEvents = useTransform(scrollY, [0, 100], ['auto', 'none']);

  return (
    <div style={{ position: 'relative', minHeight: '100vh', width: '100vw', overflowX: 'hidden' }}>
      <LandingScene />
      
      {/* Fixed Dynamic Background Globe */}
      <motion.div
        style={{
          position: 'fixed',
          top: '50%',
          left: '50%',
          marginLeft: '-300px', // Half of globe width (600px)
          marginTop: '-300px', // Half of globe height
          x: globeX,
          y: globeY,
          scale: globeScale,
          opacity: globeOpacity,
          pointerEvents: globePointerEvents as any,
          zIndex: 1,
        }}
      >
        <CustomGlobe />
      </motion.div>

      <div className="page" style={{ position: 'relative', zIndex: 10, paddingBottom: '100px' }}>
        
        {/* Kage-inspired Cinematic Hero Section */}
        <section style={{ position: 'relative', minHeight: '100vh', padding: '0 5%', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          {/* Vertical Japanese/Cyberpunk side text */}
          <div style={{ position: 'absolute', right: '2%', top: '50%', transform: 'translateY(-50%)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '14px', pointerEvents: 'none', zIndex: 2 }}>
            <div style={{ writingMode: 'vertical-rl', fontSize: '13px', letterSpacing: '.62em', color: 'rgba(223,231,224,.5)' }}>
              クラウドバース · クラスタ
            </div>
          </div>

          <motion.div 
            initial="hidden" animate="visible" variants={staggerContainer}
            style={{ flex: '1', maxWidth: '700px', position: 'relative', zIndex: 2 }}
          >
            <motion.div variants={fadeUp} className="eyebrow" style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '24px' }}>
              <div style={{ width: '5px', height: '5px', borderRadius: '50%', background: 'var(--vermilion)', boxShadow: '0 0 10px var(--vermilion)' }}></div>
              FINAL YEAR ENGINEERING PROJECT
            </motion.div>
            
            <motion.h1 variants={fadeUp} className="kage-display kage-h-hero" style={{ marginBottom: '24px', letterSpacing: '-0.02em', color: 'var(--bone)' }}>
              PREDICTIVE WORKLOAD<br />ORCHESTRATION
            </motion.h1>
            
            <motion.p variants={fadeUp} className="kage-body-lg" style={{ maxWidth: '480px', marginBottom: '40px' }}>
              Where deep reinforcement learning meets virtualized environments. Proactive scaling through predictive forecasting.
            </motion.p>
            
            <motion.div variants={fadeUp} style={{ display: 'flex', gap: '20px' }}>
              <Link to="/cloudverse" className="btn-primary" style={{ padding: '16px 36px', fontSize: '1.05rem', display: 'flex', alignItems: 'center', gap: '12px', background: 'var(--vermilion)', boxShadow: '0 4px 20px rgba(224,35,28,0.4)', borderRadius: '100px', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                <Globe size={18} /> Enter Command Center
              </Link>
            </motion.div>
          </motion.div>
          
          <div style={{ flex: '1', display: 'flex', justifyContent: 'center', position: 'relative', zIndex: 10 }}>
            {/* Empty space holder. The CustomGlobe is now fixed in the background! */}
          </div>
          
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.5, duration: 1 }} style={{ position: 'absolute', bottom: '40px', left: '5%', zIndex: 2, display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '9px', letterSpacing: '.3em', textTransform: 'uppercase', color: 'var(--muted)' }}>Scroll to Explore</span>
            <div style={{ width: '54px', height: '1px', background: 'var(--line-soft)', position: 'relative', overflow: 'hidden' }}>
              <motion.div 
                animate={{ x: ["-100%", "100%"] }} 
                transition={{ repeat: Infinity, duration: 2.8, ease: "easeInOut" }}
                style={{ position: 'absolute', inset: 0, background: 'var(--bone)' }} 
              />
            </div>
          </motion.div>
        </section>

      {/* Problem vs Solution */}
      {/* Problem vs Solution (Kage Section) */}
      <section className="kage-sec" style={{ padding: '140px 5%' }}>
        <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-10%" }} variants={staggerContainer} style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <motion.div variants={fadeUp} className="eyebrow" style={{ marginBottom: '24px' }}>01 / The Problem</motion.div>
          <motion.h2 variants={fadeUp} className="kage-display kage-h-sec" style={{ marginBottom: '60px' }}>Reimagining Scaling</motion.h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '60px' }}>
            <motion.div variants={fadeUp} className="glass-panel" style={{ padding: '40px', background: 'rgba(5,7,10,0.7)', border: '1px solid var(--line-soft)' }}>
              <h3 style={{ color: 'var(--muted)', marginBottom: '24px', fontSize: '1.3rem', display: 'flex', alignItems: 'center', gap: '12px' }}>
                <Clock size={20} /> Reactive Thresholds
              </h3>
              <p className="kage-body-lg" style={{ marginBottom: '24px' }}>
                Current cloud providers use reactive threshold-based scaling, leading to cold start delays of 2–5 minutes per VM boot and SLA violations during sudden traffic spikes.
              </p>
              <ul style={{ color: 'var(--muted)', lineHeight: 2, paddingLeft: '20px', fontSize: '0.95rem' }}>
                <li>Energy waste from over-provisioned resources</li>
                <li>No learning from historical workload patterns</li>
              </ul>
            </motion.div>
            
            <motion.div variants={fadeUp} className="glass-panel" style={{ padding: '40px', background: 'rgba(224,35,28,0.05)', border: '1px solid rgba(224,35,28,0.2)' }}>
              <h3 style={{ color: 'var(--bone)', marginBottom: '24px', fontSize: '1.3rem', display: 'flex', alignItems: 'center', gap: '12px' }}>
                <TrendingUp size={20} color="var(--vermilion)" /> Predictive DQN
              </h3>
              <p className="kage-body-lg" style={{ color: 'var(--bone)', marginBottom: '24px' }}>
                Our system forecasts future demand using Facebook Prophet, letting a Deep Q-Network agent make proactive scaling decisions <i>before</i> the workload arrives.
              </p>
              <ul style={{ color: 'var(--bone-dim)', lineHeight: 2, paddingLeft: '20px', fontSize: '0.95rem' }}>
                <li>Pre-scales VMs before demand arrives</li>
                <li>HARS dynamically adjusts reward weights</li>
                <li>Reduces energy consumption by 18.4%</li>
              </ul>
            </motion.div>
          </div>
        </motion.div>
      </section>

      {/* Pipeline Flow */}
      <section className="kage-sec" style={{ padding: '140px 5%' }}>
        <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-10%" }} variants={staggerContainer} style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <motion.div variants={fadeUp} className="eyebrow" style={{ marginBottom: '24px' }}>02 / Methodology</motion.div>
          <motion.h2 variants={fadeUp} className="kage-display kage-h-sec" style={{ marginBottom: '60px' }}>Research Pipeline</motion.h2>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
          {[
            { step: 'DATA', icon: <BarChart3 size={18} />, desc: 'Alibaba Trace' },
            { step: 'PREDICT', icon: <TrendingUp size={18} />, desc: 'Prophet Forecast' },
            { step: 'ADAPT', icon: <Zap size={18} />, desc: 'HARS Reward' },
            { step: 'DECIDE', icon: <Cpu size={18} />, desc: 'DQN Agent' },
            { step: 'SIMULATE', icon: <Server size={18} />, desc: 'CloudSim+' },
            { step: 'VISUALIZE', icon: <Globe size={18} />, desc: 'CloudVerse 3D' }
          ].map((item, i) => (
            <React.Fragment key={item.step}>
              <div className="glass-panel" style={{ padding: '20px 28px', textAlign: 'center', minWidth: '140px' }}>
                <div style={{ color: 'var(--accent-cyan)', marginBottom: '8px', display: 'flex', justifyContent: 'center' }}>{item.icon}</div>
                <div style={{ fontWeight: 700, marginBottom: '4px' }}>{item.step}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{item.desc}</div>
              </div>
              {i < 5 && <ArrowRight size={20} style={{ color: 'var(--accent-blue)', flexShrink: 0 }} />}
            </React.Fragment>
          ))}
        </div>
        </motion.div>
      </section>

      {/* Key Results */}
      <section className="kage-sec" style={{ padding: '140px 5%' }}>
        <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-10%" }} variants={staggerContainer} style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <motion.div variants={fadeUp} className="eyebrow" style={{ marginBottom: '24px' }}>03 / Validation</motion.div>
          <motion.h2 variants={fadeUp} className="kage-display kage-h-sec" style={{ marginBottom: '60px' }}>Key Results</motion.h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '24px' }}>
          {[
            { label: 'Forecast Accuracy (R²)', value: '92.4%', color: 'var(--accent-cyan)', icon: <TrendingUp size={22} /> },
            { label: 'SLA Compliance', value: '99.8%', color: 'var(--accent-purple)', icon: <ShieldCheck size={22} /> },
            { label: 'Energy Savings', value: '-18.4%', color: 'var(--accent-green)', icon: <Zap size={22} /> },
            { label: 'Avg Reward Improvement', value: '+34.2%', color: 'var(--accent-blue)', icon: <Activity size={22} /> }
          ].map((stat) => (
            <div key={stat.label} className="glass-panel" style={{ padding: '32px', textAlign: 'center' }}>
              <div style={{ color: stat.color, marginBottom: '12px', display: 'flex', justifyContent: 'center' }}>{stat.icon}</div>
              <div style={{ fontSize: '2.4rem', fontWeight: 800, color: stat.color, marginBottom: '8px' }}>{stat.value}</div>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{stat.label}</div>
            </div>
          ))}
        </div>
        </motion.div>
      </section>

      {/* Detailed Research Metrics (Graphs) */}
      <section className="kage-sec" style={{ padding: '140px 5%' }}>
        <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-10%" }} variants={staggerContainer} style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <motion.div variants={fadeUp} className="eyebrow" style={{ marginBottom: '24px' }}>04 / Data Evidence</motion.div>
          <motion.h2 variants={fadeUp} className="kage-display kage-h-sec" style={{ marginBottom: '60px' }}>Detailed Metrics</motion.h2>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '32px', marginBottom: '32px' }}>
          {/* Main Scaling Comparison Chart */}
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '8px', fontSize: '1.2rem', textAlign: 'center' }}>Proactive vs Reactive Scaling Behavior</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', textAlign: 'center', marginBottom: '24px' }}>
              Notice how the DQN model scales up slightly before the peak workload hits, preventing the SLA violations seen in the baseline threshold model.
            </p>
            <div style={{ height: '350px', width: '100%' }}>
              <ResponsiveContainer>
                <LineChart data={performanceData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" />
                  <YAxis yAxisId="left" stroke="rgba(255,255,255,0.5)" />
                  <YAxis yAxisId="right" orientation="right" stroke="rgba(255,255,255,0.5)" />
                  <RechartsTooltip 
                    contentStyle={{ backgroundColor: 'rgba(10,12,16,0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <Legend wrapperStyle={{ paddingTop: '20px' }} />
                  <Line yAxisId="left" type="monotone" dataKey="load" name="Workload Demand (%)" stroke="#ff8c00" strokeWidth={2} dot={false} strokeDasharray="5 5" />
                  <Line yAxisId="right" type="stepAfter" dataKey="thresholdVMs" name="Baseline Threshold VMs" stroke="#ef4444" strokeWidth={3} />
                  <Line yAxisId="right" type="stepAfter" dataKey="dqnVMs" name="Proposed DQN VMs" stroke="#8b5cf6" strokeWidth={3} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
          {/* Workload Area Chart */}
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '24px', fontSize: '1.2rem', textAlign: 'center' }}>Predicted vs Actual Demand</h3>
            <div style={{ height: '280px', width: '100%' }}>
              <ResponsiveContainer>
                <AreaChart data={performanceData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                  <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" />
                  <YAxis stroke="rgba(255,255,255,0.5)" />
                  <RechartsTooltip contentStyle={{ backgroundColor: 'rgba(10,12,16,0.9)', border: 'none', borderRadius: '8px' }} />
                  <Area type="monotone" dataKey="load" name="Actual Trace" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.3} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Energy Pie Chart */}
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '8px', fontSize: '1.2rem', textAlign: 'center' }}>Total Energy Consumption</h3>
            <div style={{ height: '280px', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={energyData}
                    cx="50%"
                    cy="50%"
                    innerRadius={70}
                    outerRadius={100}
                    paddingAngle={5}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${((percent || 0) * 100).toFixed(0)}%`}
                    labelLine={false}
                  >
                    {energyData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <RechartsTooltip contentStyle={{ backgroundColor: 'rgba(10,12,16,0.9)', border: 'none', borderRadius: '8px' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
        </motion.div>
      </section>

      {/* Architecture Overview */}
      <section className="kage-sec" style={{ padding: '140px 5%' }}>
        <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-10%" }} variants={staggerContainer} style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <motion.div variants={fadeUp} className="eyebrow" style={{ marginBottom: '24px' }}>02 / System Architecture</motion.div>
          <motion.h2 variants={fadeUp} className="kage-display kage-h-sec" style={{ marginBottom: '60px' }}>The Three Pillars</motion.h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '40px' }}>
            {[
              { title: 'Data Layer', color: 'var(--muted)', desc: 'Raw CPU traces from Alibaba Cluster 2018 are ingested, cleaned, and forecasted using Facebook Prophet.' },
              { title: 'RL Agent Layer', color: 'var(--bone)', desc: 'A custom Gymnasium environment simulates auto-scaling. A DQN learns the optimal policy, aided by HARS.' },
              { title: 'Visualization Layer', color: 'var(--vermilion)', desc: 'CloudVerse connects a FastAPI backend to a React Three Fiber 3D frontend, rendering decisions in real-time.' }
            ].map((col, i) => (
              <motion.div key={i} variants={fadeUp} style={{ padding: '32px 0', borderTop: '1px solid var(--line-soft)' }}>
                <h3 style={{ color: col.color, marginBottom: '20px', fontSize: '1.2rem', fontWeight: 500 }}>{col.title}</h3>
                <p className="kage-body-lg" style={{ fontSize: '0.95rem' }}>{col.desc}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>
      
      {/* Footer CTA */}
      <section className="kage-sec" style={{ padding: '100px 5% 50px', borderTop: '1px solid var(--line-soft)' }}>
        <motion.div initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '40px' }}>
          <div>
            <div className="eyebrow" style={{ marginBottom: '16px' }}>Initialize CloudVerse</div>
            <div style={{ fontFamily: 'monospace', color: 'var(--muted)', fontSize: '0.85rem' }}>python -m cloudverse.backend.main</div>
          </div>
          <Link to="/cloudverse" className="btn-primary" style={{ padding: '12px 30px', background: 'transparent', border: '1px solid var(--line-soft)', color: 'var(--bone)', boxShadow: 'none', borderRadius: '100px' }}>
            Enter Command Center
          </Link>
        </motion.div>
      </section>
      
      </div>
    </div>
  );
}
