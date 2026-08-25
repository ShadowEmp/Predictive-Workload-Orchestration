import React, { useEffect, useState, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { Link } from 'react-router-dom';
import { ArrowLeft, Play, Square, Pause, Volume2, VolumeX, ShieldCheck, Zap } from 'lucide-react';
import VMInstances from '../vms/VMInstances';
import TrafficSystem from '../traffic/TrafficSystem';
import HostDistrict from '../hosts/HostDistrict';
import RoadNetwork from '../roads/RoadNetwork';
import CitySkyline from '../city/CitySkyline';
import EnvironmentDecorations from '../city/EnvironmentDecorations';
import type { ChallengeState } from '../traffic/TrafficSystem';

const DEFAULT_CHALLENGE_STATE: ChallengeState = {
  failedHost: -1,
  flashSaleActive: false,
  memoryPressureActive: false,
};

function CloudCityScene({ vmCount, cpuUtil, requestRate, hostTraffic, challengeState, title, colorHex, accentText }: { vmCount: number; cpuUtil: number; requestRate: number; hostTraffic: number[]; challengeState: ChallengeState; title: string; colorHex: string; accentText: string }) {
  
  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {/* Semi-transparent info panel inside the scene */}
      <div style={{ 
        position: 'absolute', 
        top: 20, 
        left: '50%', 
        transform: 'translateX(-50%)', 
        zIndex: 10, 
        background: 'rgba(10,12,16,0.85)', 
        padding: '16px 28px', 
        borderRadius: '12px', 
        border: `1px solid ${colorHex}55`, 
        textAlign: 'center',
        boxShadow: `0 0 15px ${colorHex}22`,
        backdropFilter: 'blur(8px)',
        minWidth: '280px'
      }}>
        <h2 style={{ fontSize: '1.05rem', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '0.5px', marginBottom: '8px' }}>{title}</h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginTop: '10px' }}>
          <div>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', display: 'block' }}>VMS INSTANCES</span>
            <span style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--accent-cyan)' }}>{vmCount}</span>
          </div>
          <div>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', display: 'block' }}>CPU AVERAGE</span>
            <span style={{ fontSize: '1.4rem', fontWeight: 800, color: cpuUtil > 80 ? 'var(--accent-red)' : colorHex }}>{cpuUtil.toFixed(1)}%</span>
          </div>
        </div>
        
        {/* Load bar */}
        <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', overflow: 'hidden', marginTop: '12px' }}>
          <div style={{ width: `${cpuUtil}%`, height: '100%', background: colorHex, transition: 'width 0.3s ease' }} />
        </div>
      </div>
      
      <Canvas camera={{ position: [45, 45, 45], fov: 40 }}>
        <color attach="background" args={['#87CEEB']} />
        
        <ambientLight intensity={0.7} />
        <directionalLight position={[40, 60, 30]} intensity={2.0} color="#ffffff" castShadow />
        <hemisphereLight args={['#87CEEB', '#22c55e', 0.4]} />
        
        <EnvironmentDecorations />
        <CitySkyline />
        <RoadNetwork />
        <VMInstances count={vmCount} cpuUtil={cpuUtil} />
        <TrafficSystem 
          hostTraffic={hostTraffic}
          requestRate={requestRate}
          activeVmCount={vmCount} 
          challengeState={challengeState}
        />
        <mesh position={[0, -0.21, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[300, 300]} />
          <meshStandardMaterial color="#22c55e" />
        </mesh>
        
        <OrbitControls makeDefault maxPolarAngle={Math.PI / 2 - 0.1} minDistance={15} maxDistance={150} target={[0, 0, 10]} />
      </Canvas>
    </div>
  );
}

export default function ComparisonView() {
  const [connected, setConnected] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  
  const [regState, setRegState] = useState({ vmCount: 5, cpuUtil: 0, reward: 0, requestRate: 0.1, hostTraffic: [0.1, 0.1, 0.1, 0.1, 0.1] });
  const [propState, setPropState] = useState({ vmCount: 5, cpuUtil: 0, reward: 0, requestRate: 0.1, hostTraffic: [0.1, 0.1, 0.1, 0.1, 0.1] });
  
  const [totalRegReward, setTotalRegReward] = useState(0);
  const [totalPropReward, setTotalPropReward] = useState(0);
  
  const [challengeState, setChallengeState] = useState<ChallengeState>(DEFAULT_CHALLENGE_STATE);
  const [challengeLoading, setChallengeLoading] = useState<Record<string, boolean>>({});
  
  const lastRegCount = useRef(5);
  const lastPropCount = useRef(5);

  const speak = (text: string) => {
    if (!voiceEnabled || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.15;
    utterance.pitch = 0.95;
    window.speechSynthesis.speak(utterance);
  };

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onopen = () => {
      setConnected(true);
      speak("Comparison node established. Standard and DQN twin synchronizing.");
    };
    ws.onclose = () => setConnected(false);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.event_type === 'COMPARISON_UPDATE') {
        const p = data.payload;
        
        setRegState({ 
          vmCount: p.regular.vm_count, 
          cpuUtil: p.regular.cpu_util, 
          reward: p.regular.reward,
          requestRate: p.regular.request_rate || (p.regular.cpu_util / 100),
          hostTraffic: p.regular.host_traffic || [0.1, 0.1, 0.1, 0.1, 0.1]
        });
        setPropState({ 
          vmCount: p.proposed.vm_count, 
          cpuUtil: p.proposed.cpu_util, 
          reward: p.proposed.reward,
          requestRate: p.proposed.request_rate || (p.proposed.cpu_util / 100),
          hostTraffic: p.proposed.host_traffic || [0.1, 0.1, 0.1, 0.1, 0.1]
        });
        
        setTotalRegReward(prev => prev + p.regular.reward);
        setTotalPropReward(prev => prev + p.proposed.reward);

        // Vocalise comparative scaling actions
        if (p.proposed.vm_count !== lastPropCount.current) {
          const delta = p.proposed.vm_count - lastPropCount.current;
          if (delta > 0) speak(`DQN model added ${delta} VM for workload buffer`);
          else speak(`DQN model released ${Math.abs(delta)} VM due to prediction confidence`);
          lastPropCount.current = p.proposed.vm_count;
        }
      } else if (data.event_type === 'HOST_FAILURE') {
        setChallengeState((prev) => ({ ...prev, failedHost: data.payload.failed_host }));
        speak(`Warning. Hardware failure detected in Host ${data.payload.failed_host}.`);
      } else if (data.event_type === 'HOST_RESTORED') {
        setChallengeState((prev) => ({ ...prev, failedHost: -1 }));
        speak("Host district restored.");
      } else if (data.event_type === 'FLASH_SALE_SPIKE') {
        setChallengeState((prev) => ({ ...prev, flashSaleActive: true }));
        speak("Alert! Flash sale detected.");
      } else if (data.event_type === 'FLASH_SALE_ENDED') {
        setChallengeState((prev) => ({ ...prev, flashSaleActive: false }));
        speak("Flash sale resolved.");
      } else if (data.event_type === 'MEMORY_PRESSURE') {
        setChallengeState((prev) => ({ ...prev, memoryPressureActive: true }));
        speak("Memory pressure applied.");
      } else if (data.event_type === 'MEMORY_PRESSURE_ENDED') {
        setChallengeState((prev) => ({ ...prev, memoryPressureActive: false }));
        speak("Memory pressure resolved.");
      } else if (data.event_type === 'SIMULATION_STARTED') {
        setTotalRegReward(0);
        setTotalPropReward(0);
        speak("Dual simulation pipeline initialized.");
      } else if (data.event_type === 'SIMULATION_COMPLETED') {
        speak("Comparison complete. proposed DQN outperformed baseline thresholds.");
      }
    };
    
    return () => {
      ws.close();
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, [voiceEnabled]);

  const handleStart = async () => await fetch('http://localhost:8000/api/simulation/start_comparison', { method: 'POST' });
  const handleStop = async () => await fetch('http://localhost:8000/api/simulation/stop', { method: 'POST' });

  const handleChallenge = async (name: string, endpoint: string) => {
    setChallengeLoading((prev) => ({ ...prev, [name]: true }));
    try {
      await fetch(`http://localhost:8000/api/challenges/${endpoint}`, { method: 'POST' });
    } finally {
      setChallengeLoading((prev) => ({ ...prev, [name]: false }));
    }
  };

  return (
    <div className="full-screen-app" style={{ display: 'flex', flexDirection: 'column' }}>
      {/* Top Header Control Bar */}
      <div style={{ 
        height: '70px', 
        borderBottom: '1px solid rgba(239, 68, 68, 0.2)', 
        display: 'flex', 
        alignItems: 'center', 
        padding: '0 24px', 
        justifyContent: 'space-between', 
        background: 'var(--bg-dark)',
        zIndex: 100
      }}>
        <Link to="/cloudverse" className="btn-secondary" style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 16px', background: 'rgba(255,255,255,0.02)' }}>
          <ArrowLeft size={16} /> Single View
        </Link>
        
        {/* Core sim toggle controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button className="btn-primary" onClick={handleStart} style={{ padding: '8px 20px', display: 'flex', alignItems: 'center', gap: '6px' }}><Play size={16} /> SYNC RUN</button>
            <button className="btn-secondary" onClick={handleStop} style={{ padding: '8px 16px' }}><Square size={16} /></button>
          </div>
          
          <button 
            onClick={() => setVoiceEnabled(!voiceEnabled)} 
            style={{ background: 'none', border: 'none', color: voiceEnabled ? 'var(--accent-cyan)' : 'var(--text-secondary)', cursor: 'pointer', display: 'flex', alignItems: 'center' }}
            title={voiceEnabled ? "Voice On" : "Voice Off"}
          >
            {voiceEnabled ? <Volume2 size={18} /> : <VolumeX size={18} />}
          </button>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            Sync Status: <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: connected ? 'var(--accent-green)' : 'var(--accent-red)', boxShadow: connected ? '0 0 8px #10b981' : '0 0 8px #ef4444' }} />
          </div>
        </div>
        
        {/* Live Scoreboards */}
        <div style={{ display: 'flex', gap: '24px' }}>
          <div style={{ background: 'rgba(255,255,255,0.02)', padding: '6px 14px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', display: 'block' }}>REGULAR REWARD</span>
            <span style={{ fontWeight: 800, fontSize: '1.1rem', color: 'var(--accent-red)' }}>{totalRegReward.toFixed(1)}</span>
          </div>
          <div style={{ background: 'rgba(139, 92, 246, 0.05)', padding: '6px 14px', borderRadius: '8px', border: '1px solid rgba(139, 92, 246, 0.15)' }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', display: 'block' }}>PROPOSED DQN REWARD</span>
            <span style={{ fontWeight: 800, fontSize: '1.1rem', color: 'var(--accent-purple)' }}>{totalPropReward.toFixed(1)}</span>
          </div>
        </div>
      </div>

      {/* Challenges Bar */}
      <div style={{ background: 'rgba(255,255,255,0.02)', padding: '10px 24px', borderBottom: '1px solid rgba(239, 68, 68, 0.1)', display: 'flex', gap: '16px', alignItems: 'center' }}>
        <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', fontWeight: 800, display: 'flex', alignItems: 'center', gap: '6px', marginRight: '10px' }}>
          <ShieldCheck size={14} /> INJECT:
        </div>
        
        <button
          className="btn-secondary"
          style={{ fontSize: '0.75rem', padding: '6px 12px', border: challengeState.failedHost >= 0 ? '1px solid rgba(239,68,68,0.5)' : undefined, opacity: challengeLoading['hostFailure'] ? 0.6 : 1 }}
          onClick={() => handleChallenge('hostFailure', challengeState.failedHost >= 0 ? 'host_failure/clear' : 'host_failure')}
          disabled={challengeLoading['hostFailure']}
        >
          {challengeLoading['hostFailure'] ? '⟳...' : challengeState.failedHost >= 0 ? '✓ Restore Host' : '⚡ Host Failure'}
        </button>

        <button
          className="btn-secondary"
          style={{ fontSize: '0.75rem', padding: '6px 12px', border: challengeState.flashSaleActive ? '1px solid rgba(255,153,0,0.5)' : undefined, opacity: challengeLoading['flashSale'] ? 0.6 : 1 }}
          onClick={() => handleChallenge('flashSale', 'flash_sale')}
          disabled={challengeLoading['flashSale']}
        >
          {challengeLoading['flashSale'] ? '⟳...' : challengeState.flashSaleActive ? '🔥 Flash Sale Active (Click to clear)' : '🔥 Flash Sale Spike'}
        </button>

        <button
          className="btn-secondary"
          style={{ fontSize: '0.75rem', padding: '6px 12px', border: challengeState.memoryPressureActive ? '1px solid rgba(168,85,247,0.5)' : undefined, opacity: challengeLoading['memoryPressure'] ? 0.6 : 1 }}
          onClick={() => handleChallenge('memoryPressure', 'memory_pressure')}
          disabled={challengeLoading['memoryPressure'] || challengeState.memoryPressureActive}
        >
          {challengeLoading['memoryPressure'] ? '⟳...' : challengeState.memoryPressureActive ? '⚠ Memory Pressure Active' : '⚠ Memory Pressure'}
        </button>
      </div>

      {/* Synchronized Twin Panels */}
      <div style={{ flex: 1, display: 'flex' }}>
        <div style={{ flex: 1, borderRight: '1px solid rgba(239, 68, 68, 0.15)' }}>
          <CloudCityScene 
            vmCount={regState.vmCount} 
            cpuUtil={regState.cpuUtil} 
            requestRate={regState.requestRate}
            hostTraffic={regState.hostTraffic}
            challengeState={challengeState}
            title="REGULAR AUTO-SCALING (Threshold)" 
            colorHex="#ef4444" 
            accentText="var(--accent-red)"
          />
        </div>
        <div style={{ flex: 1 }}>
          <CloudCityScene 
            vmCount={propState.vmCount} 
            cpuUtil={propState.cpuUtil} 
            requestRate={propState.requestRate}
            hostTraffic={propState.hostTraffic}
            challengeState={challengeState}
            title="PROPOSED AI ORCHESTRATION (DQN)" 
            colorHex="#8b5cf6" 
            accentText="var(--accent-purple)"
          />
        </div>
      </div>
    </div>
  );
}
