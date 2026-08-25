import React, { useEffect, useState, useRef } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { Link } from 'react-router-dom';
import { ArrowLeft, Play, Square, Pause, Volume2, VolumeX, FastForward, Zap, ShieldAlert, Cpu, Activity, AlertTriangle, MemoryStick } from 'lucide-react';
import VMInstances from './vms/VMInstances';
import TrafficSystem from './traffic/TrafficSystem';
import type { ChallengeState } from './traffic/TrafficSystem';
import RoadNetwork from './roads/RoadNetwork';
import CitySkyline from './city/CitySkyline';
import EnvironmentDecorations from './city/EnvironmentDecorations';
import * as THREE from 'three';

function CameraAnimator({ viewTarget, controlsRef }: { viewTarget: { pos: number[], look: number[], ts: number }, controlsRef: React.MutableRefObject<any> }) {
  const { camera } = useThree();
  const targetPos = useRef(new THREE.Vector3());
  const targetLook = useRef(new THREE.Vector3());
  const isAnimating = useRef(false);

  useEffect(() => {
    isAnimating.current = true;
    targetPos.current.set(viewTarget.pos[0], viewTarget.pos[1], viewTarget.pos[2]);
    targetLook.current.set(viewTarget.look[0], viewTarget.look[1], viewTarget.look[2]);
  }, [viewTarget]);

  useFrame(() => {
    if (!isAnimating.current) return;
    
    camera.position.lerp(targetPos.current, 0.05);
    if (controlsRef.current) {
      controlsRef.current.target.lerp(targetLook.current, 0.05);
    }
    
    // Stop animating once we are close enough, freeing the camera for OrbitControls
    if (camera.position.distanceTo(targetPos.current) < 0.5 && controlsRef.current.target.distanceTo(targetLook.current) < 0.5) {
      isAnimating.current = false;
    }
  });

  return null;
}

const DEFAULT_CHALLENGE_STATE: ChallengeState = {
  failedHost: -1,
  flashSaleActive: false,
  memoryPressureActive: false,
};

export default function CloudVerseApp() {
  const [events, setEvents] = useState<any[]>([]);
  const [vmCount, setVmCount] = useState(5);
  const [cpuUtil, setCpuUtil] = useState(0);
  const [connected, setConnected] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [simSpeed, setSimSpeed] = useState(1);
  const [cameraView, setCameraView] = useState({ id: 'overview', pos: [150, 120, 150], look: [0, 0, 0], ts: Date.now() });
  const controlsRef = useRef<any>(null);
  const lastActionRef = useRef<string>('');

  // ── Traffic / challenge state ──
  const [requestRate, setRequestRate] = useState(0.1);
  const [hostTraffic, setHostTraffic] = useState<number[]>([0.1, 0.1, 0.1, 0.1, 0.1]);
  const [challengeState, setChallengeState] = useState<ChallengeState>(DEFAULT_CHALLENGE_STATE);

  // ── Challenge injection loading states ──
  const [challengeLoading, setChallengeLoading] = useState<Record<string, boolean>>({});

  // JARVIS Web Speech API synthesis helper
  const speak = (text: string) => {
    if (!voiceEnabled || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.15;
    utterance.pitch = 0.9;
    window.speechSynthesis.speak(utterance);
  };
  
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onopen = () => {
      setConnected(true);
      speak("Command Center connected. Telemetry online.");
    };
    ws.onclose = () => setConnected(false);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setEvents((prev) => [data, ...prev].slice(0, 40));
      
      if (data.event_type === 'VM_CREATED' || data.event_type === 'VM_DELETED' || data.event_type === 'WORKLOAD_UPDATED') {
        if (data.payload.vm_count !== undefined) {
          setVmCount(data.payload.vm_count);
        }
      }
      if (data.payload.cpu_util !== undefined) {
        setCpuUtil(data.payload.cpu_util);
      }

      // ── Request-traffic data from WORKLOAD_UPDATED ──
      if (data.event_type === 'WORKLOAD_UPDATED') {
        if (data.payload.request_rate !== undefined) {
          setRequestRate(data.payload.request_rate);
        }
        if (Array.isArray(data.payload.host_traffic) && data.payload.host_traffic.length === 5) {
          setHostTraffic(data.payload.host_traffic);
        }
        // Sync challenge flags from backend state
        setChallengeState((prev) => ({
          ...prev,
          flashSaleActive: !!data.payload.flash_sale_active,
          memoryPressureActive: !!data.payload.memory_pressure_active,
        }));
      }

      // ── Challenge events ──
      if (data.event_type === 'HOST_FAILURE') {
        const fh = data.payload.failed_host ?? -1;
        setChallengeState((prev) => ({ ...prev, failedHost: fh }));
        speak(`Warning. Host district ${fh} has gone offline. Traffic is rerouting.`);
      } else if (data.event_type === 'HOST_RECOVERED') {
        setChallengeState((prev) => ({ ...prev, failedHost: -1 }));
        speak("Host district has been restored. Normal traffic flow resuming.");
      } else if (data.event_type === 'FLASH_SALE_SPIKE') {
        setChallengeState((prev) => ({ ...prev, flashSaleActive: true }));
        speak("Alert! Flash sale detected. Request traffic surging at four times normal load.");
      } else if (data.event_type === 'FLASH_SALE_ENDED') {
        setChallengeState((prev) => ({ ...prev, flashSaleActive: false }));
        speak("Flash sale traffic spike resolved. Load normalising.");
      } else if (data.event_type === 'MEMORY_PRESSURE') {
        setChallengeState((prev) => ({ ...prev, memoryPressureActive: true }));
        speak("Memory pressure applied. Virtual machine throughput is degraded.");
      } else if (data.event_type === 'MEMORY_PRESSURE_ENDED') {
        setChallengeState((prev) => ({ ...prev, memoryPressureActive: false }));
        speak("Memory pressure resolved. System performance restored.");
      } else if (data.event_type === 'SIMULATION_STARTED') {
        speak("Cloud simulation started. Workload trace loading.");
      } else if (data.event_type === 'DQN_DECISION') {
        const action = data.payload.action_name;
        if (action !== lastActionRef.current) {
          if (action === 'SCALE_UP') speak("Scaling action triggered. Allocating new virtual instances.");
          else if (action === 'SCALE_DOWN') speak("Deallocating idle resources. Energy saving optimized.");
          lastActionRef.current = action;
        }
      } else if (data.event_type === 'QUEUE_INCREASED') {
        speak("Warning. Incoming workload queue capacity rising.");
      } else if (data.event_type === 'SIMULATION_COMPLETED') {
        speak("Simulation cycle completed. Metric payload recorded.");
      }
    };
    return () => { ws.close(); if ('speechSynthesis' in window) window.speechSynthesis.cancel(); };
  }, [voiceEnabled]);
  
  const handleStart = async () => await fetch('http://localhost:8000/api/simulation/start', { method: 'POST' });
  const handleStop = async () => await fetch('http://localhost:8000/api/simulation/stop', { method: 'POST' });
  const handlePause = async () => await fetch('http://localhost:8000/api/simulation/pause', { method: 'POST' });
  const handleResume = async () => await fetch('http://localhost:8000/api/simulation/resume', { method: 'POST' });
  
  const handleSpeedChange = async (speed: number) => {
    setSimSpeed(speed);
    await fetch(`http://localhost:8000/api/simulation/speed/${speed}`, { method: 'POST' });
    speak(`Simulation speed scaled to ${speed}x`);
  };

  // ── Challenge injection handlers ──
  const handleChallenge = async (name: string, endpoint: string, onSuccess?: () => void) => {
    setChallengeLoading((prev) => ({ ...prev, [name]: true }));
    try {
      const res = await fetch(`http://localhost:8000/api/challenges/${endpoint}`, { method: 'POST' });
      const data = await res.json();
      if (data.status === 'no_simulation_running') {
        speak("No simulation is running. Please start the simulation first.");
      } else {
        onSuccess?.();
      }
    } catch {
      speak("Challenge injection failed. Backend unreachable.");
    } finally {
      setChallengeLoading((prev) => ({ ...prev, [name]: false }));
    }
  };

  const handleHostFailure = () =>
    handleChallenge('hostFailure', 'host_failure');

  const handleClearHostFailure = () =>
    handleChallenge('hostFailure', 'host_failure/clear');

  const handleFlashSale = () =>
    handleChallenge('flashSale', 'flash_sale');

  const handleMemoryPressure = () =>
    handleChallenge('memoryPressure', 'memory_pressure');

  const cpuIntensity = Math.min(1, Math.max(0, cpuUtil / 100));

  return (
    <div className="full-screen-app">
      <div style={{ position: 'absolute', top: 20, left: 20, zIndex: 10, display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', gap: '12px' }}>
          <Link to="/" className="btn-secondary" style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 16px', background: 'rgba(10,12,16,0.85)' }}>
            <ArrowLeft size={16} /> Exit Command Center
          </Link>
          <Link to="/cloudverse/comparison" className="btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 16px', background: 'linear-gradient(135deg, var(--accent-purple) 0%, var(--accent-blue) 100%)' }}>
            Switch to Comparison Mode
          </Link>
        </div>
        
        <div className="glass-panel" style={{ padding: '24px', width: '340px', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
          <h2 style={{ fontSize: '1.3rem', fontWeight: 700, marginBottom: '18px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', letterSpacing: '1px' }}>
            <span className="premium-gradient-text" style={{ fontWeight: 800 }}>MISSION CONTROL</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <button 
                onClick={() => setVoiceEnabled(!voiceEnabled)} 
                style={{ background: 'none', border: 'none', color: voiceEnabled ? 'var(--accent-cyan)' : 'var(--text-secondary)', cursor: 'pointer', display: 'flex', alignItems: 'center' }}
                title={voiceEnabled ? "Voice On" : "Voice Off"}
              >
                {voiceEnabled ? <Volume2 size={18} /> : <VolumeX size={18} />}
              </button>
              <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: connected ? 'var(--accent-green)' : 'var(--accent-red)', boxShadow: connected ? '0 0 10px #10b981' : '0 0 10px #ef4444' }} />
            </div>
          </h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px', marginBottom: '24px' }}>
            <button className="btn-primary" onClick={handleStart} title="Start" style={{ padding: '10px' }}><Play size={16} /></button>
            <button className="btn-secondary" onClick={handlePause} title="Pause" style={{ padding: '10px' }}><Pause size={16} /></button>
            <button className="btn-secondary" onClick={handleResume} title="Resume" style={{ padding: '10px' }}><Play size={16} style={{ transform: 'rotate(90deg)', opacity: 0.5 }} /></button>
            <button className="btn-secondary" onClick={handleStop} title="Stop" style={{ padding: '10px' }}><Square size={16} /></button>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', marginBottom: '8px', letterSpacing: '0.5px' }}>CAMERA VIEWS</div>
            <div style={{ display: 'flex', gap: '4px' }}>
              <button onClick={() => setCameraView({ id: 'overview', pos: [150, 120, 150], look: [0, 0, 0], ts: Date.now() })} className={cameraView.id === 'overview' ? "btn-primary" : "btn-secondary"} style={{ flex: 1, padding: '6px', fontSize: '0.75rem' }}>Overview</button>
              <button onClick={() => setCameraView({ id: 'center', pos: [0, 60, 80], look: [0, 0, 0], ts: Date.now() })} className={cameraView.id === 'center' ? "btn-primary" : "btn-secondary"} style={{ flex: 1, padding: '6px', fontSize: '0.75rem' }}>Center</button>
              <button onClick={() => setCameraView({ id: 'topdown', pos: [0, 280, 0.1], look: [0, 0, 0], ts: Date.now() })} className={cameraView.id === 'topdown' ? "btn-primary" : "btn-secondary"} style={{ flex: 1, padding: '6px', fontSize: '0.75rem' }}>Top-Down</button>
            </div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.65rem', marginTop: '6px', fontStyle: 'italic' }}>Double-click anywhere on the map to zoom there!</div>
          </div>

          <div style={{ marginBottom: '24px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div style={{ background: 'rgba(239, 68, 68, 0.05)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(239, 68, 68, 0.1)' }}>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.7rem', display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '4px' }}><Cpu size={12} /> VMs</div>
              <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-cyan)' }}>{vmCount}</div>
            </div>
            <div style={{ background: cpuUtil > 80 ? 'rgba(239, 68, 68, 0.05)' : 'rgba(239, 68, 68, 0.02)', padding: '12px', borderRadius: '8px', border: cpuUtil > 80 ? '1px solid rgba(239, 68, 68, 0.3)' : '1px solid rgba(239, 68, 68, 0.1)' }}>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.7rem', display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '4px' }}><Zap size={12} /> CPU UTIL</div>
              <div style={{ fontSize: '1.8rem', fontWeight: 800, color: cpuUtil > 80 ? 'var(--accent-red)' : 'var(--accent-red)' }}>{cpuUtil.toFixed(1)}%</div>
            </div>
          </div>

          {/* Request traffic bar */}
          <div style={{ marginBottom: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><Activity size={10} /> REQUEST TRAFFIC</span>
              <span>{Math.round(requestRate * 100)}%</span>
            </div>
            <div style={{ width: '100%', height: '5px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ width: `${requestRate * 100}%`, height: '100%', background: 'linear-gradient(90deg, #00cfff 0%, #00ff88 100%)', transition: 'width 0.4s ease' }} />
            </div>
          </div>

          {/* Per-host mini traffic bars */}
          <div style={{ marginBottom: '20px' }}>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.7rem', marginBottom: '8px' }}>PER-HOST TRAFFIC</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {['Centre','North','South','East','West'].map((label, h) => {
                const isFailed = h === challengeState.failedHost;
                const traffic = hostTraffic[h] ?? 0;
                const hostPalette = ['#00cfff','#ff9900','#22cc55','#cc44ff','#ff3344'];
                return (
                  <div key={h} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '40px', fontSize: '0.65rem', color: isFailed ? 'var(--accent-red)' : 'var(--text-secondary)', flexShrink: 0 }}>
                      {label}{isFailed ? ' ✗' : ''}
                    </span>
                    <div style={{ flex: 1, height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', overflow: 'hidden' }}>
                      <div style={{
                        width: `${traffic * 100}%`,
                        height: '100%',
                        background: isFailed ? '#ef4444' : hostPalette[h],
                        transition: 'width 0.4s ease',
                        opacity: isFailed ? 0.4 : 1,
                      }} />
                    </div>
                    <span style={{ width: '28px', fontSize: '0.6rem', color: 'var(--text-secondary)', textAlign: 'right' }}>
                      {Math.round(traffic * 100)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
          
          <div style={{ marginBottom: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
              <span>CLUSTER CAPACITY LOAD</span>
              <span>{Math.round(cpuUtil)}%</span>
            </div>
            <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ width: `${cpuUtil}%`, height: '100%', background: 'linear-gradient(90deg, var(--accent-red) 0%, #ff8c00 60%, var(--accent-purple) 100%)', transition: 'width 0.3s ease' }} />
            </div>
          </div>

          {/* Active challenge badges */}
          {(challengeState.failedHost >= 0 || challengeState.flashSaleActive || challengeState.memoryPressureActive) && (
            <div style={{ marginBottom: '16px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {challengeState.failedHost >= 0 && (
                <span style={{ fontSize: '0.65rem', padding: '3px 8px', borderRadius: '12px', background: 'rgba(239,68,68,0.15)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.3)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <AlertTriangle size={10} /> HOST {challengeState.failedHost} OFFLINE
                </span>
              )}
              {challengeState.flashSaleActive && (
                <span style={{ fontSize: '0.65rem', padding: '3px 8px', borderRadius: '12px', background: 'rgba(255,153,0,0.15)', color: '#ff9900', border: '1px solid rgba(255,153,0,0.3)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Zap size={10} /> FLASH SALE SPIKE
                </span>
              )}
              {challengeState.memoryPressureActive && (
                <span style={{ fontSize: '0.65rem', padding: '3px 8px', borderRadius: '12px', background: 'rgba(168,85,247,0.15)', color: '#a855f7', border: '1px solid rgba(168,85,247,0.3)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  ⚠ MEMORY PRESSURE
                </span>
              )}
            </div>
          )}
          
          {/* Challenge Injections */}
          <div style={{ marginBottom: '24px', background: 'rgba(255,255,255,0.02)', padding: '14px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div style={{ color: 'var(--text-primary)', fontSize: '0.85rem', marginBottom: '12px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '6px' }}><ShieldAlert size={14} className="brand-icon" /> CHALLENGE INJECTIONS</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>

              {/* Host Failure */}
              <div style={{ display: 'flex', gap: '6px' }}>
                <button
                  className="btn-secondary"
                  style={{
                    flex: 1,
                    fontSize: '0.75rem',
                    padding: '8px',
                    textAlign: 'left',
                    border: challengeState.failedHost >= 0 ? '1px solid rgba(239,68,68,0.5)' : undefined,
                    opacity: challengeLoading['hostFailure'] ? 0.6 : 1,
                  }}
                  onClick={handleHostFailure}
                  disabled={challengeLoading['hostFailure']}
                >
                  {challengeLoading['hostFailure'] ? '⟳ Injecting...' : '⚡ Inject Host Failure'}
                </button>
                {challengeState.failedHost >= 0 && (
                  <button
                    className="btn-secondary"
                    style={{ fontSize: '0.7rem', padding: '8px', color: 'var(--accent-green)', border: '1px solid rgba(34,197,94,0.3)' }}
                    onClick={handleClearHostFailure}
                  >
                    Restore
                  </button>
                )}
              </div>

              {/* Flash Sale */}
              <button
                className="btn-secondary"
                style={{
                  fontSize: '0.75rem',
                  padding: '8px',
                  textAlign: 'left',
                  border: challengeState.flashSaleActive ? '1px solid rgba(255,153,0,0.5)' : undefined,
                  opacity: challengeLoading['flashSale'] ? 0.6 : 1,
                }}
                onClick={handleFlashSale}
                disabled={challengeLoading['flashSale']}
              >
                {challengeLoading['flashSale']
                  ? '⟳ Toggling...'
                  : challengeState.flashSaleActive
                  ? '🔥 Flash Sale Active (Click to clear)'
                  : '🔥 Flash Sale Spike'}
              </button>

              {/* Memory Pressure */}
              <button
                className="btn-secondary"
                style={{
                  fontSize: '0.75rem',
                  padding: '8px',
                  textAlign: 'left',
                  border: challengeState.memoryPressureActive ? '1px solid rgba(168,85,247,0.5)' : undefined,
                  opacity: challengeLoading['memoryPressure'] ? 0.6 : 1,
                }}
                onClick={handleMemoryPressure}
                disabled={challengeLoading['memoryPressure'] || challengeState.memoryPressureActive}
              >
                {challengeLoading['memoryPressure']
                  ? '⟳ Injecting...'
                  : challengeState.memoryPressureActive
                  ? '⚠ Memory Pressure Active (auto-clears)'
                  : '⚠ Memory Pressure'}
              </button>

            </div>
          </div>
          
          <div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', marginBottom: '8px', letterSpacing: '0.5px' }}>TELEMETRY EVENT STREAM</div>
            <div className="event-stream-scroll" style={{ maxHeight: '180px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.75rem', paddingRight: '4px' }}>
              {events.length === 0 && <span style={{ opacity: 0.4, fontStyle: 'italic' }}>Listening for active socket frames...</span>}
              {events.map((e, i) => (
                <div key={i} style={{ padding: '8px', background: 'rgba(255,255,255,0.03)', borderRadius: '6px', borderLeft: `3px solid ${
                  e.event_type.includes('FAILURE') || e.event_type.includes('PRESSURE')
                    ? 'var(--accent-red)'
                    : e.event_type.includes('SPIKE') || e.event_type.includes('FLASH')
                    ? '#ff9900'
                    : e.event_type.includes('WARNING') || e.event_type.includes('DECISION')
                    ? 'var(--accent-cyan)'
                    : 'var(--accent-purple)'
                }` }}>
                  <div style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: '2px' }}>{e.event_type}</div>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '0.7rem' }}>Time step: {e.simulation_time}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <Canvas camera={{ position: [150, 120, 150], fov: 45 }}>
        <color attach="background" args={['#87CEEB']} />
        
        <ambientLight intensity={0.7} />
        <directionalLight position={[40, 60, 30]} intensity={2.0} color="#ffffff" castShadow />
        <hemisphereLight args={['#87CEEB', '#22c55e', 0.4]} />
        
        <CameraAnimator viewTarget={cameraView} controlsRef={controlsRef} />
        
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
        
        <mesh 
          position={[0, -0.21, 0]} 
          rotation={[-Math.PI / 2, 0, 0]}
          onDoubleClick={(e) => {
            e.stopPropagation();
            setCameraView({
              id: 'custom',
              pos: [e.point.x + 60, 60, e.point.z + 60],
              look: [e.point.x, 0, e.point.z],
              ts: Date.now()
            });
          }}
        >
          <planeGeometry args={[400, 400]} />
          <meshStandardMaterial color="#22c55e" />
        </mesh>
        
        <OrbitControls ref={controlsRef} makeDefault maxPolarAngle={Math.PI / 2 - 0.1} minDistance={15} maxDistance={400} target={[0, 0, 0]} />
      </Canvas>
    </div>
  );
}
