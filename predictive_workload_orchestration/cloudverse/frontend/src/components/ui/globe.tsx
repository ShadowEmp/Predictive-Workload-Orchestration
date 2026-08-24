import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import RotatingEarth from './wireframe-dotted-globe';

// Swarm of Green Satellites
function Satellites({ count = 300 }) {
  const pointsRef = useRef<THREE.Points>(null);
  
  const particles = useMemo(() => {
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos((Math.random() * 2) - 1);
      const r = 2.7 + Math.random() * 0.7; // Orbit range from 2.7 to 3.4
      
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);
    }
    return positions;
  }, [count]);

  useFrame((state) => {
    if (pointsRef.current) {
      // Complex multi-axis rotation to simulate swarm orbiting
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.15;
      pointsRef.current.rotation.z = state.clock.elapsedTime * 0.08;
      pointsRef.current.rotation.x = state.clock.elapsedTime * 0.05;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particles.length / 3}
          array={particles}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial 
        size={0.06} 
        color="#10b981" // Glowing green
        transparent 
        opacity={0.9} 
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

const Globe: React.FC = () => {
  return (
    <div style={{ width: '600px', height: '600px', position: 'relative' }}>
      
      {/* 
        The User's D3 Wireframe Globe (Interactive)
        This handles mouse drag to rotate the earth!
      */}
      <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <RotatingEarth width={600} height={600} />
      </div>

      {/* 
        The R3F Satellites Swarm (Transparent Overlay)
        pointerEvents: 'none' allows the mouse to drag the D3 canvas underneath it!
      */}
      <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', filter: 'drop-shadow(0 0 40px rgba(16, 185, 129, 0.2))' }}>
        <Canvas camera={{ position: [0, 0, 6.5], fov: 45 }} gl={{ alpha: true }}>
          <Satellites count={400} />
        </Canvas>
      </div>

    </div>
  );
};

export default Globe;
