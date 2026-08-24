import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { PerspectiveCamera, Stars } from '@react-three/drei';
import * as THREE from 'three';
import RoadNetwork from '../cloudverse/roads/RoadNetwork';
import HostDistrict from '../cloudverse/hosts/HostDistrict';

function CinematicCamera() {
  const group = useRef<THREE.Group>(null);
  const camera = useRef<THREE.PerspectiveCamera>(null);

  useFrame((state) => {
    if (group.current) {
      // Very slow cinematic rotation
      group.current.rotation.y = state.clock.elapsedTime * 0.03;
    }
  });

  return (
    <group ref={group}>
      <PerspectiveCamera ref={camera} makeDefault position={[0, 35, 75]} fov={45} />
    </group>
  );
}

export default function LandingScene() {
  const districts = [
    { id: 'sec-1', x: -30, z: -30 },
    { id: 'sec-2', x: 30, z: -30 },
    { id: 'sec-3', x: -30, z: 30 },
    { id: 'sec-4', x: 30, z: 30 },
    { id: 'sec-center', x: 0, z: 0 },
  ];

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 0, background: 'var(--ink)' }}>
      {/* Subtle Grain Overlay for cinematic texture (Kage-style) */}
      <div 
        style={{ 
          position: 'absolute', inset: 0, zIndex: 5, pointerEvents: 'none',
          background: 'url("data:image/svg+xml,%3Csvg viewBox=%220 0 200 200%22 xmlns=%22http://www.w3.org/2000/svg%22%3E%3Cfilter id=%22noiseFilter%22%3E%3CfeTurbulence type=%22fractalNoise%22 baseFrequency=%220.75%22 numOctaves=%223%22 stitchTiles=%22stitch%22/%3E%3C/filter%3E%3Crect width=%22100%25%22 height=%22100%25%22 filter=%22url(%23noiseFilter)%22/%3E%3C/svg%3E")',
          opacity: 0.04, mixBlendMode: 'overlay'
        }} 
      />
      
      {/* Heavy vignette for focus */}
      <div 
        style={{
          position: 'absolute', inset: 0, zIndex: 4, pointerEvents: 'none',
          background: 'radial-gradient(125% 95% at 50% 42%, transparent 40%, rgba(2,4,6,0.85) 100%)'
        }}
      />

      <Canvas shadows dpr={[1, 2]}>
        <fog attach="fog" args={['#05070a', 20, 120]} />
        <ambientLight intensity={0.15} />
        <directionalLight 
          position={[50, 60, -50]} 
          intensity={0.6} 
          color="#e0231c" 
          castShadow 
        />
        <pointLight position={[-40, 20, 40]} intensity={0.4} color="#06b6d4" />

        <CinematicCamera />
        
        {districts.map(d => (
          <group key={d.id} position={[d.x, 0, d.z]}>
            <HostDistrict hostCount={3} />
          </group>
        ))}

        <RoadNetwork />
        
        <Stars radius={100} depth={50} count={3000} factor={3} saturation={0} fade speed={1} />
        
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.2, 0]} receiveShadow>
          <planeGeometry args={[600, 600]} />
          <meshStandardMaterial color="#020305" roughness={0.9} />
        </mesh>
      </Canvas>
    </div>
  );
}
