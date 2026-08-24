import React, { useRef, useMemo, useEffect } from 'react';
import * as THREE from 'three';
import { Text } from '@react-three/drei';

const SECTORS = [
  { id: 0, label: "CENTRE", sx: 0, sz: 0 },
  { id: 1, label: "NORTH", sx: 0, sz: -100 },
  { id: 2, label: "SOUTH", sx: 0, sz: 100 },
  { id: 3, label: "EAST", sx: 100, sz: 0 },
  { id: 4, label: "WEST", sx: -100, sz: 0 }
];

export default function EnvironmentDecorations() {
  const treeTrunkRef = useRef<THREE.InstancedMesh>(null);
  const treeLeavesRef = useRef<THREE.InstancedMesh>(null);
  const mountainRef = useRef<THREE.InstancedMesh>(null);
  const cloudRef = useRef<THREE.InstancedMesh>(null);

  const numTrees = 800; // Dense forest
  const numMountains = 20;
  const numClouds = 20;

  // Generate trees
  const trees = useMemo(() => {
    const data = [];
    
    for (let i = 0; i < numTrees; i++) {
      let x = (Math.random() - 0.5) * 350;
      let z = (Math.random() - 0.5) * 350;
      
      let inCity = true;
      let nearHighway = true;
      while (inCity || nearHighway) {
        x = (Math.random() - 0.5) * 350;
        z = (Math.random() - 0.5) * 350;
        
        // Check if inside any 40x40 sector
        inCity = SECTORS.some(s => Math.abs(x - s.sx) < 22 && Math.abs(z - s.sz) < 22);
        // Check if on highway (highway width is 3.6, give it a buffer of 5)
        nearHighway = Math.abs(x) < 5 || Math.abs(z) < 5;
      }
      
      const scale = 0.5 + Math.random() * 1.5;
      data.push({ x, z, scale });
    }
    return data;
  }, []);

  // Generate mountains
  const mountains = useMemo(() => {
    const data = [];
    const radius = 170; // Move mountains very far out
    let i = 0;
    while (data.length < numMountains && i < 100) {
      const angle = (i / numMountains) * Math.PI * 2 + (Math.random() * 0.2);
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      i++;
      
      // Mountains shouldn't block the front view
      if (z > 70) continue; 
      
      const height = 30 + Math.random() * 30; // taller
      const width = 30 + Math.random() * 20; // wider
      data.push({ x, z, height, width });
    }
    return data;
  }, []);
  
  // Generate clouds
  const clouds = useMemo(() => {
    const data = [];
    for (let i = 0; i < numClouds; i++) {
      data.push({
        x: (Math.random() - 0.5) * 200,
        y: 20 + Math.random() * 15,
        z: (Math.random() - 0.5) * 200,
        scaleX: 4 + Math.random() * 4,
        scaleY: 1.5 + Math.random() * 2,
        scaleZ: 3 + Math.random() * 3,
      });
    }
    return data;
  }, []);

  const dummy = useMemo(() => new THREE.Object3D(), []);

  useEffect(() => {
    if (!treeTrunkRef.current || !treeLeavesRef.current || !mountainRef.current || !cloudRef.current) return;
    
    // Setup Trees
    for (let i = 0; i < numTrees; i++) {
      const t = trees[i];
      // Trunk
      dummy.position.set(t.x, 0.5 * t.scale, t.z);
      dummy.scale.set(t.scale * 0.4, t.scale, t.scale * 0.4);
      dummy.updateMatrix();
      treeTrunkRef.current.setMatrixAt(i, dummy.matrix);
      
      // Leaves
      dummy.position.set(t.x, 1.5 * t.scale, t.z);
      dummy.scale.set(t.scale * 1.5, t.scale * 1.5, t.scale * 1.5);
      dummy.updateMatrix();
      treeLeavesRef.current.setMatrixAt(i, dummy.matrix);
    }
    treeTrunkRef.current.instanceMatrix.needsUpdate = true;
    treeLeavesRef.current.instanceMatrix.needsUpdate = true;
    
    // Setup Mountains
    for (let i = 0; i < mountains.length; i++) {
      const m = mountains[i];
      dummy.position.set(m.x, m.height / 2 - 2, m.z);
      dummy.scale.set(m.width, m.height, m.width);
      dummy.updateMatrix();
      mountainRef.current.setMatrixAt(i, dummy.matrix);
    }
    mountainRef.current.instanceMatrix.needsUpdate = true;
    
    // Setup Clouds
    for (let i = 0; i < numClouds; i++) {
      const c = clouds[i];
      dummy.position.set(c.x, c.y, c.z);
      dummy.scale.set(c.scaleX, c.scaleY, c.scaleZ);
      dummy.updateMatrix();
      cloudRef.current.setMatrixAt(i, dummy.matrix);
    }
    cloudRef.current.instanceMatrix.needsUpdate = true;
    
  }, [trees, mountains, clouds, dummy]);

  return (
    <group>
      {/* Trees */}
      <instancedMesh ref={treeTrunkRef} args={[undefined, undefined, numTrees]}>
        <cylinderGeometry args={[0.5, 0.5, 1, 8]} />
        <meshStandardMaterial color="#8b5a2b" roughness={0.9} />
      </instancedMesh>
      
      <instancedMesh ref={treeLeavesRef} args={[undefined, undefined, numTrees]}>
        <coneGeometry args={[1, 2, 8]} />
        <meshStandardMaterial color="#228b22" roughness={0.8} />
      </instancedMesh>
      
      {/* Mountains */}
      <instancedMesh ref={mountainRef} args={[undefined, undefined, numMountains]}>
        <coneGeometry args={[1, 1, 4]} />
        <meshStandardMaterial color="#556b2f" roughness={1.0} flatShading={true} />
      </instancedMesh>
      
      {/* Clouds */}
      <instancedMesh ref={cloudRef} args={[undefined, undefined, numClouds]}>
        <sphereGeometry args={[1, 8, 8]} />
        <meshStandardMaterial color="#ffffff" transparent opacity={0.8} flatShading={true} />
      </instancedMesh>
      
      {/* Sector Labels */}
      {SECTORS.map((s) => (
        <Text
          key={s.id}
          position={[s.sx, 0.2, s.sz + 30]}
          rotation={[-Math.PI / 2, 0, 0]}
          fontSize={6}
          color="rgba(255, 255, 255, 0.3)"
          anchorX="center"
          anchorY="middle"
          fontWeight={800}
        >
          {s.label}
        </Text>
      ))}
    </group>
  );
}
