import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface VMInstancesProps {
  count: number;
  cpuUtil?: number;
}

export default function VMInstances({ count, cpuUtil = 0 }: VMInstancesProps) {
  const wallsRef = useRef<THREE.InstancedMesh>(null);
  const roofsRef = useRef<THREE.InstancedMesh>(null);
  const windowsRef = useRef<THREE.InstancedMesh>(null);
  const maxVms = 100;
  const windowsPerVm = 4; // 2 on front, 2 on back
  
  const positions = useMemo(() => {
    const pos = [];
    const sectors = [
      { sx: 0, sz: 0 },
      { sx: 0, sz: -100 },
      { sx: 0, sz: 100 },
      { sx: 100, sz: 0 },
      { sx: -100, sz: 0 }
    ];
    
    // In each sector, we use 5 blocks in a wide '+' pattern across the 5x5 grid
    const blocks = [
      { r: 0, c: 0 },
      { r: -2, c: 0 },
      { r: 2, c: 0 },
      { r: 0, c: -2 },
      { r: 0, c: 2 }
    ];
    
    const sectorSpots: THREE.Vector3[][] = sectors.map(() => []);
    
    for (let s = 0; s < sectors.length; s++) {
      for (const block of blocks) {
        const blockX = sectors[s].sx + block.c * 12;
        const blockZ = sectors[s].sz + block.r * 12;
        sectorSpots[s].push(new THREE.Vector3(blockX - 2, 0, blockZ - 2));
        sectorSpots[s].push(new THREE.Vector3(blockX + 2, 0, blockZ - 2));
        sectorSpots[s].push(new THREE.Vector3(blockX - 2, 0, blockZ + 2));
        sectorSpots[s].push(new THREE.Vector3(blockX + 2, 0, blockZ + 2));
      }
    }
    
    for (let i = 0; i < 20; i++) {
      for (let s = 0; s < sectors.length; s++) {
        pos.push(sectorSpots[s][i]);
      }
    }
    
    return pos;
  }, []);

  const currentHeights = useRef<Float32Array>(new Float32Array(maxVms));
  const dummyWall = useMemo(() => new THREE.Object3D(), []);
  const dummyRoof = useMemo(() => new THREE.Object3D(), []);
  const dummyWindow = useMemo(() => new THREE.Object3D(), []);

  // Pre-generate roof colors
  const roofColors = useMemo(() => {
    const colors = [0xff4444, 0x4444ff, 0xffcc00, 0x44aa44, 0xff8800];
    return Array.from({ length: maxVms }, () => 
      new THREE.Color(colors[Math.floor(Math.random() * colors.length)])
    );
  }, []);

  // Pre-generate wall colors (more saturated so they don't wash out)
  const wallColors = useMemo(() => {
    // Light blue, light green, light yellow, peach, tan, light pink, white
    const colors = [0xadd8e6, 0x90ee90, 0xffffe0, 0xffdab9, 0xd2b48c, 0xffb6c1, 0xf5f5f5];
    return Array.from({ length: maxVms }, () => 
      new THREE.Color(colors[Math.floor(Math.random() * colors.length)])
    );
  }, []);

  // Pre-generate random sizes for each house to add variety
  const vmSizes = useMemo(() => {
    return Array.from({ length: maxVms }, () => ({
      height: 1.2 + Math.random() * 2.0, // 1.2 to 3.2
      width: 1.5 + Math.random() * 1.2,  // 1.5 to 2.7
      depth: 1.5 + Math.random() * 1.2   // 1.5 to 2.7
    }));
  }, []);

  const dormantWallColor = useMemo(() => new THREE.Color('#bbbbbb'), []); // Light concrete pad
  const dormantRoofColor = useMemo(() => new THREE.Color('#999999'), []); // Slightly darker concrete
  const dormantWindowColor = useMemo(() => new THREE.Color('#111111'), []);
  const activeWindowColor = useMemo(() => new THREE.Color('#223344'), []);

  useEffect(() => {
    // Initial color setup is now handled in useFrame dynamically
  }, []);

  useFrame(() => {
    if (!wallsRef.current || !roofsRef.current || !windowsRef.current) return;
    
    let windowIdx = 0;
    
    for (let i = 0; i < maxVms; i++) {
      const isActive = i < count;
      const size = vmSizes[i];
      // Inactive VMs are rendered as smaller gray houses (0.8 height)
      const baseTargetHeight = isActive ? size.height : 0.8;
      
      // Dynamically update colors based on active state
      wallsRef.current.setColorAt(i, isActive ? wallColors[i] : dormantWallColor);
      roofsRef.current.setColorAt(i, isActive ? roofColors[i] : dormantRoofColor);
      
      currentHeights.current[i] += (baseTargetHeight - currentHeights.current[i]) * 0.1;
      const h = currentHeights.current[i];
      
      if (h > 0.01) {
        // Walls
        dummyWall.position.set(positions[i].x, h / 2, positions[i].z);
        dummyWall.scale.set(size.width, h, size.depth);
        dummyWall.updateMatrix();
        wallsRef.current.setMatrixAt(i, dummyWall.matrix);

        // Roof (Cone)
        dummyRoof.position.set(positions[i].x, h + 0.5, positions[i].z);
        dummyRoof.scale.set(size.width / 1.4, 1, size.depth / 1.4);
        dummyRoof.rotation.y = Math.PI / 4; // Rotate 45deg so edges align with box
        dummyRoof.updateMatrix();
        roofsRef.current.setMatrixAt(i, dummyRoof.matrix);
        
        // Windows
        const wx = positions[i].x;
        const wy = h / 2 + 0.1; // middle height
        const wz = positions[i].z;
        const wScale = 0.3; // window size
        
        const wXOff = size.width * 0.25;
        const wZOff = size.depth / 2 + 0.01;
        
        // Front windows
        dummyWindow.position.set(wx - wXOff, wy, wz + wZOff);
        dummyWindow.scale.set(wScale, wScale, 0.05);
        dummyWindow.updateMatrix();
        windowsRef.current.setMatrixAt(windowIdx, dummyWindow.matrix);
        windowsRef.current.setColorAt(windowIdx++, isActive ? activeWindowColor : dormantWindowColor);
        
        dummyWindow.position.set(wx + wXOff, wy, wz + wZOff);
        dummyWindow.updateMatrix();
        windowsRef.current.setMatrixAt(windowIdx, dummyWindow.matrix);
        windowsRef.current.setColorAt(windowIdx++, isActive ? activeWindowColor : dormantWindowColor);
        
        // Back windows
        dummyWindow.position.set(wx - wXOff, wy, wz - wZOff);
        dummyWindow.updateMatrix();
        windowsRef.current.setMatrixAt(windowIdx, dummyWindow.matrix);
        windowsRef.current.setColorAt(windowIdx++, isActive ? activeWindowColor : dormantWindowColor);
        
        dummyWindow.position.set(wx + wXOff, wy, wz - wZOff);
        dummyWindow.updateMatrix();
        windowsRef.current.setMatrixAt(windowIdx, dummyWindow.matrix);
        windowsRef.current.setColorAt(windowIdx++, isActive ? activeWindowColor : dormantWindowColor);
      } else {
        // Fallback for extremely tiny sizes during transition
        for (let w = 0; w < windowsPerVm; w++) {
          dummyWindow.scale.set(0, 0, 0);
          dummyWindow.updateMatrix();
          windowsRef.current.setMatrixAt(windowIdx++, dummyWindow.matrix);
        }
      }
    }
    
    wallsRef.current.instanceMatrix.needsUpdate = true;
    roofsRef.current.instanceMatrix.needsUpdate = true;
    windowsRef.current.instanceMatrix.needsUpdate = true;
    
    if (wallsRef.current.instanceColor) wallsRef.current.instanceColor.needsUpdate = true;
    if (roofsRef.current.instanceColor) roofsRef.current.instanceColor.needsUpdate = true;
    if (windowsRef.current.instanceColor) windowsRef.current.instanceColor.needsUpdate = true;
  });

  return (
    <group>
      <instancedMesh ref={wallsRef} args={[undefined, undefined, maxVms]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial roughness={0.9} />
      </instancedMesh>
      
      <instancedMesh ref={roofsRef} args={[undefined, undefined, maxVms]}>
        <coneGeometry args={[1.2, 1, 4]} />
        <meshStandardMaterial roughness={0.8} />
      </instancedMesh>
      
      {/* House Windows */}
      <instancedMesh ref={windowsRef} args={[undefined, undefined, maxVms * windowsPerVm]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#223344" roughness={0.2} metalness={0.8} />
      </instancedMesh>
    </group>
  );
}
