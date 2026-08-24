import React, { useRef, useMemo, useEffect } from 'react';
import * as THREE from 'three';

export default function CitySkyline() {
  const wallsRef = useRef<THREE.InstancedMesh>(null);
  const roofsRef = useRef<THREE.InstancedMesh>(null);
  const windowsRef = useRef<THREE.InstancedMesh>(null);
  
  const numBuildings = 300; // 5 sectors * 20 blocks * 3 buildings per block
  const maxWindowsPerBuilding = 24; // Grid of windows for commercial, less for residential
  
  const buildings = useMemo(() => {
    const data = [];
    const sectors = [
      { sx: 0, sz: 0 },
      { sx: 0, sz: -100 },
      { sx: 0, sz: 100 },
      { sx: 100, sz: 0 },
      { sx: -100, sz: 0 }
    ];
    
    // The 5 VM blocks to skip
    const vmBlocks = [
      { r: 0, c: 0 },
      { r: -2, c: 0 },
      { r: 2, c: 0 },
      { r: 0, c: -2 },
      { r: 0, c: 2 }
    ];
    
    for (const sector of sectors) {
      for (let r = -2; r <= 2; r++) {
        for (let c = -2; c <= 2; c++) {
          // Skip if this is a VM block
          if (vmBlocks.some(b => b.r === r && b.c === c)) continue;
          
          const blockX = sector.sx + c * 12;
          const blockZ = sector.sz + r * 12;
        
        // Place 3 buildings in each corner block
        for (let i = 0; i < 3; i++) {
          const offsetX = (Math.random() - 0.5) * 6;
          const offsetZ = (Math.random() - 0.5) * 6;
          
          const height = 2 + Math.random() * 6;
          const width = 2.5 + Math.random() * 3.5;
          const depth = 2.5 + Math.random() * 3.5;
          
          const isCommercial = Math.random() > 0.6;
          
          data.push({ 
            x: blockX + offsetX, 
            z: blockZ + offsetZ, 
            height: isCommercial ? height * 2.5 : height, 
            width, 
            depth,
            isCommercial
          });
        }
        }
      }
    }
    return data;
  }, []);

  const dummyWall = useMemo(() => new THREE.Object3D(), []);
  const dummyRoof = useMemo(() => new THREE.Object3D(), []);
  const dummyWindow = useMemo(() => new THREE.Object3D(), []);

  const roofColors = useMemo(() => {
    const colors = [0xff4444, 0x4444ff, 0xffcc00, 0x44aa44, 0x884444, 0x555555];
    return Array.from({ length: numBuildings }, () => 
      new THREE.Color(colors[Math.floor(Math.random() * colors.length)])
    );
  }, []);

  const wallColors = useMemo(() => {
    // Light blue, light green, light yellow, peach, tan, light pink, white
    const colors = [0xadd8e6, 0x90ee90, 0xffffe0, 0xffdab9, 0xd2b48c, 0xffb6c1, 0xf5f5f5];
    return Array.from({ length: numBuildings }, () => 
      new THREE.Color(colors[Math.floor(Math.random() * colors.length)])
    );
  }, []);

  useEffect(() => {
    if (!wallsRef.current || !roofsRef.current || !windowsRef.current) return;
    
    let windowIdx = 0;
    
    for (let i = 0; i < numBuildings; i++) {
      const b = buildings[i];
      
      dummyWall.position.set(b.x, b.height / 2 - 0.2, b.z);
      dummyWall.scale.set(b.width, b.height, b.depth);
      dummyWall.updateMatrix();
      wallsRef.current.setMatrixAt(i, dummyWall.matrix);
      wallsRef.current.setColorAt(i, wallColors[i]);

      if (!b.isCommercial) {
        dummyRoof.position.set(b.x, b.height - 0.2 + 1.5/2, b.z);
        dummyRoof.scale.set(b.width * 0.7, 1.5, b.depth * 0.7);
        dummyRoof.rotation.y = Math.PI / 4;
        dummyRoof.updateMatrix();
        roofsRef.current.setColorAt(i, roofColors[i]);
      } else {
        dummyRoof.scale.set(0, 0, 0);
        dummyRoof.updateMatrix();
        roofsRef.current.setColorAt(i, roofColors[i]);
      }
      roofsRef.current.setMatrixAt(i, dummyRoof.matrix);
      
      // Windows
      let windowsAdded = 0;
      const wScale = 0.3;
      
      if (b.isCommercial) {
        // Grid of windows
        const rows = Math.floor(b.height / 1.2);
        const cols = Math.floor(b.width / 1.0);
        for (let r = 0; r < rows && windowsAdded < maxWindowsPerBuilding; r++) {
          for (let c = 0; c < cols && windowsAdded < maxWindowsPerBuilding; c++) {
            const wy = r * 1.2 + 0.5;
            const wx = b.x - b.width/2 + 0.5 + c * 1.0;
            // Front
            dummyWindow.position.set(wx, wy, b.z + b.depth/2 + 0.01);
            dummyWindow.scale.set(wScale, wScale, 0.05);
            dummyWindow.updateMatrix();
            windowsRef.current.setMatrixAt(windowIdx++, dummyWindow.matrix);
            windowsAdded++;
          }
        }
      } else {
        // 4 windows for houses
        const wy = b.height / 2;
        // Front
        dummyWindow.position.set(b.x - 0.5, wy, b.z + b.depth/2 + 0.01);
        dummyWindow.scale.set(wScale, wScale, 0.05);
        dummyWindow.updateMatrix();
        windowsRef.current.setMatrixAt(windowIdx++, dummyWindow.matrix);
        windowsAdded++;
        
        dummyWindow.position.set(b.x + 0.5, wy, b.z + b.depth/2 + 0.01);
        dummyWindow.updateMatrix();
        windowsRef.current.setMatrixAt(windowIdx++, dummyWindow.matrix);
        windowsAdded++;
        
        // Back
        dummyWindow.position.set(b.x - 0.5, wy, b.z - b.depth/2 - 0.01);
        dummyWindow.updateMatrix();
        windowsRef.current.setMatrixAt(windowIdx++, dummyWindow.matrix);
        windowsAdded++;
        
        dummyWindow.position.set(b.x + 0.5, wy, b.z - b.depth/2 - 0.01);
        dummyWindow.updateMatrix();
        windowsRef.current.setMatrixAt(windowIdx++, dummyWindow.matrix);
        windowsAdded++;
      }
      
      // Fill remaining with 0 scale
      while (windowsAdded < maxWindowsPerBuilding) {
        dummyWindow.scale.set(0, 0, 0);
        dummyWindow.updateMatrix();
        windowsRef.current.setMatrixAt(windowIdx++, dummyWindow.matrix);
        windowsAdded++;
      }
    }
    
    wallsRef.current.instanceMatrix.needsUpdate = true;
    roofsRef.current.instanceMatrix.needsUpdate = true;
    windowsRef.current.instanceMatrix.needsUpdate = true;
    if (roofsRef.current.instanceColor) roofsRef.current.instanceColor.needsUpdate = true;
    if (wallsRef.current.instanceColor) wallsRef.current.instanceColor.needsUpdate = true;
  }, [buildings, dummyWall, dummyRoof, roofColors, wallColors]);

  return (
    <group>
      <instancedMesh ref={wallsRef} args={[undefined, undefined, numBuildings]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial roughness={0.9} />
      </instancedMesh>

      <instancedMesh ref={roofsRef} args={[undefined, undefined, numBuildings]}>
        <coneGeometry args={[1, 1, 4]} />
        <meshStandardMaterial roughness={0.9} />
      </instancedMesh>
      
      {/* Background Building Windows */}
      <instancedMesh ref={windowsRef} args={[undefined, undefined, numBuildings * maxWindowsPerBuilding]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#334455" roughness={0.3} metalness={0.7} />
      </instancedMesh>
    </group>
  );
}
