import React from 'react';

export default function RoadNetwork() {
  const roadWidth = 2.4;
  const highwayWidth = 3.6;
  const localExtent = 30; // From center of sector, roads go to +/- 30 for a 5x5 grid
  const highwayExtent = 140; 
  
  const sectors = [
    { sx: 0, sz: 0 },
    { sx: 0, sz: -100 },
    { sx: 0, sz: 100 },
    { sx: 100, sz: 0 },
    { sx: -100, sz: 0 }
  ];

  // Local grid roads for 5x5 blocks
  const localRoadOffsets = [-30, -18, -6, 6, 18, 30];

  return (
    <group position={[0, -0.19, 0]}>
      
      {/* 1. Local Sector Roads */}
      {sectors.map((sector, sIdx) => (
        <group key={`sector-${sIdx}`}>
          {/* Vertical local roads */}
          {localRoadOffsets.map((offset, rIdx) => (
            <mesh key={`vl-${rIdx}`} position={[sector.sx + offset, 0.01, sector.sz]}>
              <boxGeometry args={[roadWidth, 0.02, localExtent * 2]} />
              <meshStandardMaterial color="#555555" roughness={0.9} />
            </mesh>
          ))}
          {/* Horizontal local roads */}
          {localRoadOffsets.map((offset, rIdx) => (
            <mesh key={`hl-${rIdx}`} position={[sector.sx, 0.01, sector.sz + offset]}>
              <boxGeometry args={[localExtent * 2, 0.02, roadWidth]} />
              <meshStandardMaterial color="#555555" roughness={0.9} />
            </mesh>
          ))}
          
          {/* Local intersections */}
          {localRoadOffsets.map((xOff) => (
            localRoadOffsets.map((zOff) => (
              <mesh key={`int-${xOff}-${zOff}`} position={[sector.sx + xOff, 0.02, sector.sz + zOff]}>
                <boxGeometry args={[roadWidth, 0.001, roadWidth]} />
                <meshStandardMaterial color="#444444" roughness={0.9} />
              </mesh>
            ))
          ))}
        </group>
      ))}

      {/* 2. Global Highways (Central Cross + Outer Ring Road) */}
      
      {/* Central Vertical Highway */}
      <mesh position={[0, 0.015, 0]}>
        <boxGeometry args={[highwayWidth, 0.02, highwayExtent * 2]} />
        <meshStandardMaterial color="#333333" roughness={0.8} />
      </mesh>
      
      {/* Central Horizontal Highway */}
      <mesh position={[0, 0.015, 0]}>
        <boxGeometry args={[highwayExtent * 2, 0.02, highwayWidth]} />
        <meshStandardMaterial color="#333333" roughness={0.8} />
      </mesh>
      
      {/* Outer Ring Road Vertical (Left & Right) */}
      <mesh position={[-100, 0.015, 0]}>
        <boxGeometry args={[highwayWidth, 0.02, 200]} />
        <meshStandardMaterial color="#333333" roughness={0.8} />
      </mesh>
      <mesh position={[100, 0.015, 0]}>
        <boxGeometry args={[highwayWidth, 0.02, 200]} />
        <meshStandardMaterial color="#333333" roughness={0.8} />
      </mesh>
      
      {/* Outer Ring Road Horizontal (Top & Bottom) */}
      <mesh position={[0, 0.015, -100]}>
        <boxGeometry args={[200, 0.02, highwayWidth]} />
        <meshStandardMaterial color="#333333" roughness={0.8} />
      </mesh>
      <mesh position={[0, 0.015, 100]}>
        <boxGeometry args={[200, 0.02, highwayWidth]} />
        <meshStandardMaterial color="#333333" roughness={0.8} />
      </mesh>
      
      {/* Highway Intersections */}
      {[-100, 0, 100].map(x => (
        [-100, 0, 100].map(z => (
          <mesh key={`hw-int-${x}-${z}`} position={[x, 0.03, z]}>
            <boxGeometry args={[highwayWidth, 0.001, highwayWidth]} />
            <meshStandardMaterial color="#222222" roughness={0.9} />
          </mesh>
        ))
      ))}
      
    </group>
  );
}
