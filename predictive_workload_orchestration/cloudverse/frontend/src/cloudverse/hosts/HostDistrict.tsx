import React from 'react';

interface HostDistrictProps {
  hostCount: number;
}

export default function HostDistrict({ hostCount }: HostDistrictProps) {
  // Let's create Host pads that sit underneath VM groups.
  // Each host supports 10 VMs.
  
  return (
    <group>
      {Array.from({ length: hostCount }).map((_, i) => {
        const yPos = -0.15;
        const zPos = i * 8 - 6;
        
        return (
          <group key={`host-district-${i}`} position={[0, yPos, zPos]}>
            {/* The primary concrete server platform */}
            <mesh>
              <boxGeometry args={[22, 0.3, 7.5]} />
              <meshStandardMaterial 
                color="#666666" 
                metalness={0.1} 
                roughness={0.8} 
              />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}
