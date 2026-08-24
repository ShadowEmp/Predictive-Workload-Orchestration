import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// ─── Props ───────────────────────────────────────────────────────────────────

export interface ChallengeState {
  failedHost: number;          // -1 = none; 0-4 = offline district
  flashSaleActive: boolean;
  memoryPressureActive: boolean;
};

interface TrafficSystemProps {
  /** Per-host normalised request-traffic (0–1) for each of the 5 host districts. */
  hostTraffic: number[];
  /** Global normalised request-rate (0–1). */
  requestRate: number;
  /** Active VM count (used for routing decisions). */
  activeVmCount: number;
  /** Current challenge state (affects car behaviour). */
  challengeState: ChallengeState;
}

// ─── City geometry constants ──────────────────────────────────────────────────

/**
 * 5 host districts (sectors) matching the VMInstances sector layout.
 */
const SECTORS = [
  { id: 0, sx: 0,    sz: 0   },  // Centre
  { id: 1, sx: 0,    sz: -100 }, // North
  { id: 2, sx: 0,    sz: 100  }, // South
  { id: 3, sx: 100,  sz: 0   },  // East
  { id: 4, sx: -100, sz: 0   },  // West
];

const BLOCKS = [
  { r: 0,  c: 0  },
  { r: -1, c: 0  },
  { r: 1,  c: 0  },
  { r: 0,  c: -1 },
  { r: 0,  c: 1  },
];

// Precompute VM positions (100 total, 20 per sector)
const VM_POSITIONS: THREE.Vector3[] = [];
{
  const sectorSpots: THREE.Vector3[][] = SECTORS.map(() => []);
  for (let s = 0; s < SECTORS.length; s++) {
    for (const b of BLOCKS) {
      const bx = SECTORS[s].sx + b.c * 12;
      const bz = SECTORS[s].sz + b.r * 12;
      sectorSpots[s].push(new THREE.Vector3(bx - 2, 0.15, bz - 2));
      sectorSpots[s].push(new THREE.Vector3(bx + 2, 0.15, bz - 2));
      sectorSpots[s].push(new THREE.Vector3(bx - 2, 0.15, bz + 2));
      sectorSpots[s].push(new THREE.Vector3(bx + 2, 0.15, bz + 2));
    }
  }
  for (let i = 0; i < 20; i++) {
    for (let s = 0; s < SECTORS.length; s++) {
      VM_POSITIONS.push(sectorSpots[s][i]);
    }
  }
}

// ─── Car data ─────────────────────────────────────────────────────────────────

/**
 * Max pool of instanced cars. We activate a subset each frame based on traffic.
 *
 * Per-host budget:
 *   carsForHost(h) = Math.floor(hostTraffic[h] * MAX_CARS_PER_HOST)
 * Total rendered  = sum of all carsForHost (capped at MAX_CARS)
 */
const MAX_CARS_PER_HOST = 80;   // max cars visible per district at 100% traffic
const MAX_CARS = MAX_CARS_PER_HOST * SECTORS.length; // 400 total pool

const HIGHWAY_EXTENT = 140;

interface CarData {
  position: THREE.Vector3;
  speed: number;
  baseSpeed: number;   // speed before challenge modifiers
  targetVm: number;
  targetX: number;
  targetZ: number;
  phase: number;
  routeX: number;
  routeZ: number;
  scaleX: number;
  scaleY: number;
  scaleZ: number;
  hostIdx: number;     // which district this car belongs to (-1 = highway)
}

function resetCar(
  car: CarData,
  hostIdx: number,
  hostTraffic: number[],
  activeVmCount: number,
  challengeState: ChallengeState,
) {
  const isHighway = Math.random() > 0.65; // ~35% highway ambient

  if (isHighway) {
    car.hostIdx = -1;
    car.targetVm = -1;
    const direction = Math.random() > 0.5 ? 1 : -1;
    const traffic = hostTraffic[hostIdx] ?? 0.3;
    car.baseSpeed = (Math.random() * 0.25 + 0.1) * (0.5 + traffic);
    car.speed = car.baseSpeed;

    const highwayOffsets = [0, 100, -100];
    const hwOffset = highwayOffsets[Math.floor(Math.random() * highwayOffsets.length)];

    if (Math.random() > 0.5) {
      car.position.set(-HIGHWAY_EXTENT * direction, 0.15, hwOffset);
      car.targetX = HIGHWAY_EXTENT * direction;
      car.phase = 3;
    } else {
      car.position.set(hwOffset, 0.15, -HIGHWAY_EXTENT * direction);
      car.targetZ = HIGHWAY_EXTENT * direction;
      car.phase = 4;
    }
  } else {
    // Local sector traffic
    car.hostIdx = hostIdx;

    // VMs for this host sector = indices [hostIdx, hostIdx+5, hostIdx+10, ...]
    const vmIndicesForHost: number[] = [];
    for (let i = hostIdx; i < Math.min(100, activeVmCount * 5); i += SECTORS.length) {
      vmIndicesForHost.push(i);
    }
    if (vmIndicesForHost.length === 0) vmIndicesForHost.push(hostIdx);

    const vmIdx = vmIndicesForHost[Math.floor(Math.random() * vmIndicesForHost.length)];
    const target = VM_POSITIONS[Math.min(vmIdx, VM_POSITIONS.length - 1)];
    car.targetVm = vmIdx;

    const sector = SECTORS[hostIdx];
    const localC = Math.round((target.x - sector.sx) / 12);
    const localR = Math.round((target.z - sector.sz) / 12);

    car.routeX = sector.sx + localC * 12 + (Math.random() > 0.5 ? 6 : -6);
    car.routeZ = sector.sz + localR * 12 + (Math.random() > 0.5 ? 6 : -6);

    const startZ = sector.sz + (Math.random() > 0.5 ? -30 : 30);
    car.position.set(car.routeX, 0.15, startZ);

    const traffic = hostTraffic[hostIdx] ?? 0.3;
    car.baseSpeed = (Math.random() * 0.1 + 0.04) * (0.4 + traffic * 1.2);
    car.speed = car.baseSpeed;
    car.targetX = target.x;
    car.targetZ = target.z;
    car.phase = 0;
  }

  // Car size variant
  const t = Math.random();
  if (t > 0.8) {
    car.scaleX = 0.4; car.scaleY = 0.5; car.scaleZ = 0.9;
  } else if (t > 0.5) {
    car.scaleX = 0.35; car.scaleY = 0.4; car.scaleZ = 0.7;
  } else {
    car.scaleX = 0.3; car.scaleY = 0.25; car.scaleZ = 0.6;
  }
}

// ─── Car colour palette ────────────────────────────────────────────────────────

/** District-tinted colours: each host has its own hue family. */
const HOST_COLORS: number[][] = [
  [0x00cfff, 0x0088cc, 0x00ffee],   // Centre  – cyan
  [0xff6600, 0xff9900, 0xffcc00],   // North   – amber
  [0x22cc55, 0x00ff88, 0x66ff44],   // South   – green
  [0xcc44ff, 0xff44cc, 0x8800ff],   // East    – purple
  [0xff3344, 0xff6688, 0xff0000],   // West    – red
];
const HIGHWAY_COLORS = [0xffffff, 0xdddddd, 0xaaaaaa];

// ─── Component ────────────────────────────────────────────────────────────────

export default function TrafficSystem({
  hostTraffic,
  requestRate,
  activeVmCount,
  challengeState,
}: TrafficSystemProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  // Sanitise incoming arrays
  const safeHostTraffic = useMemo(() => {
    const t = Array.isArray(hostTraffic) && hostTraffic.length === 5
      ? hostTraffic
      : [requestRate, requestRate, requestRate, requestRate, requestRate];
    return t.map((v) => Math.max(0, Math.min(1, v ?? 0)));
  }, [hostTraffic, requestRate]);

  // Cars per host — proportional to that host's request traffic
  const carsPerHost = useMemo(
    () =>
      SECTORS.map((s) => {
        if (s.id === challengeState.failedHost) return 0; // failed host gets no cars
        const traffic = safeHostTraffic[s.id] ?? 0;
        let count = Math.max(4, Math.floor(traffic * MAX_CARS_PER_HOST));
        if (challengeState.flashSaleActive) count = Math.min(MAX_CARS_PER_HOST, count * 2);
        return count;
      }),
    [safeHostTraffic, challengeState],
  );

  const totalActiveCars = carsPerHost.reduce((a, b) => a + b, 0);

  // One-time allocation of all car objects
  const carsData = useMemo<CarData[]>(() => {
    const data: CarData[] = [];
    for (let i = 0; i < MAX_CARS; i++) {
      const hostIdx = i % SECTORS.length;
      const car: CarData = {
        position: new THREE.Vector3(0, -20, 0),
        speed: 0,
        baseSpeed: 0,
        targetVm: 0,
        targetX: 0,
        targetZ: 0,
        phase: 0,
        routeX: 0,
        routeZ: 0,
        scaleX: 1,
        scaleY: 1,
        scaleZ: 1,
        hostIdx,
      };
      resetCar(car, hostIdx, [0.3, 0.3, 0.3, 0.3, 0.3], 5, {
        failedHost: -1,
        flashSaleActive: false,
        memoryPressureActive: false,
      });
      data.push(car);
    }
    return data;
  }, []);

  // Assign colors once, tinted by host district
  const carColors = useMemo(() => {
    return Array.from({ length: MAX_CARS }, (_, i) => {
      const hostIdx = i % SECTORS.length;
      const palette = HOST_COLORS[hostIdx] ?? HIGHWAY_COLORS;
      return new THREE.Color(palette[Math.floor(Math.random() * palette.length)]);
    });
  }, []);

  useEffect(() => {
    if (!meshRef.current) return;
    for (let i = 0; i < MAX_CARS; i++) {
      meshRef.current.setColorAt(i, carColors[i]);
    }
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
  }, [carColors]);

  const dummy = useMemo(() => new THREE.Object3D(), []);

  useFrame(() => {
    if (!meshRef.current) return;

    const safeVmCount = Math.max(1, activeVmCount);

    /**
     * We lay out the MAX_CARS pool in blocks of MAX_CARS_PER_HOST,
     * one block per host district, in sector order.
     * Within each block, indices 0..carsPerHost[h]-1 are ACTIVE,
     * the rest are hidden below the ground.
     */
    let poolOffset = 0;

    for (let h = 0; h < SECTORS.length; h++) {
      const activeCarsThisHost = carsPerHost[h];
      const traffic = safeHostTraffic[h];

      for (let j = 0; j < MAX_CARS_PER_HOST; j++) {
        const i = poolOffset + j;
        if (i >= MAX_CARS) break;

        if (j < activeCarsThisHost) {
          const car = carsData[i];

          // ── Apply challenge speed modifiers ──
          let speedMod = 1.0;
          if (challengeState.memoryPressureActive) speedMod *= 0.35; // sluggish
          if (challengeState.flashSaleActive) speedMod *= 1.6;        // frantic
          car.speed = car.baseSpeed * speedMod;

          // ── Reroute if car was heading to the failed host's VMs ──
          if (
            challengeState.failedHost >= 0 &&
            car.hostIdx === challengeState.failedHost
          ) {
            // Reassign to a random healthy host
            const healthyHosts = SECTORS.filter((s) => s.id !== challengeState.failedHost);
            const newHost = healthyHosts[Math.floor(Math.random() * healthyHosts.length)];
            resetCar(car, newHost.id, safeHostTraffic, safeVmCount, challengeState);
          }

          // ── Routing phases ──
          if (car.phase === 0) {
            const diffZ = car.routeZ - car.position.z;
            if (Math.abs(diffZ) < 0.5) {
              car.position.z = car.routeZ;
              car.phase = 1;
            } else {
              car.position.z += car.speed * Math.sign(diffZ);
            }
          } else if (car.phase === 1) {
            const diffX = car.targetX - car.position.x;
            if (Math.abs(diffX) < 0.5) {
              car.position.x = car.targetX;
              car.phase = 2;
            } else {
              car.position.x += car.speed * Math.sign(diffX);
            }
          } else if (car.phase === 2) {
            const diffZ = car.targetZ - car.position.z;
            if (Math.abs(diffZ) < 0.5) {
              resetCar(car, h, safeHostTraffic, safeVmCount, challengeState);
            } else {
              car.position.z += car.speed * 0.6 * Math.sign(diffZ);
            }
          } else if (car.phase === 3) {
            const diffX = car.targetX - car.position.x;
            if (Math.abs(diffX) < 0.5) {
              resetCar(car, h, safeHostTraffic, safeVmCount, challengeState);
            } else {
              car.position.x += car.speed * Math.sign(diffX);
            }
          } else if (car.phase === 4) {
            const diffZ = car.targetZ - car.position.z;
            if (Math.abs(diffZ) < 0.5) {
              resetCar(car, h, safeHostTraffic, safeVmCount, challengeState);
            } else {
              car.position.z += car.speed * Math.sign(diffZ);
            }
          }

          // ── Hide car if it is physically inside the failed sector bounds ──
          let hideCar = false;
          if (challengeState.failedHost >= 0) {
            const fs = SECTORS[challengeState.failedHost];
            if (
              Math.abs(car.position.x - fs.sx) < 40 &&
              Math.abs(car.position.z - fs.sz) < 40
            ) {
              hideCar = true;
            }
          }

          if (hideCar) {
            dummy.position.set(0, -20, 0);
            dummy.scale.set(0.001, 0.001, 0.001);
          } else {
            dummy.position.copy(car.position);
            dummy.position.y = car.scaleY / 2 + 0.05;
            dummy.scale.set(car.scaleX, car.scaleY, car.scaleZ);

            // Face direction of travel
            if (car.phase === 0 || car.phase === 2 || car.phase === 4) {
              dummy.rotation.y = 0;
            } else {
              dummy.rotation.y = Math.PI / 2;
            }
          }
        } else {
          // Hide inactive car under the ground
          dummy.position.set(0, -20, 0);
          dummy.scale.set(0.001, 0.001, 0.001);
        }

        dummy.updateMatrix();
        meshRef.current.setMatrixAt(i, dummy.matrix);
      }

      poolOffset += MAX_CARS_PER_HOST;
    }

    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, MAX_CARS]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial metalness={0.5} roughness={0.4} />
    </instancedMesh>
  );
}
