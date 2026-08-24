from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

from cloudverse.backend.events.bus import event_bus
from cloudverse.backend.models.events import CloudVerseEvent
from cloudverse.backend.bridge.simulator import SimulationBridge

app = FastAPI(title="CloudVerse API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_simulations = {}
active_connections = []

@app.on_event("startup")
async def startup_event():
    # Subscribe to all events and broadcast to websockets
    async def broadcast_event(event: CloudVerseEvent):
        if not active_connections:
            return
        message = event.model_dump_json()
        disconnected = []
        for connection in active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        for d in disconnected:
            if d in active_connections:
                active_connections.remove(d)

    event_bus.subscribe_all(broadcast_event)

@app.get("/")
def read_root():
    return {"status": "CloudVerse API is running"}

from cloudverse.backend.bridge.comparison_simulator import ComparisonBridge

@app.post("/api/simulation/start")
async def start_simulation():
    # Stop existing
    if "default" in active_simulations:
        active_simulations["default"].stop()

    sim = SimulationBridge()
    active_simulations["default"] = sim
    asyncio.create_task(sim.start())
    return {"status": "started", "simulation_id": sim.simulation_id, "mode": "normal"}

@app.post("/api/simulation/start_comparison")
async def start_comparison():
    # Stop existing
    if "default" in active_simulations:
        active_simulations["default"].stop()

    sim = ComparisonBridge()
    active_simulations["default"] = sim
    asyncio.create_task(sim.start())
    return {"status": "started", "simulation_id": sim.simulation_id, "mode": "comparison"}

@app.post("/api/simulation/stop")
async def stop_simulation():
    sim = active_simulations.get("default")
    if sim:
        sim.stop()
        return {"status": "stopped"}
    return {"status": "not_running"}

@app.post("/api/simulation/pause")
async def pause_simulation():
    sim = active_simulations.get("default")
    if sim:
        sim.pause()
        return {"status": "paused"}
    return {"status": "not_running"}

@app.post("/api/simulation/resume")
async def resume_simulation():
    sim = active_simulations.get("default")
    if sim:
        sim.resume()
        return {"status": "resumed"}
    return {"status": "not_running"}

@app.post("/api/simulation/speed/{speed}")
async def set_speed(speed: float):
    sim = active_simulations.get("default")
    if sim:
        sim.set_speed(speed)
        return {"status": "speed_updated", "speed": speed}
    return {"status": "not_running"}

# =============================================================================
# Challenge Injection Endpoints
# =============================================================================

@app.post("/api/challenges/host_failure")
async def inject_host_failure():
    """
    Take a random host district offline.
    Traffic in that host's sector drops to zero; cars reroute to active hosts.
    Emits HOST_FAILURE event with the failed host index (0–4).
    """
    sim = active_simulations.get("default")
    if not sim or not hasattr(sim, "inject_host_failure"):
        return {"status": "no_simulation_running"}

    result = sim.inject_host_failure()
    failed_host = result["failed_host"]

    # Emit event immediately so the frontend reacts right away
    event = CloudVerseEvent(
        simulation_id=sim.simulation_id,
        simulation_time=sim.step_idx,
        event_type="HOST_FAILURE",
        source="challenge_injector",
        severity="critical",
        payload={
            "failed_host": failed_host,
            "message": f"Host district {failed_host} has gone offline. Traffic rerouting..."
        }
    )
    await event_bus.publish(event)
    return {"status": "injected", "failed_host": failed_host}

@app.post("/api/challenges/host_failure/clear")
async def clear_host_failure():
    """Bring the failed host back online."""
    sim = active_simulations.get("default")
    if not sim or not hasattr(sim, "clear_host_failure"):
        return {"status": "no_simulation_running"}

    sim.clear_host_failure()
    event = CloudVerseEvent(
        simulation_id=sim.simulation_id,
        simulation_time=sim.step_idx,
        event_type="HOST_RECOVERED",
        source="challenge_injector",
        severity="info",
        payload={"message": "Failed host has been restored. Traffic resuming."}
    )
    await event_bus.publish(event)
    return {"status": "host_recovered"}

@app.post("/api/challenges/flash_sale")
async def inject_flash_sale():
    """
    Trigger a flash-sale traffic spike: request_rate × 4 for 20 simulation steps.
    Cars surge across all lanes to visualise the overload.
    """
    sim = active_simulations.get("default")
    if not sim or not hasattr(sim, "inject_flash_sale"):
        return {"status": "no_simulation_running"}

    result = sim.inject_flash_sale(multiplier=4.0, duration_steps=20)

    event = CloudVerseEvent(
        simulation_id=sim.simulation_id,
        simulation_time=sim.step_idx,
        event_type="FLASH_SALE_SPIKE",
        source="challenge_injector",
        severity="warning",
        payload={
            "multiplier": result["multiplier"],
            "duration_steps": result["duration_steps"],
            "message": "Flash sale detected! Request traffic surging 4×."
        }
    )
    await event_bus.publish(event)
    return {"status": "injected", **result}

@app.post("/api/challenges/memory_pressure")
async def inject_memory_pressure():
    """
    Apply memory pressure for 15 simulation steps.
    Processing slows down; cars move sluggishly to visualise degraded throughput.
    """
    sim = active_simulations.get("default")
    if not sim or not hasattr(sim, "inject_memory_pressure"):
        return {"status": "no_simulation_running"}

    result = sim.inject_memory_pressure(duration_steps=15)

    event = CloudVerseEvent(
        simulation_id=sim.simulation_id,
        simulation_time=sim.step_idx,
        event_type="MEMORY_PRESSURE",
        source="challenge_injector",
        severity="warning",
        payload={
            "duration_steps": result["duration_steps"],
            "message": "Memory pressure applied. VM throughput degraded."
        }
    )
    await event_bus.publish(event)
    return {"status": "injected", **result}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # wait for messages from client (if any)
            data = await websocket.receive_text()
            # client can send commands like start, pause, etc.
            command = json.loads(data)
            if command.get("action") == "ping":
                await websocket.send_text(json.dumps({"action": "pong"}))
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cloudverse.backend.main:app", host="0.0.0.0", port=8000, reload=True)
