from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
from datetime import datetime, timezone
import uuid

class CloudVerseEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    simulation_id: str
    simulation_time: int
    wall_time: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str
    source: str
    severity: str = "info"  # info, warning, critical
    payload: Dict[str, Any] = Field(default_factory=dict)
