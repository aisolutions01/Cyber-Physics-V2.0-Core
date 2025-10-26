# schema_incident.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Incident(BaseModel):
    timestamp: datetime
    incident_type: str
    severity: int = Field(..., ge=1, le=5)
    source: str
    source_ip: Optional[str]
    dest_ip: Optional[str]
    user: Optional[str]
    event_id: str
    cpu_load: Optional[float]
    net_bytes: Optional[int]
    system: Optional[str] = None  # e.g., 'ITSM', 'IAM', 'SIEM'
    description: Optional[str] = None
