# collector_interface.py
import json
from Unified_Incident_Schema.schema_incident import Incident
from datetime import datetime

class CollectorInterface:
    def __init__(self):
        self.events = []

    def receive_event(self, raw_event: dict, source_system: str):
   
        try:
            incident = Incident(
                timestamp=datetime.utcnow(),
                incident_type=raw_event.get("type", "unknown"),
                severity=raw_event.get("severity", 1),
                source=source_system,
                source_ip=raw_event.get("source_ip"),
                dest_ip=raw_event.get("dest_ip"),
                user=raw_event.get("user"),
                event_id=raw_event.get("id", f"evt-{len(self.events)}"),
                cpu_load=raw_event.get("cpu_load"),
                net_bytes=raw_event.get("net_bytes"),
                description=raw_event.get("description", "")
            )
            self.events.append(incident)
            print(f"Event accepted from {source_system}: {incident.incident_type}")
        except Exception as e:
            print(f"Failed to process event: {e}")

    def export_events(self):

        return [incident.dict() for incident in self.events]
