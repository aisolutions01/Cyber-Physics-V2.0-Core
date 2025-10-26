# =========================
# data_generator.py
# =========================
import time
from datetime import datetime
from typing import Generator, Dict, Any, List
from Collector_Interface.collector_interface import CollectorInterface
import random
import requests

class IncidentGenerator:
    def __init__(self, collector: CollectorInterface = None):
        """
        Initialize generator.
        If no collector is provided, it will create one and simulate sample events.
        """
        self.collector = collector or CollectorInterface()
        self.events: List[Dict[str, Any]] = []
        self.index = 0

        # If no real-time events are passed, simulate input from multiple systems
        if not self.collector.events:
            self._simulate_sources()

    def _simulate_sources(self, n: int = 50):
        """Generate simulated streaming events from different IT systems."""
        systems = ["ITSM", "IAM", "SIEM", "UEM", "NDR", "EDR"]
        for i in range(n):
            raw_event = {
                "type": random.choice([
                    "login_fail", "policy_violation", "network_anomaly",
                    "privilege_escalation", "data_exfiltration"
                ]),
                "severity": random.randint(1, 5),
                "source_ip": f"10.0.{random.randint(1,255)}.{random.randint(1,255)}",
                "dest_ip": f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
                "user": f"user{random.randint(1,100):03d}",
                "cpu_load": round(random.uniform(0.05, 0.95), 3),
                "net_bytes": random.randint(1000, 100000),
            }
            self.collector.receive_event(raw_event, random.choice(systems))

        self.events = self.collector.export_events()

    def generate_incident(self) -> Dict[str, Any]:
        """Return one incident (on-the-fly) from the collector."""
        if not self.events:
            self._simulate_sources()

        if self.index >= len(self.events):
            self.index = 0  # restart loop for continuous stream

        event = self.events[self.index]
        self.index += 1

        # normalize timestamp
        event["timestamp"] = datetime.utcnow().isoformat()
        return event

    def stream(self, n: int = 10, delay: float = 0.5) -> Generator[Dict[str, Any], None, None]:
        """Yield n incidents sequentially, with optional delay to simulate live streaming."""
        for _ in range(n):
            yield self.generate_incident()
            time.sleep(delay)


if __name__ == "__main__":

    API_ENDPOINT = "http://127.0.0.1:5000/event"
    
    print("Starting dynamic on-the-fly data generation...")
    print(f"Streaming events to API at: {API_ENDPOINT}")
    
    gen = IncidentGenerator()

    for evt in gen.stream(n=1000, delay=0.5):
        try:
            response = requests.post(API_ENDPOINT, json=evt)
            
            if response.status_code == 200:
                print(f"Event accepted from {evt.get('source', 'N/A')}: {evt.get('incident_type', 'N/A')}")
            else:
                print(f"API Error: {response.status_code} - {response.text}")
        
        except requests.exceptions.ConnectionError:
            print(f"[❌] API Connection Failed. Is ingest_api.py running at {API_ENDPOINT}?")
            time.sleep(2)
        except Exception as e:
            print(f"An error occurred: {e}")
