# # =========================
# # data_generator.py  (modified to use streams_1k.json)
# # =========================
# import json
# import time
# from datetime import datetime
# from typing import Generator, Dict, Any

# class IncidentGenerator:
#     def __init__(self, data_path: str = "streams_1k.json"):
#         """Initialize generator by loading the pre-generated streaming JSON file."""
#         with open(data_path, "r") as f:
#             self.events = json.load(f)
#         self.index = 0
#         self.total = len(self.events)

#     def generate_incident(self) -> Dict[str, Any]:
#         """Return one incident from the loaded JSON, emulating on-the-fly behavior."""
#         if self.index >= self.total:
#             # restart or stop when reaching the end
#             self.index = 0
#         event = self.events[self.index]
#         self.index += 1

#         # normalize timestamp if needed
#         if "timestamp" not in event:
#             event["timestamp"] = datetime.utcnow().isoformat()

#         return event

#     def stream(self, n: int = 10, delay: float = 0.5) -> Generator[Dict[str, Any], None, None]:
#         """Yield n incidents sequentially, with optional delay to simulate streaming."""
#         for _ in range(min(n, self.total)):
#             yield self.generate_incident()
#             time.sleep(delay)


# # Example usage:
# if __name__ == "__main__":
#     gen = IncidentGenerator("streams_1k.json")
#     for evt in gen.stream(n=5, delay=0.1):
#         print(evt)

























# =========================
# data_generator.py  (V1.1 - Now sending to API)
# =========================
import time
from datetime import datetime
from typing import Generator, Dict, Any, List
from Collector_Interface.collector_interface import CollectorInterface
import random
import requests  # (تغيير 1: استيراد مكتبة requests)

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


# (تغيير 2: تم تعديل هذا الجزء بالكامل)
if __name__ == "__main__":
    
    # تعريف عنوان الـ API الذي يجب إرسال البيانات إليه
    API_ENDPOINT = "http://127.0.0.1:5000/event"
    
    print("Starting dynamic on-the-fly data generation...")
    print(f"Streaming events to API at: {API_ENDPOINT}")
    
    gen = IncidentGenerator()
    
    # تأكد من أن الـ delay ليس سريعاً جداً، 0.5 ثانية جيد للبداية
    for evt in gen.stream(n=1000, delay=0.5):
        try:
            # هذا هو السطر الأهم: إرسال الحدث إلى الـ API
            response = requests.post(API_ENDPOINT, json=evt)
            
            if response.status_code == 200:
                # هذه الرسالة [✅] تعني الآن أن الـ API استلمها فعلاً
                print(f"[✅] Event accepted from {evt.get('source', 'N/A')}: {evt.get('incident_type', 'N/A')}")
            else:
                print(f"[❌] API Error: {response.status_code} - {response.text}")
        
        except requests.exceptions.ConnectionError:
            print(f"[❌] API Connection Failed. Is ingest_api.py running at {API_ENDPOINT}?")
            time.sleep(2) # الانتظار قبل المحاولة مرة أخرى
        except Exception as e:
            print(f"An error occurred: {e}")
