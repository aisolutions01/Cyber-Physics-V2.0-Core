# integration_sim.py
from Collector_Interface.collector_interface import CollectorInterface
import random

# Establishing complex
collector = CollectorInterface()

# Simulate a different systems
systems = ["ITSM", "IAM", "SIEM", "UEM"]

for i in range(10):
    event = {
        "type": random.choice(["login_fail", "policy_violation", "network_anomaly"]),
        "severity": random.randint(1, 5),
        "source_ip": f"10.0.0.{random.randint(1,255)}",
        "dest_ip": f"192.168.1.{random.randint(1,255)}",
        "user": f"user{random.randint(1,99)}",
        "cpu_load": round(random.uniform(0.1, 0.9), 3),
        "net_bytes": random.randint(1000, 100000),
    }
    collector.receive_event(event, random.choice(systems))

# Export Test
collected = collector.export_events()
print(f"\nCollected {len(collected)} unified incidents:")
for i in collected[:3]:
    print(i)
