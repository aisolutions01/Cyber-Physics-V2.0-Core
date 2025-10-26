# =========================
# evaluation.py
# =========================
import json
import random
import time
from datetime import datetime
import numpy as np


class IncidentGenerator:
    def __init__(self, json_path="streams_1k.json", noise=False, stream_delay=0.0):
        """
        Load pre-generated streaming incidents.
        noise: adds small gaussian noise to numeric features for variability.
        stream_delay: optional delay between yielded samples.
        """
        with open(json_path, "r") as f:
            self.events = json.load(f)

        self.noise = noise
        self.stream_delay = stream_delay
        self.index = 0
        self.total = len(self.events)

    def generate_incident(self):
        """Return one incident dict from the dataset."""
        if self.index >= self.total:
            self.index = 0
        event = self.events[self.index]
        self.index += 1
        return event

    def stream(self, n=10):
        """Yield n raw incident dicts sequentially."""
        for _ in range(min(n, self.total)):
            yield self.generate_incident()
            if self.stream_delay > 0:
                time.sleep(self.stream_delay)

    def generate(self):
        """
        Yield (x, y, inc) tuples for model training/evaluation.
        x: numpy feature vector
        y: label (0=normal, 1=incident)
        inc: raw event dict
        """
        for inc in self.events:
            # --- feature extraction ---
            # اختيار ميزات عددية لتمثيل الحادثة
            features = [
                inc.get("severity", 0),
                inc.get("cpu_load", 0.0),
                inc.get("net_bytes", 0),
            ]

            # إضافة تذبذب بسيط لزيادة تنوع البيانات
            if self.noise:
                features = [f + random.gauss(0, 0.05) if isinstance(f, (int, float)) else f for f in features]

            x = np.array(features, dtype=float)

            # --- labeling logic ---
            # اعتبر الحالات الخطيرة كـ "incident"
            y = 1 if inc.get("severity", 1) >= 3 else 0

            yield x, y, inc

            if self.stream_delay > 0:
                time.sleep(self.stream_delay)


# --- Quick test ---
if __name__ == "__main__":
    gen = IncidentGenerator("streams_1k.json", noise=True)
    for i, (x, y, inc) in enumerate(gen.generate()):
        print(f"{i:03d} | y={y} | x={x} | {inc['incident_type']}")
        if i >= 4:
            break
