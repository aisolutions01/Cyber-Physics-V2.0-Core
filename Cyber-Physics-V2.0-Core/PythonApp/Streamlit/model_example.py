# model_example.py
import numpy as np
import json
import time
import os
import collections
import redis
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score

# =============================
# GLOBAL STATE
# =============================
WINDOW_SIZE = 1000
scaler = StandardScaler()
model = SGDClassifier(loss="log_loss", random_state=42)
trained = False

X_hist = collections.deque(maxlen=WINDOW_SIZE)
y_hist = collections.deque(maxlen=WINDOW_SIZE)

metrics = {"accuracy": 0, "recall_pos": 0, "recall_neg": 0, "events": 0, "events_per_s": 0}
start_time = time.time()

redis_host = os.environ.get("REDIS_HOST", "localhost")
r = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True)
METRICS_KEY = "cyber_physics_metrics"

# =============================
# HELPER FUNCTIONS
# =============================

def _safe_float(value, default=0.0):

    try:
        return float(value)
    except (ValueError, TypeError, SystemError):

        return default

def incident_to_features(inc):
    """(مُعدّل) Convert an incident dict to numeric feature vector."""
    s = _safe_float(inc.get("severity"), default=1.0)
    c = _safe_float(inc.get("cpu_load"), default=0.0)
    n = _safe_float(inc.get("net_bytes"), default=0.0)
    

    n = max(n, 0.0) 
    
    return np.array([s, c, n, s * c, c * np.log1p(n)], dtype=float)


# =============================
# MAIN UPDATE FUNCTION
# =============================
def update_model_metrics(incident):
    """
    (مُعدّل) Incrementally update the model using one incident.
    """
    global trained, X_hist, y_hist, metrics

    try:
        # Convert incoming event
        x = incident_to_features(incident)
        y = incident_to_label(incident)

        X_hist.append(x)
        y_hist.append(y)

        total_events = metrics["events"] + 1
        
        if len(X_hist) > 5:  # (العتبة 5)
            
            X = np.vstack(X_hist)
            y = np.array(y_hist)


            if not trained:
                scaler.fit(X)
                model.partial_fit(scaler.transform(X), y, classes=np.array([0, 1]))
                trained = True
            else:
                model.partial_fit(scaler.transform([x]), np.array([y]))

            y_pred = model.predict(scaler.transform(X))
            
            acc = accuracy_score(y, y_pred)
            rec_p = recall_score(y, y_pred, pos_label=1, zero_division=0)
            rec_n = recall_score(y, y_pred, pos_label=0, zero_division=0)

            elapsed = time.time() - start_time
            metrics.update({
                "accuracy": round(acc, 3),
                "recall_pos": round(rec_p, 3),
                "recall_neg": round(rec_n, 3),
                "events": total_events,
                "events_per_s": round(total_events / elapsed, 3)
            })


            try:
                r.set(METRICS_KEY, json.dumps(metrics))
            except Exception as e:
                print(f"Error writing to Redis: {e}")
        
    except Exception as e:

        print(f"[MODEL ERROR] Failed to process incident. Error: {e}")
        print(f"[MODEL ERROR] Incident data: {incident}")

        return metrics

    return metrics
