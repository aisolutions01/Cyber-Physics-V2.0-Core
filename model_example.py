# # model_example_pa.py
# """
# Improved streaming training using PassiveAggressiveClassifier, stratified replay,
# dynamic class weighting, PDE-inspired temporal regularization, and feature interactions.

# Usage:
#     python model_example_pa.py

# Notes:
# - It will try to import IncidentGenerator from data_generator.py and call `.generate()` or `.stream()`.
#   If not available, it will fall back to reading streams_1k.json directly.
# - Adjust constants below (BATCH_SIZE, REPLAY_MEMORY, REPLAY_RATIO) to taste.
# """

# import json
# import numpy as np
# import random
# from collections import deque, Counter
# import time
# import os

# from sklearn.linear_model import PassiveAggressiveClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# # --------------------
# # Config
# # --------------------
# STREAM_PATH = "streams_1k.json"
# BATCH_SIZE = 100
# REPLAY_MEMORY = 1000   # larger memory for diversity
# REPLAY_RATIO = 0.4     # fraction of combined batch drawn from replay
# WARMUP = 100           # samples to warm up scaler
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)

# # PDE-inspired hyperparam
# PDE_ALPHA = 0.5        # weight of PDE temporal penalty when adjusting sample weights
# PDE_SMOOTH = 0.1       # smoothing factor for predicted mean history

# # --------------------
# # Helper functions
# # --------------------
# def incident_to_features(inc):
#     """Create base + interaction features from an incident dict."""
#     severity = float(inc.get("severity", 1))
#     cpu = float(inc.get("cpu_load", 0.0))
#     net = float(inc.get("net_bytes", 0))
#     # interactions / engineered
#     interaction1 = cpu * severity
#     interaction2 = cpu * np.log1p(net)
#     return np.array([severity, cpu, net, interaction1, interaction2], dtype=float)

# def incident_to_label(inc):
#     critical = {"network_anomaly", "privilege_escalation", "policy_violation", "data_exfil", "malware_alert"}
#     return 1 if inc.get("incident_type") in critical else 0

# def stratified_sample_from_replay(replay_buffer, k):
#     """Return X, y arrays sampled stratified by class from replay_buffer (list of (x,y))."""
#     if k <= 0 or len(replay_buffer) == 0:
#         return np.empty((0,)), np.empty((0,))
#     by_class = {}
#     for x, y in replay_buffer:
#         by_class.setdefault(y, []).append((x, y))
#     classes = sorted(by_class.keys())
#     per_class = max(1, k // len(classes))
#     sampled = []
#     for cls in classes:
#         pool = by_class[cls]
#         if len(pool) <= per_class:
#             sampled += pool
#         else:
#             sampled += random.sample(pool, per_class)
#     # fill up if short
#     while len(sampled) < k and len(replay_buffer) > 0:
#         sampled.append(random.choice(replay_buffer))
#     Xs = np.array([s[0] for s in sampled])
#     ys = np.array([s[1] for s in sampled])
#     return Xs, ys

# def compute_balance_score(y_true, y_pred):
#     """1 - |Recall_0 - Recall_1|, closer to 1 is more balanced."""
#     cm = confusion_matrix(y_true, y_pred, labels=[0,1])
#     # rows=true classes
#     with np.errstate(divide='ignore', invalid='ignore'):
#         recall0 = cm[0,0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
#         recall1 = cm[1,1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
#     return 1.0 - abs(recall0 - recall1), recall0, recall1

# # --------------------
# # Try to import user's generator gracefully (dynamic mode)
# # --------------------
# from data_generator import IncidentGenerator

# try:
#     # ✅ نستخدم الجيل الديناميكي مباشرة بدون أي ملف
#     generator = IncidentGenerator()
#     use_generator = True
#     print("[INFO] Using dynamic on-the-fly IncidentGenerator (no JSON file).")
# except Exception as e:
#     print(f"[WARN] Could not initialize dynamic generator: {e}")
#     use_generator = False
#     generator = None

# # --------------------
# # Define streaming iterator
# # --------------------
# if use_generator:
#     def stream_iter():
#         # عدد الأحداث التقريبي لتجربة قصيرة
#         for e in generator.stream(n=1000, delay=0.05):
#             yield e
# else:
#     raise RuntimeError("No valid data source found. Please check data_generator.py.")


# # Fallback: read JSON file directly
# if not use_generator:
#     if not os.path.exists(STREAM_PATH):
#         raise FileNotFoundError(f"Could not find generator and {STREAM_PATH} missing.")
#     with open(STREAM_PATH, "r") as f:
#         events_all = json.load(f)
#     def stream_iter():
#         for e in events_all:
#             yield e
# else:
#     # try several method names
#     if hasattr(generator, "generate"):
#         def stream_iter():
#             for e in generator.generate():
#                 yield e
#     elif hasattr(generator, "stream"):
#         def stream_iter():
#             for e in generator.stream(n=len(getattr(generator, "events", [])), delay=0.0):
#                 yield e
#     else:
#         # as fallback, try iterating generator if it's an iterable
#         try:
#             def stream_iter():
#                 for e in generator:
#                     yield e
#         except Exception:
#             # final fallback: load file
#             with open(STREAM_PATH, "r") as f:
#                 events_all = json.load(f)
#             def stream_iter():
#                 for e in events_all:
#                     yield e

# # --------------------
# # Model, scaler, replay buffer
# # --------------------
# from sklearn.linear_model import SGDClassifier
# model = SGDClassifier(loss="log_loss", learning_rate="optimal", random_state=42)

# # model = PassiveAggressiveClassifier(C=0.01, max_iter=1000, random_state=SEED, tol=1e-3)
# scaler = StandardScaler()
# replay_buffer = deque(maxlen=REPLAY_MEMORY)

# # warmup buffer for scaler
# warmup = []

# # bookkeeping for PDE temporal regularization
# prev_mean_pred = None  # running previous mean prediction
# smoothed_prev_mean = 0.5  # smoothed history

# # training loop
# X_batch_raw, y_batch = [], []
# batch_count = 0
# start_time = time.time()

# for i, inc in enumerate(stream_iter()):
#     x_raw = incident_to_features(inc)
#     y = incident_to_label(inc)

#     X_batch_raw.append(x_raw)
#     y_batch.append(y)

#     if len(warmup) < WARMUP:
#         warmup.append(x_raw)

#     # when batch ready or end
#     if (i + 1) % BATCH_SIZE == 0:
#         batch_count += 1
#         Xb = np.vstack(X_batch_raw)
#         yb = np.array(y_batch)

#         # stratified replay sampling
#         replay_k = int(REPLAY_RATIO * len(Xb))
#         X_replay, y_replay = stratified_sample_from_replay(list(replay_buffer), replay_k)

#         if X_replay.shape[0] > 0:
#             X_comb = np.vstack([Xb, X_replay])
#             y_comb = np.concatenate([yb, y_replay])
#         else:
#             X_comb = Xb.copy()
#             y_comb = yb.copy()

#         # feature scaling: fit scaler on warmup if not fitted
#         if len(warmup) >= WARMUP and not hasattr(scaler, "mean_"):
#             scaler.fit(np.vstack(warmup))
#         if hasattr(scaler, "mean_"):
#             X_comb_scaled = scaler.transform(X_comb)
#             Xb_scaled = scaler.transform(Xb)
#         else:
#             X_comb_scaled = X_comb
#             Xb_scaled = Xb

#         # dynamic class weights -> sample weights
#         classes_present = np.unique(y_comb)
#         if len(classes_present) == 1:
#             # if missing class, try to augment from replay (or jitter)
#             missing_class = 0 if classes_present[0] == 1 else 1
#             # try to get candidates from replay of missing class
#             candidates = [r for r in list(replay_buffer) if r[1] == missing_class]
#             if len(candidates) >= 1:
#                 # duplicate some
#                 add_n = min(len(candidates), 5)
#                 addX = np.array([c[0] for c in random.sample(candidates, add_n)])
#                 addy = np.array([c[1] for c in random.sample(candidates, add_n)])
#                 X_comb = np.vstack([X_comb, addX])
#                 y_comb = np.concatenate([y_comb, addy])
#                 # re-scale
#                 if hasattr(scaler, "mean_"):
#                     X_comb_scaled = scaler.transform(X_comb)
#             else:
#                 # jitter dominant class to synthesize minority examples (last resort)
#                 dom = classes_present[0]
#                 dom_idx = np.where(y_comb == dom)[0]
#                 synth_count = min(5, len(dom_idx))
#                 synth = []
#                 for _ in range(synth_count):
#                     src = X_comb[ random.choice(dom_idx) ]
#                     jitter = src + np.random.normal(scale=0.02, size=src.shape)
#                     synth.append(jitter)
#                 if len(synth) > 0:
#                     addX = np.vstack(synth)
#                     addy = np.array([1-dom]*len(synth))
#                     X_comb = np.vstack([X_comb, addX])
#                     y_comb = np.concatenate([y_comb, addy])
#                     if hasattr(scaler, "mean_"):
#                         X_comb_scaled = scaler.transform(X_comb)

#         # recompute classes present after augmentation
#         classes_present = np.unique(y_comb)
#         # compute balanced class weights (per-batch)
#         try:
#             cw = compute_class_weight(class_weight="balanced", classes=classes_present, y=y_comb)
#             weight_dict = {c: w for c, w in zip(classes_present, cw)}
#             sample_weights = np.array([weight_dict[yy] for yy in y_comb])
#         except Exception:
#             # fallback uniform
#             sample_weights = np.ones(len(y_comb))

#         # PDE-inspired temporal penalty: compute mean prediction change and adjust sample weights
#         # get current mean prediction (on combined scaled data)
#         try:
#             preds_proba = None
#             # PassiveAggressive doesn't provide predict_proba; use predict (0/1) mean as proxy
#             cur_mean_pred = None
#             if hasattr(model, "predict"):
#                 cur_mean_pred = np.mean(model.predict(X_comb_scaled)) if hasattr(model, "coef_") else 0.5
#             else:
#                 cur_mean_pred = 0.5
#         except Exception:
#             cur_mean_pred = 0.5

#         if prev_mean_pred is None:
#             prev_mean_pred = cur_mean_pred
#             smoothed_prev_mean = cur_mean_pred

#         # compute temporal residual
#         residual = cur_mean_pred - smoothed_prev_mean
#         # update smoothed_prev_mean with PDE_SMOOTH
#         smoothed_prev_mean = (1 - PDE_SMOOTH) * smoothed_prev_mean + PDE_SMOOTH * cur_mean_pred

#         # apply penalty: if residual large, slightly reduce weights of currently dominant class
#         adjust = np.exp(-PDE_ALPHA * (residual**2))
#         sample_weights = sample_weights * adjust

#         # final partial_fit (PassiveAggressive supports partial_fit)
#         try:
#             model.partial_fit(X_comb_scaled, y_comb, classes=np.array([0,1]), sample_weight=sample_weights)
#         except Exception as e:
#             # in case of any issue, try without sample_weight
#             model.partial_fit(X_comb_scaled, y_comb, classes=np.array([0,1]))

#         # update replay buffer with raw (unscaled) Xb and yb
#         for xr, yr in zip(Xb, yb):
#             replay_buffer.append((xr, yr))

#         # evaluation on combined (quick)
#         y_pred_comb = model.predict(X_comb_scaled)
#         acc = accuracy_score(y_comb, y_pred_comb)
#         bal_score, r0, r1 = compute_balance_score(y_comb, y_pred_comb)
#         rep = classification_report(y_comb, y_pred_comb, digits=3, zero_division=0)
#         cm = confusion_matrix(y_comb, y_pred_comb, labels=[0,1])

#         print(f"\n--- Batch {batch_count} trained ---")
#         print(f"Batch samples: {len(Xb)} | Combined: {len(X_comb)} | Replay size: {len(replay_buffer)}")
#         print(f"Accuracy (combined): {acc:.3f} | Balance score: {bal_score:.3f} (recalls: {r0:.3f}, {r1:.3f})")
#         print("Confusion matrix (rows=true, cols=pred):")
#         print(cm)
#         print("Classification report:")
#         print(rep)
#         print("-"*50)

#         # reset
#         X_batch_raw, y_batch = [], []

#         # update prev_mean_pred for next iteration
#         prev_mean_pred = cur_mean_pred

#     # optional early stop for debugging
#     if i > 5000:
#         break

# end_time = time.time()
# print(f"Training finished in {end_time - start_time:.2f}s")

# # Save final model & scaler
# try:
#     import joblib
#     joblib.dump(model, "model_pa.joblib")
#     joblib.dump(scaler, "scaler_pa.joblib")
#     print("Saved model_pa.joblib and scaler_pa.joblib")
# except Exception:
#     pass

























# model_example_pa_fast.py
"""
Very low-latency online training for on-the-fly incidents.
Per-sample updates, online scaling, dynamic per-sample weights to mitigate bias.
"""

import time
import random
from collections import deque, defaultdict
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, recall_score

# -------------------------
# Config (tune these)
# -------------------------
DELAY = 0.01            # stream delay per event (s) -> you can set 0.0 for maximum throughput
MAX_EVENTS = 1000       # total events to process (for test)
PRINT_EVERY = 200       # print light stats every N events
REPLAY_MEMORY = 200     # small replay if desired
REPLAY_USE = False      # set True to use tiny replay (affects latency)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# PDE-like smoothing for temporal regularization (very light)
PDE_ALPHA = 0.3
PDE_SMOOTH = 0.05

# -------------------------
# Online scaler (Welford)
# -------------------------
class OnlineScaler:
    def __init__(self, n_features):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=float)
        self.M2 = np.zeros(n_features, dtype=float)
        self.eps = 1e-6

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def transform(self, x):
        if self.n < 2:
            return x
        var = self.M2 / (self.n - 1)
        std = np.sqrt(var + self.eps)
        return (x - self.mean) / std

# -------------------------
# Features & label helpers
# -------------------------
def incident_to_features(inc):
    severity = float(inc.get("severity", 1))
    cpu = float(inc.get("cpu_load", 0.0))
    net = float(inc.get("net_bytes", 0))
    interaction1 = cpu * severity
    interaction2 = cpu * np.log1p(net)
    return np.array([severity, cpu, net, interaction1, interaction2], dtype=float)

def incident_to_label(inc):
    critical = {"network_anomaly", "privilege_escalation", "policy_violation", "data_exfiltration", "malware_alert"}
    return 1 if inc.get("incident_type") in critical else 0

# -------------------------
# Model init (lightweight)
# -------------------------
# Use SGDClassifier with partial_fit; single-epoch updates are fast.
model = SGDClassifier(loss="log_loss", learning_rate="optimal", max_iter=1, tol=None, warm_start=False)
# We'll call partial_fit with classes=[0,1] on first seen sample.
initialized = False

# -------------------------
# Replay buffer (optional)
# -------------------------
replay = deque(maxlen=REPLAY_MEMORY)

# -------------------------
# Streaming source
# -------------------------
from data_generator import IncidentGenerator
gen = IncidentGenerator()  # dynamic generator
stream = gen.stream(n=MAX_EVENTS, delay=DELAY)

# -------------------------
# Online training loop
# -------------------------
scaler = None
class_counts = defaultdict(int)
event_count = 0
prev_mean_pred = 0.5
smoothed_prev = 0.5

# lightweight rolling metrics
rolling_preds = []
rolling_trues = []

t_start = time.time()
for evt in stream:
    event_count += 1
    x_raw = incident_to_features(evt)
    y = incident_to_label(evt)

    # init scaler once we know feature dim
    if scaler is None:
        scaler = OnlineScaler(len(x_raw))

    # update scaler (use before transform or after? we'll update then transform)
    scaler.update(x_raw)
    x_scaled = scaler.transform(x_raw).reshape(1, -1)

    # dynamic per-sample weight (inverse class frequency)
    class_counts[y] += 1
    total_seen = class_counts[0] + class_counts[1]
    # avoid zero division
    freq0 = class_counts[0] / total_seen if total_seen > 0 else 0.5
    freq1 = class_counts[1] / total_seen if total_seen > 0 else 0.5
    # per-class weight inversely proportional to freq
    w0 = (1.0 / (freq0 + 1e-6)) if freq0 > 0 else 1.0
    w1 = (1.0 / (freq1 + 1e-6)) if freq1 > 0 else 1.0
    sample_weight = w1 if y == 1 else w0
    # normalize weights to about 1
    sample_weight = sample_weight / max(w0, w1)

    # temporal PDE-inspired adjustment (light)
    # predict current model output (if available) and compute residual
    if initialized:
        cur_pred = model.predict(x_scaled)[0]
    else:
        cur_pred = 0.5
    residual = cur_pred - smoothed_prev
    smoothed_prev = (1 - PDE_SMOOTH) * smoothed_prev + PDE_SMOOTH * cur_pred
    # adjust sample weight slightly by residual (reduce when big swings)
    sample_weight *= np.exp(-PDE_ALPHA * (residual ** 2))

    # initialize model with classes on first call
    if not initialized:
        # partial_fit requires at least one call with classes specified
        model.partial_fit(x_scaled, np.array([y]), classes=np.array([0,1]), sample_weight=np.array([sample_weight]))
        initialized = True
    else:
        model.partial_fit(x_scaled, np.array([y]), sample_weight=np.array([sample_weight]))

    # add to replay optionally
    if REPLAY_USE:
        replay.append((x_raw.copy(), y))

    # rolling metrics
    y_pred = model.predict(x_scaled)[0]
    rolling_preds.append(y_pred)
    rolling_trues.append(y)
    if len(rolling_trues) > 1000:
        rolling_trues.pop(0); rolling_preds.pop(0)

    # periodic light printing
    if event_count % PRINT_EVERY == 0:
        # compute simple accuracy and recall over rolling window
        acc = accuracy_score(rolling_trues, rolling_preds)
        rec1 = recall_score(rolling_trues, rolling_preds, zero_division=0)
        rec0 = recall_score(rolling_trues, rolling_preds, pos_label=0, zero_division=0)
        t_now = time.time()
        elapsed = t_now - t_start
        per_event = elapsed / event_count
        print(f"[{event_count}] elapsed={elapsed:.2f}s avg_per_event={per_event:.4f}s acc={acc:.3f} recall_pos={rec1:.3f} recall_neg={rec0:.3f} replay_size={len(replay)}")
    # small safety stop
    if event_count >= MAX_EVENTS:
        break

t_total = time.time() - t_start
print(f"Done. Processed {event_count} events in {t_total:.2f}s -> avg {t_total/event_count:.4f}s per event")

# save model quickly
try:
    import joblib
    joblib.dump(model, "model_fast.joblib")
    print("[SAVE] model_fast.joblib")
except Exception as e:
    print("[WARN] saving failed:", e)
