# ===============================
# Streamlit Real-Time Dashboard
# ===============================
import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime
import plotly.express as px
import redis
import os

st.set_page_config(
    page_title="Cyber Physics - Live Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Cyber-Physics Live Analytics Dashboard")
st.markdown("### Real-Time AI Learning Performance")

# --- Redis Connection ---
redis_host = os.environ.get("REDIS_HOST", "localhost")
METRICS_KEY = "cyber_physics_metrics"
try:
    r = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True)
    r.ping()
    st.sidebar.success(f"Connected to Redis ({redis_host})")
except Exception as e:
    st.sidebar.error(f"Redis connection failed: {e}")
    r = None
# --- End Redis Connection ---

metrics_placeholder = st.empty()
chart_placeholder = st.empty()

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["timestamp", "accuracy", "recall_pos", "recall_neg", "events_per_s"])

def load_metrics():
    if r is None: return None
    try:
        metrics_data = r.get(METRICS_KEY)
        if metrics_data:
            data = json.loads(metrics_data)
            data["timestamp"] = datetime.now().isoformat(timespec="seconds")
            return data
    except Exception as e:
        print(f"Error reading from Redis: {e}")
    return None

while True:
    metrics = load_metrics()
    
    if metrics:
        df = pd.DataFrame([metrics])
        st.session_state.history = pd.concat([st.session_state.history, df], ignore_index=True).tail(500) # (استخدم قيمة max_points)

        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            col2.metric("Recall (Pos)", f"{metrics['recall_pos']:.3f}")
            col3.metric("Recall (Neg)", f"{metrics['recall_neg']:.3f}")
            col4.metric("Events/sec", f"{metrics['events_per_s']:.3f}")

        chart = px.line(
            st.session_state.history,
            x="timestamp", y=["accuracy", "recall_pos", "recall_neg"],
            title="Model Performance Over Time",
        )
        chart_placeholder.plotly_chart(chart, use_container_width=True)

    time.sleep(1.0)
