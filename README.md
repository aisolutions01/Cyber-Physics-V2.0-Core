# Cyber-Physics V2.0: The Adaptive AI Core

**Cyber-Physics V2.0** is a high-performance, lightweight, on-the-fly learning engine designed to act as an adaptive AI "immune system" for cybersecurity and AIOps.

This repository demonstrates a **radical shift** from traditional, high-cost batch-learning models to a cost-effective, real-time **online learning** paradigm.

---

## The Core Problem: Static, Slow, and Expensive AI

Traditional AI in security (SIEM, AIOps) is fundamentally broken. It relies on a "batch training" model, which suffers from three critical flaws:

1. **SLOW (Weeks to Months):**  
   Models require terabytes of pre-collected data and long offline training cycles, leaving systems blind in the meantime.

2. **STATIC (Always Outdated):**  
   Once trained, a model is frozen — incapable of recognizing new "zero-day" attacks or dynamic changes in network behavior.

3. **EXPENSIVE (Hardware & HR):**  
   - **Hardware Costs:** Massive GPU clusters are needed for retraining.  
   - **Human Costs:** Teams of data scientists are required to manage the retraining lifecycle.

---

## Our Solution: The On-the-Fly Adaptive Core

**Cyber-Physics V2.0** eliminates the entire “training phase.”  
The model learns from **every event as it arrives**, shifting from *Batch Learning* to true **Online Learning**.

| Property | Traditional AI | Cyber-Physics Core |
|-----------|----------------|--------------------|
| Training | Offline, periodic | Continuous, per-event |
| Adaptation Time | Weeks–Months | Sub-second |
| Hardware | GPU clusters | CPU-only |
| Data Requirement | Static datasets | Live streams |
| Cost | High | Minimal |

### ⚙️ Key Advantages

- **INSTANT (Seconds, not Months):**  
  Learning and adaptation occur in milliseconds.  
  “Time-to-adapt” is reduced from months to the time between two incoming events.

- **ADAPTIVE (Immune System):**  
  The model continuously evolves its definition of “normal” and “threat,” adapting autonomously to unseen attack patterns.

- **COST-EFFECTIVE (CPU, not GPU):**  
  Runs efficiently on standard CPUs using a constant-size *sliding window* for memory stability — removing the need for GPUs entirely.

---

## From V1.0 to V2.0

| Feature | V1.0 | V2.0 |
|----------|------|------|
| Focus | Streamlit Dashboard (Visualization) | Modular Core Engine |
| Storage | JSON / CSV metrics files | In-memory pipeline (Redis) |
| Processing | File-based | Real-time stream |
| Architecture | Single-threaded | Event-driven, scalable |
| Goal | Demonstration | Integration-ready AI Core |

---

## V2.0 Core Architecture

This version introduces a high-performance modular design enabling true on-the-fly learning.

* **Core Ingest API (`ingest_api.py`):**  
  Lightweight Flask API — the "nervous system" that receives incident streams from ITSM, SIEM, or IoT sensors.

* **On-the-Fly Learner (`model_example.py`):**  
  - Incremental model powered by `SGDClassifier (partial_fit)`.  
  - Uses a fixed-size `deque` for **constant memory (O(1))**.  
  - Tracks and updates metrics in real time.

* **In-Memory Bus (Redis):**  
  - Eliminates disk I/O bottlenecks from V1.0.  
  - Provides near-instantaneous metric availability for downstream systems or visualization tools.

---

## Repository Structure

```text
Cyber_Physics_Dataset_main/
│
├── schema_incident.py
├── data_generator.py
├── collector_interface.py
├── integration_sim.py
│
└── Streamlit/
    ├── streamlit_app.py
    ├── ingest_api.py
    ├── model_example.py
    ├── requirements.txt
    ├── Dockerfile
    └── docker-compose.yml
```
### Running the Core Engine and Demo

- Prerequisites
  
- Python 3.9+
  
- Docker and Docker Compose
  
- pip

### Installation 
git clone https://github.com/aisolutions01/Cyber-Physics-V2.0-Core.git

cd cyber-physics-dataset/Streamlit

pip install -r requirements.txt

### Run the Core

python ingest_api.py

### Run the Streamlit Demo

streamlit run streamlit_app.py

You can now send live events through the API and visualize system adaptation in real time.

### Use Cases

- Cybersecurity (SIEM/XDR/NDR/EDR): Real-time adaptive anomaly detection.

- IT Operations (AIOps/ITSM): Continuous learning from infrastructure events.

- IoT Networks: Lightweight real-time pattern recognition on the edge.

### Citation

If you use Cyber-Physics V2.0 in your research, please cite:

Kazem, M. (2025). Cyber-Physics V2.0: The Adaptive AI Core. Zenodo.

DOI: [to be added upon publication]

### Contact

Author: Munther Kazem

muntherkz2018@gmail.com

LinkedIn: https://www.linkedin.com/company/101435653/admin/dashboard/

Medium: https://medium.com/@muntherkz2018
