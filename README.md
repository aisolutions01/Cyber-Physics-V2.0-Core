# cyber-physics-dataset
On-the-fly synthetic data generator for cybersecurity and enterprise automation. Inspired by physics-informed PDEs, this repo demonstrates dynamic incident log generation, policy-based constraints, and real-time model training to replace static datasets.

### On-the-Fly Dataset Generator

This repository demonstrates **on-the-fly synthetic data generation** for ITSM, SIEM, and cybersecurity use cases.  
Inspired by **physics-informed PDE's**, the project simulates dynamic incident logs with **policy-based constraints**, enabling real-time AI training without relying on static datasets.

---

## Features
- Synthetic **incident log generator** with configurable constraints.
- **Policy-informed data rules** (severity, type, workflow).
- Real-time **model training** on generated data.
- Evaluation scripts to compare dynamic vs static dataset performance.
- Jupyter notebook demo for quick experimentation.

---

## Repository Structure
on_the_fly_dataset/

├── data_generator.py # On-the-fly incident generator

├── constraints.json # Example rules and policies

├── model_example.py # Simple classification model

├── evaluation.py # Evaluation script

├── notebooks/ # Demos & experiments

└── README.md


---

## Quick Start
```bash
# Clone repository
git clone https://github.com/aisolutions01/cyber-physics-dataset
cd on-the-fly-dataset

# Install requirements
pip install -r requirements.txt

# Run generator
python data_generator.py
