---
title: Cal Triage Env
emoji: 🗓️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

<p align="center">
  <img src="assets/infographic.png" alt="CalTriage-Env Infographic" width="100%" style="border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);">
</p>

# 🗓️ CalTriage-Env: Smart Calendar Triage

<div align="center">
  
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Framework: OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-purple.svg)](https://github.com/meta-pytorch/OpenEnv)
[![PyTorch Hackathon](https://img.shields.io/badge/Meta_PyTorch_Hackathon-Round_1-orange.svg)]()

> *An advanced Reinforcement Learning environment built to solve real-world scheduling paradigms.*

</div>

---

## 🚀 The Challenge: Executive Assistant AI

**CalTriage-Env** is a state-of-the-art **Reinforcement Learning environment** built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. It simulates a highly practical, product-relevant domain: **Calendar / Scheduling Triage**. 

As professionals, our schedules constantly devolve into chaotic, overlapping meeting blocks. We envision an AI Executive Assistant capable of elegantly resolving these overlaps while preserving an individual's intricate preferences.

Unlike simple toy problems, CalTriage-Env thrusts your Agent into a daily schedule riddled with conflicting meetings. To win, the Agent must surgically act—choosing to **reschedule**, **cancel**, or **keep** meetings—all while strictly adhering to user constraints and maximizing schedule fluidity.

---

## 🌟 Key Features

- **🧠 Real-World Complexity:** Handles both **Hard Constraints** (locked unmovable meetings) and **Soft Constraints** (preferred hours, lunch blocks, mental breaks).
- **🕸️ Microservices Architecture:** The environment seamlessly integrates a WebSocket interface designed explicitly to securely wrap OpenEnv loops.
- **⚡ Fast Local Execution:** Optimized `uvicorn` setup with a lightweight container layer for ultra-fast throughput during model testing.
- **📈 Dynamic Difficulty:** Three progressive difficulty settings (Easy, Medium, and Hard) testing simple pairing logic to cascading multi-block constraints.
- **🌐 HuggingFace Ready:** Packaged efficiently with OpenEnv specifications to run flawlessly on HF Spaces.

---

## 💡 How It Works

### The Action Space
On every step, the agent parses the current schedule and outputs an intervention for a single overlapping meeting:

| Argument | Type | Description |
|:---|:---:|:---|
| `meeting_id` | `str` | The unique ID of the target meeting to manage. |
| `action_type` | `Enum` | Must be `"reschedule"`, `"cancel"`, or `"keep"`. |
| `new_start_hour` | `int` | (If Rescheduling) Hour parameter (0-23).|
| `new_start_minute`| `int` | (If Rescheduling) Minute parameter (`0` or `30`). |

### The Observation Space
The environment yields rich context arrays representing the state vector:
- `current_schedule`: List of active Meetings.
- `active_conflicts`: List of active pairwise/multi-node meeting overlaps.
- `constraints`: List of localized constraints the agent must solve for.
- `last_action_error`: Real-time semantic feedback on invalid action attempts.
- `done` / `reward`: Episode termination blocks.

---

## 🏗️ Technical Architecture & Progressive Tasks

Depending on user invocation, CalTriage scales its internal complexity:

| Task Tier | Active Meetings | Initial Conflicts | Locked Events | Soft Constraints Imposed | Expected Difficulty |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 🟢 **Easy** | 6 | 2 | 1 | 0 | Beginner pairwise overlaps. |
| 🟡 **Medium** | 10 | 4 | 2 | 2 (Preferred Time, Lunch) | Overlapping clusters. |
| 🔴 **Hard** | 15 | 6 | 3 | 4 (Time, Lunch, Gaps) | Intensive dense cascading overlaps. |

#### Reward Mechanism (0.0 to 1.0)
Rewards are strictly normalized. The engine penalizes agents for breaking hard constraints (`reward = 0.0`) and issues continuous partial rewards (`0.0 - 0.5`) for conflict resolution combined with soft constraint violations, scaling to a perfect `1.0` for a unified schedule.

---

## 🛠️ Quick Start & Setup

### Requirements
- **Python:** 3.10 or higher.
- **Docker:** (Optional but recommended for robust testing).

### Local Installation
```bash
# Clone the repository
git clone https://github.com/Shreyas-cpu/cal_triage_env.git
cd cal_triage_env

# Install locally as an editable package
pip install -e .
```

### Launch the Environment
```bash
# Start the OpenEnv server bridge
uvicorn server.app:app --host 0.0.0.0 --port 8000
```
Navigate to `http://localhost:8000/web` for the manual interaction interface.

### Running with Docker (Isolated)
```bash
docker build -t cal-triage-env .
docker run -p 8000:8000 cal-triage-env
```

---

## 🚀 Deploying to HuggingFace Spaces
Ready for submission? Deploy directly to a GPU-backed OpenEnv HF Space using the CLI:
```bash
openenv push --repo-id your-username/cal-triage-env
```

---

## 📊 Evaluation & Baselines
We benchmark standard Large Language Models acting as agents:

| Baseline Agent | Easy | Medium | Hard |
|:---|:---:|:---:|:---:|
| **Qwen 2.5 72B Instruct** | ~0.7 | ~0.4 | ~0.2 |

*It's clear that while simpler models handle 2-conflict limits, the Hard mode cascaded overlaps require much higher-tier reasoning algorithms.*

---

<div align="center">

*Engineered for Round 1 of the **Meta PyTorch OpenEnv Hackathon**.*<br/>
[Open an Issue](https://github.com/Shreyas-cpu/cal_triage_env/issues) · [BSD 3-Clause License](LICENSE)

</div>
