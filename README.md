---
title: CalTriage Env
emoji: 🗓️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---

# CalTriage-Env — Smart Calendar Triage
A **real-world** Reinforcement Learning environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. An AI executive assistant must resolve overlapping meeting conflicts on a daily calendar schedule while strictly obeying user constraints.

> Built for Round 1 of the **Meta PyTorch OpenEnv Hackathon**.

---

## Environment Description

**Domain:** Calendar/scheduling triage — a practical, product-relevant task (not a game or toy).

The agent receives a daily schedule with several overlapping meetings and must resolve every conflict by choosing to **reschedule**, **cancel**, or **keep** one meeting per step. Hard constraints (locked meetings) cannot be violated; soft constraints (preferred hours, lunch blocks, minimum gaps) incur penalties.

---

## Action Space

| Field | Type | Description |
|---|---|---|
| `meeting_id` | `str` | ID of the meeting to act upon |
| `action_type` | `"reschedule" \| "cancel" \| "keep"` | Decision for this meeting |
| `new_start_hour` | `Optional[int]` | New hour (required for reschedule) |
| `new_start_minute` | `Optional[int]` | New minute: 0 or 30 (required for reschedule) |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `current_schedule` | `List[Meeting]` | All active meetings |
| `active_conflicts` | `List[Conflict]` | Remaining overlaps |
| `constraints` | `List[Constraint]` | Rules agent must obey |
| `num_conflicts` | `int` | Count of active conflicts |
| `done` | `bool` | Episode terminated? |
| `reward` | `float` | Normalized reward ∈ [0.0, 1.0] |
| `last_action_error` | `Optional[str]` | Error from invalid action |
| `task_name` | `str` | Current difficulty level |

---

## Tasks (3 difficulty levels)

| Task | Meetings | Conflicts | Locked | Soft Constraints | Max Steps | Expected Difficulty |
|---|---|---|---|---|---|---|
| **easy** | 6 | 2 | 1 | 0 | 5 | Simple pair-wise overlaps |
| **medium** | 10 | 4 | 2 | 2 (preferred time, lunch) | 8 | Chained conflicts + soft rules |
| **hard** | 15 | 6 | 3 | 4 (time, lunch, gap, time) | 12 | Dense schedule, cascading effects |

---

## Reward Function (0.0 – 1.0)

| Outcome | Reward |
|---|---|
| Hard constraint violated (moved/cancelled locked meeting) | **0.0** |
| Invalid action (bad ID, missing fields) | **0.0** |
| All conflicts resolved, no soft violations | **1.0** |
| All conflicts resolved, with soft violations | **0.3 – 0.5** |
| Partial progress (some conflicts resolved) | **0.0 – 0.3** |

Partial rewards provide useful training signal throughout the trajectory.

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- Docker

### Install
```bash
pip install -e .
```

### Run locally
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t cal-triage-env .
docker run -p 8000:8000 cal-triage-env
```

### Web Interface
Open `http://localhost:8000/web` to interact manually.

### Deploy to HF Spaces
```bash
openenv push --repo-id your-username/cal-triage-env
```

---

## Baseline Scores

| Task | Baseline Model | Score |
|---|---|---|
| easy | Qwen2.5-72B-Instruct | ~0.7 |
| medium | Qwen2.5-72B-Instruct | ~0.4 |
| hard | Qwen2.5-72B-Instruct | ~0.2 |

---

## License

BSD 3-Clause License
