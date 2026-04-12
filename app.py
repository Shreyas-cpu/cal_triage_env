"""
CalTriage-Env FastAPI Server.

Plain FastAPI server with no openenv framework dependency.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from environment import CalTriageEnvironment


app = FastAPI(
    title="CalTriage OpenEnv",
    description="Calendar conflict resolution RL environment for OpenEnv Hackathon",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "server", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

env = CalTriageEnvironment()


class ResetRequest(BaseModel):
    task_name: Optional[Literal["easy", "medium", "hard"]] = Field(
        default="easy",
        description="Task difficulty: easy, medium, or hard",
    )
    seed: Optional[int] = None


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    meeting_id: str = Field(..., description="ID of the meeting to act upon")
    action_type: Literal["reschedule", "cancel", "keep"] = Field(
        ..., description="Action to take"
    )
    new_start_hour: Optional[int] = Field(default=None, ge=0, le=23)
    new_start_minute: Optional[int] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def unpack_action_payload(cls, data):
        if isinstance(data, dict) and isinstance(data.get("action"), dict):
            merged = dict(data["action"])
            for key, value in data.items():
                if key != "action":
                    merged[key] = value
            return merged
        return data


def _oracle_action(obs):
    from models import CalAction

    conflicts = obs.active_conflicts
    schedule = {meeting.meeting_id: meeting for meeting in obs.current_schedule}
    if not conflicts:
        first = obs.current_schedule[0] if obs.current_schedule else None
        meeting_id = first.meeting_id if first else "mtg_000"
        return CalAction(meeting_id=meeting_id, action_type="keep")

    conflict = conflicts[0]
    meeting_a = schedule.get(conflict.meeting_a_id)
    meeting_b = schedule.get(conflict.meeting_b_id)
    priority_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    candidates = [meeting for meeting in [meeting_a, meeting_b] if meeting and not meeting.is_locked]
    if not candidates:
        return CalAction(meeting_id=conflict.meeting_a_id, action_type="keep")
    candidates.sort(key=lambda meeting: priority_rank.get(meeting.priority, 0))
    return CalAction(meeting_id=candidates[0].meeting_id, action_type="cancel")


@app.get("/", include_in_schema=False)
def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "env": "cal-triage-env", "version": "1.0.0"}


@app.get("/web", include_in_schema=False)
def web():
    return root()


@app.get("/health", tags=["General"], summary="Health check")
def health():
    return {"status": "ok", "env": "cal-triage-env", "version": "1.0.0"}


@app.post("/reset", tags=["OpenEnv API"], summary="Start a new episode")
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    obs = env.reset(task_name=req.task_name, seed=req.seed)
    return obs.model_dump()


@app.post("/step", tags=["OpenEnv API"], summary="Submit an action")
def step(req: StepRequest):
    from models import CalAction

    try:
        action = CalAction(
            meeting_id=req.meeting_id,
            action_type=req.action_type,
            new_start_hour=req.new_start_hour,
            new_start_minute=req.new_start_minute,
        )
        result = env.step(action)
        observation = result[0]
        reward = result[1]
        done = result[2]
        info = result[3]
        return {
            "observation": observation.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", tags=["OpenEnv API"], summary="Get current environment state")
def state():
    return env.get_state()


@app.get("/tasks", tags=["Tasks & Grading"], summary="List all 3 tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "task_easy",
                "name": "Simple Conflict Resolution",
                "difficulty": "easy",
                "num_meetings": 6,
                "num_conflicts": 2,
                "num_locked": 1,
                "description": "6 meetings, 2 simple conflicts, 1 locked meeting.",
            },
            {
                "id": "task_medium",
                "name": "Multi-Overlap Schedule",
                "difficulty": "medium",
                "num_meetings": 10,
                "num_conflicts": 4,
                "num_locked": 2,
                "description": "10 meetings, 4 conflicts, soft constraints active.",
            },
            {
                "id": "task_hard",
                "name": "Dense Cascade Conflicts",
                "difficulty": "hard",
                "num_meetings": 15,
                "num_conflicts": 6,
                "num_locked": 3,
                "description": "15 meetings, 6 cascading conflicts, all constraint types.",
            },
        ]
    }


@app.post("/grade/{task_id}", tags=["Tasks & Grading"], summary="Run oracle grader for a task")
def grade_task(task_id: str):
    difficulty_map = {
        "task_easy": "easy",
        "task_medium": "medium",
        "task_hard": "hard",
    }
    if task_id not in difficulty_map:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id '{task_id}'. Valid: task_easy, task_medium, task_hard",
        )

    grader_env = CalTriageEnvironment()
    scores = []
    for episode_seed in range(5):
        obs = grader_env.reset(task_name=difficulty_map[task_id], seed=episode_seed)
        episode_score = 0.0
        done = False
        steps = 0
        while not done and steps < 20:
            if not obs.active_conflicts:
                break
            action = _oracle_action(obs)
            obs, reward, done, info = grader_env.step(action)
            episode_score += float(reward)
            steps += 1
        scores.append(round(episode_score / max(steps, 1), 3))

    average_score = round(sum(scores) / len(scores), 3)
    return {
        "task_id": task_id,
        "episodes_run": 5,
        "average_score": average_score,
        "scores": scores,
        "grader": "oracle (keep/cancel policy)",
        "score_range": "0.0-1.0",
    }


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()