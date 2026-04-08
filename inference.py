"""
Inference Script for CalTriage-Env
===================================
MANDATORY evaluation script for the Meta PyTorch OpenEnv Hackathon.

Runs an LLM agent against the CalTriage environment across all 3 tasks
(easy, medium, hard) and produces structured [START]/[STEP]/[END] logs.

Environment variables required:
    API_BASE_URL   HF Router endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Hugging Face API token
    IMAGE_NAME     Docker image name (if using from_docker_image)

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK = "cal_triage_env"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS_MAP = {"easy": 5, "medium": 8, "hard": 12}
TEMPERATURE = 0.3
MAX_TOKENS = 200
SUCCESS_THRESHOLD = 0.5

# ──────────────────────────────────────────────────────────────────────────────
# Structured Logging  (must match hackathon format exactly)
# ──────────────────────────────────────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# System Prompt  (instructs the LLM to be an AI calendar assistant)
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an AI executive assistant whose job is to resolve meeting conflicts
    on a daily calendar. Each step you must act on ONE meeting.

    ACTIONS (pick exactly one):
      reschedule — Move the meeting to a new, conflict-free time.
      cancel     — Remove the meeting from the calendar entirely.
      keep       — Leave the meeting at its current time (use when it's locked).

    CRITICAL RULES:
    1. NEVER reschedule or cancel a meeting marked is_locked=true. Always "keep" it.
    2. Prefer rescheduling over cancelling — cancelling is penalised.
    3. When rescheduling, pick a time that does NOT overlap any other meeting.
    4. Resolve the conflict involving the LOWER-priority meeting.
       Priority order: critical > high > medium > low
    5. Times use 30-minute granularity: minutes must be 0 or 30.
    6. Keep meetings within working hours (day boundaries shown in schedule).

    RESPONSE FORMAT — output ONLY a single JSON object, nothing else:
    For reschedule:
      {"meeting_id": "mtg_XXX", "action_type": "reschedule", "new_start_hour": 14, "new_start_minute": 0}
    For cancel:
      {"meeting_id": "mtg_XXX", "action_type": "cancel"}
    For keep:
      {"meeting_id": "mtg_XXX", "action_type": "keep"}

    Do NOT add any explanation, markdown, or extra text. Only valid JSON.
""")

# ──────────────────────────────────────────────────────────────────────────────
# Prompt Builder  (formats the observation for the LLM)
# ──────────────────────────────────────────────────────────────────────────────


def format_meeting(m: Dict[str, Any]) -> str:
    """Format a single meeting dict into a human-readable line."""
    ts = m.get("time_slot", {})
    sh, sm = ts.get("start_hour", 0), ts.get("start_minute", 0)
    dur = ts.get("duration_min", 30)
    eh, em = divmod(sh * 60 + sm + dur, 60)
    locked = " [LOCKED]" if m.get("is_locked") else ""
    prio = m.get("priority", "medium")
    return (
        f'  {m["meeting_id"]}: "{m.get("title", "?")}" '
        f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d} "
        f"(priority={prio}{locked})"
    )


def format_conflict(c: Dict[str, Any]) -> str:
    """Format a conflict dict."""
    return (
        f"  {c['meeting_a_id']} overlaps with {c['meeting_b_id']} "
        f"by {c['overlap_minutes']} minutes"
    )


def format_constraint(c: Dict[str, Any]) -> str:
    """Format a constraint dict."""
    hard = "HARD" if c.get("is_hard") else "SOFT"
    return f"  [{hard}] {c.get('constraint_type', '?')} → target={c.get('target', '?')}"


def build_user_prompt(obs: Dict[str, Any], step: int, max_steps: int) -> str:
    """Build the user prompt from the current observation."""
    schedule_lines = [
        format_meeting(m) for m in obs.get("current_schedule", [])
    ]
    conflict_lines = [
        format_conflict(c) for c in obs.get("active_conflicts", [])
    ]
    constraint_lines = [
        format_constraint(c) for c in obs.get("constraints", [])
    ]

    return textwrap.dedent(f"""\
        Step {step}/{max_steps} — {obs.get('num_conflicts', 0)} conflict(s) remaining.

        CURRENT SCHEDULE:
        {chr(10).join(schedule_lines) if schedule_lines else "  (empty)"}

        ACTIVE CONFLICTS:
        {chr(10).join(conflict_lines) if conflict_lines else "  None — all resolved!"}

        CONSTRAINTS:
        {chr(10).join(constraint_lines) if constraint_lines else "  None"}

        Your action (JSON only):
    """)


# ──────────────────────────────────────────────────────────────────────────────
# LLM Interaction
# ──────────────────────────────────────────────────────────────────────────────


def get_llm_action(
    client: OpenAI,
    obs: Dict[str, Any],
    step: int,
    max_steps: int,
    history: List[str],
) -> Dict[str, Any]:
    """Call the LLM and parse its response into an action dict."""
    user_prompt = build_user_prompt(obs, step, max_steps)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    # Add last 2 history entries for context
    for h in history[-2:]:
        messages.append({"role": "assistant", "content": h})
    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        raw = ""

    return parse_llm_response(raw, obs)


def parse_llm_response(raw: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the LLM's text response into a valid action dict.
    Falls back to a safe heuristic if parsing fails.
    """
    # Try direct JSON parse
    action = try_parse_json(raw)
    if action and is_valid_action(action):
        return clean_action(action)

    # Try extracting JSON from markdown or mixed text
    json_match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
    if json_match:
        action = try_parse_json(json_match.group())
        if action and is_valid_action(action):
            return clean_action(action)

    # Fallback: use heuristic — cancel/reschedule the lowest-priority
    # unlocked meeting involved in a conflict
    return fallback_action(obs)


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse text as JSON."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def is_valid_action(action: Dict[str, Any]) -> bool:
    """Check if an action dict has the minimum required fields."""
    return (
        isinstance(action, dict)
        and "meeting_id" in action
        and "action_type" in action
        and action["action_type"] in ("reschedule", "cancel", "keep")
    )


def clean_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure action only contains valid fields."""
    result: Dict[str, Any] = {
        "meeting_id": str(action["meeting_id"]),
        "action_type": action["action_type"],
    }
    if action["action_type"] == "reschedule":
        result["new_start_hour"] = int(action.get("new_start_hour", 14))
        result["new_start_minute"] = int(action.get("new_start_minute", 0))
        # Snap to valid 30-min values
        if result["new_start_minute"] not in (0, 30):
            result["new_start_minute"] = 0
    return result


def fallback_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristic fallback when LLM output can't be parsed.
    Picks the lowest-priority unlocked meeting from the first conflict
    and cancels it.
    """
    conflicts = obs.get("active_conflicts", [])
    schedule = obs.get("current_schedule", [])

    if not conflicts or not schedule:
        # Nothing to do — keep first meeting
        if schedule:
            return {"meeting_id": schedule[0]["meeting_id"], "action_type": "keep"}
        return {"meeting_id": "mtg_000", "action_type": "keep"}

    # Build lookup
    meetings_by_id = {m["meeting_id"]: m for m in schedule}
    priority_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}

    # Pick first conflict, find the lower-priority unlocked meeting
    conflict = conflicts[0]
    m_a = meetings_by_id.get(conflict["meeting_a_id"])
    m_b = meetings_by_id.get(conflict["meeting_b_id"])

    if not m_a or not m_b:
        return {"meeting_id": conflict["meeting_a_id"], "action_type": "keep"}

    # Pick the lower-priority, non-locked meeting to cancel
    candidates = []
    for m in [m_a, m_b]:
        if not m.get("is_locked", False):
            candidates.append(m)

    if not candidates:
        # Both locked — keep one (will result in 0 reward but no crash)
        return {"meeting_id": m_a["meeting_id"], "action_type": "keep"}

    # Sort by priority ascending (lowest first)
    candidates.sort(key=lambda x: priority_rank.get(x.get("priority", "low"), 0))
    target = candidates[0]

    # Try to find a free slot for rescheduling instead of cancelling
    free_hour = find_free_slot(schedule, target.get("time_slot", {}).get("duration_min", 30))
    if free_hour is not None:
        return {
            "meeting_id": target["meeting_id"],
            "action_type": "reschedule",
            "new_start_hour": free_hour,
            "new_start_minute": 0,
        }

    return {"meeting_id": target["meeting_id"], "action_type": "cancel"}


def find_free_slot(
    schedule: List[Dict[str, Any]], duration: int
) -> Optional[int]:
    """Find a free hour slot that doesn't overlap with any meeting."""
    occupied = set()
    for m in schedule:
        ts = m.get("time_slot", {})
        start = ts.get("start_hour", 0) * 60 + ts.get("start_minute", 0)
        end = start + ts.get("duration_min", 30)
        for t in range(start, end):
            occupied.add(t)

    # Search from 8 AM to 6 PM
    for hour in range(8, 18):
        for minute in (0, 30):
            start = hour * 60 + minute
            end = start + duration
            if end > 18 * 60:
                continue
            slot_minutes = set(range(start, end))
            if not slot_minutes & occupied:
                return hour
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Environment Communication via WebSocket
#
# IMPORTANT: OpenEnv HTTP endpoints (/reset, /step) are stateless — they
# create a NEW environment per request.  A WebSocket connection maintains
# a persistent session so that reset() state carries through to step().
# ──────────────────────────────────────────────────────────────────────────────


async def ws_run_episode(
    ws_url: str,
    task_name: str,
    llm_client: OpenAI,
) -> None:
    """Run one episode for *task_name* over a WebSocket connection."""
    import websockets

    max_steps = MAX_STEPS_MAP.get(task_name, 8)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with websockets.connect(ws_url) as ws:
            # ── RESET ────────────────────────────────────────────────
            reset_msg = json.dumps({
                "type": "reset",
                "data": {"task_name": task_name, "seed": 42},
            })
            await ws.send(reset_msg)
            raw_resp = await ws.recv()
            resp = json.loads(raw_resp)

            # WSObservationResponse.data = serialize_observation(obs) =
            #   {"observation": {schedule, conflicts, ...}, "reward": ..., "done": ...}
            ws_data = resp.get("data", resp)
            obs = ws_data.get("observation", ws_data)
            obs["reward"] = ws_data.get("reward", 0.0)
            obs["done"] = ws_data.get("done", False)

            for step in range(1, max_steps + 1):
                if obs.get("done"):
                    break

                # Get LLM action
                action_dict = get_llm_action(llm_client, obs, step, max_steps, history)

                # ── STEP ─────────────────────────────────────────────
                step_msg = json.dumps({
                    "type": "step",
                    "data": action_dict,
                })
                await ws.send(step_msg)
                raw_resp = await ws.recv()
                resp = json.loads(raw_resp)

                # Handle error responses from the server
                if resp.get("type") == "error":
                    err_data = resp.get("data", {})
                    error_msg = err_data.get("message", str(err_data))
                    rewards.append(0.0)
                    steps_taken = step
                    log_step(step=step, action=json.dumps(action_dict, separators=(",", ":")),
                             reward=0.0, done=True, error=error_msg)
                    break

                ws_data = resp.get("data", resp)
                obs = ws_data.get("observation", ws_data)
                reward = ws_data.get("reward", 0.0)
                if reward is None:
                    reward = 0.0
                done = ws_data.get("done", False)
                error = obs.get("last_action_error")
                # Merge reward/done into obs for consistency
                obs["reward"] = reward
                obs["done"] = done

                rewards.append(float(reward))
                steps_taken = step

                action_str = json.dumps(action_dict, separators=(",", ":"))
                log_step(step=step, action=action_str, reward=float(reward),
                         done=done, error=error)
                history.append(action_str)

                if done:
                    break

            # ── STATE (optional — log episode metadata) ──────────
            try:
                state_msg = json.dumps({"type": "state"})
                await ws.send(state_msg)
                raw_state = await ws.recv()
                state_resp = json.loads(raw_state)
                state_data = state_resp.get("data", state_resp)
                print(f"[DEBUG] Final state: {json.dumps(state_data)}", flush=True)
            except Exception:
                pass

            # ── CLOSE ────────────────────────────────────────────────
            try:
                close_msg = json.dumps({"type": "close"})
                await ws.send(close_msg)
            except Exception:
                pass

        # Final score = last reward (which encodes full episode outcome)
        score = rewards[-1] if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ──────────────────────────────────────────────────────────────────────────────
# Main — run all 3 tasks via WebSocket
# ──────────────────────────────────────────────────────────────────────────────


async def main() -> None:
    """Run all tasks against the environment server."""
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Build WebSocket URL from ENV_URL
    # http://localhost:8000 → ws://localhost:8000/ws
    ws_base = ENV_URL.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_base}/ws"

    for task in TASKS:
        await ws_run_episode(ws_url, task, llm_client)


if __name__ == "__main__":
    asyncio.run(main())
