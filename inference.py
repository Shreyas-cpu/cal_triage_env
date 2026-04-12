"""
inference.py  CalTriage-Env
OpenEnv Hackathon  Mandatory evaluation script.

STDOUT FORMAT (exact):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
import re
import sys

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY") or HF_TOKEN or ""
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

BENCHMARK = "cal_triage_env"
MAX_STEPS = {"easy": 5, "medium": 8, "hard": 12}
SUCCESS_THRESHOLD = 0.4


def _clean(error):
    text = re.sub(r"<[^>]+>", "", str(error))
    text = re.sub(r"\s+", " ", text).strip()[:120]
    return text if text else "error"


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    err = _clean(error) if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    values = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={values}",
        flush=True,
    )


def ensure_packages():
    for package_name, install_spec in [("openai", "openai==1.30.1"), ("requests", "requests")]:
        try:
            __import__(package_name)
        except ImportError:
            os.system(f"{sys.executable} -m pip install {install_spec} --quiet")


ensure_packages()


client = None
try:
    if API_KEY:
        from openai import OpenAI

        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
except Exception:
    client = None


SYSTEM_PROMPT = """You are an AI executive assistant resolving calendar conflicts.

Given a schedule with overlapping meetings, choose ONE meeting to act on:
- reschedule: move it to a free time slot
- cancel: remove it entirely
- keep: leave it (use for locked meetings)

RULES:
1. NEVER reschedule or cancel a meeting marked is_locked=true. Always "keep" locked meetings.
2. Prefer reschedule over cancel.
3. Target the LOWEST-priority meeting in the conflict.
4. Priority order (highest to lowest): critical > high > medium > low

Reply with EXACTLY one line:
ACTION: <meeting_id> <reschedule|cancel|keep> [new_hour] [new_minute]

Examples:
ACTION: mtg_003 cancel
ACTION: mtg_002 reschedule 14 0
ACTION: mtg_000 keep"""


PRIORITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def _fallback_action(obs_dict):
    conflicts = obs_dict.get("active_conflicts", [])
    schedule = {meeting["meeting_id"]: meeting for meeting in obs_dict.get("current_schedule", [])}

    if not conflicts or not schedule:
        first_id = obs_dict.get("current_schedule", [{}])[0].get("meeting_id", "mtg_000") if obs_dict.get("current_schedule") else "mtg_000"
        return {"meeting_id": first_id, "action_type": "keep"}

    conflict = conflicts[0]
    meeting_a = schedule.get(conflict["meeting_a_id"])
    meeting_b = schedule.get(conflict["meeting_b_id"])
    candidates = [meeting for meeting in [meeting_a, meeting_b] if meeting and not meeting.get("is_locked", False)]

    if not candidates:
        return {"meeting_id": conflict["meeting_a_id"], "action_type": "keep"}

    candidates.sort(key=lambda meeting: PRIORITY_RANK.get(meeting.get("priority", "low"), 0))
    target = candidates[0]

    free_hour = _find_free_slot(obs_dict.get("current_schedule", []), 30)
    if free_hour is not None:
        return {
            "meeting_id": target["meeting_id"],
            "action_type": "reschedule",
            "new_start_hour": free_hour,
            "new_start_minute": 0,
        }

    return {"meeting_id": target["meeting_id"], "action_type": "cancel"}


def _find_free_slot(schedule, duration):
    occupied = set()
    for meeting in schedule:
        time_slot = meeting.get("time_slot", {})
        start = time_slot.get("start_hour", 0) * 60 + time_slot.get("start_minute", 0)
        for minute in range(start, start + time_slot.get("duration_min", 30)):
            occupied.add(minute)

    for hour in range(8, 18):
        for minute in (0, 30):
            start = hour * 60 + minute
            if start + duration > 18 * 60:
                continue
            if not any((start + offset) in occupied for offset in range(duration)):
                return hour
    return None


def _build_prompt(obs_dict, step, max_steps):
    schedule = obs_dict.get("current_schedule", [])
    conflicts = obs_dict.get("active_conflicts", [])

    lines = [f"Step {step}/{max_steps}  {len(conflicts)} conflict(s) remaining.\n"]
    lines.append("SCHEDULE:")
    for meeting in schedule:
        time_slot = meeting.get("time_slot", {})
        locked = " [LOCKED]" if meeting.get("is_locked") else ""
        lines.append(
            f"  {meeting['meeting_id']}: \"{meeting.get('title','?')}\" "
            f"{time_slot.get('start_hour',0):02d}:{time_slot.get('start_minute',0):02d} "
            f"(priority={meeting.get('priority','?')}{locked})"
        )

    lines.append("\nCONFLICTS:")
    for conflict in conflicts:
        lines.append(
            f"  {conflict['meeting_a_id']} overlaps {conflict['meeting_b_id']} by {conflict['overlap_minutes']} min"
        )

    lines.append("\nYour action (one line: ACTION: <id> <type> [hour] [min]):")
    return "\n".join(lines)


def _parse_llm_response(raw, obs_dict):
    raw = raw.strip().upper()
    match = re.search(r"ACTION:\s*(\S+)\s+(RESCHEDULE|CANCEL|KEEP)(?:\s+(\d+)(?:\s+(\d+))?)?", raw)
    if match:
        meeting_id = match.group(1).lower()
        action_type = match.group(2).lower()
        hour = int(match.group(3)) if match.group(3) else None
        minute = int(match.group(4)) if match.group(4) else 0
        action = {"meeting_id": meeting_id, "action_type": action_type}
        if action_type == "reschedule" and hour is not None:
            action["new_start_hour"] = hour
            action["new_start_minute"] = minute if minute in (0, 30) else 0
        return action
    return None


def _agent_act(obs_dict, step, max_steps):
    if client is None:
        return _fallback_action(obs_dict), "client_unavailable"

    prompt = _build_prompt(obs_dict, step, max_steps)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        action = _parse_llm_response(raw, obs_dict)
        if action:
            return action, None
        return _fallback_action(obs_dict), "parse_failed"
    except Exception as exc:
        return _fallback_action(obs_dict), _clean(exc)


def _http_reset(task_name):
    response = requests.post(f"{ENV_URL}/reset", json={"task_name": task_name}, timeout=30)
    response.raise_for_status()
    return response.json()


def _http_step(action_dict):
    response = requests.post(f"{ENV_URL}/step", json=action_dict, timeout=30)
    response.raise_for_status()
    return response.json()


def _action_to_str(action_dict):
    parts = [f"meeting={action_dict.get('meeting_id', '?')}", f"action={action_dict.get('action_type', '?')}" ]
    if action_dict.get("action_type") == "reschedule":
        parts.append(f"hour={action_dict.get('new_start_hour','?')}")
    return ",".join(parts)


def run_task(task_id, difficulty):
    max_steps = MAX_STEPS.get(difficulty, 8)
    log_start(task_id, BENCHMARK, MODEL_NAME)

    rewards = []
    steps = 0
    done = False

    try:
        obs_dict = _http_reset(difficulty)
    except Exception as exc:
        log_step(1, "meeting=none,action=keep", 0.0, True, _clean(exc))
        log_end(False, 1, 0.0, [0.0])
        return 0.0

    while not done and steps < max_steps:
        error_str = None
        action_str = "meeting=none,action=keep"
        reward = 0.0

        try:
            action, error_str = _agent_act(obs_dict, steps + 1, max_steps)
            action_str = _action_to_str(action)
            result = _http_step(action)
            obs_dict = result.get("observation", obs_dict)
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            if isinstance(obs_dict, dict):
                error_str = obs_dict.get("last_action_error")
        except Exception as exc:
            error_str = _clean(exc)
            done = True

        steps += 1
        rewards.append(reward)
        log_step(steps, action_str, reward, done, error_str)

        if done:
            break

    score = round(sum(rewards) / max(len(rewards), 1), 3)
    log_end(score >= SUCCESS_THRESHOLD, steps, score, rewards)
    return score


def main():
    tasks = [("task_easy", "easy"), ("task_medium", "medium"), ("task_hard", "hard")]
    results = {}

    for task_id, difficulty in tasks:
        try:
            results[task_id] = run_task(task_id, difficulty)
        except Exception:
            log_start(task_id, BENCHMARK, MODEL_NAME)
            log_step(1, "meeting=none,action=keep", 0.0, True, "null")
            log_end(False, 1, 0.0, [0.0])
            results[task_id] = 0.0

    overall = round(sum(results.values()) / len(results), 3)
    with open("baseline_scores.json", "w", encoding="utf-8") as handle:
        json.dump({"tasks": results, "overall": overall, "model": MODEL_NAME}, handle, indent=2)


if __name__ == "__main__":
    main()

