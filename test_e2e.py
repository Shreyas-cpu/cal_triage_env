"""End-to-end HTTP test for all 3 tasks."""
import json
import requests

ENV = "http://localhost:8000"

print("=== END-TO-END HTTP TEST ===")
for task in ["easy", "medium", "hard"]:
    # Reset
    r = requests.post(f"{ENV}/reset", json={"task_name": task, "seed": 42})
    resp = r.json()
    obs = resp.get("observation", resp)
    n_meetings = len(obs["current_schedule"])
    n_conflicts = obs["num_conflicts"]
    print(f"\n[{task.upper()}] Reset: {n_meetings} meetings, {n_conflicts} conflicts")

    reward = 0.0
    done = False
    for step in range(1, 15):
        conflicts = obs.get("active_conflicts", [])
        schedule = obs.get("current_schedule", [])
        if not conflicts:
            break

        mtg_map = {m["meeting_id"]: m for m in schedule}
        c = conflicts[0]
        target = None
        for mid in [c["meeting_b_id"], c["meeting_a_id"]]:
            m = mtg_map.get(mid)
            if m and not m.get("is_locked"):
                target = m
                break
        if not target:
            break

        action = {"meeting_id": target["meeting_id"], "action_type": "cancel"}
        r = requests.post(f"{ENV}/step", json={"action": action})
        step_resp = r.json()
        obs = step_resp.get("observation", step_resp)
        reward = step_resp.get("reward", 0.0)
        done = step_resp.get("done", False)

        print(f"  Step {step}: cancel {target['meeting_id']} -> reward={reward}, done={done}, conflicts={obs.get('num_conflicts')}")

        if done:
            break

    # Final state
    r = requests.get(f"{ENV}/state")
    state = r.json()
    print(f"  Final: steps={state.get('step_count')}, resolved={state.get('conflicts_resolved')}")

    # Verify reward bounds
    assert 0.0 <= float(reward) <= 1.0, f"REWARD OUT OF BOUNDS: {reward}"
    print(f"  Reward {reward} is in [0,1] -- OK")

print("\n=== ALL TASKS PASSED ===")
