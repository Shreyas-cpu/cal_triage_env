"""Quick smoke test — validates models, environment, and reward function."""
import json
import sys

print("=" * 60)
print("SMOKE TEST: CalTriage-Env")
print("=" * 60)

# 1. Test imports
print("\n[1] Testing imports...")
try:
    from models import CalAction, CalObservation, CalState, Meeting, TimeSlot, Conflict, Constraint
    print("    ✅ models.py imports OK")
except Exception as e:
    print(f"    ❌ models.py import FAILED: {e}")
    sys.exit(1)

try:
    from server.environment import CalTriageEnvironment
    print("    ✅ server/environment.py imports OK")
except Exception as e:
    print(f"    ❌ server/environment.py import FAILED: {e}")
    sys.exit(1)

# 2. Test environment instantiation
print("\n[2] Testing CalTriageEnvironment()...")
env = CalTriageEnvironment()
print("    ✅ Environment created")

# 3. Test reset for each task
for task in ["easy", "medium", "hard"]:
    print(f"\n[3-{task}] Testing reset(task_name='{task}')...")
    obs = env.reset(seed=42, task_name=task)
    
    assert isinstance(obs, CalObservation), f"Expected CalObservation, got {type(obs)}"
    assert obs.done == False, f"done should be False after reset, got {obs.done}"
    assert obs.reward == 0.0, f"reward should be 0.0 after reset, got {obs.reward}"
    assert len(obs.current_schedule) > 0, "Schedule should not be empty"
    assert len(obs.active_conflicts) > 0, "There should be conflicts"
    assert obs.num_conflicts == len(obs.active_conflicts), "num_conflicts mismatch"
    assert obs.task_name == task, f"task_name should be '{task}', got {obs.task_name}"
    
    print(f"    ✅ Reset OK: {len(obs.current_schedule)} meetings, {obs.num_conflicts} conflicts")
    print(f"    Meetings: {[m.meeting_id + ' (' + m.title + ')' for m in obs.current_schedule[:3]]}...")
    
    # Print conflicts for debugging
    for c in obs.active_conflicts:
        print(f"    Conflict: {c.meeting_a_id} ↔ {c.meeting_b_id} ({c.overlap_minutes}min overlap)")

# 4. Test hard constraint violation → reward=0.0
print("\n[4] Testing HARD CONSTRAINT violation (cancel locked meeting)...")
obs = env.reset(seed=42, task_name="easy")
locked_meetings = [m for m in obs.current_schedule if m.is_locked]
assert len(locked_meetings) > 0, "Should have at least 1 locked meeting"
locked_id = locked_meetings[0].meeting_id
print(f"    Locked meeting: {locked_id} ({locked_meetings[0].title})")

action = CalAction(meeting_id=locked_id, action_type="cancel")
result = env.step(action)
assert result.reward == 0.0, f"Hard constraint violation should give reward=0.0, got {result.reward}"
assert result.done == True, f"Hard constraint violation should set done=True, got {result.done}"
assert result.last_action_error is not None, "Should have an error message"
print(f"    ✅ Reward={result.reward}, done={result.done}, error='{result.last_action_error}'")

# 5. Test valid action (cancel non-locked meeting)
print("\n[5] Testing valid cancel action...")
obs = env.reset(seed=42, task_name="easy")
non_locked = [m for m in obs.current_schedule if not m.is_locked]
assert len(non_locked) > 0, "Should have non-locked meetings"

# Find a non-locked meeting involved in a conflict
conflict_mtg_ids = set()
for c in obs.active_conflicts:
    conflict_mtg_ids.add(c.meeting_a_id)
    conflict_mtg_ids.add(c.meeting_b_id)

target = None
for m in non_locked:
    if m.meeting_id in conflict_mtg_ids:
        target = m
        break

if target:
    print(f"    Cancelling: {target.meeting_id} ({target.title})")
    action = CalAction(meeting_id=target.meeting_id, action_type="cancel")
    result = env.step(action)
    assert 0.0 <= result.reward <= 1.0, f"Reward out of range: {result.reward}"
    print(f"    ✅ Reward={result.reward}, done={result.done}, conflicts_left={result.num_conflicts}")
else:
    print("    ⚠️ No non-locked meeting in conflict (unexpected)")

# 6. Test reschedule action
print("\n[6] Testing reschedule action...")
obs = env.reset(seed=42, task_name="easy")
non_locked_in_conflict = [m for m in obs.current_schedule 
                          if not m.is_locked and m.meeting_id in conflict_mtg_ids]

if non_locked_in_conflict:
    target = non_locked_in_conflict[0]
    print(f"    Rescheduling: {target.meeting_id} ({target.title}) to 16:00")
    action = CalAction(
        meeting_id=target.meeting_id,
        action_type="reschedule",
        new_start_hour=16,
        new_start_minute=0,
    )
    result = env.step(action)
    assert 0.0 <= result.reward <= 1.0, f"Reward out of range: {result.reward}"
    print(f"    ✅ Reward={result.reward}, done={result.done}, conflicts_left={result.num_conflicts}")

# 7. Test full episode (resolve all conflicts)
print("\n[7] Testing full easy episode (cancel all conflicting non-locked meetings)...")
obs = env.reset(seed=42, task_name="easy")
step = 0
while not obs.done and step < 10:
    # Find non-locked meeting in a conflict
    conflict_ids = set()
    for c in obs.active_conflicts:
        conflict_ids.add(c.meeting_a_id)
        conflict_ids.add(c.meeting_b_id)
    
    target = None
    for m in obs.current_schedule:
        if not m.is_locked and m.meeting_id in conflict_ids:
            target = m
            break
    
    if target is None:
        break
    
    action = CalAction(meeting_id=target.meeting_id, action_type="cancel")
    obs = env.step(action)
    step += 1
    print(f"    Step {step}: cancelled {target.meeting_id}, reward={obs.reward}, conflicts={obs.num_conflicts}, done={obs.done}")
    assert 0.0 <= obs.reward <= 1.0, f"REWARD OUT OF RANGE: {obs.reward}"

# 8. Verify state
print("\n[8] Testing state property...")
state = env.state
assert isinstance(state, CalState), f"Expected CalState, got {type(state)}"
assert state.step_count == step, f"Step count mismatch: {state.step_count} vs {step}"
print(f"    ✅ State: steps={state.step_count}, resolved={state.conflicts_resolved}, "
      f"task={state.task_name}")

# 9. Test server app import
print("\n[9] Testing server/app.py import...")
try:
    from server.app import app
    print(f"    ✅ FastAPI app created: {app}")
except Exception as e:
    print(f"    ❌ app.py import FAILED: {e}")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED ✅")
print("=" * 60)
