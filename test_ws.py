"""WebSocket E2E test — verifies stateful reset/step over WebSocket."""
import asyncio
import json

async def test_ws():
    import websockets
    
    ws_url = "ws://localhost:8000/ws"
    
    for task in ["easy", "medium", "hard"]:
        print(f"\n=== [{task.upper()}] WebSocket Test ===")
        
        async with websockets.connect(ws_url) as ws:
            # RESET
            await ws.send(json.dumps({"type": "reset", "data": {"task_name": task, "seed": 42}}))
            resp = json.loads(await ws.recv())
            
            # Response: {type: "observation", data: {observation: {...}, reward: ..., done: ...}}
            ws_data = resp.get("data", resp)
            obs = ws_data.get("observation", ws_data)
            reward = ws_data.get("reward", 0.0)
            done = ws_data.get("done", False)
            
            n_schedule = len(obs.get("current_schedule", []))
            n_conflicts = obs.get("num_conflicts", 0)
            print(f"  Reset: {n_schedule} meetings, {n_conflicts} conflicts, done={done}, reward={reward}")
            
            assert n_schedule > 0, f"Expected meetings but got 0 for {task}"
            assert n_conflicts > 0, f"Expected conflicts but got 0 for {task}"
            
            # STEP — cancel non-locked meetings in conflicts
            step_count = 0
            while obs.get("active_conflicts"):
                conflicts = obs.get("active_conflicts", [])
                schedule = obs.get("current_schedule", [])
                mtg_map = {m["meeting_id"]: m for m in schedule}
                
                c = conflicts[0]
                target = None
                for mid in [c["meeting_b_id"], c["meeting_a_id"]]:
                    m = mtg_map.get(mid)
                    if m and not m.get("is_locked"):
                        target = m
                        break
                if not target:
                    print("  No non-locked target found, breaking")
                    break
                
                action = {"meeting_id": target["meeting_id"], "action_type": "cancel"}
                await ws.send(json.dumps({"type": "step", "data": action}))
                resp = json.loads(await ws.recv())
                
                if resp.get("type") == "error":
                    print(f"  ERROR: {resp.get('data')}")
                    break
                
                ws_data = resp.get("data", resp)
                obs = ws_data.get("observation", ws_data)
                reward = ws_data.get("reward", 0.0)
                done = ws_data.get("done", False)
                step_count += 1
                
                n_conflicts = obs.get("num_conflicts", len(obs.get("active_conflicts", [])))
                print(f"  Step {step_count}: cancel {target['meeting_id']} -> reward={reward}, done={done}, conflicts={n_conflicts}")
                
                assert 0.0 <= float(reward or 0) <= 1.0, f"REWARD OUT OF BOUNDS: {reward}"
                
                if done:
                    break
            
            # STATE
            await ws.send(json.dumps({"type": "state"}))
            resp = json.loads(await ws.recv())
            state = resp.get("data", resp)
            print(f"  State: steps={state.get('step_count')}, resolved={state.get('conflicts_resolved')}")
            
            # CLOSE
            await ws.send(json.dumps({"type": "close"}))
    
    print("\n=== ALL WebSocket TESTS PASSED ===")

asyncio.run(test_ws())
