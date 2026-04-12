"""
CalTriage Environment  Core Business Logic.

An RL environment where an AI executive assistant must resolve overlapping
meeting conflicts on a daily calendar while respecting hard and soft constraints.

Implements the OpenEnv Environment interface:
    - reset()   generates a conflicted daily schedule
    - step()    applies an agent action, returns observation + reward [0,1]
    - state     episode metadata
"""

import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# Our strictly-typed Pydantic models
from models import (
    CalAction,
    CalObservation,
    CalState,
    Conflict,
    Constraint,
    Meeting,
    TimeSlot,
)

# 
# Meeting template pool  realistic executive-assistant scenario
# 

MEETING_TEMPLATES: List[Dict[str, Any]] = [
    {"title": "CEO 1-on-1",          "priority": "critical", "duration": 60,  "attendees": ["ceo", "you"],              "locked": True},
    {"title": "Board Prep",          "priority": "critical", "duration": 60,  "attendees": ["ceo", "cfo", "you"],       "locked": True},
    {"title": "Investor Call",       "priority": "critical", "duration": 60,  "attendees": ["ceo", "investors"],         "locked": True},
    {"title": "Team Standup",        "priority": "high",     "duration": 30,  "attendees": ["engineering_team"],         "locked": False},
    {"title": "Client Call",         "priority": "high",     "duration": 60,  "attendees": ["client_a", "pm"],           "locked": False},
    {"title": "Sprint Planning",     "priority": "high",     "duration": 90,  "attendees": ["engineering_team", "pm"],   "locked": False},
    {"title": "Budget Review",       "priority": "high",     "duration": 60,  "attendees": ["cfo", "finance"],           "locked": False},
    {"title": "Project Review",      "priority": "medium",   "duration": 60,  "attendees": ["pm", "engineering_lead"],   "locked": False},
    {"title": "Design Review",       "priority": "medium",   "duration": 60,  "attendees": ["design_team", "pm"],        "locked": False},
    {"title": "HR Sync",             "priority": "medium",   "duration": 30,  "attendees": ["hr", "you"],                "locked": False},
    {"title": "Marketing Sync",      "priority": "medium",   "duration": 30,  "attendees": ["marketing", "pm"],          "locked": False},
    {"title": "1-on-1 with Report",  "priority": "medium",   "duration": 30,  "attendees": ["direct_report", "you"],     "locked": False},
    {"title": "Lunch Break",         "priority": "low",      "duration": 30,  "attendees": ["you"],                      "locked": False},
    {"title": "Team Lunch",          "priority": "low",      "duration": 60,  "attendees": ["team"],                     "locked": False},
    {"title": "Strategy Session",    "priority": "high",     "duration": 90,  "attendees": ["leadership", "you"],        "locked": False},
]

# 
# Task configurations  easy / medium / hard
# 

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "num_meetings": 6,
        "num_conflicts": 2,       # 2 simple pair-wise overlaps
        "num_locked": 1,
        "soft_constraints": [],   # no soft constraints
        "max_steps": 5,
        "day_start": 9,           # 9 AM
        "day_end": 17,            # 5 PM
    },
    "medium": {
        "num_meetings": 10,
        "num_conflicts": 4,
        "num_locked": 2,
        "soft_constraints": ["preferred_time", "lunch_block"],
        "max_steps": 8,
        "day_start": 9,
        "day_end": 17,
    },
    "hard": {
        "num_meetings": 15,
        "num_conflicts": 6,
        "num_locked": 3,
        "soft_constraints": ["preferred_time", "lunch_block", "min_gap", "preferred_time"],
        "max_steps": 12,
        "day_start": 8,           # 8 AM
        "day_end": 18,            # 6 PM
    },
}
class StepResult:
    def __init__(self, observation: CalObservation, reward: float, done: bool, info: Dict[str, Any]) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info

    def __iter__(self):
        yield self.observation
        yield self.reward
        yield self.done
        yield self.info

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        return (self.observation, self.reward, self.done, self.info)[index]

    def __getattr__(self, name: str):
        return getattr(self.observation, name)


class CalTriageEnvironment:
    """
    Calendar Triage RL Environment.

    The agent acts as an AI executive assistant resolving overlapping meeting
    conflicts.  Each step the agent picks ONE meeting and chooses to
    *reschedule*, *cancel*, or *keep* it.  The episode ends when all conflicts
    are resolved, a hard constraint is violated, or the step budget is
    exhausted.

    Reward is normalized to [0.0, 1.0]:
        0.0   hard-constraint violation or invalid action
        0.5   all conflicts resolved but soft-constraint(s) violated
        1.0   perfect resolution (no conflicts, no violations)
        0.00.3  partial progress (intermediate steps)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._state: CalState = CalState(episode_id=str(uuid4()), step_count=0)
        self._schedule: List[Meeting] = []
        self._cancelled_ids: List[str] = []
        self._constraints: List[Constraint] = []
        self._initial_conflict_count: int = 0
        self._task_name: str = "easy"
        self._hard_violated: bool = False
        self._rng: random.Random = random.Random(42)

    # 
    #  OpenEnv API  reset()
    # 

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CalObservation:
        """
        Generate a fresh daily schedule with deliberate conflicts.

        Keyword Args:
            task_name: "easy" | "medium" | "hard" (default "easy")
        """
        self._task_name = kwargs.get("task_name", "easy")
        if self._task_name not in TASK_CONFIGS:
            self._task_name = "easy"

        config = TASK_CONFIGS[self._task_name]
        self._rng = random.Random(seed if seed is not None else random.randint(0, 2**31))
        self._cancelled_ids = []
        self._hard_violated = False

        # Build schedule with deliberate overlaps
        self._generate_schedule(config)

        # Build constraints
        self._generate_constraints(config)

        # Detect initial conflicts
        conflicts = self._detect_conflicts()
        self._initial_conflict_count = len(conflicts)

        # Initialize state
        self._state = CalState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            max_steps=config["max_steps"],
            initial_conflict_count=self._initial_conflict_count,
            conflicts_resolved=0,
            hard_constraint_violations=0,
            soft_constraint_violations=0,
            cancelled_meetings=0,
            task_name=self._task_name,
        )

        return CalObservation(
            current_schedule=self._active_meetings(),
            active_conflicts=conflicts,
            constraints=self._constraints,
            num_conflicts=len(conflicts),
            done=False,
            reward=0.0,
            last_action_error=None,
            task_name=self._task_name,
        )

    # 
    #  OpenEnv API  step()
    # 

    def step(
        self,
        action: CalAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> StepResult:
        """
        Apply the agent's action and return the updated observation.

        Reward (strictly 0.01.0):
            0.0  hard constraint violated or invalid action (done=True)
            0.5  all conflicts resolved with soft violations (done=True)
            1.0  perfect resolution (done=True)
            0.00.3  partial progress (done=False)
        """
        # Parse action into CalAction
        if isinstance(action, CalAction):
            cal_action = action
        else:
            try:
                cal_action = CalAction(**action.model_dump(exclude={"metadata"}))
            except Exception as e:
                self._state.step_count += 1
                return StepResult(self._make_error_obs(f"Invalid action format: {e}"), 0.0, True, {})

        self._state.step_count += 1

        #  Validate meeting exists 
        meeting = self._find_meeting(cal_action.meeting_id)
        if meeting is None:
                return StepResult(
                    self._make_error_obs(
                        f"Meeting '{cal_action.meeting_id}' not found in schedule"
                    ),
                    0.0,
                    True,
                    {},
                )

        #  Apply action 
        if cal_action.action_type == "keep":
            pass  # no-op

        elif cal_action.action_type == "cancel":
            if meeting.is_locked:
                self._hard_violated = True
                self._state.hard_constraint_violations += 1
                return StepResult(
                    self._make_terminal_obs(
                        reward=0.0,
                        error="HARD CONSTRAINT VIOLATED: Cannot cancel a locked meeting"
                    ),
                    0.0,
                    True,
                    {},
                )
            self._cancelled_ids.append(meeting.meeting_id)
            self._state.cancelled_meetings += 1

        elif cal_action.action_type == "reschedule":
            if meeting.is_locked:
                self._hard_violated = True
                self._state.hard_constraint_violations += 1
                return StepResult(
                    self._make_terminal_obs(
                        reward=0.0,
                        error="HARD CONSTRAINT VIOLATED: Cannot reschedule a locked meeting"
                    ),
                    0.0,
                    True,
                    {},
                )
            if cal_action.new_start_hour is None or cal_action.new_start_minute is None:
                return StepResult(
                    self._make_error_obs(
                        "Reschedule requires new_start_hour and new_start_minute"
                    ),
                    0.0,
                    True,
                    {},
                )
            # Apply the reschedule
            self._reschedule_meeting(
                meeting.meeting_id,
                cal_action.new_start_hour,
                cal_action.new_start_minute,
            )
        else:
            return StepResult(
                self._make_error_obs(
                    f"Unknown action_type: '{cal_action.action_type}'"
                ),
                0.0,
                True,
                {},
            )

        #  Recalculate state 
        conflicts = self._detect_conflicts()
        current_resolved = self._initial_conflict_count - len(conflicts)
        self._state.conflicts_resolved = max(0, current_resolved)

        soft_violations = self._count_soft_violations()
        self._state.soft_constraint_violations = soft_violations

        all_resolved = len(conflicts) == 0
        out_of_steps = self._state.step_count >= self._state.max_steps
        done = all_resolved or out_of_steps

        #  Compute reward  [0.0  1.0] 
        reward = self._compute_reward(
            conflicts_remaining=len(conflicts),
            soft_violations=soft_violations,
            all_resolved=all_resolved,
            done=done,
        )

        return StepResult(
            CalObservation(
                current_schedule=self._active_meetings(),
                active_conflicts=conflicts,
                constraints=self._constraints,
                num_conflicts=len(conflicts),
                done=done,
                reward=reward,
                last_action_error=None,
                task_name=self._task_name,
            ),
            float(reward),
            done,
            {"step": self._state.step_count, "conflicts_remaining": len(conflicts)},
        )

    # 
    #  OpenEnv API  state (property)
    # 

    @property
    def state(self) -> CalState:
        """Return current episode metadata."""
        return self._state

    def get_state(self) -> Dict[str, Any]:
        return self._state.model_dump()

    # 
    #  PRIVATE HELPERS
    # 

    #  Schedule Generation 

    def _generate_schedule(self, config: Dict[str, Any]) -> None:
        """Build a daily schedule with deliberate overlaps."""
        num_meetings = config["num_meetings"]
        num_conflicts = config["num_conflicts"]
        num_locked = config["num_locked"]
        day_start = config["day_start"]
        day_end = config["day_end"]

        # Shuffle and pick from template pool
        templates = list(MEETING_TEMPLATES)
        self._rng.shuffle(templates)
        selected = templates[:num_meetings]

        meetings: List[Meeting] = []
        used_slots: List[TimeSlot] = []

        # Phase 1: Place non-conflicting meetings first
        non_conflict_count = num_meetings - num_conflicts
        for i in range(non_conflict_count):
            tmpl = selected[i]
            slot = self._find_free_slot(
                used_slots, tmpl["duration"], day_start, day_end
            )
            locked = i < num_locked  # first N meetings are locked
            meeting = Meeting(
                meeting_id=f"mtg_{i:03d}",
                title=tmpl["title"],
                time_slot=slot,
                priority=tmpl["priority"],
                is_locked=locked,
                attendees=tmpl["attendees"],
                preferred_hours=(9, 12) if tmpl["priority"] in ("critical", "high") else None,
            )
            meetings.append(meeting)
            used_slots.append(slot)

        # Phase 2: Create deliberate overlaps
        for j in range(num_conflicts):
            idx = non_conflict_count + j
            tmpl = selected[idx] if idx < len(selected) else self._rng.choice(templates)

            # Pick a random existing meeting to overlap with
            target = self._rng.choice(meetings[:non_conflict_count])
            slot = self._create_overlapping_slot(
                target.time_slot, tmpl["duration"], day_start, day_end
            )

            meeting = Meeting(
                meeting_id=f"mtg_{idx:03d}",
                title=tmpl["title"],
                time_slot=slot,
                priority=tmpl["priority"],
                is_locked=False,  # overlapping meetings are never locked
                attendees=tmpl["attendees"],
                preferred_hours=None,
            )
            meetings.append(meeting)

        self._schedule = meetings

    def _find_free_slot(
        self,
        used: List[TimeSlot],
        duration: int,
        day_start: int,
        day_end: int,
    ) -> TimeSlot:
        """Find a non-overlapping time slot for a meeting."""
        for _ in range(200):  # safety limit
            hour = self._rng.randint(day_start, day_end - 1)
            minute = self._rng.choice([0, 30])
            # Ensure meeting fits within the day
            end_total = hour * 60 + minute + duration
            if end_total > day_end * 60:
                continue
            candidate = TimeSlot(
                start_hour=hour, start_minute=minute, duration_min=duration
            )
            if not any(candidate.overlaps(s) for s in used):
                return candidate
        # Fallback: place at day_start (may overlap, but that's OK for variety)
        return TimeSlot(
            start_hour=day_start, start_minute=0, duration_min=duration
        )

    def _create_overlapping_slot(
        self,
        target: TimeSlot,
        duration: int,
        day_start: int,
        day_end: int,
    ) -> TimeSlot:
        """Create a time slot that deliberately overlaps with the target."""
        target_start = target.start_hour * 60 + target.start_minute
        target_end = target_start + target.duration_min

        # Start the overlapping meeting partway through the target
        offset = self._rng.choice([15, 30])  # start 15 or 30 min into target
        overlap_start = target_start + offset
        overlap_start_hour = overlap_start // 60
        overlap_start_min = (overlap_start // 30) * 30 % 60  # snap to 30-min grid
        overlap_start_hour = (overlap_start // 30 * 30) // 60

        # Clamp to day boundaries
        end_total = overlap_start_hour * 60 + overlap_start_min + duration
        if end_total > day_end * 60:
            # Shift earlier
            overlap_start_hour = max(day_start, target.start_hour)
            overlap_start_min = 0 if target.start_minute == 30 else 30

        return TimeSlot(
            start_hour=max(day_start, min(overlap_start_hour, day_end - 1)),
            start_minute=overlap_start_min if overlap_start_min in (0, 30) else 0,
            duration_min=duration,
        )

    #  Constraint Generation 

    def _generate_constraints(self, config: Dict[str, Any]) -> None:
        """Build constraints based on task difficulty."""
        constraints: List[Constraint] = []

        # Hard constraints: locked meetings cannot be moved or cancelled
        for m in self._schedule:
            if m.is_locked:
                constraints.append(
                    Constraint(
                        constraint_type="no_move",
                        target=m.meeting_id,
                        value=None,
                        is_hard=True,
                    )
                )
                constraints.append(
                    Constraint(
                        constraint_type="no_cancel",
                        target=m.meeting_id,
                        value=None,
                        is_hard=True,
                    )
                )

        # Soft constraints based on config
        for sc_type in config.get("soft_constraints", []):
            if sc_type == "lunch_block":
                constraints.append(
                    Constraint(
                        constraint_type="lunch_block",
                        target="global",
                        value={"start_hour": 12, "end_hour": 13},
                        is_hard=False,
                    )
                )
            elif sc_type == "preferred_time":
                # Pick a random non-locked meeting and add preferred time
                non_locked = [m for m in self._schedule if not m.is_locked]
                if non_locked:
                    target_mtg = self._rng.choice(non_locked)
                    constraints.append(
                        Constraint(
                            constraint_type="preferred_time",
                            target=target_mtg.meeting_id,
                            value={"start_hour": 9, "end_hour": 12},
                            is_hard=False,
                        )
                    )
            elif sc_type == "min_gap":
                constraints.append(
                    Constraint(
                        constraint_type="min_gap",
                        target="global",
                        value={"gap_minutes": 15},
                        is_hard=False,
                    )
                )

        self._constraints = constraints

    #  Conflict Detection 

    def _detect_conflicts(self) -> List[Conflict]:
        """Find all pairwise overlaps among active (non-cancelled) meetings."""
        active = self._active_meetings()
        conflicts: List[Conflict] = []
        conflict_counter = 0

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                overlap = active[i].time_slot.overlap_minutes(active[j].time_slot)
                if overlap > 0:
                    conflicts.append(
                        Conflict(
                            conflict_id=f"conf_{conflict_counter:03d}",
                            meeting_a_id=active[i].meeting_id,
                            meeting_b_id=active[j].meeting_id,
                            overlap_minutes=overlap,
                        )
                    )
                    conflict_counter += 1

        return conflicts

    #  Meeting Helpers 

    def _active_meetings(self) -> List[Meeting]:
        """Return all non-cancelled meetings."""
        return [m for m in self._schedule if m.meeting_id not in self._cancelled_ids]

    def _find_meeting(self, meeting_id: str) -> Optional[Meeting]:
        """Find an active meeting by ID."""
        for m in self._active_meetings():
            if m.meeting_id == meeting_id:
                return m
        return None

    def _reschedule_meeting(
        self, meeting_id: str, new_hour: int, new_minute: int
    ) -> None:
        """Move a meeting to a new time, keeping its duration."""
        for i, m in enumerate(self._schedule):
            if m.meeting_id == meeting_id:
                self._schedule[i] = m.model_copy(
                    update={
                        "time_slot": TimeSlot(
                            start_hour=new_hour,
                            start_minute=new_minute,
                            duration_min=m.time_slot.duration_min,
                        )
                    }
                )
                break

    #  Soft Constraint Checking 

    def _count_soft_violations(self) -> int:
        """Count the number of soft constraint violations in the current schedule."""
        violations = 0
        active = self._active_meetings()

        for constraint in self._constraints:
            if constraint.is_hard:
                continue  # hard constraints are checked during action application

            if constraint.constraint_type == "lunch_block":
                # Check if any meeting overlaps with the lunch block (12-1 PM)
                lunch_start = constraint.value["start_hour"] * 60
                lunch_end = constraint.value["end_hour"] * 60
                for m in active:
                    m_start = m.time_slot.start_hour * 60 + m.time_slot.start_minute
                    m_end = m_start + m.time_slot.duration_min
                    if m_start < lunch_end and m_end > lunch_start:
                        # Only count if this isn't the lunch meeting itself
                        if "lunch" not in m.title.lower():
                            violations += 1
                            break  # count once per constraint

            elif constraint.constraint_type == "preferred_time":
                target_mtg = self._find_meeting(constraint.target)
                if target_mtg is not None:
                    pref_start = constraint.value["start_hour"]
                    pref_end = constraint.value["end_hour"]
                    mtg_hour = target_mtg.time_slot.start_hour
                    if mtg_hour < pref_start or mtg_hour >= pref_end:
                        violations += 1

            elif constraint.constraint_type == "min_gap":
                gap_min = constraint.value["gap_minutes"]
                # Check if any two adjacent meetings have less than the required gap
                sorted_meetings = sorted(
                    active,
                    key=lambda m: m.time_slot.start_hour * 60 + m.time_slot.start_minute,
                )
                for k in range(len(sorted_meetings) - 1):
                    curr_end = (
                        sorted_meetings[k].time_slot.start_hour * 60
                        + sorted_meetings[k].time_slot.start_minute
                        + sorted_meetings[k].time_slot.duration_min
                    )
                    next_start = (
                        sorted_meetings[k + 1].time_slot.start_hour * 60
                        + sorted_meetings[k + 1].time_slot.start_minute
                    )
                    gap = next_start - curr_end
                    if 0 < gap < gap_min:
                        violations += 1
                        break  # count once per constraint

        return violations

    #  Reward Computation 

    def _compute_reward(
        self,
        conflicts_remaining: int,
        soft_violations: int,
        all_resolved: bool,
        done: bool,
    ) -> float:
        """
        Compute a normalized reward strictly in [0.0, 1.0].

        Reward schedule:
            - All conflicts resolved, zero soft violations    1.0
            - All conflicts resolved, with soft violations     0.5
            - Partial progress (conflicts still remain)        0.00.3
            - No progress at all                               0.0
        """
        if self._initial_conflict_count == 0:
            return 1.0  # no conflicts to begin with

        if all_resolved:
            if soft_violations > 0:
                # Penalty for soft violations, but still partially successful
                return max(0.3, 0.5 - 0.05 * soft_violations)
            return 1.0

        # Partial progress: proportional to conflicts resolved
        resolved = self._initial_conflict_count - conflicts_remaining
        progress = resolved / self._initial_conflict_count
        # Scale partial rewards to [0.0, 0.3] range
        reward = round(progress * 0.3, 2)

        # Small penalty for excessive cancellations
        cancel_penalty = self._state.cancelled_meetings * 0.02
        reward = max(0.0, reward - cancel_penalty)

        return min(reward, 0.3)  # cap intermediate rewards

    #  Observation Builders 

    def _make_error_obs(self, error_msg: str) -> CalObservation:
        """Return an observation for an invalid action (reward=0.0, done=True)."""
        conflicts = self._detect_conflicts()
        return CalObservation(
            current_schedule=self._active_meetings(),
            active_conflicts=conflicts,
            constraints=self._constraints,
            num_conflicts=len(conflicts),
            done=True,
            reward=0.0,
            last_action_error=error_msg,
            task_name=self._task_name,
        )

    def _make_terminal_obs(self, reward: float, error: Optional[str] = None) -> CalObservation:
        """Return a terminal observation (done=True) with the given reward."""
        conflicts = self._detect_conflicts()
        return CalObservation(
            current_schedule=self._active_meetings(),
            active_conflicts=conflicts,
            constraints=self._constraints,
            num_conflicts=len(conflicts),
            done=True,
            reward=max(0.0, min(1.0, reward)),  # clamp [0, 1]
            last_action_error=error,
            task_name=self._task_name,
        )

