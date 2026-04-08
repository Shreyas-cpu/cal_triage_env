"""
Data models for the CalTriage Environment.

CalTriage-Env is a real-world RL environment where an AI executive assistant
must resolve overlapping meeting conflicts on a daily calendar schedule
while strictly obeying user constraints.

All models use strict Pydantic typing as required by the OpenEnv specification.
"""

from typing import Any, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

# OpenEnv base types — Action, Observation, State
from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Supporting Sub-Models (not OpenEnv base types, just Pydantic data containers)
# ---------------------------------------------------------------------------


class TimeSlot(BaseModel):
    """Represents a time window on the calendar with 30-minute granularity."""

    start_hour: int = Field(
        ..., ge=0, le=23, description="Hour of day (0-23)"
    )
    start_minute: int = Field(
        ..., description="Minute of hour (0 or 30)"
    )
    duration_min: int = Field(
        ..., ge=30, le=180, description="Duration in minutes (30-180)"
    )

    @field_validator("start_minute")
    @classmethod
    def validate_minute_granularity(cls, v: int) -> int:
        if v not in (0, 30):
            raise ValueError("start_minute must be 0 or 30 (30-min granularity)")
        return v

    @property
    def end_hour(self) -> int:
        total_minutes = self.start_hour * 60 + self.start_minute + self.duration_min
        return total_minutes // 60

    @property
    def end_minute(self) -> int:
        total_minutes = self.start_hour * 60 + self.start_minute + self.duration_min
        return total_minutes % 60

    def overlaps(self, other: "TimeSlot") -> bool:
        """Check if this time slot overlaps with another."""
        self_start = self.start_hour * 60 + self.start_minute
        self_end = self_start + self.duration_min
        other_start = other.start_hour * 60 + other.start_minute
        other_end = other_start + other.duration_min
        return self_start < other_end and other_start < self_end

    def overlap_minutes(self, other: "TimeSlot") -> int:
        """Calculate the number of overlapping minutes with another slot."""
        self_start = self.start_hour * 60 + self.start_minute
        self_end = self_start + self.duration_min
        other_start = other.start_hour * 60 + other.start_minute
        other_end = other_start + other.duration_min
        overlap = min(self_end, other_end) - max(self_start, other_start)
        return max(0, overlap)

    def __str__(self) -> str:
        return f"{self.start_hour:02d}:{self.start_minute:02d}-{self.end_hour:02d}:{self.end_minute:02d}"


class Meeting(BaseModel):
    """Represents a single calendar meeting."""

    meeting_id: str = Field(..., description="Unique meeting identifier, e.g. 'mtg_001'")
    title: str = Field(..., description="Meeting title, e.g. 'CEO 1-on-1'")
    time_slot: TimeSlot = Field(..., description="Scheduled time window")
    priority: Literal["critical", "high", "medium", "low"] = Field(
        ..., description="Meeting priority level"
    )
    is_locked: bool = Field(
        default=False,
        description="If True, this meeting CANNOT be moved or cancelled (hard constraint)",
    )
    attendees: List[str] = Field(
        default_factory=list, description="List of attendee names"
    )
    preferred_hours: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Preferred hour range as (start_hour, end_hour), e.g. (9, 12) for morning",
    )


class Conflict(BaseModel):
    """Represents an overlap between two meetings."""

    conflict_id: str = Field(..., description="Unique conflict identifier")
    meeting_a_id: str = Field(..., description="First overlapping meeting ID")
    meeting_b_id: str = Field(..., description="Second overlapping meeting ID")
    overlap_minutes: int = Field(
        ..., ge=1, description="Number of minutes the meetings overlap"
    )


class Constraint(BaseModel):
    """A rule the agent must respect when resolving conflicts."""

    constraint_type: Literal[
        "no_move", "no_cancel", "preferred_time", "min_gap", "lunch_block"
    ] = Field(..., description="Type of constraint")
    target: str = Field(
        ...,
        description="Target meeting_id, or 'global' for schedule-wide constraints",
    )
    value: Any = Field(
        default=None,
        description="Constraint parameter: e.g. (12,13) for lunch block, 15 for min_gap",
    )
    is_hard: bool = Field(
        default=True,
        description="If True, violation kills reward to 0.0; if False, soft penalty",
    )


# ---------------------------------------------------------------------------
# Core OpenEnv Models (inherit from framework base types)
# ---------------------------------------------------------------------------


class CalAction(Action):
    """
    The agent's decision for resolving a calendar conflict.

    Each step, the agent acts on ONE meeting: keep it in place,
    reschedule it to a new time, or cancel it entirely.
    """

    meeting_id: str = Field(
        ..., description="ID of the meeting to act upon"
    )
    action_type: Literal["reschedule", "cancel", "keep"] = Field(
        ..., description="Action to take: reschedule, cancel, or keep"
    )
    new_start_hour: Optional[int] = Field(
        default=None,
        ge=0,
        le=23,
        description="New start hour (required if action_type is 'reschedule')",
    )
    new_start_minute: Optional[int] = Field(
        default=None,
        description="New start minute — 0 or 30 (required if action_type is 'reschedule')",
    )

    @field_validator("new_start_minute")
    @classmethod
    def validate_new_minute(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v not in (0, 30):
            raise ValueError("new_start_minute must be 0 or 30")
        return v


class CalObservation(Observation):
    """
    The environment's response after each step (or on reset).

    Provides the agent with the full current schedule, remaining conflicts,
    active constraints, and feedback from the last action.

    Inherits from Observation which provides:
        - done (bool): Whether the episode has ended
        - reward (float): Reward for this step, normalized 0.0-1.0
        - metadata (dict): Additional metadata
    """

    current_schedule: List[Meeting] = Field(
        default_factory=list,
        description="All active (non-cancelled) meetings on the calendar",
    )
    active_conflicts: List[Conflict] = Field(
        default_factory=list,
        description="Remaining unresolved time overlaps between meetings",
    )
    constraints: List[Constraint] = Field(
        default_factory=list,
        description="Rules the agent must obey",
    )
    num_conflicts: int = Field(
        default=0, ge=0, description="Number of active conflicts"
    )
    last_action_error: Optional[str] = Field(
        default=None,
        description="Error from last action, e.g. 'Cannot cancel a locked meeting'",
    )
    task_name: str = Field(
        default="easy",
        description="Current task difficulty: easy, medium, or hard",
    )


class CalState(State):
    """
    Internal episode metadata tracked by the environment.

    Inherits from State which provides:
        - episode_id (str): Unique episode identifier
        - step_count (int): Current step number
    """

    max_steps: int = Field(
        default=5, description="Maximum steps allowed in this episode"
    )
    initial_conflict_count: int = Field(
        default=0, ge=0, description="Number of conflicts at episode start"
    )
    conflicts_resolved: int = Field(
        default=0, ge=0, description="Number of conflicts resolved so far"
    )
    hard_constraint_violations: int = Field(
        default=0, ge=0, description="Count of hard constraint violations"
    )
    soft_constraint_violations: int = Field(
        default=0, ge=0, description="Count of soft constraint violations"
    )
    cancelled_meetings: int = Field(
        default=0, ge=0, description="Number of meetings cancelled"
    )
    task_name: str = Field(
        default="easy", description="Current task: easy, medium, or hard"
    )
