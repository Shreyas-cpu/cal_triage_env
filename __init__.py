"""
CalTriage-Env — Smart Calendar Triage RL Environment.

A real-world OpenEnv environment where an AI executive assistant resolves
overlapping meeting conflicts on a daily schedule while respecting constraints.

Usage:
    >>> from cal_triage_env import CalTriageEnv, CalAction
    >>>
    >>> with CalTriageEnv(base_url="http://localhost:8000").sync() as env:
    ...     result = env.reset(task_name="easy")
    ...     result = env.step(CalAction(
    ...         meeting_id="mtg_004",
    ...         action_type="cancel",
    ...     ))
"""

from .client import CalTriageEnv
from .models import CalAction, CalObservation, CalState

__all__ = ["CalTriageEnv", "CalAction", "CalObservation", "CalState"]
