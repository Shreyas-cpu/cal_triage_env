"""
CalTriage Environment Client.

Provides the client for connecting to a CalTriage Environment server,
either via a remote Hugging Face Space URL or a local Docker container.

Example:
    >>> from cal_triage_env import CalTriageEnv, CalAction
    >>>
    >>> async with CalTriageEnv(base_url="http://localhost:8000") as env:
    ...     result = await env.reset(task_name="easy")
    ...     obs = result.observation
    ...     print(f"Conflicts: {obs.num_conflicts}")
    ...
    ...     result = await env.step(CalAction(
    ...         meeting_id="mtg_004",
    ...         action_type="reschedule",
    ...         new_start_hour=15,
    ...         new_start_minute=0,
    ...     ))
    ...     print(f"Reward: {result.reward}, Done: {result.done}")
"""

from openenv.core.env_client import EnvClient

from models import CalAction, CalObservation


class CalTriageEnv(EnvClient[CalAction, CalObservation]):
    """
    Client for the CalTriage Environment.

    Inherits async/sync connection management from EnvClient.
    Use .sync() for synchronous context manager.
    """

    pass  # EnvClient provides all needed functionality
