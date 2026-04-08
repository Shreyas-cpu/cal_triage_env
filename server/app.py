"""
FastAPI application for the CalTriage Environment.

Wraps the CalTriageEnvironment using OpenEnv's create_app() to expose
the standard HTTP + WebSocket endpoints: /reset, /step, /state, /health, /web

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production (Docker):
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv.core.env_server import create_app

from server.environment import CalTriageEnvironment
from models import CalAction, CalObservation

# Create the FastAPI app with web interface support
# Pass the CLASS (factory) for WebSocket session support
app = create_app(
    CalTriageEnvironment, CalAction, CalObservation, env_name="cal_triage_env"
)


def main():
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
