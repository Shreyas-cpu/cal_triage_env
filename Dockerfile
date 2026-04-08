# CalTriage-Env Dockerfile
# ⚠️  This file MUST remain at the project root (not in server/)
#     as required by the hackathon pre-submission validator.

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for better Docker layer caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv.git@v0.2.3" \
    "fastapi>=0.115.0" \
    "pydantic>=2.0.0" \
    "uvicorn>=0.24.0" \
    "requests>=2.31.0" \
    "websockets>=12.0"

# Copy application code
COPY models.py ./
COPY client.py ./
COPY __init__.py ./
COPY openenv.yaml ./
COPY README.md ./
COPY server/ ./server/

# Expose the server port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Launch the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
