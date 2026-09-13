# Production Dockerfile for Emotion Detection AI Backend (FastAPI + CPU PyTorch)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8000 \
    APP_ENV=production

# Install system utilities & ffmpeg for video stream processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight CPU-only PyTorch first to keep image small & build fast
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy project specification and install remaining dependencies
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

# Copy application and model directories
COPY apps/api/ apps/api/
COPY ml/ ml/
COPY artifacts/ artifacts/

# Create directory for persistent local data/uploads/db
RUN mkdir -p /app/data /app/data/videos /app/data/processed

WORKDIR /app/apps/api

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/v1/health || exit 1

EXPOSE 8000

# Start Uvicorn binding to Render/Railway dynamic $PORT or default 8000
CMD sh -c "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"
