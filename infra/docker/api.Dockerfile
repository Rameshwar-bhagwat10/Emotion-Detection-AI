# Production Dockerfile for Emotion Detection AI Backend (FastAPI + CPU PyTorch)
FROM python:3.11-slim

# Set environment variables (crucial: PYTHONPATH enables ml and app imports, thread limits eliminate CPU CFS throttling)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8000 \
    APP_ENV=production \
    PYTHONPATH="/app:/app/apps/api" \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    TORCH_NUM_THREADS=2

# Install system utilities & ffmpeg for video stream processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight CPU-only PyTorch first to keep image small & build fast
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy project specification
COPY pyproject.toml README.md ./

# Copy packages and code before pip install so setuptools discovers ml and apps packages
COPY apps/ apps/
COPY ml/ ml/
COPY artifacts/ artifacts/

# Ensure Champion model weights are fully hydrated if cloned as an un-pulled Git LFS pointer
RUN if [ ! -s /app/artifacts/optimized/champion/model.pt ] || [ $(wc -c < /app/artifacts/optimized/champion/model.pt) -lt 1000 ]; then \
      echo "Hydrating Phase 09 Champion model weights from GitHub Media CDN..." && \
      curl -fSL -o /app/artifacts/optimized/champion/model.pt \
        https://media.githubusercontent.com/media/Rameshwar-bhagwat10/Emotion-Detection-AI/main/artifacts/optimized/champion/model.pt && \
      echo "Model hydration complete: $(ls -lh /app/artifacts/optimized/champion/model.pt)"; \
    fi

# Install dependencies and editable project package
RUN pip install --no-cache-dir -e .

# Create directory for persistent local data/uploads/db
RUN mkdir -p /app/data /app/data/videos /app/data/processed

# Ensure working directory remains /app so relative paths (artifacts/..., data/...) resolve cleanly
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/v1/health || exit 1

EXPOSE 8000

# Start Uvicorn binding to Render/Railway dynamic $PORT from /app
CMD sh -c "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"

