# AI-Based Facial Expression Emotion Detection & Analytics System

Production-grade real-time facial expression analysis, emotion recognition, and behavioral analytics web application.

---

## 🏛️ System Architecture

```text
                           ┌────────────────────────┐
                           │   Next.js 15 Web App   │
                           │   (App Router, UI)     │
                           └───────────┬────────────┘
                                       │ HTTP / WebSocket
                                       ▼
                           ┌────────────────────────┐
                           │   FastAPI Backend API  │
                           │   (v1 API, REST/WS)    │
                           └─────┬────────────┬─────┘
                                 │            │
                    ┌────────────┘            └────────────┐
                    ▼                                      ▼
         ┌─────────────────────┐                ┌─────────────────────┐
         │ PostgreSQL Database │                │ Redis Message Broker│
         │ (Sessions, Reports) │                │ & Cache             │
         └─────────────────────┘                └──────────┬──────────┘
                                                           │
                                                           ▼
                                                ┌─────────────────────┐
                                                │ Celery Async Worker │
                                                │ (Batch Analysis)    │
                                                └─────────────────────┘
                                                           │
                                                           ▼
                                                ┌─────────────────────┐
                                                │  ML / Deep Learning │
                                                │  (PyTorch / OpenCV) │
                                                └─────────────────────┘
```

---

## 📋 Prerequisites

Ensure the following tools are installed on your host machine:

* **Node.js**: v20.x or higher
* **pnpm**: v9.x or higher
* **Python**: 3.11 or 3.12
* **uv** / **pip**: Package manager for Python
* **Docker & Docker Compose**: For PostgreSQL & Redis container orchestration
* **Git**: Version control

---

## 🚀 Quick Setup Guide

### 1. Clone Repository & Setup Environment
```bash
git clone <repository-url> emotion-detection-ai
cd emotion-detection-ai

# Copy environment template
cp .env.example .env
cp apps/web/.env.local.example apps/web/.env.local
```

### 2. Install Frontend Dependencies
```bash
cd apps/web
pnpm install
cd ../..
```

### 3. Setup Python Virtual Environment & Dependencies
```bash
# Create Python virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# (Linux / macOS)
# source .venv/bin/activate

# Install Python dependencies using uv or pip
pip install uv
uv pip install -e .
```

### 4. Start Infrastructure (PostgreSQL & Redis)
```bash
docker compose up -d postgres redis
```

### 5. Run Backend API
```bash
cd apps/api
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
API Documentation will be available at: `http://localhost:8000/docs`

### 6. Run Celery Background Worker
```bash
cd apps/worker
celery -A worker.celery_app worker --loglevel=info
```

### 7. Run Frontend Web Dashboard
```bash
cd apps/web
pnpm dev
```
Web application will be accessible at: `http://localhost:3000`

---

## 🔍 Verification & Health Checks

### Verify Backend & Services Health
```bash
# Using curl or browser
curl http://localhost:8000/api/v1/health

# Response format:
# {
#   "status": "ok",
#   "service": "emotion-detection-api",
#   "version": "0.1.0",
#   "environment": "development",
#   "dependencies": {
#     "database": { "status": "healthy", ... },
#     "redis": { "status": "healthy", ... }
#   }
# }
```

### Run Test Suites
```bash
# Run all unit and integration smoke tests
pytest
```

### Run Code Quality Linters & Formatters
```bash
# Python lint & format check
ruff check .
black --check .

# Frontend lint & build
cd apps/web
pnpm lint
pnpm build
```

---

## ⚙️ Environment Variables Reference

| Variable | Description | Default |
| :--- | :--- | :--- |
| `APP_ENV` | Application environment mode | `development` |
| `APP_NAME` | Service name | `emotion-detection-api` |
| `API_HOST` | API host binding | `0.0.0.0` |
| `API_PORT` | API port | `8000` |
| `NEXT_PUBLIC_API_URL` | Public backend API URL | `http://localhost:8000` |
| `NEXT_PUBLIC_WS_URL` | Public WebSocket endpoint | `ws://localhost:8000/ws` |
| `POSTGRES_DB` | Database name | `emotion_detection` |
| `POSTGRES_USER` | PostgreSQL user | `emotion_user` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `change_me` |
| `DATABASE_URL` | Asyncpg database connection URL | `postgresql+asyncpg://...` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `CELERY_BROKER_URL` | Celery task broker URL | `redis://localhost:6379/0` |
| `CELERY_RESULT_BACKEND` | Celery task result backend | `redis://localhost:6379/1` |
| `MODEL_PATH` | Directory containing model weights | `models/production` |

---

## 🛠️ Development Commands

```bash
make install     # Install all dependencies (Node & Python)
make dev         # Run local dev environment
make web         # Start Next.js frontend
make api         # Start FastAPI server
make worker      # Start Celery worker
make infra-up    # Start PostgreSQL & Redis in Docker
make infra-down  # Stop Docker containers
make test        # Run Pytest suite
make lint        # Run linters (Ruff & ESLint)
make format      # Format code (Black & Prettier)
make typecheck   # Typecheck (mypy & tsc)
make health      # Query API health status
```

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
