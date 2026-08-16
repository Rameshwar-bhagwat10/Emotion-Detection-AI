# ==============================================================================
# Makefile — Emotion Detection AI Development & Operations
# ==============================================================================

PYTHON ?= ./.venv/Scripts/python.exe
UV ?= ./.venv/Scripts/uv.exe
PNPM ?= pnpm

.PHONY: help install dev web api worker infra-up infra-down test lint format typecheck health clean

help:
	@echo "Available commands:"
	@echo "  make install     - Install all Python and Node.js dependencies"
	@echo "  make dev         - Start development servers (Web & API)"
	@echo "  make web         - Start Next.js web application"
	@echo "  make api         - Start FastAPI backend server"
	@echo "  make worker      - Start Celery worker"
	@echo "  make infra-up    - Start PostgreSQL & Redis with Docker Compose"
	@echo "  make infra-down  - Stop Docker Compose services"
	@echo "  make test        - Run all test suites with pytest"
	@echo "  make lint        - Run Python (Ruff) and TypeScript (ESLint) linters"
	@echo "  make format      - Format Python (Black/Ruff) and Frontend (Prettier)"
	@echo "  make typecheck   - Run type checkers (mypy & tsc)"
	@echo "  make health      - Check API & dependency health status"
	@echo "  make clean       - Clean temporary build and cache artifacts"

# Dependency Management
install:
	@echo "--> Installing Node.js dependencies..."
	cd apps/web && $(PNPM) install
	@echo "--> Installing Python dependencies..."
	$(UV) pip install -e .

# Application Services
web:
	cd apps/web && $(PNPM) dev

api:
	cd apps/api && $(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker:
	cd apps/worker && $(PYTHON) -m celery -A worker.celery_app worker --loglevel=info

dev:
	@echo "Starting development environment (run make api and make web in separate terminals)"

# Infrastructure Services
infra-up:
	docker compose up -d postgres redis

infra-down:
	docker compose down

# Quality & Testing
test:
	$(PYTHON) -m pytest tests/ apps/api/tests/ apps/worker/tests/

lint:
	$(PYTHON) -m ruff check .
	cd apps/web && $(PNPM) lint

format:
	$(PYTHON) -m black .
	$(PYTHON) -m ruff check --fix .
	cd apps/web && $(PNPM) format

typecheck:
	cd apps/web && $(PNPM) exec tsc --noEmit
	$(PYTHON) -m mypy apps/ ml/

health:
	$(PYTHON) -c "import urllib.request, json; res = urllib.request.urlopen('http://localhost:8000/api/v1/health'); print(json.dumps(json.loads(res.read().decode()), indent=2))"

clean:
	@echo "--> Cleaning cache and artifacts..."
	-rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__ apps/web/.next apps/web/out
