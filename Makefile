# Makefile for Buy Me That Look API
# Common commands for development, testing, and deployment

.PHONY: help install install-dev test test-unit test-integration test-cov lint format run docker-build docker-run docker-stop clean

# Default target
help:
	@echo "Buy Me That Look API - Available Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install development dependencies"
	@echo "  make run           Run the API server locally"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run all tests"
	@echo "  make test-unit     Run unit tests only"
	@echo "  make test-integration  Run integration tests only"
	@echo "  make test-cov      Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code with black and isort"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo "  make docker-stop   Stop Docker container"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Remove cache and temporary files"

# ============================================================================
# Installation
# ============================================================================

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-asyncio httpx black flake8 isort mypy

# ============================================================================
# Running
# ============================================================================

run:
	python main.py

run-dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

# ============================================================================
# Testing
# ============================================================================

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

test-cov:
	pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v --ignore=tests/integration -x

# ============================================================================
# Code Quality
# ============================================================================

lint:
	flake8 app/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy app/ --ignore-missing-imports

format:
	black app/ tests/ --line-length=100
	isort app/ tests/ --profile=black

check: lint test
	@echo "All checks passed!"

# ============================================================================
# Docker
# ============================================================================

DOCKER_IMAGE := buy-me-that-look-api
DOCKER_TAG := latest
DOCKER_CONTAINER := bmtl-api

docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	docker run -d \
		--name $(DOCKER_CONTAINER) \
		-p 8000:8000 \
		-v $(PWD)/uploads:/app/uploads \
		-v $(PWD)/logs:/app/logs \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-stop:
	docker stop $(DOCKER_CONTAINER) || true
	docker rm $(DOCKER_CONTAINER) || true

docker-logs:
	docker logs -f $(DOCKER_CONTAINER)

docker-shell:
	docker exec -it $(DOCKER_CONTAINER) /bin/bash

# ============================================================================
# Utilities
# ============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/ 2>/dev/null || true

# Create required directories
setup-dirs:
	mkdir -p uploads logs models data

# Download sample data (placeholder)
download-data:
	@echo "Downloading sample data..."
	@echo "Note: Implement actual data download logic"

# ── MLOps ─────────────────────────────────────────────────────────────────────

# ── Frontend ───────────────────────────────────────────────────────────────────

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

# ── MLOps ─────────────────────────────────────────────────────────────────────

build-catalog:
	KMP_DUPLICATE_LIB_OK=TRUE python scripts/build_catalog.py

evaluate:
	KMP_DUPLICATE_LIB_OK=TRUE python scripts/evaluate.py

retrain:
	KMP_DUPLICATE_LIB_OK=TRUE python scripts/retrain.py

retrain-dry:
	KMP_DUPLICATE_LIB_OK=TRUE python scripts/retrain.py --dry-run

retrain-force:
	KMP_DUPLICATE_LIB_OK=TRUE python scripts/retrain.py --force-promote

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5001

# ============================================================================
# CI/CD Helpers
# ============================================================================

ci-test:
	pip install pytest pytest-cov pytest-asyncio httpx
	pytest tests/ -v --cov=app --cov-report=xml

ci-lint:
	pip install flake8 black isort
	flake8 app/ tests/ --max-line-length=100 --ignore=E203,W503
	black app/ tests/ --check --line-length=100
	isort app/ tests/ --check --profile=black
