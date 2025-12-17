.PHONY: install dev-install test lint format clean docs run-server benchmark

# Installation
install:
	pip install -e .

dev-install:
	pip install -e ".[dev,server]"

# Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest tests/ --cov=vectordb --cov-report=html

benchmark:
	pytest tests/benchmark/ -v --benchmark-only

# Code quality
lint:
	black --check vectordb/ tests/
	isort --check-only vectordb/ tests/
	mypy vectordb/

format:
	black vectordb/ tests/
	isort vectordb/ tests/

# Documentation
docs:
	mkdocs serve

docs-build:
	mkdocs build

# Server
run-server:
	uvicorn vectordb.server.app:app --reload --host 0.0.0.0 --port 8000

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete