.PHONY: help install dev test test-verbose clean format lint typecheck all-checks docs docs-clean docs-serve

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install dependencies from pyproject.toml (uv sync)"
	@echo "  dev          - Set up development environment (install + show tools)"
	@echo "  test         - Run tests (quiet)"
	@echo "  test-verbose - Run tests with verbose output"
	@echo "  format       - Format code with black"
	@echo "  lint         - Lint code with ruff"
	@echo "  typecheck    - Type check with mypy"
	@echo "  all-checks   - Run lint, typecheck, and tests"
	@echo "  docs         - Build documentation"
	@echo "  docs-serve   - Build and serve documentation locally"
	@echo "  clean        - Remove build artifacts and caches"

# Install all deps (main + dev) according to pyproject.toml
install:
	uv sync

# Dev setup: install deps and print versions of tools
dev: install
	@echo "✔ Dev environment ready. Tool versions:"
	uv run python -m pytest --version || true
	uv run black --version || true
	uv run ruff --version || true
	uv run mypy --version || true

# Testing
test:
	uv run pytest tests/ -q

test-verbose:
	uv run pytest tests/ -v

# Code quality
format:
	uv run black src/ tests/

lint:
	uv run ruff check src/ tests/

typecheck:
	uv run mypy src/

# Run all quality checks (excluding auto-format)
all-checks: lint typecheck test
	@echo "✅ All checks passed!"

# Documentation
docs:
	uv run --extra docs sphinx-build -b html docs docs/_build/html

docs-clean:
	rm -rf docs/_build

docs-serve: docs
	python -m http.server 8000 --directory docs/_build/html

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf docs/_build/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -not -path "./.venv/*" -delete 2>/dev/null || true
	find . -type f -name "*.so" -not -path "./.venv/*" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete"
