.PHONY: help install dev test test-verbose clean format lint typecheck all-checks

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

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
