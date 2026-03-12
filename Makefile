PLUGIN_PKG := reeln_openai_plugin

.PHONY: dev-install test lint format check

VENV := .venv/bin

dev-install:
	uv venv --clear
	uv pip install -e ../reeln-cli
	uv pip install -e ".[dev]"

test:
	$(VENV)/python -m pytest tests/ -n auto --cov=$(PLUGIN_PKG) --cov-branch --cov-fail-under=100 -q

lint:
	$(VENV)/ruff check .

format:
	$(VENV)/ruff format .

check: lint
	$(VENV)/mypy $(PLUGIN_PKG)/
	$(MAKE) test
