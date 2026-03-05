# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**plugin-name** is a reeln-cli plugin that provides ... integration.

- **Package:** `plugin-name` | **License:** AGPL-3.0
- **Python:** 3.11+ | **Plugin framework:** reeln-cli plugin system
- Entry point: `reeln.plugins` → `myplugin = "plugin_name:PluginName"`

## Dev Commands

```bash
make dev-install    # uv venv + editable install with dev deps (also installs sibling ../reeln-cli)
make test           # pytest with 100% line+branch coverage, parallel via xdist
make lint           # ruff check
make format         # ruff format
make check          # lint → mypy → test (sequential)
```

Run a single test file or test:
```bash
.venv/bin/python -m pytest tests/unit/test_plugin.py -q
.venv/bin/python -m pytest tests/unit/test_plugin.py::TestClassName::test_method -q
```

## Architecture

This plugin hooks into reeln-cli lifecycle events via the plugin system.

### Implemented Modules

| Module | Responsibility |
|---|---|
| `plugin.py` | `PluginName` — plugin lifecycle, hook handlers, config schema |
| `__init__.py` | Exports `PluginName` and `__version__` |

### Key Hook

**`ON_GAME_INIT`** — handles game initialization events.

### Shared Context Convention

Plugins communicate via `HookContext.shared` (mutable dict on frozen dataclass):
```python
context.shared["livestreams"]["google"] = "https://youtube.com/live/abc123"
```

### External Dependencies

- `reeln` — plugin hooks, capabilities, and models (sibling install)

## Versioning

Every code change **must** bump the version following [Semantic Versioning](https://semver.org/):

- **Major** — breaking changes to plugin behavior or config schema
- **Minor** — new features, new capabilities, new config options
- **Patch** — bug fixes, internal refactors, test-only changes

Update all three locations in lockstep:

1. `plugin_name/__init__.py` — `__version__`
2. `plugin_name/plugin.py` — `version` class attribute
3. `CHANGELOG.md` — new section under `[Unreleased]` with date and description

## Conventions

- `from __future__ import annotations` in every module
- 4-space indent, snake_case, type hints on all signatures
- `pathlib.Path` for all file paths
- 100% line + branch coverage — no exceptions
- Keep a Changelog format in CHANGELOG.md
- Tests use `tmp_path` for all file I/O and mock external API clients
