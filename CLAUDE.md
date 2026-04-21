# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**reeln-plugin-openai** is a reeln-cli plugin that provides OpenAI-powered LLM integration for video metadata generation, intelligent zoom/crop, translation, and game summaries.

- **Package:** `reeln-plugin-openai` | **License:** AGPL-3.0
- **Python:** 3.11+ | **Plugin framework:** reeln-cli plugin system
- Entry point: `reeln.plugins` → `openai = "reeln_openai_plugin:OpenAIPlugin"`

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

This plugin provides OpenAI LLM capabilities to reeln-cli via hooks and the MetadataEnricher capability. It replicates the OpenAI features originally built in `streamn-dad-highlights/replay_publisher/llm.py`.

### Modules

| Module | Responsibility |
|---|---|
| `plugin.py` | `OpenAIPlugin` — plugin lifecycle, hook handlers, config schema |
| `__init__.py` | Exports `OpenAIPlugin` and `__version__` |
| `client.py` | OpenAI API client — `/v1/responses` endpoint, JSON schema + image generation + vision |
| `prompts.py` | Prompt template engine — `{{variable}}` substitution, template loading |
| `livestream.py` | Livestream title + description generation via OpenAI |
| `playlist.py` | Playlist title + description generation via OpenAI |
| `translate.py` | Translation — batch or per-language with custom commentator personas |
| `game_image.py` | Game thumbnail generation — team logos + prompt → broadcast-style image |
| `zoom.py` | Smart zoom — per-frame vision analysis for action area detection |
| `frames.py` | Frame description — multi-frame vision analysis for play-by-play summary |
| `render_metadata.py` | Render metadata — LLM-generated title + description for rendered clips |

### Key Hooks

- **`ON_GAME_INIT`** — generate livestream/playlist metadata, game image, cache game_info
- **`ON_FRAMES_EXTRACTED`** — smart zoom target detection + frame description generation
- **`POST_RENDER`** — generate render metadata (title/description) for uploaded clips

### Shared Context Convention

Plugins communicate via `HookContext.shared` (mutable dict on frozen dataclass):
```python
context.shared["livestream_metadata"] = {"title": "...", "description": "..."}
context.shared["playlist_metadata"] = {"title": "...", "description": "..."}
context.shared["game_image"] = {"image_path": "/path/to/image.png"}
context.shared["smart_zoom"] = {"zoom_path": ZoomPath(...)}
context.shared["frame_descriptions"] = {"descriptions": [...], "summary": "..."}
context.shared["render_metadata"] = {"title": "...", "description": "..."}
```

### External Dependencies

- `reeln` — plugin hooks, capabilities, and models (sibling install)
- OpenAI API — `/v1/responses` endpoint (direct HTTP, no SDK dependency)

### OpenAI API Details

- **Endpoint:** `https://api.openai.com/v1/responses`
- **Auth:** Bearer token from file or config
- **Models:** configurable (e.g., `gpt-5-mini` for metadata, `gpt-5.2` for spatial analysis)
- **Response format:** JSON schema validation for structured output
- **Vision:** base64-encoded JPEG frames sent as `input_image` content blocks

## Versioning

Every code change **must** bump the version following [Semantic Versioning](https://semver.org/):

- **Major** — breaking changes to plugin behavior or config schema
- **Minor** — new features, new capabilities, new config options
- **Patch** — bug fixes, internal refactors, test-only changes

Update all three locations in lockstep:

1. `reeln_openai_plugin/__init__.py` — `__version__`
2. `reeln_openai_plugin/plugin.py` — `version` class attribute
3. `CHANGELOG.md` — new section under `[Unreleased]` with date and description

## Game State Boundary

**Plugins MUST NOT directly read or mutate `game.json`.** Plugins interact with game
state exclusively through `HookContext`:

- **Read** game info via `context.data["game_info"]` (passed by the hook emitter)
- **Share** computed results via `context.shared` (mutable dict on the frozen dataclass)
- **Never** import `load_game_state`, `save_game_state`, or any `reeln-state` function
- **Never** open, read, or write `game.json` directly

The host application (dock or CLI) loads state before hooks fire and persists any state
changes after hooks complete. Plugins are pure capability providers — they compute results
and deposit them in `context.shared` for the host to consume.

If a plugin needs to cause a state change (e.g., writing a livestream URL), it writes
to `context.shared` and the host application calls the appropriate `reeln-state` mutation.

## Conventions

- `from __future__ import annotations` in every module
- 4-space indent, snake_case, type hints on all signatures
- `pathlib.Path` for all file paths
- 100% line + branch coverage — no exceptions
- Keep a Changelog format in CHANGELOG.md
- Tests use `tmp_path` for all file I/O and mock external API clients
- Mock OpenAI HTTP responses — never make real API calls in tests
- Use `httpx` or `urllib.request` for HTTP (no `openai` SDK dependency)
