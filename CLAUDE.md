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

### Implemented Modules

| Module | Responsibility |
|---|---|
| `plugin.py` | `OpenAIPlugin` — plugin lifecycle, hook handlers, config schema |
| `__init__.py` | Exports `OpenAIPlugin` and `__version__` |
| `client.py` | OpenAI API client — `/v1/responses` endpoint, JSON schema + image generation |
| `prompts.py` | Prompt template engine — `{{variable}}` substitution, template loading |
| `livestream.py` | Livestream title + description generation via OpenAI |
| `playlist.py` | Playlist title + description generation via OpenAI |
| `translate.py` | Translation — batch or per-language with custom commentator personas |
| `game_image.py` | Game thumbnail generation — team logos + prompt → broadcast-style image |

### Planned Modules

| Module | Responsibility |
|---|---|
| `frames.py` | FFmpeg frame extraction — sample N frames, scale, base64 encode |
| `metadata.py` | Video metadata generation — title, description, hashtags from frames |
| `zoom.py` | Intelligent zoom — smart crop (multi-frame) and replay zoom (single-frame) |
| `cache.py` | Result caching — zoom point cache with configurable directory |

### Key Features (ported from streamn-dad-highlights)

1. **Video metadata** — analyze video frames via vision API to generate title/description/hashtags
2. **Text-only metadata** — generate metadata from text prompts (game summaries, playlist descriptions)
3. **Smart zoom** — multi-frame puck/action tracking for vertical video crops
4. **Replay zoom** — single-frame net center detection for crop centering
5. **Translation** — batch or per-language translation with custom commentator persona prompts
6. **Prompt templates** — `{{variable}}` substitution with external template files

### Key Hooks

- **`ON_CLIP_AVAILABLE`** — enrich clip metadata with LLM-generated title/description/hashtags
- **`ON_HIGHLIGHTS_MERGED`** — generate game summary metadata
- **`PRE_RENDER`** — compute smart zoom / replay zoom points for render plan

### Plugin Capabilities

- **MetadataEnricher** — `enrich(event_data)` returns enriched metadata with LLM fields

### Shared Context Convention

Plugins communicate via `HookContext.shared` (mutable dict on frozen dataclass):
```python
context.shared["metadata"]["title"] = "Amazing Goal by #12"
context.shared["metadata"]["description"] = "..."
context.shared["metadata"]["hashtags"] = ["#hockey", "#goal"]
context.shared["zoom_points"] = [(0.45, 0.52), (0.48, 0.55), ...]
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

## Conventions

- `from __future__ import annotations` in every module
- 4-space indent, snake_case, type hints on all signatures
- `pathlib.Path` for all file paths
- 100% line + branch coverage — no exceptions
- Keep a Changelog format in CHANGELOG.md
- Tests use `tmp_path` for all file I/O and mock external API clients
- Mock OpenAI HTTP responses — never make real API calls in tests
- Use `httpx` or `urllib.request` for HTTP (no `openai` SDK dependency)
