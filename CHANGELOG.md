# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.4.0] - 2026-03-11

### Added

- Game image thumbnail generation (`game_image.py`) — ESPN/NBC Sports broadcast-style thumbnails via OpenAI image generation tool
- Bundled prompt template: `game_image`
- Per-feature model override config fields: `game_image_model` (default `gpt-5.2`), `game_image_renderer_model` (default `gpt-image-1.5`)
- `game_image_enabled` and `game_image_output_dir` config fields
- `OpenAIClient.request_image()` method for tool-based image generation requests
- Optional timeout override on `OpenAIClient._post()` (120s for image generation)
- `Pillow` dependency for image resizing (1536x1024 → 1280x720 YouTube thumbnail)
- Game image written to `context.shared["game_image"]` with `image_path`
- `OPENAI_API_KEY` env var fallback when `api_key_file` is not configured or missing
- `OpenAIClient.__repr__` redacts API key to prevent accidental logging
- Path traversal protection via `_slugify` sanitization on output filenames
- Team profiles read from `context.data["home_profile"]` / `context.data["away_profile"]` (matches reeln-cli hook contract)

### Changed

- `api_key_file` is now optional — falls back to `OPENAI_API_KEY` environment variable
- Game image handler extracted to `_maybe_generate_game_image()` for reduced nesting
- `resize_image()` wraps PIL errors as `OpenAIError` for consistent error handling

## [0.3.0] - 2026-03-05

### Added

- Playlist metadata generation (`playlist.py`) — LLM-generated title + description for YouTube playlists
- Bundled prompt templates: `playlist_title`, `playlist_description`
- `playlist_enabled` config field to opt in to playlist metadata generation
- Playlist translation support — reuses existing `translate_metadata()` pipeline
- Playlist metadata written to `context.shared["playlist_metadata"]`

## [0.2.0] - 2026-03-05

### Added

- OpenAI HTTP client (`client.py`) — direct `/v1/responses` API, no SDK dependency
- Prompt template engine (`prompts.py`) with bundled defaults and config overrides
- Livestream metadata generation (`livestream.py`) — title + description via OpenAI
- Multi-language translation (`translate.py`) — batch and per-language modes
- Full plugin config schema with JSON-string dict fields
- `ON_GAME_INIT` hook handler wiring: metadata generation + optional translation
- Bundled prompt templates: `livestream_title`, `livestream_description`, `translate_batch`, `translate_single`

### Changed

- Renamed scaffold from `plugin_name`/`PluginName` to `reeln_openai_plugin`/`OpenAIPlugin`

## [0.1.0] - 2026-03-05

### Added

- Initial plugin scaffolding with `OpenAIPlugin` class
- `ON_GAME_INIT` hook handler
- 100% line + branch test coverage
