# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.8.2] - 2026-04-02

### Fixed

- Use correct `team_name` attribute (instead of `name`) on team profile objects in game image prompt variables and filename slugs — fixes home/away team names rendering as empty strings (issue #6)
- Use `level` attribute (instead of `game_level`) on team profile objects in game image prompt variables — fixes game level rendering as empty string
- Read team profile `summary` from `metadata` dict (instead of non-existent `summary` attribute) in livestream prompt variables — fixes home/away profile context always being empty

## [0.8.1] - 2026-03-27

### Added

- `{{level}}`, `{{tournament}}`, and `{{description}}` template variables for game image, livestream title, and livestream description prompts
- `level`, `description`, `tournament` parameters on `generate_game_image()` and `build_prompt_variables()`

### Fixed

- Removed dead `zoom_path is not None` check — `_analyze_frames_for_zoom` always returns a `ZoomPath` or raises

## [0.8.0] - 2026-03-25

### Added

- Smart zoom target detection (`zoom.py`) — analyze video frames via OpenAI vision API to identify action areas for dynamic crop panning
- `ON_FRAMES_EXTRACTED` hook handler — iterates extracted frames, calls vision API per frame, writes `ZoomPath` to `context.shared["smart_zoom"]`
- `smart_zoom_enabled` and `smart_zoom_model` config fields
- Vision support on `request_structured()` — optional `images` and `model_override` keyword arguments for sending base64 images with structured JSON requests
- Per-frame fallback to center `(0.5, 0.5)` on API failure; no shared write when all frames fail
- `POST_RENDER` hook handler — generates LLM-powered title and description for rendered clips via `render_metadata_enabled` config flag
- `render_metadata.py` module with `RenderMetadata` model and `generate_render_metadata()` function
- Bundled prompt templates: `render_title`, `render_description`, `smart_zoom_detect`, `frame_describe`
- `render_metadata_enabled` config field (default `false`) — opt-in LLM metadata for rendered clips
- Game info cached during `ON_GAME_INIT` for use in `POST_RENDER` metadata generation
- Render metadata written to `context.shared["render_metadata"]` (platform-agnostic)
- Frame description generation (`frames.py`) — analyze all extracted frames via OpenAI vision to generate per-frame descriptions and overall play summary
- `frame_description_enabled` and `frame_description_model` config fields
- Frame descriptions cached on plugin instance and fed as `{{frame_summary}}` context to render metadata prompts
- Frame descriptions written to `context.shared["frame_descriptions"]` with `descriptions` list and `summary`
- Player name, assists, event type, and team level passed as context to render metadata prompt templates (`{{player}}`, `{{assists}}`, `{{event}}`, `{{team_level}}`)

## [0.4.2] - 2026-03-15

### Added

- User-provided broadcast description (`game_info.description`) is now passed as `{{description}}` context to livestream prompt templates, allowing the LLM to incorporate tournament round, rivalry, or event details
- `Context: {{description}}` added to `livestream_title` and `livestream_description` prompt templates

### Changed

- Skip game image generation when user provides a thumbnail path during `game init`
- Extract `_generate_livestream_metadata()` and `_generate_playlist_metadata()` helper methods for reduced nesting

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
