# reeln-plugin-openai

A [reeln-cli](https://github.com/StreamnDad/reeln-cli) plugin for OpenAI-powered LLM integration — livestream metadata, game thumbnails, playlist descriptions, and multi-language translation.

## Features

- **Livestream metadata** — LLM-generated title and description for game livestreams
- **Playlist metadata** — creative playlist titles and descriptions
- **Game image thumbnails** — ESPN/NBC Sports broadcast-style thumbnails from team logos via OpenAI image generation
- **Multi-language translation** — batch or per-language translation with custom commentator personas
- **Prompt templates** — customizable `{{variable}}` templates with override support

## Install

```bash
pip install reeln-plugin-openai
```

Or for development:

```bash
git clone https://github.com/StreamnDad/reeln-plugin-openai
cd reeln-plugin-openai
make dev-install
```

## Configuration

Add the plugin to your reeln config:

```json
{
  "plugins": {
    "enabled": ["openai"],
    "settings": {
      "openai": {
        "enabled": true,
        "api_key_file": "~/.config/reeln/secrets/openai_api_key.txt",
        "model": "gpt-4.1",
        "playlist_enabled": true,
        "game_image_enabled": true,
        "game_image_output_dir": "~/Documents/game_images"
      }
    }
  }
}
```

### API Key

The plugin resolves the OpenAI API key in this order:

1. **`api_key_file`** config — reads key from the specified file (recommended)
2. **`OPENAI_API_KEY`** environment variable — fallback for dev/CI

### Config Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable OpenAI LLM integration |
| `api_key_file` | str | `""` | Path to file containing the OpenAI API key |
| `model` | str | `"gpt-4.1"` | OpenAI model for text generation |
| `request_timeout_seconds` | float | `30.0` | API request timeout |
| `prompt_overrides` | str (JSON) | `"{}"` | Dict of prompt name to override file path |
| `prompt_context` | str (JSON) | `"{}"` | Dict of prompt name to extra context lines |
| `translate_enabled` | bool | `false` | Enable multi-language translation |
| `translate_languages` | str (JSON) | `"{}"` | Dict of language code to name |
| `translate_per_language_prompts` | str (JSON) | `"{}"` | Dict of language code to prompt name |
| `playlist_enabled` | bool | `false` | Enable LLM-generated playlist metadata |
| `game_image_enabled` | bool | `false` | Enable game image thumbnail generation |
| `game_image_model` | str | `"gpt-5.2"` | Model for image generation orchestration |
| `game_image_renderer_model` | str | `"gpt-image-1.5"` | Model for image rendering |
| `game_image_output_dir` | str | `""` | Directory to save generated game images |

## Hooks

| Hook | What it does |
|---|---|
| `ON_GAME_INIT` | Generates livestream metadata, playlist metadata, game image thumbnail, and translations |

### Shared Context

The plugin writes to `context.shared` for other plugins to consume:

```python
context.shared["livestream_metadata"]  # {"title": "...", "description": "...", "translations": {...}}
context.shared["playlist_metadata"]    # {"title": "...", "description": "...", "translations": {...}}
context.shared["game_image"]           # {"image_path": "/path/to/thumbnail.png"}
```

## Development

```bash
make dev-install    # uv venv + editable install with dev deps
make test           # pytest with 100% coverage
make lint           # ruff check
make format         # ruff format
make check          # lint + mypy + test
```

## License

AGPL-3.0-only
