# reeln-plugin-openai

A [reeln-cli](https://github.com/StreamnDad/reeln-cli) plugin for OpenAI-powered LLM integration — livestream metadata, game thumbnails, playlist descriptions, and multi-language translation.

## Features

- **Livestream metadata** — LLM-generated title and description for game livestreams
- **Playlist metadata** — creative playlist titles and descriptions
- **Game image thumbnails** — broadcast-style backgrounds from OpenAI with official team logos and an optional game sponsor added locally
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

Set `game_image_model` to a Responses model that supports the image generation tool, such as `gpt-5.2`. Set `game_image_renderer_model` to the tool’s image model, such as `gpt-image-2.5-flare`. [OpenAI image generation tool documentation](https://developers.openai.com/api/docs/guides/tools-image-generation)

## Hooks

| Hook | What it does |
|---|---|
| `ON_GAME_INIT` | Generates livestream metadata, playlist metadata, game image thumbnail, and translations |

### Shared Context

The plugin writes to `context.shared` for other plugins to consume:

```python
context.shared["livestream_metadata"]  # {"title": "...", "description": "...", "translations": {...}}
context.shared["playlist_metadata"]    # {"title": "...", "description": "...", "translations": {...}}
context.shared["game_image"]           # {"image_path": "...", "team_logos_composited": True, "sponsor_logo_applied": False}
```

### Game thumbnail artwork

Choose a logo in each team's profile. The plugin uses the files from those two profiles and fits them proportionally into the matchup. OpenAI receives a background prompt with zero image inputs. The plugin adds the team artwork, names, game date and time, and venue to the 1280 by 720 thumbnail.

Supply an optional sponsor via the `Game Sponsor Logo` prompt or `reeln game init --plugin-input sponsor_logo_path=/path/to/logo.png`. The plugin keeps the original static sRGB PNG at `<game directory>/game-sponsor.png` and scales it proportionally into the “This game is brought to you in part by” footer. Use the sponsor’s approved full-resolution export with up to 8 bits per channel. Accepted files are at most 32 MiB, 8192 pixels per edge, and 16 million pixels total.

For a finished graphic, the game image action in Dock accepts a PNG, JPEG, or WebP. It fits the whole graphic above the same sponsor footer and writes a separate `sponsored-game-image.png` beside the generated image. The original file is retained, and this action requires no OpenAI request.

The plugin saves per-game thumbnails at `<game_image_output_dir>/games/<game directory name>/<game path hash>/game-image.png`. The hash is the first 16 lowercase hexadecimal digits of the SHA-256 of the UTF-8 canonical absolute game directory path.

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
