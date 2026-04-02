"""Game image thumbnail generation via OpenAI image generation tool."""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from reeln_openai_plugin.client import OpenAIClient, OpenAIError
from reeln_openai_plugin.prompts import PromptRegistry

log: logging.Logger = logging.getLogger(__name__)

IMAGE_TIMEOUT_SECONDS: float = 120.0
TARGET_WIDTH: int = 1280
TARGET_HEIGHT: int = 720


@dataclass(frozen=True)
class GameImageResult:
    """Path to the generated game thumbnail image."""

    image_path: Path


def encode_logo(path: Path) -> str:
    """Read a logo file and return its base64-encoded contents."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise OpenAIError(f"Cannot read logo file {path}: {exc}") from exc
    return base64.b64encode(raw).decode()


def resize_image(raw_bytes: bytes, width: int, height: int) -> bytes:
    """Resize raw image bytes to *width* x *height* and return PNG bytes."""
    try:
        img = Image.open(io.BytesIO(raw_bytes))
    except OSError as exc:
        raise OpenAIError(f"Cannot decode image from API response: {exc}") from exc
    resized = img.resize((width, height), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, "PNG")
    return buf.getvalue()


def _build_prompt_variables(
    home: object,
    away: object,
    rink: str,
    game_date: str,
    game_time: str,
    level: str,
    description: str,
    tournament: str,
) -> dict[str, str]:
    """Extract template variables from team objects and game info."""
    return {
        "home_team": str(getattr(home, "team_name", "")),
        "away_team": str(getattr(away, "team_name", "")),
        "home_colors": str(getattr(home, "colors", "")),
        "away_colors": str(getattr(away, "colors", "")),
        "game_level": str(getattr(home, "game_level", "")),
        "rink": rink,
        "game_date": game_date,
        "game_time": game_time,
        "level": level,
        "description": description,
        "tournament": tournament,
    }


def _slugify(name: str) -> str:
    """Convert a name to a filesystem-safe slug."""
    return re.sub(r"[^\w-]", "_", name).lower()


def generate_game_image(
    client: OpenAIClient,
    prompt_registry: PromptRegistry,
    home: object,
    away: object,
    rink: str,
    game_date: str,
    game_time: str,
    output_dir: Path,
    model: str,
    renderer_model: str,
    level: str = "",
    description: str = "",
    tournament: str = "",
) -> GameImageResult:
    """Generate a game thumbnail image and save it as a 1280x720 PNG.

    Encodes team logos, renders the ``game_image`` prompt template,
    calls the OpenAI image generation API, resizes the result, and
    saves it to *output_dir*.
    """
    home_logo_path = getattr(home, "logo_path", None)
    away_logo_path = getattr(away, "logo_path", None)

    images: list[str] = []
    if home_logo_path is not None:
        images.append(encode_logo(Path(home_logo_path)))
    if away_logo_path is not None:
        images.append(encode_logo(Path(away_logo_path)))

    variables = _build_prompt_variables(
        home, away, rink, game_date, game_time, level, description, tournament,
    )
    prompt = prompt_registry.render("game_image", variables)

    raw_bytes = client.request_image(
        prompt=prompt,
        images=images,
        model_override=model,
        renderer_model=renderer_model,
        renderer_size="1536x1024",
        output_format="png",
        timeout_override=IMAGE_TIMEOUT_SECONDS,
    )

    resized = resize_image(raw_bytes, TARGET_WIDTH, TARGET_HEIGHT)

    output_dir.mkdir(parents=True, exist_ok=True)

    home_slug = _slugify(str(getattr(home, "short_name", getattr(home, "team_name", "home"))))
    away_slug = _slugify(str(getattr(away, "short_name", getattr(away, "team_name", "away"))))
    date_slug = _slugify(game_date.replace("/", "-"))
    filename = f"{date_slug}_{home_slug}_vs_{away_slug}.png"
    out_path = output_dir / filename
    out_path.write_bytes(resized)
    return GameImageResult(image_path=out_path)
