"""Smart zoom target detection via OpenAI vision API."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from reeln_openai_plugin.client import OpenAIClient, OpenAIError
from reeln_openai_plugin.prompts import PromptRegistry

log: logging.Logger = logging.getLogger(__name__)

ZOOM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "center_x": {"type": "number"},
        "center_y": {"type": "number"},
    },
    "required": ["center_x", "center_y"],
    "additionalProperties": False,
}

FALLBACK_CENTER: tuple[float, float] = (0.5, 0.5)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp *value* to the range [*low*, *high*]."""
    return max(low, min(high, value))


def _encode_frame(frame_path: Path) -> str:
    """Read a frame file and return its base64-encoded contents."""
    try:
        raw = frame_path.read_bytes()
    except OSError as exc:
        raise OpenAIError(f"Cannot read frame file {frame_path}: {exc}") from exc
    return base64.b64encode(raw).decode()


def analyze_frame_for_zoom(
    client: OpenAIClient,
    prompt_registry: PromptRegistry,
    frame_path: Path,
    *,
    model: str = "gpt-4.1",
) -> tuple[float, float]:
    """Send a frame to OpenAI vision and get zoom target coordinates.

    Returns ``(center_x, center_y)`` as normalized 0.0-1.0 floats.
    Raises :class:`OpenAIError` on failure — caller decides fallback.
    """
    b64_image = _encode_frame(frame_path)
    prompt = prompt_registry.render("smart_zoom_detect")
    result = client.request_structured(
        prompt,
        ZOOM_SCHEMA,
        "smart_zoom_detect",
        images=[b64_image],
        model_override=model,
    )
    center_x = _clamp(float(result["center_x"]))
    center_y = _clamp(float(result["center_y"]))
    return (center_x, center_y)
