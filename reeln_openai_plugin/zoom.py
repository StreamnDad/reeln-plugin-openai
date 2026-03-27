"""Smart zoom target detection via OpenAI vision API."""

from __future__ import annotations

import base64
import logging
import time
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

# Retry defaults
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_INITIAL_BACKOFF: float = 2.0
DEFAULT_BACKOFF_MULTIPLIER: float = 2.0
DEFAULT_MAX_BACKOFF: float = 30.0


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


def _is_retryable(exc: OpenAIError) -> bool:
    """Return True if the error is transient and worth retrying.

    Retries on: HTTP 500, 502, 503, 429 (rate limit), network errors, timeouts.
    Does NOT retry on: 400, 401, 403, 404, JSON parse errors, file I/O errors.
    """
    msg = str(exc)
    if "HTTP 5" in msg:
        return True
    if "HTTP 429" in msg:
        return True
    if "Network error" in msg:
        return True
    return "timed out" in msg.lower()


def analyze_frame_for_zoom(
    client: OpenAIClient,
    prompt_registry: PromptRegistry,
    frame_path: Path,
    *,
    model: str = "gpt-4.1",
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
    max_backoff: float = DEFAULT_MAX_BACKOFF,
) -> tuple[float, float]:
    """Send a frame to OpenAI vision and get zoom target coordinates.

    Returns ``(center_x, center_y)`` as normalized 0.0-1.0 floats.

    Retries transient errors (HTTP 5xx, 429, network, timeout) with
    exponential backoff. Raises :class:`OpenAIError` after all retries
    are exhausted or on non-retryable errors.
    """
    b64_image = _encode_frame(frame_path)
    prompt = prompt_registry.render("smart_zoom_detect")

    last_exc: OpenAIError | None = None
    backoff = initial_backoff

    for attempt in range(1, max_retries + 1):
        try:
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
        except OpenAIError as exc:
            last_exc = exc
            if not _is_retryable(exc):
                raise
            log.warning(
                "frame %s attempt %d/%d failed (retrying in %.1fs): %s",
                frame_path.name,
                attempt,
                max_retries,
                backoff,
                exc,
            )
            time.sleep(backoff)
            backoff = min(backoff * backoff_multiplier, max_backoff)

    # All retries exhausted — raise the last error
    raise last_exc  # type: ignore[misc]
