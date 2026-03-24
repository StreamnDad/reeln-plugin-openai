"""Frame description generation via OpenAI vision API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reeln_openai_plugin.client import OpenAIClient
from reeln_openai_plugin.prompts import PromptRegistry
from reeln_openai_plugin.zoom import _encode_frame

log: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameDescriptions:
    """Per-frame descriptions and an overall play summary."""

    descriptions: tuple[str, ...]
    summary: str


FRAME_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "descriptions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "summary": {"type": "string"},
    },
    "required": ["descriptions", "summary"],
    "additionalProperties": False,
}


def describe_frames(
    client: OpenAIClient,
    prompt_registry: PromptRegistry,
    frame_paths: tuple[Path, ...],
    *,
    model: str = "gpt-4.1",
) -> FrameDescriptions:
    """Send all frames to OpenAI vision in a single call and get descriptions.

    Returns a :class:`FrameDescriptions` with per-frame descriptions and
    an overall summary.  Raises :class:`OpenAIError` on failure.
    """
    images = [_encode_frame(p) for p in frame_paths]
    prompt = prompt_registry.render("frame_describe")

    result = client.request_structured(
        prompt=prompt,
        schema=FRAME_DESCRIPTION_SCHEMA,
        schema_name="frame_descriptions",
        images=images,
        model_override=model,
    )

    return FrameDescriptions(
        descriptions=tuple(result["descriptions"]),
        summary=result["summary"],
    )
