"""Playlist title and description generation via OpenAI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from reeln_openai_plugin.client import OpenAIClient
from reeln_openai_plugin.livestream import build_prompt_variables
from reeln_openai_plugin.prompts import PromptRegistry

log: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaylistMetadata:
    """Generated playlist title and description."""

    title: str
    description: str


PLAYLIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["title", "description"],
    "additionalProperties": False,
}


def generate_playlist_metadata(
    client: OpenAIClient,
    prompt_registry: PromptRegistry,
    game_info: object,
    livestream_title: str | None = None,
) -> PlaylistMetadata:
    """Generate a playlist title and description from *game_info*.

    Renders the ``playlist_title`` and ``playlist_description`` prompts,
    combines them into a single API call, and returns the structured result.

    When *livestream_title* is provided it is included as a template variable
    so the playlist can reference the livestream name.
    """
    variables = build_prompt_variables(game_info)

    if livestream_title is not None:
        variables["livestream_title"] = livestream_title

    title_prompt = prompt_registry.render("playlist_title", variables)
    desc_prompt = prompt_registry.render("playlist_description", variables)

    combined_prompt = f"{title_prompt}\n\n---\n\n{desc_prompt}"

    result = client.request_structured(
        prompt=combined_prompt,
        schema=PLAYLIST_SCHEMA,
        schema_name="playlist",
    )

    return PlaylistMetadata(
        title=result["title"],
        description=result["description"],
    )
