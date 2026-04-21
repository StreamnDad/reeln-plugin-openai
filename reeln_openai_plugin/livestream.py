"""Livestream title and description generation via OpenAI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from reeln_openai_plugin.client import OpenAIClient
from reeln_openai_plugin.prompts import PromptRegistry

log: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LivestreamMetadata:
    """Generated livestream title and description."""

    title: str
    description: str


LIVESTREAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["title", "description"],
    "additionalProperties": False,
}


def _profile_summary(profile: object) -> str:
    """Return the ``summary`` field from a profile's ``metadata``.

    The ``metadata`` field may arrive as either a real ``dict`` (when the
    plugin is invoked in-process with a ``TeamProfile`` dataclass) or as a
    ``types.SimpleNamespace`` (when invoked via ``reeln hooks run``, which
    recursively converts JSON dicts to namespaces for ``getattr`` access).
    Handle both shapes here so callers don't have to.
    """
    meta = getattr(profile, "metadata", None)
    if meta is None:
        return ""
    if isinstance(meta, dict):
        return str(meta.get("summary", ""))
    return str(getattr(meta, "summary", ""))


def build_prompt_variables(
    game_info: object,
    home_profile: object | None = None,
    away_profile: object | None = None,
) -> dict[str, str]:
    """Extract template variables from *game_info* and optional team profiles."""
    variables: dict[str, str] = {
        "home_team": str(getattr(game_info, "home_team", "")),
        "away_team": str(getattr(game_info, "away_team", "")),
        "date": str(getattr(game_info, "date", "")),
        "sport": str(getattr(game_info, "sport", "")),
        "venue": str(getattr(game_info, "venue", "")),
        "game_time": str(getattr(game_info, "game_time", "")),
        "description": str(getattr(game_info, "description", "")),
        "level": str(getattr(game_info, "level", "")),
        "tournament": str(getattr(game_info, "tournament", "")),
    }

    if home_profile is not None:
        variables["home_profile"] = _profile_summary(home_profile)
    if away_profile is not None:
        variables["away_profile"] = _profile_summary(away_profile)

    return variables


def generate_livestream_metadata(
    client: OpenAIClient,
    prompt_registry: PromptRegistry,
    game_info: object,
    home_profile: object | None = None,
    away_profile: object | None = None,
) -> LivestreamMetadata:
    """Generate a livestream title and description from *game_info*.

    Renders the ``livestream_title`` and ``livestream_description`` prompts,
    combines them into a single API call, and returns the structured result.
    """
    variables = build_prompt_variables(game_info, home_profile, away_profile)

    title_prompt = prompt_registry.render("livestream_title", variables)
    desc_prompt = prompt_registry.render("livestream_description", variables)

    combined_prompt = f"{title_prompt}\n\n---\n\n{desc_prompt}"

    result = client.request_structured(
        prompt=combined_prompt,
        schema=LIVESTREAM_SCHEMA,
        schema_name="livestream",
    )

    return LivestreamMetadata(
        title=result["title"],
        description=result["description"],
    )
