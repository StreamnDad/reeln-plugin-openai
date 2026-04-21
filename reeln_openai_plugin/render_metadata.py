"""Render metadata (title and description) generation via OpenAI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from reeln_openai_plugin.client import OpenAIClient
from reeln_openai_plugin.livestream import build_prompt_variables
from reeln_openai_plugin.prompts import PromptRegistry

log: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderMetadata:
    """Generated render metadata (title and description)."""

    title: str
    description: str


RENDER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["title", "description"],
    "additionalProperties": False,
}


def generate_render_metadata(
    client: OpenAIClient,
    prompt_registry: PromptRegistry,
    game_info: object,
    clip_name: str = "",
    frame_summary: str = "",
    player: str = "",
    assists: str = "",
    event_type: str = "",
    level: str = "",
    scoring_team: str = "",
    opposing_team: str = "",
) -> RenderMetadata:
    """Generate render metadata (title and description) from *game_info*.

    Renders the ``render_title`` and ``render_description`` prompts,
    combines them into a single API call, and returns the structured result.

    When *clip_name* is provided it is included as a template variable
    so the LLM can incorporate the clip identifier.

    When *scoring_team* and *opposing_team* are provided, they're exposed
    as prompt variables so the LLM can correctly attribute the play to
    the scoring team. Without these, the LLM defaults to guessing based
    on team name order in the prompt — which reliably gets away-team
    plays wrong (e.g. "Player scores for HomeTeam" when Player is on
    AwayTeam). The caller is expected to determine the scoring team
    from ``game_event.metadata['team']``.
    """
    variables = build_prompt_variables(game_info)

    if clip_name:
        variables["clip_name"] = clip_name

    if frame_summary:
        variables["frame_summary"] = frame_summary

    if player:
        variables["player"] = player

    if assists:
        variables["assists"] = assists

    if event_type:
        variables["event"] = event_type

    if level:
        variables["team_level"] = level

    if scoring_team:
        variables["scoring_team"] = scoring_team

    if opposing_team:
        variables["opposing_team"] = opposing_team

    title_prompt = prompt_registry.render("render_title", variables)
    desc_prompt = prompt_registry.render("render_description", variables)

    combined_prompt = f"{title_prompt}\n\n---\n\n{desc_prompt}"

    result = client.request_structured(
        prompt=combined_prompt,
        schema=RENDER_SCHEMA,
        schema_name="render",
    )

    return RenderMetadata(
        title=result["title"],
        description=result["description"],
    )
