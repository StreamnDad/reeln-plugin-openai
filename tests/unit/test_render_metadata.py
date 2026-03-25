"""Tests for render metadata generation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reeln_openai_plugin.client import OpenAIError
from reeln_openai_plugin.prompts import PromptRegistry
from reeln_openai_plugin.render_metadata import (
    RENDER_SCHEMA,
    RenderMetadata,
    generate_render_metadata,
)
from tests.conftest import FakeGameInfo

# ------------------------------------------------------------------
# RenderMetadata
# ------------------------------------------------------------------


class TestRenderMetadata:
    def test_frozen(self) -> None:
        m = RenderMetadata(title="T", description="D")
        with pytest.raises(AttributeError):
            m.title = "X"  # type: ignore[misc]

    def test_fields(self) -> None:
        m = RenderMetadata(title="T", description="D")
        assert m.title == "T"
        assert m.description == "D"


# ------------------------------------------------------------------
# RENDER_SCHEMA
# ------------------------------------------------------------------


class TestRenderSchema:
    def test_required_fields(self) -> None:
        assert set(RENDER_SCHEMA["required"]) == {"title", "description"}

    def test_no_additional_properties(self) -> None:
        assert RENDER_SCHEMA["additionalProperties"] is False


# ------------------------------------------------------------------
# generate_render_metadata
# ------------------------------------------------------------------


class TestGenerateRenderMetadata:
    def test_success(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "title": "Eagles vs Hawks Goal Highlight",
            "description": "Amazing goal in the Eagles vs Hawks game!",
        }
        registry = PromptRegistry()
        info = FakeGameInfo()

        result = generate_render_metadata(client, registry, info)

        assert isinstance(result, RenderMetadata)
        assert result.title == "Eagles vs Hawks Goal Highlight"
        assert result.description == "Amazing goal in the Eagles vs Hawks game!"
        client.request_structured.assert_called_once()

        call_kwargs = client.request_structured.call_args[1]
        assert call_kwargs["schema_name"] == "render"

    def test_with_clip_name(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "title": "Short Title",
            "description": "Short Desc",
        }
        registry = PromptRegistry()
        info = FakeGameInfo()

        result = generate_render_metadata(
            client, registry, info, clip_name="goal_001",
        )

        assert result.title == "Short Title"
        call_kwargs = client.request_structured.call_args[1]
        assert "goal_001" in call_kwargs["prompt"]

    def test_without_clip_name(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "title": "Title",
            "description": "Desc",
        }
        registry = PromptRegistry()
        info = FakeGameInfo()

        result = generate_render_metadata(client, registry, info)

        assert result.title == "Title"
        # clip_name placeholder should remain unrendered
        call_kwargs = client.request_structured.call_args[1]
        assert "{{clip_name}}" in call_kwargs["prompt"]

    def test_api_error_propagates(self) -> None:
        client = MagicMock()
        client.request_structured.side_effect = OpenAIError("API down")
        registry = PromptRegistry()
        info = FakeGameInfo()

        with pytest.raises(OpenAIError, match="API down"):
            generate_render_metadata(client, registry, info)

    def test_with_frame_summary(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "title": "Amazing Goal!",
            "description": "Player scores top corner.",
        }
        registry = PromptRegistry()
        info = FakeGameInfo()

        result = generate_render_metadata(
            client, registry, info, frame_summary="Wrist shot finds the net",
        )

        assert result.title == "Amazing Goal!"
        call_kwargs = client.request_structured.call_args[1]
        assert "Wrist shot finds the net" in call_kwargs["prompt"]

    def test_without_frame_summary(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "title": "T",
            "description": "D",
        }
        registry = PromptRegistry()
        info = FakeGameInfo()

        result = generate_render_metadata(client, registry, info)

        assert result.title == "T"
        # frame_summary placeholder should remain unrendered
        call_kwargs = client.request_structured.call_args[1]
        assert "{{frame_summary}}" in call_kwargs["prompt"]

    def test_uses_game_info_variables(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "title": "T",
            "description": "D",
        }
        registry = PromptRegistry()
        info = FakeGameInfo(home_team="Storm", away_team="Thunder", sport="hockey")

        generate_render_metadata(client, registry, info)

        call_kwargs = client.request_structured.call_args[1]
        assert "Storm" in call_kwargs["prompt"]
        assert "Thunder" in call_kwargs["prompt"]
        assert "hockey" in call_kwargs["prompt"]

    def test_with_player_and_assists(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "title": "#48 Remitz Scores!",
            "description": "Great play.",
        }
        registry = PromptRegistry()
        info = FakeGameInfo()

        result = generate_render_metadata(
            client, registry, info,
            player="#48 Benjamin Remitz",
            assists="#7 John Smith, #22 Jane Doe",
        )

        assert result.title == "#48 Remitz Scores!"
        call_kwargs = client.request_structured.call_args[1]
        assert "#48 Benjamin Remitz" in call_kwargs["prompt"]
        assert "#7 John Smith, #22 Jane Doe" in call_kwargs["prompt"]

    def test_with_event_type_and_level(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "title": "Goal!",
            "description": "Desc.",
        }
        registry = PromptRegistry()
        info = FakeGameInfo()

        result = generate_render_metadata(
            client, registry, info,
            event_type="goal",
            level="2016",
        )

        assert result.title == "Goal!"
        call_kwargs = client.request_structured.call_args[1]
        assert "goal" in call_kwargs["prompt"]
        assert "2016" in call_kwargs["prompt"]
