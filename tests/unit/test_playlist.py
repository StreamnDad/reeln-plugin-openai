"""Tests for playlist metadata generation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reeln_openai_plugin.client import OpenAIError
from reeln_openai_plugin.playlist import (
    PLAYLIST_SCHEMA,
    PlaylistMetadata,
    generate_playlist_metadata,
)
from reeln_openai_plugin.prompts import PromptRegistry
from tests.conftest import FakeGameInfo

# ------------------------------------------------------------------
# PlaylistMetadata
# ------------------------------------------------------------------


class TestPlaylistMetadata:
    def test_frozen(self) -> None:
        m = PlaylistMetadata(title="T", description="D")
        with pytest.raises(AttributeError):
            m.title = "X"  # type: ignore[misc]

    def test_fields(self) -> None:
        m = PlaylistMetadata(title="T", description="D")
        assert m.title == "T"
        assert m.description == "D"


# ------------------------------------------------------------------
# PLAYLIST_SCHEMA
# ------------------------------------------------------------------


class TestPlaylistSchema:
    def test_required_fields(self) -> None:
        assert set(PLAYLIST_SCHEMA["required"]) == {"title", "description"}

    def test_no_additional_properties(self) -> None:
        assert PLAYLIST_SCHEMA["additionalProperties"] is False


# ------------------------------------------------------------------
# generate_playlist_metadata
# ------------------------------------------------------------------


class TestGeneratePlaylistMetadata:
    def test_success(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "title": "Eagles vs Hawks Highlights - 2026-01-15",
            "description": "Game highlights from the Eagles vs Hawks matchup.",
        }
        registry = PromptRegistry()
        info = FakeGameInfo()

        result = generate_playlist_metadata(client, registry, info)

        assert isinstance(result, PlaylistMetadata)
        assert result.title == "Eagles vs Hawks Highlights - 2026-01-15"
        assert result.description == "Game highlights from the Eagles vs Hawks matchup."
        client.request_structured.assert_called_once()

        # Verify schema name
        call_kwargs = client.request_structured.call_args[1]
        assert call_kwargs["schema_name"] == "playlist"

    def test_with_livestream_title(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "title": "Playlist Title",
            "description": "Playlist Desc",
        }
        registry = PromptRegistry()
        info = FakeGameInfo()

        result = generate_playlist_metadata(
            client,
            registry,
            info,
            livestream_title="Live: Eagles vs Hawks",
        )

        assert result.title == "Playlist Title"
        # Verify livestream_title was passed to the prompt
        call_kwargs = client.request_structured.call_args[1]
        assert "Live: Eagles vs Hawks" in call_kwargs["prompt"]

    def test_without_livestream_title(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "title": "Title",
            "description": "Desc",
        }
        registry = PromptRegistry()
        info = FakeGameInfo()

        result = generate_playlist_metadata(client, registry, info)

        assert result.title == "Title"
        # livestream_title placeholder should remain unrendered
        call_kwargs = client.request_structured.call_args[1]
        assert "{{livestream_title}}" in call_kwargs["prompt"]

    def test_api_error_propagates(self) -> None:
        client = MagicMock()
        client.request_structured.side_effect = OpenAIError("API down")
        registry = PromptRegistry()
        info = FakeGameInfo()

        with pytest.raises(OpenAIError, match="API down"):
            generate_playlist_metadata(client, registry, info)

    def test_uses_game_info_variables(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "title": "T",
            "description": "D",
        }
        registry = PromptRegistry()
        info = FakeGameInfo(home_team="Storm", away_team="Thunder", sport="hockey")

        generate_playlist_metadata(client, registry, info)

        call_kwargs = client.request_structured.call_args[1]
        assert "Storm" in call_kwargs["prompt"]
        assert "Thunder" in call_kwargs["prompt"]
        assert "hockey" in call_kwargs["prompt"]
