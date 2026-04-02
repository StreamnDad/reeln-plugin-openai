"""Tests for livestream metadata generation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reeln_openai_plugin.client import OpenAIError
from reeln_openai_plugin.livestream import (
    LIVESTREAM_SCHEMA,
    LivestreamMetadata,
    build_prompt_variables,
    generate_livestream_metadata,
)
from reeln_openai_plugin.prompts import PromptRegistry
from tests.conftest import FakeGameInfo, FakeTeamProfile

# ------------------------------------------------------------------
# LivestreamMetadata
# ------------------------------------------------------------------


class TestLivestreamMetadata:
    def test_frozen(self) -> None:
        m = LivestreamMetadata(title="T", description="D")
        with pytest.raises(AttributeError):
            m.title = "X"  # type: ignore[misc]

    def test_fields(self) -> None:
        m = LivestreamMetadata(title="T", description="D")
        assert m.title == "T"
        assert m.description == "D"


# ------------------------------------------------------------------
# LIVESTREAM_SCHEMA
# ------------------------------------------------------------------


class TestLivestreamSchema:
    def test_required_fields(self) -> None:
        assert set(LIVESTREAM_SCHEMA["required"]) == {"title", "description"}

    def test_no_additional_properties(self) -> None:
        assert LIVESTREAM_SCHEMA["additionalProperties"] is False


# ------------------------------------------------------------------
# build_prompt_variables
# ------------------------------------------------------------------


class TestBuildPromptVariables:
    def test_basic_variables(self) -> None:
        info = FakeGameInfo(home_team="Storm", away_team="Thunder", date="2026-03-01", sport="hockey")
        v = build_prompt_variables(info)
        assert v["home_team"] == "Storm"
        assert v["away_team"] == "Thunder"
        assert v["date"] == "2026-03-01"
        assert v["sport"] == "hockey"
        assert v["venue"] == ""
        assert v["game_time"] == ""

    def test_with_venue(self) -> None:
        info = FakeGameInfo(venue="Rink Arena")
        v = build_prompt_variables(info)
        assert v["venue"] == "Rink Arena"

    def test_with_profiles(self) -> None:
        info = FakeGameInfo()
        home = FakeTeamProfile(metadata={"summary": "Home summary"})
        away = FakeTeamProfile(metadata={"summary": "Away summary"})
        v = build_prompt_variables(info, home_profile=home, away_profile=away)
        assert v["home_profile"] == "Home summary"
        assert v["away_profile"] == "Away summary"

    def test_without_profiles(self) -> None:
        info = FakeGameInfo()
        v = build_prompt_variables(info)
        assert "home_profile" not in v
        assert "away_profile" not in v

    def test_with_description(self) -> None:
        info = FakeGameInfo(description="Semifinals tournament game")
        v = build_prompt_variables(info)
        assert v["description"] == "Semifinals tournament game"

    def test_empty_description(self) -> None:
        info = FakeGameInfo()
        v = build_prompt_variables(info)
        assert v["description"] == ""

    def test_with_level(self) -> None:
        info = FakeGameInfo(level="2016")
        v = build_prompt_variables(info)
        assert v["level"] == "2016"

    def test_with_tournament(self) -> None:
        info = FakeGameInfo(tournament="State Championship")
        v = build_prompt_variables(info)
        assert v["tournament"] == "State Championship"

    def test_empty_level_and_tournament(self) -> None:
        info = FakeGameInfo()
        v = build_prompt_variables(info)
        assert v["level"] == ""
        assert v["tournament"] == ""

    def test_missing_attributes(self) -> None:
        """Duck-typed object without all attributes still works."""

        class Minimal:
            home_team: str = "X"

        v = build_prompt_variables(Minimal())
        assert v["home_team"] == "X"
        assert v["away_team"] == ""


# ------------------------------------------------------------------
# generate_livestream_metadata
# ------------------------------------------------------------------


class TestGenerateLivestreamMetadata:
    def test_success(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "title": "Eagles vs Hawks - Game Day!",
            "description": "Watch the Eagles take on the Hawks.",
        }
        registry = PromptRegistry()
        info = FakeGameInfo()

        result = generate_livestream_metadata(client, registry, info)

        assert isinstance(result, LivestreamMetadata)
        assert result.title == "Eagles vs Hawks - Game Day!"
        assert result.description == "Watch the Eagles take on the Hawks."
        client.request_structured.assert_called_once()

        # Verify prompt was rendered with variables
        call_args = client.request_structured.call_args
        assert "Eagles" in call_args[1]["prompt"] or "Eagles" in call_args[0][0]

    def test_api_error_propagates(self) -> None:
        client = MagicMock()
        client.request_structured.side_effect = OpenAIError("API down")
        registry = PromptRegistry()
        info = FakeGameInfo()

        with pytest.raises(OpenAIError, match="API down"):
            generate_livestream_metadata(client, registry, info)

    def test_with_profiles(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "title": "Title",
            "description": "Desc",
        }
        registry = PromptRegistry()
        info = FakeGameInfo()
        home = FakeTeamProfile()
        away = FakeTeamProfile()

        result = generate_livestream_metadata(client, registry, info, home, away)
        assert result.title == "Title"
