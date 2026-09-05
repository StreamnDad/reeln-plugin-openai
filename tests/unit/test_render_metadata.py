"""Tests for render metadata generation."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reeln_openai_plugin.client import OpenAIError
from reeln_openai_plugin.prompts import PromptRegistry
from reeln_openai_plugin.render_metadata import (
    DEFAULT_FRAME_SAMPLE_CAP,
    RENDER_SCHEMA,
    RenderMetadata,
    _sample_frames,
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

    def test_with_clip_name_does_not_crash(self) -> None:
        """clip_name is no longer a template variable in the persona-driven
        prompts — the call surface still accepts it for backwards-compatible
        wiring, but the value is intentionally ignored. This test guards
        against an error when callers continue to pass it."""
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

    def test_with_scoring_team_and_opposing_team(self) -> None:
        """REGRESSION: scoring_team and opposing_team kwargs flow into
        the rendered prompt so GPT can correctly attribute the play.
        Without these, descriptions like "Cozine scores for Machine
        Orange" were generated when Cozine is on Blades Maroon.
        """
        client = MagicMock()
        client.request_structured.return_value = {
            "title": "Cozine scores for Blades Maroon!",
            "description": "Colton Cozine nets one for Blades Maroon against Machine Orange.",
        }
        registry = PromptRegistry()
        info = FakeGameInfo(
            home_team="Machine Orange", away_team="Blades Maroon"
        )

        result = generate_render_metadata(
            client,
            registry,
            info,
            player="#16 Colton Cozine",
            event_type="goal",
            scoring_team="Blades Maroon",
            opposing_team="Machine Orange",
        )

        assert "Blades Maroon" in result.title
        call_kwargs = client.request_structured.call_args[1]
        prompt = call_kwargs["prompt"]
        # Both variables should appear in the rendered prompt — substituted
        # into the persona-style narrative rather than labeled fields.
        assert "Blades Maroon" in prompt
        assert "Machine Orange" in prompt
        # And the rules section should reference the concept so GPT attributes
        # the play to the right team.
        assert "scoring team" in prompt.lower()


# ------------------------------------------------------------------
# _sample_frames
# ------------------------------------------------------------------


class TestSampleFrames:
    def test_under_cap_returns_all(self) -> None:
        frames = [Path(f"f{i}.png") for i in range(3)]
        assert _sample_frames(frames, 5) == frames

    def test_over_cap_samples_evenly(self) -> None:
        """Endpoints (first + last) MUST land in the sample so the model
        sees how the play started and how it ended. With 10 frames and
        cap=5, evenly spaced indices are 0, 2, 4, 6, 9."""
        frames = [Path(f"f{i}.png") for i in range(10)]
        sampled = _sample_frames(frames, 5)
        assert len(sampled) == 5
        assert sampled[0] == frames[0]
        assert sampled[-1] == frames[-1]

    def test_cap_zero_returns_empty(self) -> None:
        frames = [Path("a.png"), Path("b.png")]
        assert _sample_frames(frames, 0) == []

    def test_empty_input_returns_empty(self) -> None:
        assert _sample_frames([], 5) == []


# ------------------------------------------------------------------
# Frame images in render_metadata
# ------------------------------------------------------------------


class TestRenderMetadataFrames:
    def test_no_frame_paths_sends_no_images(self) -> None:
        """Without frame_paths, ``images`` arg must be None — preserves the
        text-only request path so callers that don't have frames don't pay
        for vision tokens."""
        client = MagicMock()
        client.request_structured.return_value = {"title": "T", "description": "D"}
        registry = PromptRegistry()

        generate_render_metadata(client, registry, FakeGameInfo())

        call_kwargs = client.request_structured.call_args[1]
        assert call_kwargs.get("images") is None

    def test_frame_paths_encoded_and_passed(self, tmp_path: Path) -> None:
        """Frame files become base64-encoded ``images`` on the API call."""
        frame_a = tmp_path / "frame_a.png"
        frame_b = tmp_path / "frame_b.png"
        frame_a.write_bytes(b"AAAA")
        frame_b.write_bytes(b"BBBB")

        client = MagicMock()
        client.request_structured.return_value = {"title": "T", "description": "D"}
        registry = PromptRegistry()

        generate_render_metadata(
            client,
            registry,
            FakeGameInfo(),
            frame_paths=[frame_a, frame_b],
        )

        call_kwargs = client.request_structured.call_args[1]
        assert call_kwargs["images"] == [
            base64.b64encode(b"AAAA").decode("ascii"),
            base64.b64encode(b"BBBB").decode("ascii"),
        ]

    def test_frame_paths_sampled_when_over_cap(self, tmp_path: Path) -> None:
        """Smart-zoom routinely extracts 16+ frames; we sample down to the
        cap before sending so token usage stays bounded."""
        frame_paths = []
        for i in range(10):
            p = tmp_path / f"f{i}.png"
            p.write_bytes(f"frame-{i}".encode())
            frame_paths.append(p)

        client = MagicMock()
        client.request_structured.return_value = {"title": "T", "description": "D"}
        registry = PromptRegistry()

        generate_render_metadata(
            client,
            registry,
            FakeGameInfo(),
            frame_paths=frame_paths,
            frame_sample_cap=3,
        )

        call_kwargs = client.request_structured.call_args[1]
        assert len(call_kwargs["images"]) == 3

    def test_missing_frame_file_is_skipped(self, tmp_path: Path) -> None:
        """A missing file MUST NOT abort the whole call — drop the unreadable
        frame and keep going so a single I/O error doesn't kill the title
        for an otherwise-good render."""
        good = tmp_path / "good.png"
        good.write_bytes(b"OK")
        missing = tmp_path / "missing.png"
        # Intentionally not created.

        client = MagicMock()
        client.request_structured.return_value = {"title": "T", "description": "D"}
        registry = PromptRegistry()

        generate_render_metadata(
            client,
            registry,
            FakeGameInfo(),
            frame_paths=[good, missing],
        )

        call_kwargs = client.request_structured.call_args[1]
        assert call_kwargs["images"] == [base64.b64encode(b"OK").decode("ascii")]

    def test_all_frames_missing_falls_back_to_text(self, tmp_path: Path) -> None:
        """When every frame fails to read, ``images`` is None (not an empty
        list) so the client takes the text-only payload path."""
        missing = tmp_path / "missing.png"

        client = MagicMock()
        client.request_structured.return_value = {"title": "T", "description": "D"}
        registry = PromptRegistry()

        generate_render_metadata(
            client,
            registry,
            FakeGameInfo(),
            frame_paths=[missing],
        )

        call_kwargs = client.request_structured.call_args[1]
        assert call_kwargs.get("images") is None

    def test_default_cap_is_documented(self) -> None:
        """Pin the documented default so reducing it elsewhere is an explicit
        cost-/quality-tradeoff decision, not a silent regression."""
        assert DEFAULT_FRAME_SAMPLE_CAP == 5
