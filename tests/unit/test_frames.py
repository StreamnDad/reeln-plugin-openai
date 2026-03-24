"""Tests for frame description generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reeln_openai_plugin.client import OpenAIError
from reeln_openai_plugin.frames import (
    FRAME_DESCRIPTION_SCHEMA,
    FrameDescriptions,
    describe_frames,
)
from reeln_openai_plugin.prompts import PromptRegistry

# ------------------------------------------------------------------
# FrameDescriptions
# ------------------------------------------------------------------


class TestFrameDescriptions:
    def test_frozen(self) -> None:
        fd = FrameDescriptions(descriptions=("a",), summary="s")
        with pytest.raises(AttributeError):
            fd.summary = "x"  # type: ignore[misc]

    def test_fields(self) -> None:
        fd = FrameDescriptions(descriptions=("a", "b"), summary="s")
        assert fd.descriptions == ("a", "b")
        assert fd.summary == "s"

    def test_empty_descriptions(self) -> None:
        fd = FrameDescriptions(descriptions=(), summary="nothing")
        assert fd.descriptions == ()
        assert fd.summary == "nothing"


# ------------------------------------------------------------------
# FRAME_DESCRIPTION_SCHEMA
# ------------------------------------------------------------------


class TestFrameDescriptionSchema:
    def test_required_fields(self) -> None:
        assert set(FRAME_DESCRIPTION_SCHEMA["required"]) == {"descriptions", "summary"}

    def test_no_additional_properties(self) -> None:
        assert FRAME_DESCRIPTION_SCHEMA["additionalProperties"] is False

    def test_descriptions_is_array_of_strings(self) -> None:
        desc_prop = FRAME_DESCRIPTION_SCHEMA["properties"]["descriptions"]
        assert desc_prop["type"] == "array"
        assert desc_prop["items"]["type"] == "string"

    def test_summary_is_string(self) -> None:
        assert FRAME_DESCRIPTION_SCHEMA["properties"]["summary"]["type"] == "string"


# ------------------------------------------------------------------
# describe_frames
# ------------------------------------------------------------------


class TestDescribeFrames:
    def test_success(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "descriptions": ["Player shoots", "Goalie dives", "Goal scored"],
            "summary": "A quick wrist shot finds the top corner.",
        }
        registry = PromptRegistry()
        frames = tuple(self._make_frames(tmp_path, 3))

        result = describe_frames(client, registry, frames)

        assert isinstance(result, FrameDescriptions)
        assert len(result.descriptions) == 3
        assert result.descriptions[0] == "Player shoots"
        assert result.summary == "A quick wrist shot finds the top corner."

        call_kwargs = client.request_structured.call_args[1]
        assert call_kwargs["schema_name"] == "frame_descriptions"
        assert len(call_kwargs["images"]) == 3

    def test_api_error_propagates(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.request_structured.side_effect = OpenAIError("API down")
        registry = PromptRegistry()
        frames = tuple(self._make_frames(tmp_path, 1))

        with pytest.raises(OpenAIError, match="API down"):
            describe_frames(client, registry, frames)

    def test_model_override(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "descriptions": ["action"],
            "summary": "summary",
        }
        registry = PromptRegistry()
        frames = tuple(self._make_frames(tmp_path, 1))

        describe_frames(client, registry, frames, model="gpt-5")

        call_kwargs = client.request_structured.call_args[1]
        assert call_kwargs["model_override"] == "gpt-5"

    def test_uses_default_model(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "descriptions": ["action"],
            "summary": "summary",
        }
        registry = PromptRegistry()
        frames = tuple(self._make_frames(tmp_path, 1))

        describe_frames(client, registry, frames)

        call_kwargs = client.request_structured.call_args[1]
        assert call_kwargs["model_override"] == "gpt-4.1"

    def test_single_frame(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "descriptions": ["A save is made"],
            "summary": "Goaltender makes a save.",
        }
        registry = PromptRegistry()
        frames = tuple(self._make_frames(tmp_path, 1))

        result = describe_frames(client, registry, frames)

        assert len(result.descriptions) == 1
        assert result.summary == "Goaltender makes a save."

    def test_renders_frame_describe_prompt(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "descriptions": ["d"],
            "summary": "s",
        }
        registry = PromptRegistry()
        frames = tuple(self._make_frames(tmp_path, 1))

        describe_frames(client, registry, frames)

        call_kwargs = client.request_structured.call_args[1]
        assert "chronological order" in call_kwargs["prompt"]

    def test_encodes_frames_as_base64(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "descriptions": ["d1", "d2"],
            "summary": "s",
        }
        registry = PromptRegistry()
        frames = tuple(self._make_frames(tmp_path, 2))

        describe_frames(client, registry, frames)

        call_kwargs = client.request_structured.call_args[1]
        images = call_kwargs["images"]
        assert len(images) == 2
        # base64 of b"\x89PNG" should be a non-empty string
        for img in images:
            assert isinstance(img, str)
            assert len(img) > 0

    @staticmethod
    def _make_frames(tmp_path: Path, count: int) -> list[Path]:
        paths = []
        for i in range(count):
            p = tmp_path / f"frame_{i:04d}.png"
            p.write_bytes(b"\x89PNG")
            paths.append(p)
        return paths
