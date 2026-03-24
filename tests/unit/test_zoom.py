"""Tests for the smart zoom module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reeln_openai_plugin.client import OpenAIError
from reeln_openai_plugin.zoom import (
    FALLBACK_CENTER,
    ZOOM_SCHEMA,
    _clamp,
    _encode_frame,
    analyze_frame_for_zoom,
)

# ------------------------------------------------------------------
# _clamp
# ------------------------------------------------------------------


class TestClamp:
    def test_within_range(self) -> None:
        assert _clamp(0.5) == 0.5

    def test_below_low(self) -> None:
        assert _clamp(-0.3) == 0.0

    def test_above_high(self) -> None:
        assert _clamp(1.7) == 1.0

    def test_exact_low(self) -> None:
        assert _clamp(0.0) == 0.0

    def test_exact_high(self) -> None:
        assert _clamp(1.0) == 1.0

    def test_custom_range(self) -> None:
        assert _clamp(5.0, low=2.0, high=4.0) == 4.0
        assert _clamp(1.0, low=2.0, high=4.0) == 2.0


# ------------------------------------------------------------------
# _encode_frame
# ------------------------------------------------------------------


class TestEncodeFrame:
    def test_success(self, tmp_path: Path) -> None:
        frame = tmp_path / "frame.png"
        frame.write_bytes(b"\x89PNG")
        result = _encode_frame(frame)
        assert isinstance(result, str)
        import base64

        assert base64.b64decode(result) == b"\x89PNG"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OpenAIError, match="Cannot read frame file"):
            _encode_frame(tmp_path / "missing.png")


# ------------------------------------------------------------------
# ZOOM_SCHEMA
# ------------------------------------------------------------------


class TestZoomSchema:
    def test_structure(self) -> None:
        assert ZOOM_SCHEMA["type"] == "object"
        assert "center_x" in ZOOM_SCHEMA["properties"]
        assert "center_y" in ZOOM_SCHEMA["properties"]
        assert ZOOM_SCHEMA["additionalProperties"] is False

    def test_required_fields(self) -> None:
        assert set(ZOOM_SCHEMA["required"]) == {"center_x", "center_y"}


# ------------------------------------------------------------------
# FALLBACK_CENTER
# ------------------------------------------------------------------


class TestFallbackCenter:
    def test_value(self) -> None:
        assert FALLBACK_CENTER == (0.5, 0.5)


# ------------------------------------------------------------------
# analyze_frame_for_zoom
# ------------------------------------------------------------------


class TestAnalyzeFrameForZoom:
    def test_success(self, tmp_path: Path) -> None:
        frame = tmp_path / "frame.png"
        frame.write_bytes(b"\x89PNG")

        client = MagicMock()
        client.request_structured.return_value = {"center_x": 0.3, "center_y": 0.7}
        registry = MagicMock()

        result = analyze_frame_for_zoom(client, registry, frame)

        assert result == (0.3, 0.7)
        client.request_structured.assert_called_once()
        call_kwargs = client.request_structured.call_args
        assert call_kwargs[1]["images"] is not None
        assert len(call_kwargs[1]["images"]) == 1
        assert call_kwargs[1]["model_override"] == "gpt-4.1"

    def test_clamping(self, tmp_path: Path) -> None:
        frame = tmp_path / "frame.png"
        frame.write_bytes(b"\x89PNG")

        client = MagicMock()
        client.request_structured.return_value = {"center_x": -0.5, "center_y": 1.8}
        registry = MagicMock()

        result = analyze_frame_for_zoom(client, registry, frame)

        assert result == (0.0, 1.0)

    def test_api_error_propagates(self, tmp_path: Path) -> None:
        frame = tmp_path / "frame.png"
        frame.write_bytes(b"\x89PNG")

        client = MagicMock()
        client.request_structured.side_effect = OpenAIError("API down")
        registry = MagicMock()

        with pytest.raises(OpenAIError, match="API down"):
            analyze_frame_for_zoom(client, registry, frame)

    def test_model_override_passed(self, tmp_path: Path) -> None:
        frame = tmp_path / "frame.png"
        frame.write_bytes(b"\x89PNG")

        client = MagicMock()
        client.request_structured.return_value = {"center_x": 0.5, "center_y": 0.5}
        registry = MagicMock()

        analyze_frame_for_zoom(client, registry, frame, model="gpt-5")

        call_kwargs = client.request_structured.call_args[1]
        assert call_kwargs["model_override"] == "gpt-5"

    def test_encode_failure_propagates(self, tmp_path: Path) -> None:
        client = MagicMock()
        registry = MagicMock()

        with pytest.raises(OpenAIError, match="Cannot read frame file"):
            analyze_frame_for_zoom(client, registry, tmp_path / "missing.png")

    def test_prompt_rendered(self, tmp_path: Path) -> None:
        frame = tmp_path / "frame.png"
        frame.write_bytes(b"\x89PNG")

        client = MagicMock()
        client.request_structured.return_value = {"center_x": 0.5, "center_y": 0.5}
        registry = MagicMock()
        registry.render.return_value = "custom prompt"

        analyze_frame_for_zoom(client, registry, frame)

        registry.render.assert_called_once_with("smart_zoom_detect")
        # First positional arg to request_structured should be the rendered prompt
        assert client.request_structured.call_args[0][0] == "custom prompt"

    def test_schema_passed(self, tmp_path: Path) -> None:
        frame = tmp_path / "frame.png"
        frame.write_bytes(b"\x89PNG")

        client = MagicMock()
        client.request_structured.return_value = {"center_x": 0.5, "center_y": 0.5}
        registry = MagicMock()

        analyze_frame_for_zoom(client, registry, frame)

        call_args = client.request_structured.call_args[0]
        assert call_args[1] is ZOOM_SCHEMA
        assert call_args[2] == "smart_zoom_detect"
