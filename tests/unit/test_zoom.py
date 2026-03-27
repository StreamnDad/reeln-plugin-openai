"""Tests for the smart zoom module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reeln_openai_plugin.client import OpenAIError
from reeln_openai_plugin.zoom import (
    DEFAULT_BACKOFF_MULTIPLIER,
    DEFAULT_INITIAL_BACKOFF,
    DEFAULT_MAX_BACKOFF,
    DEFAULT_MAX_RETRIES,
    FALLBACK_CENTER,
    ZOOM_SCHEMA,
    _clamp,
    _encode_frame,
    _is_retryable,
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

    def test_non_retryable_error_raises_immediately(self, tmp_path: Path) -> None:
        frame = tmp_path / "frame.png"
        frame.write_bytes(b"\x89PNG")

        client = MagicMock()
        client.request_structured.side_effect = OpenAIError("HTTP 401 Unauthorized")
        registry = MagicMock()

        with pytest.raises(OpenAIError, match="HTTP 401"):
            analyze_frame_for_zoom(client, registry, frame)

        # Should not retry — called only once
        assert client.request_structured.call_count == 1

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

    @patch("reeln_openai_plugin.zoom.time.sleep")
    def test_retries_on_transient_error(
        self, mock_sleep: MagicMock, tmp_path: Path
    ) -> None:
        """Transient error retries and succeeds on second attempt."""
        frame = tmp_path / "frame.png"
        frame.write_bytes(b"\x89PNG")

        client = MagicMock()
        client.request_structured.side_effect = [
            OpenAIError("HTTP 500 Internal Server Error"),
            {"center_x": 0.4, "center_y": 0.6},
        ]
        registry = MagicMock()

        result = analyze_frame_for_zoom(client, registry, frame)

        assert result == (0.4, 0.6)
        assert client.request_structured.call_count == 2
        mock_sleep.assert_called_once_with(DEFAULT_INITIAL_BACKOFF)

    @patch("reeln_openai_plugin.zoom.time.sleep")
    def test_retries_exhausted_raises(
        self, mock_sleep: MagicMock, tmp_path: Path
    ) -> None:
        """All retries exhausted raises the last error."""
        frame = tmp_path / "frame.png"
        frame.write_bytes(b"\x89PNG")

        client = MagicMock()
        client.request_structured.side_effect = OpenAIError("HTTP 502 Bad Gateway")
        registry = MagicMock()

        with pytest.raises(OpenAIError, match="HTTP 502"):
            analyze_frame_for_zoom(client, registry, frame, max_retries=3)

        assert client.request_structured.call_count == 3
        assert mock_sleep.call_count == 3

    @patch("reeln_openai_plugin.zoom.time.sleep")
    def test_backoff_increases_exponentially(
        self, mock_sleep: MagicMock, tmp_path: Path
    ) -> None:
        frame = tmp_path / "frame.png"
        frame.write_bytes(b"\x89PNG")

        client = MagicMock()
        client.request_structured.side_effect = OpenAIError("HTTP 500 error")
        registry = MagicMock()

        with pytest.raises(OpenAIError):
            analyze_frame_for_zoom(
                client,
                registry,
                frame,
                max_retries=3,
                initial_backoff=1.0,
                backoff_multiplier=2.0,
                max_backoff=100.0,
            )

        assert mock_sleep.call_args_list[0][0][0] == 1.0
        assert mock_sleep.call_args_list[1][0][0] == 2.0
        assert mock_sleep.call_args_list[2][0][0] == 4.0

    @patch("reeln_openai_plugin.zoom.time.sleep")
    def test_backoff_capped_at_max(
        self, mock_sleep: MagicMock, tmp_path: Path
    ) -> None:
        frame = tmp_path / "frame.png"
        frame.write_bytes(b"\x89PNG")

        client = MagicMock()
        client.request_structured.side_effect = OpenAIError("HTTP 500 error")
        registry = MagicMock()

        with pytest.raises(OpenAIError):
            analyze_frame_for_zoom(
                client,
                registry,
                frame,
                max_retries=3,
                initial_backoff=10.0,
                backoff_multiplier=5.0,
                max_backoff=15.0,
            )

        assert mock_sleep.call_args_list[0][0][0] == 10.0
        assert mock_sleep.call_args_list[1][0][0] == 15.0  # capped
        assert mock_sleep.call_args_list[2][0][0] == 15.0  # still capped

    @patch("reeln_openai_plugin.zoom.time.sleep")
    def test_rate_limit_retries(
        self, mock_sleep: MagicMock, tmp_path: Path
    ) -> None:
        frame = tmp_path / "frame.png"
        frame.write_bytes(b"\x89PNG")

        client = MagicMock()
        client.request_structured.side_effect = [
            OpenAIError("HTTP 429 Too Many Requests"),
            {"center_x": 0.5, "center_y": 0.5},
        ]
        registry = MagicMock()

        result = analyze_frame_for_zoom(client, registry, frame)
        assert result == (0.5, 0.5)
        assert client.request_structured.call_count == 2


# ------------------------------------------------------------------
# _is_retryable
# ------------------------------------------------------------------


class TestIsRetryable:
    def test_http_500(self) -> None:
        assert _is_retryable(OpenAIError("HTTP 500 Internal Server Error")) is True

    def test_http_502(self) -> None:
        assert _is_retryable(OpenAIError("HTTP 502 Bad Gateway")) is True

    def test_http_503(self) -> None:
        assert _is_retryable(OpenAIError("HTTP 503 Service Unavailable")) is True

    def test_http_429(self) -> None:
        assert _is_retryable(OpenAIError("HTTP 429 Too Many Requests")) is True

    def test_network_error(self) -> None:
        assert _is_retryable(OpenAIError("Network error: connection refused")) is True

    def test_timeout(self) -> None:
        assert _is_retryable(OpenAIError("Request timed out")) is True

    def test_timeout_case_insensitive(self) -> None:
        assert _is_retryable(OpenAIError("Connection Timed Out")) is True

    def test_http_400_not_retryable(self) -> None:
        assert _is_retryable(OpenAIError("HTTP 400 Bad Request")) is False

    def test_http_401_not_retryable(self) -> None:
        assert _is_retryable(OpenAIError("HTTP 401 Unauthorized")) is False

    def test_http_403_not_retryable(self) -> None:
        assert _is_retryable(OpenAIError("HTTP 403 Forbidden")) is False

    def test_http_404_not_retryable(self) -> None:
        assert _is_retryable(OpenAIError("HTTP 404 Not Found")) is False

    def test_json_parse_error_not_retryable(self) -> None:
        assert _is_retryable(OpenAIError("JSON parse error")) is False

    def test_file_io_error_not_retryable(self) -> None:
        assert _is_retryable(OpenAIError("Cannot read frame file /tmp/f.png")) is False


# ------------------------------------------------------------------
# Retry defaults
# ------------------------------------------------------------------


class TestRetryDefaults:
    def test_max_retries(self) -> None:
        assert DEFAULT_MAX_RETRIES == 3

    def test_initial_backoff(self) -> None:
        assert DEFAULT_INITIAL_BACKOFF == 2.0

    def test_backoff_multiplier(self) -> None:
        assert DEFAULT_BACKOFF_MULTIPLIER == 2.0

    def test_max_backoff(self) -> None:
        assert DEFAULT_MAX_BACKOFF == 30.0
