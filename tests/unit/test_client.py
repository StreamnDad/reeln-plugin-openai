"""Tests for the OpenAI HTTP client."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from reeln_openai_plugin.client import API_URL, OpenAIClient, OpenAIError

# ------------------------------------------------------------------
# __repr__
# ------------------------------------------------------------------


class TestRepr:
    def test_redacts_api_key(self) -> None:
        client = OpenAIClient(api_key="sk-secret-key", model="gpt-test")
        r = repr(client)
        assert "sk-secret-key" not in r
        assert "[REDACTED]" in r
        assert "gpt-test" in r

# ------------------------------------------------------------------
# read_api_key
# ------------------------------------------------------------------


class TestReadApiKey:
    def test_reads_and_strips(self, tmp_path: Path) -> None:
        key_file = tmp_path / "key.txt"
        key_file.write_text("  sk-abc123  \n")
        assert OpenAIClient.read_api_key(key_file) == "sk-abc123"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(OpenAIError, match="Cannot read API key file"):
            OpenAIClient.read_api_key(tmp_path / "missing.txt")


# ------------------------------------------------------------------
# _build_payload
# ------------------------------------------------------------------


class TestBuildPayload:
    def test_payload_structure(self) -> None:
        client = OpenAIClient(api_key="k", model="gpt-test")
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        payload = client._build_payload("hello", schema, "test_schema")

        assert payload["model"] == "gpt-test"
        assert payload["input"][0]["role"] == "user"
        assert payload["input"][0]["content"][0]["type"] == "input_text"
        assert payload["input"][0]["content"][0]["text"] == "hello"
        assert payload["text"]["format"]["type"] == "json_schema"
        assert payload["text"]["format"]["name"] == "test_schema"
        assert payload["text"]["format"]["schema"] is schema


# ------------------------------------------------------------------
# _parse_response
# ------------------------------------------------------------------


class TestParseResponse:
    def test_nested_message_format(self) -> None:
        raw: dict[str, Any] = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": '{"title": "hi"}'},
                    ],
                },
            ],
        }
        assert OpenAIClient._parse_response(raw) == {"title": "hi"}

    def test_output_text_fallback(self) -> None:
        raw: dict[str, Any] = {"output_text": '{"title": "fallback"}'}
        assert OpenAIClient._parse_response(raw) == {"title": "fallback"}

    def test_no_output_raises(self) -> None:
        with pytest.raises(OpenAIError, match="No structured output"):
            OpenAIClient._parse_response({"output": []})

    def test_malformed_json_nested(self) -> None:
        raw: dict[str, Any] = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "not json"}],
                },
            ],
        }
        with pytest.raises(OpenAIError, match="Malformed JSON in response"):
            OpenAIClient._parse_response(raw)

    def test_malformed_json_fallback(self) -> None:
        raw: dict[str, Any] = {"output_text": "bad json"}
        with pytest.raises(OpenAIError, match="Malformed JSON in output_text"):
            OpenAIClient._parse_response(raw)

    def test_skips_non_message_output(self) -> None:
        raw: dict[str, Any] = {
            "output": [{"type": "other"}],
            "output_text": '{"ok": true}',
        }
        assert OpenAIClient._parse_response(raw) == {"ok": True}

    def test_skips_non_output_text_content(self) -> None:
        raw: dict[str, Any] = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "other"}],
                },
            ],
            "output_text": '{"ok": true}',
        }
        assert OpenAIClient._parse_response(raw) == {"ok": True}


# ------------------------------------------------------------------
# _post
# ------------------------------------------------------------------


class TestPost:
    def _mock_response(self, body: dict[str, Any]) -> MagicMock:
        mock = MagicMock()
        mock.read.return_value = json.dumps(body).encode()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        return mock

    @patch("reeln_openai_plugin.client.urllib.request.urlopen")
    def test_successful_post(self, mock_urlopen: MagicMock) -> None:
        body = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}]}
        mock_urlopen.return_value = self._mock_response(body)

        client = OpenAIClient(api_key="sk-test", model="gpt-test", timeout_seconds=10.0)
        result = client._post({"model": "gpt-test"})
        assert result == body

        # Verify request details
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.full_url == API_URL
        assert req.get_header("Authorization") == "Bearer sk-test"
        assert req.get_header("Content-type") == "application/json"
        assert call_args[1]["timeout"] == 10.0

    @patch("reeln_openai_plugin.client.urllib.request.urlopen")
    def test_http_error_unreadable_body(self, mock_urlopen: MagicMock) -> None:
        """HTTPError whose body cannot be read falls back to empty message."""
        exc = urllib.error.HTTPError(API_URL, 429, "Rate limited", {}, None)  # type: ignore[arg-type]
        exc.read = MagicMock(side_effect=OSError("cannot read"))  # type: ignore[assignment]
        mock_urlopen.side_effect = exc

        client = OpenAIClient(api_key="sk-test")
        with pytest.raises(OpenAIError, match="HTTP 429"):
            client._post({"model": "test"})

    @patch("reeln_openai_plugin.client.urllib.request.urlopen")
    def test_http_error_with_body(self, mock_urlopen: MagicMock) -> None:
        import io

        body = io.BytesIO(b"rate limit exceeded")
        exc = urllib.error.HTTPError(API_URL, 429, "Rate limited", {}, body)  # type: ignore[arg-type]
        mock_urlopen.side_effect = exc

        client = OpenAIClient(api_key="sk-test")
        with pytest.raises(OpenAIError, match="rate limit exceeded"):
            client._post({"model": "test"})

    @patch("reeln_openai_plugin.client.urllib.request.urlopen")
    def test_url_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("DNS failure")

        client = OpenAIClient(api_key="sk-test")
        with pytest.raises(OpenAIError, match="Network error"):
            client._post({"model": "test"})

    @patch("reeln_openai_plugin.client.urllib.request.urlopen")
    def test_timeout_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = TimeoutError()

        client = OpenAIClient(api_key="sk-test", timeout_seconds=5.0)
        with pytest.raises(OpenAIError, match=r"timed out after 5\.0s"):
            client._post({"model": "test"})


# ------------------------------------------------------------------
# request_structured (integration of the above)
# ------------------------------------------------------------------


class TestRequestStructured:
    @patch("reeln_openai_plugin.client.urllib.request.urlopen")
    def test_end_to_end(self, mock_urlopen: MagicMock) -> None:
        response_body = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": '{"title": "Test"}'},
                    ],
                },
            ],
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_body).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = OpenAIClient(api_key="sk-test", model="gpt-test")
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        }
        result = client.request_structured("Generate title", schema, "test")
        assert result == {"title": "Test"}


# ------------------------------------------------------------------
# _build_image_payload
# ------------------------------------------------------------------


class TestBuildImagePayload:
    def test_payload_structure(self) -> None:
        client = OpenAIClient(api_key="k", model="gpt-default")
        images = ["base64_logo_home", "base64_logo_away"]
        payload = client._build_image_payload(
            prompt="Generate thumbnail",
            images=images,
            model="gpt-5.2",
            renderer_model="gpt-image-1.5",
            renderer_size="1536x1024",
            output_format="png",
        )

        assert payload["model"] == "gpt-5.2"
        assert len(payload["input"]) == 1
        content = payload["input"][0]["content"]
        # Two images + one text prompt
        assert len(content) == 3
        assert content[0]["type"] == "input_image"
        assert content[0]["image_url"] == "data:image/png;base64,base64_logo_home"
        assert content[1]["type"] == "input_image"
        assert content[1]["image_url"] == "data:image/png;base64,base64_logo_away"
        assert content[2]["type"] == "input_text"
        assert content[2]["text"] == "Generate thumbnail"

    def test_tools_structure(self) -> None:
        client = OpenAIClient(api_key="k")
        payload = client._build_image_payload(
            prompt="p",
            images=[],
            model="gpt-5.2",
            renderer_model="gpt-image-1.5",
            renderer_size="1536x1024",
            output_format="png",
        )
        assert len(payload["tools"]) == 1
        tool = payload["tools"][0]
        assert tool["type"] == "image_generation"
        assert tool["model"] == "gpt-image-1.5"
        assert tool["size"] == "1536x1024"
        assert tool["output_format"] == "png"

    def test_no_images(self) -> None:
        client = OpenAIClient(api_key="k")
        payload = client._build_image_payload(
            prompt="p", images=[], model="m", renderer_model="r",
            renderer_size="s", output_format="png",
        )
        content = payload["input"][0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "input_text"


# ------------------------------------------------------------------
# _parse_image_response
# ------------------------------------------------------------------


class TestParseImageResponse:
    def test_extracts_b64_image(self) -> None:
        raw: dict[str, Any] = {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "..."}]},
                {"type": "image_generation_call", "result": "AQID"},
            ],
        }
        assert OpenAIClient._parse_image_response(raw) == "AQID"

    def test_no_image_output_raises(self) -> None:
        raw: dict[str, Any] = {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "oops"}]},
            ],
        }
        with pytest.raises(OpenAIError, match="No image output"):
            OpenAIClient._parse_image_response(raw)

    def test_empty_output_raises(self) -> None:
        with pytest.raises(OpenAIError, match="No image output"):
            OpenAIClient._parse_image_response({"output": []})


# ------------------------------------------------------------------
# _post with timeout override
# ------------------------------------------------------------------


class TestPostTimeoutOverride:
    def _mock_response(self, body: dict[str, Any]) -> MagicMock:
        mock = MagicMock()
        mock.read.return_value = json.dumps(body).encode()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        return mock

    @patch("reeln_openai_plugin.client.urllib.request.urlopen")
    def test_default_timeout(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = self._mock_response({"ok": True})
        client = OpenAIClient(api_key="k", timeout_seconds=10.0)
        client._post({"model": "x"})
        assert mock_urlopen.call_args[1]["timeout"] == 10.0

    @patch("reeln_openai_plugin.client.urllib.request.urlopen")
    def test_timeout_override(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = self._mock_response({"ok": True})
        client = OpenAIClient(api_key="k", timeout_seconds=10.0)
        client._post({"model": "x"}, timeout=120.0)
        assert mock_urlopen.call_args[1]["timeout"] == 120.0

    @patch("reeln_openai_plugin.client.urllib.request.urlopen")
    def test_timeout_override_in_error_message(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = TimeoutError()
        client = OpenAIClient(api_key="k", timeout_seconds=10.0)
        with pytest.raises(OpenAIError, match=r"timed out after 120\.0s"):
            client._post({"model": "x"}, timeout=120.0)


# ------------------------------------------------------------------
# request_image (integration)
# ------------------------------------------------------------------


class TestRequestImage:
    def _mock_response(self, body: dict[str, Any]) -> MagicMock:
        mock = MagicMock()
        mock.read.return_value = json.dumps(body).encode()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        return mock

    @patch("reeln_openai_plugin.client.urllib.request.urlopen")
    def test_end_to_end(self, mock_urlopen: MagicMock) -> None:
        import base64

        img_b64 = base64.b64encode(b"fakepng").decode()
        response_body: dict[str, Any] = {
            "output": [
                {"type": "image_generation_call", "result": img_b64},
            ],
        }
        mock_urlopen.return_value = self._mock_response(response_body)

        client = OpenAIClient(api_key="sk-test", model="gpt-default")
        result = client.request_image(
            prompt="Make thumbnail",
            images=["logo_b64"],
            model_override="gpt-5.2",
            renderer_model="gpt-image-1.5",
            renderer_size="1536x1024",
            output_format="png",
            timeout_override=120.0,
        )
        assert result == b"fakepng"

        # Verify timeout was passed through
        assert mock_urlopen.call_args[1]["timeout"] == 120.0

    @patch("reeln_openai_plugin.client.urllib.request.urlopen")
    def test_uses_defaults_without_overrides(self, mock_urlopen: MagicMock) -> None:
        import base64

        img_b64 = base64.b64encode(b"img").decode()
        response_body: dict[str, Any] = {
            "output": [{"type": "image_generation_call", "result": img_b64}],
        }
        mock_urlopen.return_value = self._mock_response(response_body)

        client = OpenAIClient(api_key="sk-test", model="gpt-default", timeout_seconds=30.0)
        client.request_image(
            prompt="p",
            images=[],
            model_override="gpt-5.2",
            renderer_model="gpt-image-1.5",
        )
        # Uses default timeout when not overridden
        assert mock_urlopen.call_args[1]["timeout"] == 30.0
