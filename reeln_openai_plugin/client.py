"""OpenAI HTTP client — direct ``/v1/responses`` API without SDK dependency."""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import ssl
import urllib.request
from pathlib import Path
from typing import Any

log: logging.Logger = logging.getLogger(__name__)

API_URL: str = "https://api.openai.com/v1/responses"


class OpenAIError(Exception):
    """Raised when the OpenAI API returns an error or the response is invalid."""


class OpenAIClient:
    """Thin wrapper around the OpenAI ``/v1/responses`` endpoint.

    Uses :mod:`urllib.request` directly so the plugin has no SDK dependency.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key: str = api_key
        self._model: str = model
        self._timeout: float = timeout_seconds

    def __repr__(self) -> str:
        return f"OpenAIClient(model={self._model!r}, api_key='[REDACTED]')"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> dict[str, Any]:
        """Send a text prompt and return the parsed JSON response.

        The API is instructed to return JSON conforming to *schema* via the
        ``json_schema`` response format.
        """
        payload = self._build_payload(prompt, schema, schema_name)
        raw = self._post(payload)
        return self._parse_response(raw)

    def request_image(
        self,
        prompt: str,
        images: list[str],
        model_override: str,
        renderer_model: str,
        renderer_size: str = "1536x1024",
        output_format: str = "png",
        timeout_override: float | None = None,
    ) -> bytes:
        """Send a prompt with reference images and return the generated image bytes.

        Uses the ``image_generation`` tool in the Responses API.  *images* is
        a list of base64-encoded PNG strings.  Returns raw image bytes.
        """
        payload = self._build_image_payload(
            prompt=prompt,
            images=images,
            model=model_override,
            renderer_model=renderer_model,
            renderer_size=renderer_size,
            output_format=output_format,
        )
        raw = self._post(payload, timeout=timeout_override)
        b64_data = self._parse_image_response(raw)
        return base64.b64decode(b64_data)

    @staticmethod
    def read_api_key(path: Path) -> str:
        """Read and return the API key from *path*, stripping whitespace."""
        try:
            return path.read_text().strip()
        except OSError as exc:
            raise OpenAIError(f"Cannot read API key file {path}: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                },
            },
        }

    def _build_image_payload(
        self,
        prompt: str,
        images: list[str],
        model: str,
        renderer_model: str,
        renderer_size: str,
        output_format: str,
    ) -> dict[str, Any]:
        content: list[dict[str, str]] = [
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}
            for b64 in images
        ]
        content.append({"type": "input_text", "text": prompt})

        return {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "tools": [
                {
                    "type": "image_generation",
                    "model": renderer_model,
                    "size": renderer_size,
                    "output_format": output_format,
                },
            ],
        }

    @staticmethod
    def _parse_response(raw: dict[str, Any]) -> dict[str, Any]:
        """Extract the structured JSON from the API response."""
        for item in raw.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text = content["text"]
                        try:
                            result: dict[str, Any] = json.loads(text)
                        except json.JSONDecodeError as exc:
                            raise OpenAIError(f"Malformed JSON in response: {exc}") from exc
                        return result

        # Fallback: check top-level output_text
        if "output_text" in raw:
            try:
                result = json.loads(raw["output_text"])
            except json.JSONDecodeError as exc:
                raise OpenAIError(f"Malformed JSON in output_text: {exc}") from exc
            return result

        raise OpenAIError("No structured output found in API response")

    @staticmethod
    def _parse_image_response(raw: dict[str, Any]) -> str:
        """Extract the base64 image data from an image generation response."""
        for item in raw.get("output", []):
            if item.get("type") == "image_generation_call":
                return str(item["result"])

        raise OpenAIError("No image output found in API response")

    def _post(
        self, payload: dict[str, Any], *, timeout: float | None = None,
    ) -> dict[str, Any]:
        """POST *payload* to the OpenAI API and return the parsed response."""
        effective_timeout = timeout if timeout is not None else self._timeout
        data = json.dumps(payload).encode()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")
        ctx = ssl.create_default_context()

        try:
            with urllib.request.urlopen(req, timeout=effective_timeout, context=ctx) as resp:
                body: dict[str, Any] = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            msg = ""
            with contextlib.suppress(Exception):
                msg = exc.read().decode()
            raise OpenAIError(f"HTTP {exc.code}: {msg}") from exc
        except urllib.error.URLError as exc:
            raise OpenAIError(f"Network error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise OpenAIError(f"Request timed out after {effective_timeout}s") from exc

        return body
