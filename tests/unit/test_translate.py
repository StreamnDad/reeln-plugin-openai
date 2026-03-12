"""Tests for multi-language translation."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from reeln_openai_plugin.client import OpenAIError
from reeln_openai_plugin.prompts import PromptRegistry
from reeln_openai_plugin.translate import (
    BATCH_TRANSLATION_SCHEMA,
    SINGLE_TRANSLATION_SCHEMA,
    TranslatedMetadata,
    translate_metadata,
)

# ------------------------------------------------------------------
# TranslatedMetadata
# ------------------------------------------------------------------


class TestTranslatedMetadata:
    def test_frozen(self) -> None:
        t = TranslatedMetadata(language_code="fi", title="T", description="D")
        with pytest.raises(AttributeError):
            t.title = "X"  # type: ignore[misc]

    def test_fields(self) -> None:
        t = TranslatedMetadata(language_code="sv", title="T", description="D")
        assert t.language_code == "sv"
        assert t.title == "T"
        assert t.description == "D"


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------


class TestSchemas:
    def test_batch_schema_required(self) -> None:
        assert "translations" in BATCH_TRANSLATION_SCHEMA["required"]

    def test_single_schema_required(self) -> None:
        assert set(SINGLE_TRANSLATION_SCHEMA["required"]) == {"title", "description"}


# ------------------------------------------------------------------
# translate_metadata — batch mode
# ------------------------------------------------------------------


class TestTranslateBatch:
    def test_batch_success(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "translations": [
                {"language": "fi", "title": "Otsikko", "description": "Kuvaus"},
                {"language": "sv", "title": "Titel", "description": "Beskrivning"},
            ],
        }
        registry = PromptRegistry()

        result = translate_metadata(
            client, registry, "Title", "Desc",
            languages={"fi": "Finnish", "sv": "Swedish"},
        )

        assert len(result) == 2
        assert result["fi"].title == "Otsikko"
        assert result["sv"].language_code == "sv"
        client.request_structured.assert_called_once()

    def test_batch_filters_unknown_languages(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {
            "translations": [
                {"language": "fi", "title": "T", "description": "D"},
                {"language": "xx", "title": "T", "description": "D"},
            ],
        }
        registry = PromptRegistry()

        result = translate_metadata(
            client, registry, "Title", "Desc",
            languages={"fi": "Finnish"},
        )

        assert "fi" in result
        assert "xx" not in result

    def test_empty_languages(self) -> None:
        client = MagicMock()
        registry = PromptRegistry()

        result = translate_metadata(client, registry, "Title", "Desc", languages={})
        assert result == {}
        client.request_structured.assert_not_called()


# ------------------------------------------------------------------
# translate_metadata — per-language mode
# ------------------------------------------------------------------


class TestTranslatePerLanguage:
    def test_per_language_success(self) -> None:
        client = MagicMock()
        client.request_structured.side_effect = [
            {"title": "Otsikko", "description": "Kuvaus"},
            {"title": "Titel", "description": "Beskrivning"},
        ]
        registry = PromptRegistry()

        result = translate_metadata(
            client, registry, "Title", "Desc",
            languages={"fi": "Finnish", "sv": "Swedish"},
            per_language_prompts={"fi": "translate_single", "sv": "translate_single"},
        )

        assert len(result) == 2
        assert result["fi"].title == "Otsikko"
        assert result["sv"].title == "Titel"
        assert client.request_structured.call_count == 2

    def test_per_language_partial_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        client = MagicMock()
        client.request_structured.side_effect = [
            OpenAIError("API error"),
            {"title": "Titel", "description": "Beskrivning"},
        ]
        registry = PromptRegistry()

        with caplog.at_level(logging.WARNING):
            result = translate_metadata(
                client, registry, "Title", "Desc",
                languages={"fi": "Finnish", "sv": "Swedish"},
                per_language_prompts={"fi": "translate_single", "sv": "translate_single"},
            )

        assert "fi" not in result
        assert "sv" in result
        assert "Translation failed" in caplog.text

    def test_per_language_uses_default_prompt(self) -> None:
        """Falls back to translate_single when no per-language prompt is configured."""
        client = MagicMock()
        client.request_structured.return_value = {"title": "T", "description": "D"}
        registry = PromptRegistry()

        result = translate_metadata(
            client, registry, "Title", "Desc",
            languages={"fi": "Finnish"},
            per_language_prompts={"sv": "custom_sv_prompt"},
        )

        assert "fi" in result

    def test_per_language_key_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """KeyError in response parsing is caught and logged."""
        client = MagicMock()
        client.request_structured.return_value = {"title": "T"}  # missing description
        registry = PromptRegistry()

        with caplog.at_level(logging.WARNING):
            result = translate_metadata(
                client, registry, "Title", "Desc",
                languages={"fi": "Finnish"},
                per_language_prompts={"fi": "translate_single"},
            )

        assert "fi" not in result
        assert "Translation failed" in caplog.text


# ------------------------------------------------------------------
# Mode selection
# ------------------------------------------------------------------


class TestModeSelection:
    def test_batch_mode_when_no_per_language(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {"translations": []}
        registry = PromptRegistry()

        translate_metadata(
            client, registry, "Title", "Desc",
            languages={"fi": "Finnish"},
            per_language_prompts=None,
        )

        # Batch mode uses "translations" schema name
        call_kwargs = client.request_structured.call_args[1]
        assert call_kwargs["schema_name"] == "translations"

    def test_per_language_mode_when_configured(self) -> None:
        client = MagicMock()
        client.request_structured.return_value = {"title": "T", "description": "D"}
        registry = PromptRegistry()

        translate_metadata(
            client, registry, "Title", "Desc",
            languages={"fi": "Finnish"},
            per_language_prompts={"fi": "translate_single"},
        )

        # Per-language mode uses "translation" schema name
        call_kwargs = client.request_structured.call_args[1]
        assert call_kwargs["schema_name"] == "translation"
