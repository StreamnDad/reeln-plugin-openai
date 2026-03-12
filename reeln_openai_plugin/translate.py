"""Multi-language translation of metadata via OpenAI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from reeln_openai_plugin.client import OpenAIClient, OpenAIError
from reeln_openai_plugin.prompts import PromptRegistry

log: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranslatedMetadata:
    """A single language translation of title and description."""

    language_code: str
    title: str
    description: str


BATCH_TRANSLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["language", "title", "description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["translations"],
    "additionalProperties": False,
}

SINGLE_TRANSLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["title", "description"],
    "additionalProperties": False,
}


def translate_metadata(
    client: OpenAIClient,
    prompt_registry: PromptRegistry,
    title: str,
    description: str,
    languages: dict[str, str],
    per_language_prompts: dict[str, str] | None = None,
) -> dict[str, TranslatedMetadata]:
    """Translate *title* and *description* into *languages*.

    If *per_language_prompts* maps language codes to prompt names, each
    language is translated in a separate API call using that prompt.
    Otherwise a single batch call translates all languages at once.

    Individual language failures in per-language mode are logged and skipped.
    """
    if not languages:
        return {}

    if per_language_prompts:
        return _translate_per_language(
            client, prompt_registry, title, description, languages, per_language_prompts,
        )
    return _translate_batch(client, prompt_registry, title, description, languages)


def _translate_batch(
    client: OpenAIClient,
    prompt_registry: PromptRegistry,
    title: str,
    description: str,
    languages: dict[str, str],
) -> dict[str, TranslatedMetadata]:
    """Translate all languages in a single API call."""
    lang_lines = "\n".join(f"- {code}: {name}" for code, name in languages.items())
    variables = {
        "title": title,
        "description": description,
        "languages": lang_lines,
    }
    prompt = prompt_registry.render("translate_batch", variables)
    result = client.request_structured(
        prompt=prompt,
        schema=BATCH_TRANSLATION_SCHEMA,
        schema_name="translations",
    )

    translations: dict[str, TranslatedMetadata] = {}
    for item in result.get("translations", []):
        code = item["language"]
        if code in languages:
            translations[code] = TranslatedMetadata(
                language_code=code,
                title=item["title"],
                description=item["description"],
            )
    return translations


def _translate_per_language(
    client: OpenAIClient,
    prompt_registry: PromptRegistry,
    title: str,
    description: str,
    languages: dict[str, str],
    per_language_prompts: dict[str, str],
) -> dict[str, TranslatedMetadata]:
    """Translate each language with its own prompt and API call."""
    translations: dict[str, TranslatedMetadata] = {}

    for code, name in languages.items():
        prompt_name = per_language_prompts.get(code, "translate_single")
        variables = {
            "title": title,
            "description": description,
            "language": name,
            "language_code": code,
        }
        try:
            prompt = prompt_registry.render(prompt_name, variables)
            result = client.request_structured(
                prompt=prompt,
                schema=SINGLE_TRANSLATION_SCHEMA,
                schema_name="translation",
            )
            translations[code] = TranslatedMetadata(
                language_code=code,
                title=result["title"],
                description=result["description"],
            )
        except (OpenAIError, KeyError) as exc:
            log.warning("Translation failed for %s (%s): %s", name, code, exc)

    return translations
