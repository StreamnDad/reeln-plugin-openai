"""Tests for the prompt template engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from reeln_openai_plugin.prompts import PromptError, PromptRegistry, PromptTemplate

# ------------------------------------------------------------------
# PromptTemplate
# ------------------------------------------------------------------


class TestPromptTemplate:
    def test_text_property(self) -> None:
        t = PromptTemplate("Hello {{name}}")
        assert t.text == "Hello {{name}}"

    def test_render(self) -> None:
        t = PromptTemplate("Hello {{name}}, welcome to {{place}}")
        assert t.render({"name": "Alice", "place": "Wonderland"}) == "Hello Alice, welcome to Wonderland"

    def test_render_no_variables(self) -> None:
        t = PromptTemplate("Static text")
        assert t.render({}) == "Static text"

    def test_render_missing_variable_left_as_is(self) -> None:
        t = PromptTemplate("Hello {{name}}")
        assert t.render({}) == "Hello {{name}}"


# ------------------------------------------------------------------
# PromptRegistry — bundled loading
# ------------------------------------------------------------------


class TestPromptRegistryBundled:
    def test_load_livestream_title(self) -> None:
        registry = PromptRegistry()
        template = registry.load("livestream_title")
        assert "{{home_team}}" in template.text
        assert "{{away_team}}" in template.text

    def test_load_livestream_description(self) -> None:
        registry = PromptRegistry()
        template = registry.load("livestream_description")
        assert "{{venue}}" in template.text

    def test_load_translate_batch(self) -> None:
        registry = PromptRegistry()
        template = registry.load("translate_batch")
        assert "{{languages}}" in template.text

    def test_load_translate_single(self) -> None:
        registry = PromptRegistry()
        template = registry.load("translate_single")
        assert "{{language_code}}" in template.text

    def test_missing_prompt_raises(self) -> None:
        registry = PromptRegistry()
        with pytest.raises(PromptError, match="not found"):
            registry.load("nonexistent_prompt")


# ------------------------------------------------------------------
# PromptRegistry — override paths
# ------------------------------------------------------------------


class TestPromptRegistryOverrides:
    def test_override_path(self, tmp_path: Path) -> None:
        override_file = tmp_path / "custom.txt"
        override_file.write_text("Custom prompt for {{team}}")

        registry = PromptRegistry(overrides={"livestream_title": str(override_file)})
        template = registry.load("livestream_title")
        assert template.text == "Custom prompt for {{team}}"

    def test_override_missing_file(self, tmp_path: Path) -> None:
        registry = PromptRegistry(overrides={"livestream_title": str(tmp_path / "missing.txt")})
        with pytest.raises(PromptError, match="Cannot read prompt file"):
            registry.load("livestream_title")


# ------------------------------------------------------------------
# PromptRegistry — rendering
# ------------------------------------------------------------------


class TestPromptRegistryRender:
    def test_render_with_variables(self) -> None:
        registry = PromptRegistry()
        variables = {"home_team": "Storm", "away_team": "Thunder", "date": "2026-01-15", "sport": "hockey"}
        result = registry.render("livestream_title", variables)
        assert "Storm" in result
        assert "Thunder" in result

    def test_render_with_context(self) -> None:
        registry = PromptRegistry(prompt_context={"livestream_title": ["Extra line 1", "Extra line 2"]})
        result = registry.render("livestream_title", {"home_team": "A", "away_team": "B", "date": "D", "sport": "S"})
        assert "Extra line 1" in result
        assert "Extra line 2" in result

    def test_render_no_variables(self) -> None:
        registry = PromptRegistry()
        result = registry.render("livestream_title")
        assert "{{home_team}}" in result


# ------------------------------------------------------------------
# PromptRegistry — caching
# ------------------------------------------------------------------


class TestPromptRegistryCache:
    def test_load_caches(self) -> None:
        registry = PromptRegistry()
        t1 = registry.load("livestream_title")
        t2 = registry.load("livestream_title")
        assert t1 is t2
