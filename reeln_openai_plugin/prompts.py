"""Prompt template engine — bundled defaults with config overrides."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any


class PromptError(Exception):
    """Raised when a prompt template cannot be loaded or rendered."""


class PromptTemplate:
    """A loaded prompt template with ``{{variable}}`` placeholders."""

    def __init__(self, text: str) -> None:
        self._text: str = text

    @property
    def text(self) -> str:
        return self._text

    def render(self, variables: dict[str, str]) -> str:
        """Substitute ``{{key}}`` placeholders with *variables* values."""
        result = self._text
        for key, value in variables.items():
            result = result.replace(f"{{{{{key}}}}}", value)
        return result


class PromptRegistry:
    """Resolve and render prompt templates.

    Resolution order for a given *name*:

    1. Config override path (if present in *overrides*)
    2. Bundled default from ``reeln_openai_plugin.prompts``

    After loading, optional *prompt_context* lines are appended.
    """

    def __init__(
        self,
        overrides: dict[str, str] | None = None,
        prompt_context: dict[str, list[str]] | None = None,
    ) -> None:
        self._overrides: dict[str, str] = overrides or {}
        self._prompt_context: dict[str, list[str]] = prompt_context or {}
        self._cache: dict[str, PromptTemplate] = {}

    def load(self, name: str) -> PromptTemplate:
        """Load a :class:`PromptTemplate` by *name* (without ``.txt`` extension)."""
        if name in self._cache:
            return self._cache[name]

        text = self._read(name)
        template = PromptTemplate(text)
        self._cache[name] = template
        return template

    def render(self, name: str, variables: dict[str, str] | None = None) -> str:
        """Load, append context lines, and render the template *name*."""
        template = self.load(name)
        text = template.text

        # Append context lines if configured
        context_lines = self._prompt_context.get(name)
        if context_lines:
            text = text + "\n" + "\n".join(context_lines)

        # Render variables
        if variables:
            for key, value in variables.items():
                text = text.replace(f"{{{{{key}}}}}", value)
        return text

    def _read(self, name: str) -> str:
        """Read raw template text from override path or bundled resource."""
        override_path = self._overrides.get(name)
        if override_path is not None:
            return self._read_file(Path(override_path))
        return self._read_bundled(name)

    @staticmethod
    def _read_file(path: Path) -> str:
        """Read a prompt template from a file path."""
        try:
            return path.read_text().strip()
        except OSError as exc:
            raise PromptError(f"Cannot read prompt file {path}: {exc}") from exc

    @staticmethod
    def _read_bundled(name: str) -> str:
        """Read a prompt template from the bundled ``prompts`` package."""
        filename = f"{name}.txt"
        try:
            ref = importlib.resources.files("reeln_openai_plugin.prompt_templates").joinpath(filename)
            text: Any = ref.read_text(encoding="utf-8")
            return str(text).strip()
        except (FileNotFoundError, TypeError, ModuleNotFoundError) as exc:
            raise PromptError(f"Bundled prompt '{name}' not found") from exc
