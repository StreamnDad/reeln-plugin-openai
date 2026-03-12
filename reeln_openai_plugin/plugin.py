"""OpenAIPlugin — reeln-cli plugin for OpenAI-powered LLM integration."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from reeln.models.plugin_schema import ConfigField, PluginConfigSchema
from reeln.plugins.hooks import Hook, HookContext
from reeln.plugins.registry import HookRegistry

from reeln_openai_plugin.client import OpenAIClient, OpenAIError
from reeln_openai_plugin.game_image import generate_game_image
from reeln_openai_plugin.livestream import generate_livestream_metadata
from reeln_openai_plugin.playlist import generate_playlist_metadata
from reeln_openai_plugin.prompts import PromptRegistry
from reeln_openai_plugin.translate import translate_metadata

log: logging.Logger = logging.getLogger(__name__)


class OpenAIPlugin:
    """Plugin that provides OpenAI LLM integration for reeln-cli.

    Subscribes to ``ON_GAME_INIT`` to generate livestream metadata
    (title, description) and optional multi-language translations via
    the OpenAI ``/v1/responses`` API.
    """

    name: str = "openai"
    version: str = "0.4.0"
    api_version: int = 1

    config_schema: PluginConfigSchema = PluginConfigSchema(
        fields=(
            ConfigField(
                name="enabled",
                field_type="bool",
                default=False,
                description="Enable OpenAI LLM integration",
            ),
            ConfigField(
                name="api_key_file",
                field_type="str",
                default="",
                description="Path to file containing the OpenAI API key (falls back to OPENAI_API_KEY env var)",
                secret=True,
            ),
            ConfigField(
                name="model",
                field_type="str",
                default="gpt-4.1",
                description="OpenAI model to use for requests",
            ),
            ConfigField(
                name="request_timeout_seconds",
                field_type="float",
                default=30.0,
                description="Timeout in seconds for API requests",
            ),
            ConfigField(
                name="prompt_overrides",
                field_type="str",
                default="{}",
                description="JSON dict of prompt name to override file path",
            ),
            ConfigField(
                name="prompt_context",
                field_type="str",
                default="{}",
                description="JSON dict of prompt name to list of extra context lines",
            ),
            ConfigField(
                name="translate_enabled",
                field_type="bool",
                default=False,
                description="Enable multi-language translation",
            ),
            ConfigField(
                name="translate_languages",
                field_type="str",
                default="{}",
                description='JSON dict of language code to name (e.g. {"fi": "Finnish"})',
            ),
            ConfigField(
                name="translate_per_language_prompts",
                field_type="str",
                default="{}",
                description="JSON dict of language code to prompt name for per-language mode",
            ),
            ConfigField(
                name="playlist_enabled",
                field_type="bool",
                default=False,
                description="Enable LLM-generated playlist title and description",
            ),
            ConfigField(
                name="game_image_enabled",
                field_type="bool",
                default=False,
                description="Enable game image thumbnail generation",
            ),
            ConfigField(
                name="game_image_model",
                field_type="str",
                default="gpt-5.2",
                description="OpenAI model for game image orchestration",
            ),
            ConfigField(
                name="game_image_renderer_model",
                field_type="str",
                default="gpt-image-1.5",
                description="OpenAI model for image rendering",
            ),
            ConfigField(
                name="game_image_output_dir",
                field_type="str",
                default="",
                description="Directory to save generated game images",
            ),
        )
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config: dict[str, Any] = config or {}
        self._client: OpenAIClient | None = None

        # Parse JSON string configs
        self._prompt_overrides: dict[str, str] = self._parse_json_config("prompt_overrides")
        self._prompt_context: dict[str, list[str]] = self._parse_json_config("prompt_context")
        self._translate_languages: dict[str, str] = self._parse_json_config("translate_languages")
        self._per_language_prompts: dict[str, str] = self._parse_json_config(
            "translate_per_language_prompts",
        )

        self._prompt_registry: PromptRegistry = PromptRegistry(
            overrides=self._prompt_overrides,
            prompt_context=self._prompt_context,
        )

    def register(self, registry: HookRegistry) -> None:
        """Register hook handlers with the reeln plugin registry."""
        registry.register(Hook.ON_GAME_INIT, self.on_game_init)

    def on_game_init(self, context: HookContext) -> None:
        """Handle ``ON_GAME_INIT`` — generate livestream metadata."""
        if not self._config.get("enabled", False):
            return

        game_info = context.data.get("game_info")
        if game_info is None:
            log.warning("%s plugin: no game_info in context, skipping", self.name)
            return

        try:
            client = self._get_client()
        except OpenAIError as exc:
            log.warning("%s plugin: client setup failed: %s", self.name, exc)
            return

        home_profile = context.data.get("home_profile")
        away_profile = context.data.get("away_profile")

        # Generate livestream metadata
        try:
            metadata = generate_livestream_metadata(
                client, self._prompt_registry, game_info,
                home_profile=home_profile,
                away_profile=away_profile,
            )
        except OpenAIError as exc:
            log.warning("%s plugin: metadata generation failed: %s", self.name, exc)
            return

        result: dict[str, Any] = {
            "title": metadata.title,
            "description": metadata.description,
        }

        # Translate if enabled
        if self._config.get("translate_enabled", False) and self._translate_languages:
            try:
                translations = translate_metadata(
                    client,
                    self._prompt_registry,
                    title=metadata.title,
                    description=metadata.description,
                    languages=self._translate_languages,
                    per_language_prompts=self._per_language_prompts or None,
                )
                result["translations"] = {
                    code: {"title": t.title, "description": t.description}
                    for code, t in translations.items()
                }
            except OpenAIError as exc:
                log.warning("%s plugin: translation failed: %s", self.name, exc)
                result["translations"] = {}

        context.shared["livestream_metadata"] = result
        log.info("%s plugin: generated livestream metadata: %s", self.name, result["title"])

        # Generate playlist metadata
        if self._config.get("playlist_enabled", False):
            try:
                playlist = generate_playlist_metadata(
                    client,
                    self._prompt_registry,
                    game_info,
                    livestream_title=metadata.title,
                )
            except OpenAIError as exc:
                log.warning("%s plugin: playlist generation failed: %s", self.name, exc)
                return

            playlist_result: dict[str, Any] = {
                "title": playlist.title,
                "description": playlist.description,
            }

            if self._config.get("translate_enabled", False) and self._translate_languages:
                try:
                    playlist_translations = translate_metadata(
                        client,
                        self._prompt_registry,
                        title=playlist.title,
                        description=playlist.description,
                        languages=self._translate_languages,
                        per_language_prompts=self._per_language_prompts or None,
                    )
                    playlist_result["translations"] = {
                        code: {"title": t.title, "description": t.description}
                        for code, t in playlist_translations.items()
                    }
                except OpenAIError as exc:
                    log.warning("%s plugin: playlist translation failed: %s", self.name, exc)
                    playlist_result["translations"] = {}

            context.shared["playlist_metadata"] = playlist_result
            log.info("%s plugin: generated playlist metadata: %s", self.name, playlist_result["title"])

        # Generate game image
        if self._config.get("game_image_enabled", False):
            self._maybe_generate_game_image(client, game_info, context, home_profile, away_profile)

    def _maybe_generate_game_image(
        self,
        client: OpenAIClient,
        game_info: object,
        context: HookContext,
        home_profile: object | None,
        away_profile: object | None,
    ) -> None:
        """Generate a game thumbnail if team profiles and logos are available."""
        if home_profile is None or away_profile is None:
            log.warning("%s plugin: missing team profiles for game image, skipping", self.name)
            return

        if not getattr(home_profile, "logo_path", None) or not getattr(
            away_profile, "logo_path", None,
        ):
            log.warning("%s plugin: missing team logos for game image, skipping", self.name)
            return

        output_dir_str = self._config.get("game_image_output_dir", "")
        if not output_dir_str:
            log.warning("%s plugin: game_image_output_dir not configured, skipping", self.name)
            return

        try:
            image_result = generate_game_image(
                client=client,
                prompt_registry=self._prompt_registry,
                home=home_profile,
                away=away_profile,
                rink=str(getattr(game_info, "venue", "")),
                game_date=str(getattr(game_info, "date", "")),
                game_time=str(getattr(game_info, "game_time", "")),
                output_dir=Path(output_dir_str),
                model=str(self._config.get("game_image_model", "gpt-5.2")),
                renderer_model=str(
                    self._config.get("game_image_renderer_model", "gpt-image-1.5"),
                ),
            )
            context.shared["game_image"] = {"image_path": str(image_result.image_path)}
            log.info("%s plugin: generated game image: %s", self.name, image_result.image_path)
        except OpenAIError as exc:
            log.warning("%s plugin: game image generation failed: %s", self.name, exc)

    def _get_client(self) -> OpenAIClient:
        """Lazily create and return the :class:`OpenAIClient`.

        API key resolution order:
        1. ``api_key_file`` config — read key from file
        2. ``OPENAI_API_KEY`` environment variable — fallback
        """
        if self._client is not None:
            return self._client

        api_key = self._resolve_api_key()
        model = self._config.get("model", "gpt-4.1")
        timeout = self._config.get("request_timeout_seconds", 30.0)

        self._client = OpenAIClient(
            api_key=api_key,
            model=str(model),
            timeout_seconds=float(timeout),
        )
        return self._client

    def _resolve_api_key(self) -> str:
        """Resolve the OpenAI API key from file or environment.

        Tries ``api_key_file`` config first, then falls back to the
        ``OPENAI_API_KEY`` environment variable.
        """
        api_key_file = self._config.get("api_key_file")
        if api_key_file:
            path = Path(api_key_file)
            if path.exists():
                return OpenAIClient.read_api_key(path)
            log.warning(
                "%s plugin: api_key_file %s not found, trying OPENAI_API_KEY env var",
                self.name,
                api_key_file,
            )

        env_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if env_key:
            return env_key

        raise OpenAIError(
            "No API key: set api_key_file in config or OPENAI_API_KEY env var",
        )

    def _parse_json_config(self, key: str) -> Any:
        """Parse a JSON string config value, defaulting to ``{}``."""
        raw = self._config.get(key, "{}")
        if not isinstance(raw, str):
            return raw
        try:
            result: Any = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("%s plugin: invalid JSON for config '%s', using default", self.name, key)
            return {}
        return result
