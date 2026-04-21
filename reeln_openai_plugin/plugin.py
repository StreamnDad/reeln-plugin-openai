"""OpenAIPlugin — reeln-cli plugin for OpenAI-powered LLM integration."""

from __future__ import annotations

import json
import logging
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from reeln.models.auth import AuthCheckResult, AuthStatus
from reeln.models.plugin_schema import ConfigField, PluginConfigSchema
from reeln.models.zoom import ExtractedFrames, ZoomPath, ZoomPoint
from reeln.plugins.hooks import Hook, HookContext
from reeln.plugins.registry import HookRegistry

from reeln_openai_plugin.client import OpenAIClient, OpenAIError
from reeln_openai_plugin.frames import FrameDescriptions, describe_frames
from reeln_openai_plugin.game_image import generate_game_image
from reeln_openai_plugin.livestream import generate_livestream_metadata
from reeln_openai_plugin.playlist import generate_playlist_metadata
from reeln_openai_plugin.prompts import PromptRegistry
from reeln_openai_plugin.render_metadata import generate_render_metadata
from reeln_openai_plugin.translate import translate_metadata
from reeln_openai_plugin.zoom import analyze_frame_for_zoom

log: logging.Logger = logging.getLogger(__name__)


def _resolve_scoring_opposing(
    game_event: object | None,
    game_info: object | None,
) -> tuple[str, str]:
    """Determine (scoring_team_name, opposing_team_name) from an event.

    Resolution priority (highest wins):

    1. ``game_event.metadata["team"]`` — explicit dock tag of ``"home"``
       or ``"away"``. This is the canonical dock tagging convention.
    2. ``game_event.event_type`` prefix — ``home_*``/``away_*``.
       Legacy CLI convention.
    3. Empty strings when neither is available — the LLM prompt will
       fall back to "infer from context".

    Returns a tuple of (scoring_team, opposing_team) as human-readable
    team names, sourced from ``game_info.home_team`` / ``away_team``.
    """
    if game_info is None:
        return ("", "")

    home_team = str(getattr(game_info, "home_team", "") or "")
    away_team = str(getattr(game_info, "away_team", "") or "")

    # 1. metadata["team"] wins
    if game_event is not None:
        metadata = getattr(game_event, "metadata", None) or {}
        raw_hint = metadata.get("team") if isinstance(metadata, dict) else None
        if isinstance(raw_hint, str):
            hint = raw_hint.lower()
            if hint == "away":
                return (away_team, home_team)
            if hint == "home":
                return (home_team, away_team)

    # 2. event_type prefix
    if game_event is not None:
        event_type = str(getattr(game_event, "event_type", "") or "").lower()
        if event_type.startswith("away_"):
            return (away_team, home_team)
        if event_type.startswith("home_"):
            return (home_team, away_team)

    # 3. Unknown — return empty so the LLM infers from context.
    return ("", "")


class OpenAIPlugin:
    """Plugin that provides OpenAI LLM integration for reeln-cli.

    Subscribes to ``ON_GAME_INIT`` to generate livestream metadata
    (title, description) and optional multi-language translations via
    the OpenAI ``/v1/responses`` API.
    """

    name: str = "openai"
    version: str = "0.9.0"
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
            ConfigField(
                name="render_metadata_enabled",
                field_type="bool",
                default=False,
                description="Enable LLM-generated title and description for rendered clips",
            ),
            ConfigField(
                name="smart_zoom_enabled",
                field_type="bool",
                default=False,
                description="Enable smart zoom target detection via OpenAI vision",
            ),
            ConfigField(
                name="smart_zoom_model",
                field_type="str",
                default="gpt-4.1",
                description="OpenAI model for smart zoom vision analysis",
            ),
            ConfigField(
                name="frame_description_enabled",
                field_type="bool",
                default=False,
                description="Enable frame description generation via OpenAI vision",
            ),
            ConfigField(
                name="frame_description_model",
                field_type="str",
                default="gpt-4.1",
                description="OpenAI model for frame description generation",
            ),
        )
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config: dict[str, Any] = config or {}
        self._client: OpenAIClient | None = None
        self._game_info: object | None = None
        self._frame_descriptions: FrameDescriptions | None = None

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
        registry.register(Hook.ON_QUEUE, self.on_queue)
        registry.register(Hook.POST_RENDER, self.on_post_render)
        registry.register(Hook.ON_FRAMES_EXTRACTED, self.on_frames_extracted)

    def on_game_init(self, context: HookContext) -> None:
        """Handle ``ON_GAME_INIT`` — generate livestream metadata."""
        if not self._config.get("enabled", False):
            return

        game_info = context.data.get("game_info")
        if game_info is None:
            log.warning("%s plugin: no game_info in context, skipping", self.name)
            return

        self._game_info = game_info

        home_profile = context.data.get("home_profile")
        away_profile = context.data.get("away_profile")
        user_thumbnail = getattr(game_info, "thumbnail", "")
        image_only = context.data.get("regenerate_image_only", False)

        try:
            client = self._get_client()
        except OpenAIError as exc:
            log.warning("%s plugin: client setup failed: %s", self.name, exc)
            return

        if not image_only:
            # Generate livestream metadata (description is passed as LLM context)
            self._generate_livestream_metadata(client, game_info, context, home_profile, away_profile)

            # Generate playlist metadata (only if livestream metadata was generated)
            livestream_meta = context.shared.get("livestream_metadata")
            if self._config.get("playlist_enabled", False) and livestream_meta:
                self._generate_playlist_metadata(client, game_info, context, livestream_meta["title"])

        # Generate game image (skip if user provided a thumbnail)
        if self._config.get("game_image_enabled", False):
            if user_thumbnail:
                log.info(
                    "%s plugin: user-provided thumbnail found, skipping image generation",
                    self.name,
                )
            else:
                self._maybe_generate_game_image(client, game_info, context, home_profile, away_profile)

    def _generate_livestream_metadata(
        self,
        client: OpenAIClient,
        game_info: object,
        context: HookContext,
        home_profile: object | None,
        away_profile: object | None,
    ) -> None:
        """Generate livestream metadata via LLM and optionally translate."""
        try:
            metadata = generate_livestream_metadata(
                client,
                self._prompt_registry,
                game_info,
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
                    code: {"title": t.title, "description": t.description} for code, t in translations.items()
                }
            except OpenAIError as exc:
                log.warning("%s plugin: translation failed: %s", self.name, exc)
                result["translations"] = {}

        context.shared["livestream_metadata"] = result
        log.info("%s plugin: generated livestream metadata: %s", self.name, result["title"])

    def _generate_playlist_metadata(
        self,
        client: OpenAIClient,
        game_info: object,
        context: HookContext,
        livestream_title: str,
    ) -> None:
        """Generate playlist metadata via LLM and optionally translate."""
        try:
            playlist = generate_playlist_metadata(
                client,
                self._prompt_registry,
                game_info,
                livestream_title=livestream_title,
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
                    code: {"title": t.title, "description": t.description} for code, t in playlist_translations.items()
                }
            except OpenAIError as exc:
                log.warning("%s plugin: playlist translation failed: %s", self.name, exc)
                playlist_result["translations"] = {}

        context.shared["playlist_metadata"] = playlist_result
        log.info("%s plugin: generated playlist metadata: %s", self.name, playlist_result["title"])

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
            away_profile,
            "logo_path",
            None,
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
                level=str(getattr(game_info, "level", "")),
                description=str(getattr(game_info, "description", "")),
                tournament=str(getattr(game_info, "tournament", "")),
            )
            context.shared["game_image"] = {"image_path": str(image_result.image_path)}
            log.info("%s plugin: generated game image: %s", self.name, image_result.image_path)
        except OpenAIError as exc:
            log.warning("%s plugin: game image generation failed: %s", self.name, exc)

    def on_queue(self, context: HookContext) -> None:
        """Handle ``ON_QUEUE`` — enrich title and description for queued renders.

        Runs after a render is added to the queue (via ``--queue``), generating
        an AI title and description that the user can review before publishing.
        Updates the queue item in-place so the enriched metadata is visible in
        the dock's Review & Publish panel.
        """
        if not self._config.get("render_metadata_enabled", False):
            return

        queue_item = context.data.get("queue_item")
        if queue_item is None:
            return

        game_info = self._game_info or context.data.get("game_info")
        if game_info is None:
            log.debug("%s plugin: no game_info available, skipping queue metadata", self.name)
            return

        try:
            client = self._get_client()
        except OpenAIError as exc:
            log.warning("%s plugin: client setup failed: %s", self.name, exc)
            return

        clip_name = Path(queue_item.output).stem if queue_item.output else ""

        frame_summary = ""
        if self._frame_descriptions is not None:
            frame_summary = self._frame_descriptions.summary
            self._frame_descriptions = None

        player_name = getattr(queue_item, "player", "")
        assists_str = getattr(queue_item, "assists", "")
        event_type = getattr(queue_item, "event_type", "")
        level = getattr(queue_item, "level", "") or getattr(game_info, "level", "")

        # Resolve the scoring team from the GameEvent's metadata so the
        # LLM prompt can attribute the play correctly. Without this, GPT
        # defaults to naming the home team as the scoring team because
        # the prompt has no other signal about which side the player is
        # on — this reliably generates wrong descriptions for away-team
        # plays (e.g. "Cozine scores for Machine Orange" when Cozine is
        # on Blades Maroon).
        game_event = context.data.get("game_event")
        scoring_team, opposing_team = _resolve_scoring_opposing(
            game_event, game_info
        )

        try:
            metadata = generate_render_metadata(
                client,
                self._prompt_registry,
                game_info,
                clip_name=clip_name,
                frame_summary=frame_summary,
                player=player_name,
                assists=assists_str,
                event_type=event_type,
                level=level,
                scoring_team=scoring_team,
                opposing_team=opposing_team,
            )
        except OpenAIError as exc:
            log.warning("%s plugin: queue metadata generation failed: %s", self.name, exc)
            return

        # Persist enriched metadata back to the queue item on disk
        try:
            from reeln.core.queue import update_queue_item

            update_queue_item(
                Path(queue_item.game_dir),
                queue_item.id,
                title=metadata.title,
                description=metadata.description,
            )
        except Exception as exc:
            log.warning("%s plugin: failed to update queue item: %s", self.name, exc)
            return

        context.shared["render_metadata"] = {
            "title": metadata.title,
            "description": metadata.description,
        }
        log.info("%s plugin: enriched queue item %s: %s", self.name, queue_item.id, metadata.title)

    def on_post_render(self, context: HookContext) -> None:
        """Handle ``POST_RENDER`` — generate short title and description."""
        if not self._config.get("render_metadata_enabled", False):
            return

        # game_info may be cached from ON_GAME_INIT (same process) or passed
        # through the hook data (separate CLI invocation like render short).
        game_info = self._game_info or context.data.get("game_info")
        if game_info is None:
            log.debug("%s plugin: no game_info available, skipping render metadata", self.name)
            return

        plan = context.data.get("plan")
        if plan is None:
            return

        try:
            client = self._get_client()
        except OpenAIError as exc:
            log.warning("%s plugin: client setup failed: %s", self.name, exc)
            return

        clip_name = getattr(getattr(plan, "output", None), "stem", "")

        frame_summary = ""
        if self._frame_descriptions is not None:
            frame_summary = self._frame_descriptions.summary
            self._frame_descriptions = None

        # Extract event context from hook data
        player_name = context.data.get("player", "")
        assists_str = context.data.get("assists", "")
        game_event = context.data.get("game_event")
        event_type = getattr(game_event, "event_type", "") if game_event else ""
        level = getattr(game_info, "level", "")
        scoring_team, opposing_team = _resolve_scoring_opposing(
            game_event, game_info
        )

        try:
            metadata = generate_render_metadata(
                client,
                self._prompt_registry,
                game_info,
                clip_name=clip_name,
                frame_summary=frame_summary,
                player=player_name,
                assists=assists_str,
                event_type=event_type,
                level=level,
                scoring_team=scoring_team,
                opposing_team=opposing_team,
            )
        except OpenAIError as exc:
            log.warning("%s plugin: render metadata generation failed: %s", self.name, exc)
            return

        context.shared["render_metadata"] = {
            "title": metadata.title,
            "description": metadata.description,
        }
        log.info("%s plugin: generated render metadata: %s", self.name, metadata.title)

    def on_frames_extracted(self, context: HookContext) -> None:
        """Handle ``ON_FRAMES_EXTRACTED`` — analyze frames for zoom targets and describe frames."""
        smart_zoom_enabled = self._config.get("smart_zoom_enabled", False)
        frame_desc_enabled = self._config.get("frame_description_enabled", False)

        if not smart_zoom_enabled and not frame_desc_enabled:
            return

        frames_data = context.data.get("frames")
        if frames_data is None:
            log.warning("%s plugin: no frames in context, skipping", self.name)
            return

        frames: ExtractedFrames = frames_data

        if not frames.frame_paths:
            log.warning("%s plugin: empty frame list, skipping", self.name)
            return

        try:
            client = self._get_client()
        except OpenAIError as exc:
            log.warning("%s plugin: client setup failed: %s", self.name, exc)
            return

        if smart_zoom_enabled:
            try:
                zoom_path = self._analyze_frames_for_zoom(client, frames)
            except OpenAIError as exc:
                log.error(
                    "%s plugin: smart zoom analysis failed after retries: %s",
                    self.name,
                    exc,
                )
                context.shared["smart_zoom"] = {"error": str(exc)}
                return
            context.shared["smart_zoom"] = {"zoom_path": zoom_path}

        if frame_desc_enabled:
            self._describe_frames(client, frames, context)

    def _analyze_frames_for_zoom(
        self,
        client: OpenAIClient,
        frames: ExtractedFrames,
    ) -> ZoomPath | None:
        """Analyze each frame and return a :class:`ZoomPath`.

        Each frame is retried with exponential backoff on transient errors.
        If any frame fails after all retries, the entire analysis fails
        (raises ``OpenAIError``) rather than producing a jittery zoom path
        from fallback coordinates.
        """
        model = str(self._config.get("smart_zoom_model", "gpt-4.1"))
        points: list[ZoomPoint] = []

        for frame_path, timestamp in zip(frames.frame_paths, frames.timestamps, strict=True):
            center_x, center_y = analyze_frame_for_zoom(
                client=client,
                prompt_registry=self._prompt_registry,
                frame_path=frame_path,
                model=model,
            )

            points.append(
                ZoomPoint(
                    timestamp=timestamp,
                    center_x=center_x,
                    center_y=center_y,
                ),
            )

        log.info(
            "%s plugin: smart zoom analysis complete — %d/%d frames succeeded",
            self.name,
            len(frames.frame_paths),
            len(frames.frame_paths),
        )

        return ZoomPath(
            points=tuple(points),
            source_width=frames.source_width,
            source_height=frames.source_height,
            duration=frames.duration,
        )

    def _describe_frames(
        self,
        client: OpenAIClient,
        frames: ExtractedFrames,
        context: HookContext,
    ) -> None:
        """Generate frame descriptions and cache for POST_RENDER use."""
        model = str(self._config.get("frame_description_model", "gpt-4.1"))
        try:
            descriptions = describe_frames(
                client=client,
                prompt_registry=self._prompt_registry,
                frame_paths=frames.frame_paths,
                model=model,
            )
        except OpenAIError as exc:
            log.warning("%s plugin: frame description failed: %s", self.name, exc)
            return

        self._frame_descriptions = descriptions
        context.shared["frame_descriptions"] = {
            "descriptions": list(descriptions.descriptions),
            "summary": descriptions.summary,
        }
        log.info(
            "%s plugin: described %d frames — %s",
            self.name,
            len(descriptions.descriptions),
            descriptions.summary[:80],
        )

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

    # ------------------------------------------------------------------
    # Authenticator capability
    # ------------------------------------------------------------------

    def auth_check(self) -> list[AuthCheckResult]:
        """Validate the OpenAI API key by listing models."""
        try:
            api_key = self._resolve_api_key()
        except OpenAIError as exc:
            return [
                AuthCheckResult(
                    service="OpenAI API",
                    status=AuthStatus.NOT_CONFIGURED,
                    message=str(exc),
                    hint="Set api_key_file in config or OPENAI_API_KEY env var",
                )
            ]

        redacted = api_key[:7] + "..." if len(api_key) > 7 else "***"

        try:
            req = urllib.request.Request(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                method="GET",
            )
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                resp.read()
        except urllib.error.HTTPError as exc:
            return [
                AuthCheckResult(
                    service="OpenAI API",
                    status=AuthStatus.FAIL,
                    message=f"API key validation failed: HTTP {exc.code}",
                    identity=redacted,
                    hint="Check that your API key is valid and has not been revoked",
                )
            ]
        except Exception as exc:
            return [
                AuthCheckResult(
                    service="OpenAI API",
                    status=AuthStatus.WARN,
                    message=f"Could not validate API key: {exc}",
                    identity=redacted,
                    hint="Network error — key may still be valid",
                )
            ]

        return [
            AuthCheckResult(
                service="OpenAI API",
                status=AuthStatus.OK,
                message="Authenticated",
                identity=redacted,
            )
        ]

    def auth_refresh(self) -> list[AuthCheckResult]:
        """API keys cannot be refreshed automatically."""
        return [
            AuthCheckResult(
                service="OpenAI API",
                status=AuthStatus.FAIL,
                message="API keys cannot be refreshed automatically",
                hint="Generate a new key at https://platform.openai.com/api-keys",
            )
        ]

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
