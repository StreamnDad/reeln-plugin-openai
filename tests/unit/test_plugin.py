"""Tests for plugin module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from reeln.plugins.hooks import Hook, HookContext
from reeln.plugins.registry import HookRegistry

from reeln_openai_plugin.client import OpenAIError
from reeln_openai_plugin.plugin import OpenAIPlugin
from tests.conftest import FakeExtractedFrames, FakeGameInfo, FakeQueueItem, FakeTeamInfo

# ------------------------------------------------------------------
# Attributes
# ------------------------------------------------------------------


class TestPluginAttributes:
    def test_name(self) -> None:
        plugin = OpenAIPlugin()
        assert plugin.name == "openai"

    def test_version(self) -> None:
        plugin = OpenAIPlugin()
        assert plugin.version == "0.8.2"

    def test_api_version(self) -> None:
        plugin = OpenAIPlugin()
        assert plugin.api_version == 1


# ------------------------------------------------------------------
# Config schema validation
# ------------------------------------------------------------------


class TestPluginConfigSchema:
    def test_has_required_fields(self) -> None:
        names = [f.name for f in OpenAIPlugin.config_schema.fields]
        assert "enabled" in names
        assert "api_key_file" in names
        assert "model" in names

    def test_api_key_file_is_secret(self) -> None:
        field = OpenAIPlugin.config_schema.field_by_name("api_key_file")
        assert field is not None
        assert field.secret is True

    def test_has_game_image_fields(self) -> None:
        names = [f.name for f in OpenAIPlugin.config_schema.fields]
        assert "game_image_enabled" in names
        assert "game_image_model" in names
        assert "game_image_renderer_model" in names
        assert "game_image_output_dir" in names

    def test_has_render_metadata_field(self) -> None:
        names = [f.name for f in OpenAIPlugin.config_schema.fields]
        assert "render_metadata_enabled" in names

    def test_has_smart_zoom_fields(self) -> None:
        names = [f.name for f in OpenAIPlugin.config_schema.fields]
        assert "smart_zoom_enabled" in names
        assert "smart_zoom_model" in names

    def test_has_frame_description_fields(self) -> None:
        names = [f.name for f in OpenAIPlugin.config_schema.fields]
        assert "frame_description_enabled" in names
        assert "frame_description_model" in names

    def test_defaults(self) -> None:
        defaults = OpenAIPlugin.config_schema.defaults_dict()
        assert defaults["enabled"] is False
        assert defaults["model"] == "gpt-4.1"
        assert defaults["request_timeout_seconds"] == 30.0
        assert defaults["translate_enabled"] is False
        assert defaults["game_image_enabled"] is False
        assert defaults["game_image_model"] == "gpt-5.2"
        assert defaults["game_image_renderer_model"] == "gpt-image-1.5"
        assert defaults["game_image_output_dir"] == ""
        assert defaults["render_metadata_enabled"] is False
        assert defaults["smart_zoom_enabled"] is False
        assert defaults["smart_zoom_model"] == "gpt-4.1"
        assert defaults["frame_description_enabled"] is False
        assert defaults["frame_description_model"] == "gpt-4.1"


# ------------------------------------------------------------------
# Init
# ------------------------------------------------------------------


class TestPluginInit:
    def test_no_config(self) -> None:
        plugin = OpenAIPlugin()
        assert plugin._config == {}

    def test_empty_config(self) -> None:
        plugin = OpenAIPlugin({})
        assert plugin._config == {}

    def test_with_config(self, plugin_config: dict[str, Any]) -> None:
        plugin = OpenAIPlugin(plugin_config)
        assert plugin._config == plugin_config

    def test_parses_json_configs(self) -> None:
        config: dict[str, Any] = {
            "prompt_overrides": '{"livestream_title": "/tmp/custom.txt"}',
            "translate_languages": '{"fi": "Finnish"}',
        }
        plugin = OpenAIPlugin(config)
        assert plugin._prompt_overrides == {"livestream_title": "/tmp/custom.txt"}
        assert plugin._translate_languages == {"fi": "Finnish"}

    def test_invalid_json_config_defaults(self, caplog: pytest.LogCaptureFixture) -> None:
        config: dict[str, Any] = {"prompt_overrides": "not json"}
        with caplog.at_level(logging.WARNING):
            plugin = OpenAIPlugin(config)
        assert plugin._prompt_overrides == {}
        assert "invalid JSON" in caplog.text

    def test_non_string_json_config_passthrough(self) -> None:
        """When config value is already a dict (not a JSON string), pass through."""
        config: dict[str, Any] = {"prompt_overrides": {"key": "val"}}
        plugin = OpenAIPlugin(config)
        assert plugin._prompt_overrides == {"key": "val"}


# ------------------------------------------------------------------
# Register
# ------------------------------------------------------------------


class TestPluginRegister:
    def test_registers_on_game_init(self) -> None:
        plugin = OpenAIPlugin()
        registry = HookRegistry()
        plugin.register(registry)
        assert registry.has_handlers(Hook.ON_GAME_INIT)

    def test_registers_post_render(self) -> None:
        plugin = OpenAIPlugin()
        registry = HookRegistry()
        plugin.register(registry)
        assert registry.has_handlers(Hook.POST_RENDER)

    def test_registers_on_frames_extracted(self) -> None:
        plugin = OpenAIPlugin()
        registry = HookRegistry()
        plugin.register(registry)
        assert registry.has_handlers(Hook.ON_FRAMES_EXTRACTED)

    def test_does_not_register_other_hooks(self) -> None:
        plugin = OpenAIPlugin()
        registry = HookRegistry()
        plugin.register(registry)
        assert not registry.has_handlers(Hook.ON_GAME_FINISH)


# ------------------------------------------------------------------
# on_game_init
# ------------------------------------------------------------------


class TestOnGameInit:
    def test_disabled_skips(self) -> None:
        plugin = OpenAIPlugin({"enabled": False})
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": FakeGameInfo()})
        plugin.on_game_init(context)
        assert "livestream_metadata" not in context.shared

    def test_no_game_info_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        plugin = OpenAIPlugin({"enabled": True})
        context = HookContext(hook=Hook.ON_GAME_INIT, data={})

        with caplog.at_level(logging.WARNING):
            plugin.on_game_init(context)

        assert "no game_info" in caplog.text

    @patch.dict("os.environ", {}, clear=True)
    def test_client_failure_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        """Missing API key file and no env var causes client setup failure."""
        config: dict[str, Any] = {"enabled": True, "api_key_file": str(tmp_path / "missing.txt")}
        plugin = OpenAIPlugin(config)
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": FakeGameInfo()})

        with caplog.at_level(logging.WARNING):
            plugin.on_game_init(context)

        assert "client setup failed" in caplog.text

    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_generation_failure(
        self,
        mock_gen: MagicMock,
        caplog: pytest.LogCaptureFixture,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.client import OpenAIError

        mock_gen.side_effect = OpenAIError("API down")
        plugin = OpenAIPlugin(plugin_config)
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": FakeGameInfo()})

        with caplog.at_level(logging.WARNING):
            plugin.on_game_init(context)

        assert "metadata generation failed" in caplog.text

    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_success(
        self,
        mock_gen: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.livestream import LivestreamMetadata

        mock_gen.return_value = LivestreamMetadata(title="Title!", description="Desc!")
        plugin = OpenAIPlugin(plugin_config)
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": FakeGameInfo()})

        plugin.on_game_init(context)

        assert context.shared["livestream_metadata"]["title"] == "Title!"
        assert context.shared["livestream_metadata"]["description"] == "Desc!"

    @patch("reeln_openai_plugin.plugin.translate_metadata")
    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_success_with_translation(
        self,
        mock_gen: MagicMock,
        mock_translate: MagicMock,
        api_key_file: Path,
    ) -> None:
        from reeln_openai_plugin.livestream import LivestreamMetadata
        from reeln_openai_plugin.translate import TranslatedMetadata

        mock_gen.return_value = LivestreamMetadata(title="T", description="D")
        mock_translate.return_value = {
            "fi": TranslatedMetadata(language_code="fi", title="Otsikko", description="Kuvaus"),
        }

        config: dict[str, Any] = {
            "enabled": True,
            "api_key_file": str(api_key_file),
            "translate_enabled": True,
            "translate_languages": '{"fi": "Finnish"}',
        }
        plugin = OpenAIPlugin(config)
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": FakeGameInfo()})

        plugin.on_game_init(context)

        meta = context.shared["livestream_metadata"]
        assert meta["translations"]["fi"]["title"] == "Otsikko"

    @patch("reeln_openai_plugin.plugin.translate_metadata")
    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_translation_failure_non_fatal(
        self,
        mock_gen: MagicMock,
        mock_translate: MagicMock,
        caplog: pytest.LogCaptureFixture,
        api_key_file: Path,
    ) -> None:
        from reeln_openai_plugin.client import OpenAIError
        from reeln_openai_plugin.livestream import LivestreamMetadata

        mock_gen.return_value = LivestreamMetadata(title="T", description="D")
        mock_translate.side_effect = OpenAIError("Translate failed")

        config: dict[str, Any] = {
            "enabled": True,
            "api_key_file": str(api_key_file),
            "translate_enabled": True,
            "translate_languages": '{"fi": "Finnish"}',
        }
        plugin = OpenAIPlugin(config)
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": FakeGameInfo()})

        with caplog.at_level(logging.WARNING):
            plugin.on_game_init(context)

        # Metadata still set, translations empty
        assert context.shared["livestream_metadata"]["title"] == "T"
        assert context.shared["livestream_metadata"]["translations"] == {}
        assert "translation failed" in caplog.text

    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_user_description_passed_as_context(
        self,
        mock_gen: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        """When game_info.description is set, it is passed through to LLM as context."""
        from reeln_openai_plugin.livestream import LivestreamMetadata

        mock_gen.return_value = LivestreamMetadata(title="Semis!", description="Big game!")
        plugin = OpenAIPlugin(plugin_config)
        game_info = FakeGameInfo(description="Semifinals tournament game")
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": game_info})

        plugin.on_game_init(context)

        mock_gen.assert_called_once()
        assert context.shared["livestream_metadata"]["title"] == "Semis!"

    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_caches_game_info(
        self,
        mock_gen: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.livestream import LivestreamMetadata

        mock_gen.return_value = LivestreamMetadata(title="T", description="D")
        plugin = OpenAIPlugin(plugin_config)
        game_info = FakeGameInfo(home_team="Storm")
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": game_info})

        plugin.on_game_init(context)

        assert plugin._game_info is game_info

    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_translate_disabled_no_translations(
        self,
        mock_gen: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.livestream import LivestreamMetadata

        mock_gen.return_value = LivestreamMetadata(title="T", description="D")
        plugin = OpenAIPlugin({**plugin_config, "translate_enabled": False})
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": FakeGameInfo()})

        plugin.on_game_init(context)

        assert "translations" not in context.shared["livestream_metadata"]


# ------------------------------------------------------------------
# on_game_init — playlist
# ------------------------------------------------------------------


class TestOnGameInitPlaylist:
    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_playlist_disabled_skips(
        self,
        mock_gen: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.livestream import LivestreamMetadata

        mock_gen.return_value = LivestreamMetadata(title="T", description="D")
        plugin = OpenAIPlugin({**plugin_config, "playlist_enabled": False})
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": FakeGameInfo()})

        plugin.on_game_init(context)

        assert "livestream_metadata" in context.shared
        assert "playlist_metadata" not in context.shared

    @patch("reeln_openai_plugin.plugin.generate_playlist_metadata")
    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_playlist_success(
        self,
        mock_livestream: MagicMock,
        mock_playlist: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.livestream import LivestreamMetadata
        from reeln_openai_plugin.playlist import PlaylistMetadata

        mock_livestream.return_value = LivestreamMetadata(title="Live!", description="Go!")
        mock_playlist.return_value = PlaylistMetadata(title="Highlights!", description="Best of!")
        plugin = OpenAIPlugin({**plugin_config, "playlist_enabled": True})
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": FakeGameInfo()})

        plugin.on_game_init(context)

        assert context.shared["playlist_metadata"]["title"] == "Highlights!"
        assert context.shared["playlist_metadata"]["description"] == "Best of!"
        # Verify livestream_title was passed to playlist generator
        mock_playlist.assert_called_once()
        call_kwargs = mock_playlist.call_args[1]
        assert call_kwargs["livestream_title"] == "Live!"

    @patch("reeln_openai_plugin.plugin.translate_metadata")
    @patch("reeln_openai_plugin.plugin.generate_playlist_metadata")
    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_playlist_with_translation(
        self,
        mock_livestream: MagicMock,
        mock_playlist: MagicMock,
        mock_translate: MagicMock,
        api_key_file: Path,
    ) -> None:
        from reeln_openai_plugin.livestream import LivestreamMetadata
        from reeln_openai_plugin.playlist import PlaylistMetadata
        from reeln_openai_plugin.translate import TranslatedMetadata

        mock_livestream.return_value = LivestreamMetadata(title="T", description="D")
        mock_playlist.return_value = PlaylistMetadata(title="PT", description="PD")
        # translate_metadata called twice: once for livestream, once for playlist
        mock_translate.side_effect = [
            {"fi": TranslatedMetadata(language_code="fi", title="Otsikko", description="Kuvaus")},
            {"fi": TranslatedMetadata(language_code="fi", title="Soittolista", description="Kuvaus2")},
        ]

        config: dict[str, Any] = {
            "enabled": True,
            "api_key_file": str(api_key_file),
            "playlist_enabled": True,
            "translate_enabled": True,
            "translate_languages": '{"fi": "Finnish"}',
        }
        plugin = OpenAIPlugin(config)
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": FakeGameInfo()})

        plugin.on_game_init(context)

        playlist_meta = context.shared["playlist_metadata"]
        assert playlist_meta["translations"]["fi"]["title"] == "Soittolista"
        assert mock_translate.call_count == 2

    @patch("reeln_openai_plugin.plugin.generate_playlist_metadata")
    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_playlist_failure_non_fatal(
        self,
        mock_livestream: MagicMock,
        mock_playlist: MagicMock,
        caplog: pytest.LogCaptureFixture,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.client import OpenAIError
        from reeln_openai_plugin.livestream import LivestreamMetadata

        mock_livestream.return_value = LivestreamMetadata(title="T", description="D")
        mock_playlist.side_effect = OpenAIError("Playlist API failed")
        plugin = OpenAIPlugin({**plugin_config, "playlist_enabled": True})
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": FakeGameInfo()})

        with caplog.at_level(logging.WARNING):
            plugin.on_game_init(context)

        # Livestream metadata still set
        assert context.shared["livestream_metadata"]["title"] == "T"
        # Playlist not set
        assert "playlist_metadata" not in context.shared
        assert "playlist generation failed" in caplog.text

    @patch("reeln_openai_plugin.plugin.translate_metadata")
    @patch("reeln_openai_plugin.plugin.generate_playlist_metadata")
    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_playlist_translation_failure_non_fatal(
        self,
        mock_livestream: MagicMock,
        mock_playlist: MagicMock,
        mock_translate: MagicMock,
        caplog: pytest.LogCaptureFixture,
        api_key_file: Path,
    ) -> None:
        from reeln_openai_plugin.client import OpenAIError
        from reeln_openai_plugin.livestream import LivestreamMetadata
        from reeln_openai_plugin.playlist import PlaylistMetadata

        mock_livestream.return_value = LivestreamMetadata(title="T", description="D")
        mock_playlist.return_value = PlaylistMetadata(title="PT", description="PD")
        # First translate (livestream) succeeds, second (playlist) fails
        mock_translate.side_effect = [
            {"fi": MagicMock(title="X", description="Y")},
            OpenAIError("Translate failed"),
        ]

        config: dict[str, Any] = {
            "enabled": True,
            "api_key_file": str(api_key_file),
            "playlist_enabled": True,
            "translate_enabled": True,
            "translate_languages": '{"fi": "Finnish"}',
        }
        plugin = OpenAIPlugin(config)
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": FakeGameInfo()})

        with caplog.at_level(logging.WARNING):
            plugin.on_game_init(context)

        # Playlist metadata still set, translations empty
        assert context.shared["playlist_metadata"]["title"] == "PT"
        assert context.shared["playlist_metadata"]["translations"] == {}
        assert "playlist translation failed" in caplog.text

    @patch("reeln_openai_plugin.plugin.generate_playlist_metadata")
    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_full_lifecycle_with_playlist(
        self,
        mock_livestream: MagicMock,
        mock_playlist: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        """Simulate full lifecycle: init -> register -> emit with playlist."""
        from reeln_openai_plugin.livestream import LivestreamMetadata
        from reeln_openai_plugin.playlist import PlaylistMetadata

        mock_livestream.return_value = LivestreamMetadata(title="Live!", description="Go!")
        mock_playlist.return_value = PlaylistMetadata(title="Highlights!", description="Best!")
        plugin = OpenAIPlugin({**plugin_config, "playlist_enabled": True})
        registry = HookRegistry()
        plugin.register(registry)

        game_info = FakeGameInfo(home_team="Storm", away_team="Thunder")
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": game_info})
        registry.emit(Hook.ON_GAME_INIT, context)

        assert context.shared["livestream_metadata"]["title"] == "Live!"
        assert context.shared["playlist_metadata"]["title"] == "Highlights!"


# ------------------------------------------------------------------
# on_queue — metadata enrichment for queued renders
# ------------------------------------------------------------------


class TestOnQueue:
    def test_disabled_skips(self, plugin_config: dict[str, Any]) -> None:
        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": False})
        queue_item = FakeQueueItem()
        context = HookContext(
            hook=Hook.ON_QUEUE,
            data={"queue_item": queue_item, "game_info": FakeGameInfo()},
        )

        plugin.on_queue(context)

        assert "render_metadata" not in context.shared

    def test_no_queue_item_skips(self, plugin_config: dict[str, Any]) -> None:
        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        context = HookContext(hook=Hook.ON_QUEUE, data={})

        plugin.on_queue(context)

        assert "render_metadata" not in context.shared

    def test_no_game_info_skips(self, plugin_config: dict[str, Any]) -> None:
        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        queue_item = FakeQueueItem()
        context = HookContext(hook=Hook.ON_QUEUE, data={"queue_item": queue_item})

        plugin.on_queue(context)

        assert "render_metadata" not in context.shared

    @patch("reeln.core.queue.update_queue_item")
    @patch("reeln_openai_plugin.plugin.generate_render_metadata")
    @patch("reeln_openai_plugin.plugin.OpenAIPlugin._get_client")
    def test_generates_and_persists_metadata(
        self,
        mock_client: MagicMock,
        mock_gen: MagicMock,
        mock_update: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.render_metadata import RenderMetadata

        mock_gen.return_value = RenderMetadata(title="AI Title", description="AI Desc")
        mock_update.return_value = MagicMock()

        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        plugin._game_info = FakeGameInfo()
        queue_item = FakeQueueItem(player="#48 Ben", assists="#3 Charlie")
        context = HookContext(
            hook=Hook.ON_QUEUE,
            data={"queue_item": queue_item, "game_info": FakeGameInfo()},
        )

        plugin.on_queue(context)

        assert context.shared["render_metadata"]["title"] == "AI Title"
        assert context.shared["render_metadata"]["description"] == "AI Desc"

        mock_update.assert_called_once()
        call_args = mock_update.call_args
        assert str(call_args[0][0]) == "/games/test"
        assert call_args[0][1] == "abc123def456"
        assert call_args[1]["title"] == "AI Title"
        assert call_args[1]["description"] == "AI Desc"

    @patch("reeln.core.queue.update_queue_item")
    @patch("reeln_openai_plugin.plugin.generate_render_metadata")
    @patch("reeln_openai_plugin.plugin.OpenAIPlugin._get_client")
    def test_passes_player_context(
        self,
        mock_client: MagicMock,
        mock_gen: MagicMock,
        mock_update: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.render_metadata import RenderMetadata

        mock_gen.return_value = RenderMetadata(title="T", description="D")
        mock_update.return_value = MagicMock()

        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        plugin._game_info = FakeGameInfo()
        queue_item = FakeQueueItem(
            player="#48 Benjamin Remitz",
            assists="#3 Charles Pillsbury",
            event_type="goal",
            level="11u",
        )
        context = HookContext(
            hook=Hook.ON_QUEUE,
            data={"queue_item": queue_item, "game_info": FakeGameInfo()},
        )

        plugin.on_queue(context)

        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs["player"] == "#48 Benjamin Remitz"
        assert call_kwargs["assists"] == "#3 Charles Pillsbury"
        assert call_kwargs["event_type"] == "goal"
        assert call_kwargs["level"] == "11u"
        assert call_kwargs["clip_name"] == "clip_short"

    @patch("reeln_openai_plugin.plugin.OpenAIPlugin._get_client")
    def test_client_failure_skips(
        self,
        mock_client: MagicMock,
        plugin_config: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_client.side_effect = OpenAIError("bad key")

        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        plugin._game_info = FakeGameInfo()
        queue_item = FakeQueueItem()
        context = HookContext(
            hook=Hook.ON_QUEUE,
            data={"queue_item": queue_item, "game_info": FakeGameInfo()},
        )

        with caplog.at_level(logging.WARNING):
            plugin.on_queue(context)

        assert "client setup failed" in caplog.text
        assert "render_metadata" not in context.shared

    @patch("reeln_openai_plugin.plugin.generate_render_metadata")
    @patch("reeln_openai_plugin.plugin.OpenAIPlugin._get_client")
    def test_generation_failure_skips(
        self,
        mock_client: MagicMock,
        mock_gen: MagicMock,
        plugin_config: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_gen.side_effect = OpenAIError("API error")

        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        plugin._game_info = FakeGameInfo()
        queue_item = FakeQueueItem()
        context = HookContext(
            hook=Hook.ON_QUEUE,
            data={"queue_item": queue_item, "game_info": FakeGameInfo()},
        )

        with caplog.at_level(logging.WARNING):
            plugin.on_queue(context)

        assert "queue metadata generation failed" in caplog.text
        assert "render_metadata" not in context.shared

    @patch("reeln.core.queue.update_queue_item", side_effect=Exception("disk full"))
    @patch("reeln_openai_plugin.plugin.generate_render_metadata")
    @patch("reeln_openai_plugin.plugin.OpenAIPlugin._get_client")
    def test_update_failure_skips(
        self,
        mock_client: MagicMock,
        mock_gen: MagicMock,
        mock_update: MagicMock,
        plugin_config: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from reeln_openai_plugin.render_metadata import RenderMetadata

        mock_gen.return_value = RenderMetadata(title="T", description="D")

        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        plugin._game_info = FakeGameInfo()
        queue_item = FakeQueueItem()
        context = HookContext(
            hook=Hook.ON_QUEUE,
            data={"queue_item": queue_item, "game_info": FakeGameInfo()},
        )

        with caplog.at_level(logging.WARNING):
            plugin.on_queue(context)

        assert "failed to update queue item" in caplog.text
        assert "render_metadata" not in context.shared

    @patch("reeln.core.queue.update_queue_item")
    @patch("reeln_openai_plugin.plugin.generate_render_metadata")
    @patch("reeln_openai_plugin.plugin.OpenAIPlugin._get_client")
    def test_frame_descriptions_passed_and_cleared(
        self,
        mock_client: MagicMock,
        mock_gen: MagicMock,
        mock_update: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.frames import FrameDescriptions
        from reeln_openai_plugin.render_metadata import RenderMetadata

        mock_gen.return_value = RenderMetadata(title="T", description="D")
        mock_update.return_value = MagicMock()

        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        plugin._game_info = FakeGameInfo()
        plugin._frame_descriptions = FrameDescriptions(
            descriptions=("wrist shot", "goal"), summary="Quick wrist shot goal"
        )
        queue_item = FakeQueueItem()
        context = HookContext(
            hook=Hook.ON_QUEUE,
            data={"queue_item": queue_item, "game_info": FakeGameInfo()},
        )

        plugin.on_queue(context)

        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs["frame_summary"] == "Quick wrist shot goal"
        assert plugin._frame_descriptions is None


# ------------------------------------------------------------------
# on_post_render — short metadata
# ------------------------------------------------------------------


class TestOnPostRender:
    def test_disabled_skips(self, plugin_config: dict[str, Any]) -> None:
        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": False})
        plugin._game_info = FakeGameInfo()
        plan = MagicMock()
        plan.filter_complex = "filter"
        context = HookContext(hook=Hook.POST_RENDER, data={"plan": plan, "result": MagicMock()})

        plugin.on_post_render(context)

        assert "render_metadata" not in context.shared

    def test_no_game_info_skips(self, plugin_config: dict[str, Any]) -> None:
        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        # _game_info not set (no game init happened)
        plan = MagicMock()
        plan.filter_complex = "filter"
        context = HookContext(hook=Hook.POST_RENDER, data={"plan": plan, "result": MagicMock()})

        plugin.on_post_render(context)

        assert "render_metadata" not in context.shared

    def test_no_plan_skips(self, plugin_config: dict[str, Any]) -> None:
        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        plugin._game_info = FakeGameInfo()
        context = HookContext(hook=Hook.POST_RENDER, data={"result": MagicMock()})

        plugin.on_post_render(context)

        assert "render_metadata" not in context.shared

    @patch.dict("os.environ", {}, clear=True)
    def test_client_failure_warns(
        self,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        config: dict[str, Any] = {
            "render_metadata_enabled": True,
            "api_key_file": str(tmp_path / "missing.txt"),
        }
        plugin = OpenAIPlugin(config)
        plugin._game_info = FakeGameInfo()
        plan = MagicMock()
        plan.filter_complex = "filter"
        context = HookContext(hook=Hook.POST_RENDER, data={"plan": plan, "result": MagicMock()})

        with caplog.at_level(logging.WARNING):
            plugin.on_post_render(context)

        assert "client setup failed" in caplog.text
        assert "render_metadata" not in context.shared

    @patch("reeln_openai_plugin.plugin.generate_render_metadata")
    def test_success(
        self,
        mock_gen: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.render_metadata import RenderMetadata

        mock_gen.return_value = RenderMetadata(title="Amazing Goal!", description="Great play!")
        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        plugin._game_info = FakeGameInfo()

        plan = MagicMock()
        plan.filter_complex = "filter"
        plan.output = MagicMock()
        plan.output.stem = "goal_001"
        context = HookContext(hook=Hook.POST_RENDER, data={"plan": plan, "result": MagicMock()})

        plugin.on_post_render(context)

        assert context.shared["render_metadata"]["title"] == "Amazing Goal!"
        assert context.shared["render_metadata"]["description"] == "Great play!"
        mock_gen.assert_called_once()

    @patch("reeln_openai_plugin.plugin.generate_render_metadata")
    def test_passes_clip_name(
        self,
        mock_gen: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.render_metadata import RenderMetadata

        mock_gen.return_value = RenderMetadata(title="T", description="D")
        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        plugin._game_info = FakeGameInfo()

        plan = MagicMock()
        plan.filter_complex = "filter"
        plan.output = MagicMock()
        plan.output.stem = "clip_highlight"
        context = HookContext(hook=Hook.POST_RENDER, data={"plan": plan, "result": MagicMock()})

        plugin.on_post_render(context)

        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs["clip_name"] == "clip_highlight"

    @patch("reeln_openai_plugin.plugin.generate_render_metadata")
    def test_generation_failure_non_fatal(
        self,
        mock_gen: MagicMock,
        caplog: pytest.LogCaptureFixture,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.client import OpenAIError

        mock_gen.side_effect = OpenAIError("API down")
        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        plugin._game_info = FakeGameInfo()

        plan = MagicMock()
        plan.filter_complex = "filter"
        context = HookContext(hook=Hook.POST_RENDER, data={"plan": plan, "result": MagicMock()})

        with caplog.at_level(logging.WARNING):
            plugin.on_post_render(context)

        assert "render metadata generation failed" in caplog.text
        assert "render_metadata" not in context.shared

    @patch("reeln_openai_plugin.plugin.generate_render_metadata")
    def test_preserves_existing_shared(
        self,
        mock_gen: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        """Render metadata is written alongside existing shared data."""
        from reeln_openai_plugin.render_metadata import RenderMetadata

        mock_gen.return_value = RenderMetadata(title="T", description="D")
        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        plugin._game_info = FakeGameInfo()

        plan = MagicMock()
        plan.output = MagicMock()
        plan.output.stem = "clip"
        context = HookContext(
            hook=Hook.POST_RENDER,
            data={"plan": plan, "result": MagicMock()},
            shared={"existing_key": "data"},
        )

        plugin.on_post_render(context)

        assert context.shared["existing_key"] == "data"
        assert context.shared["render_metadata"]["title"] == "T"
        assert context.shared["render_metadata"]["description"] == "D"

    @patch("reeln_openai_plugin.plugin.generate_render_metadata")
    def test_uses_game_info_from_hook_data(
        self,
        mock_gen: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        """When _game_info is None, fall back to context.data['game_info']."""
        from reeln_openai_plugin.render_metadata import RenderMetadata

        mock_gen.return_value = RenderMetadata(title="From Hook", description="D")
        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        assert plugin._game_info is None

        game_info = FakeGameInfo()
        plan = MagicMock()
        plan.filter_complex = "filter"
        plan.output = MagicMock()
        plan.output.stem = "clip"
        context = HookContext(
            hook=Hook.POST_RENDER,
            data={"plan": plan, "result": MagicMock(), "game_info": game_info},
        )

        plugin.on_post_render(context)

        assert context.shared["render_metadata"]["title"] == "From Hook"
        # Verify the game_info was passed to generate_render_metadata
        call_args = mock_gen.call_args
        assert call_args[0][2] is game_info

    @patch("reeln_openai_plugin.plugin.generate_render_metadata")
    def test_passes_player_and_event_context(
        self,
        mock_gen: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        """Player, assists, event_type, and level are forwarded from hook data."""
        from reeln_openai_plugin.render_metadata import RenderMetadata
        from tests.conftest import FakeGameEvent

        mock_gen.return_value = RenderMetadata(title="T", description="D")
        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})

        game_info = FakeGameInfo(level="2016")
        game_event = FakeGameEvent(event_type="goal")
        plan = MagicMock()
        plan.filter_complex = "filter"
        plan.output = MagicMock()
        plan.output.stem = "clip"
        context = HookContext(
            hook=Hook.POST_RENDER,
            data={
                "plan": plan,
                "result": MagicMock(),
                "game_info": game_info,
                "game_event": game_event,
                "player": "#48 Benjamin Remitz",
                "assists": "#7 John Smith",
            },
        )

        plugin.on_post_render(context)

        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs["player"] == "#48 Benjamin Remitz"
        assert call_kwargs["assists"] == "#7 John Smith"
        assert call_kwargs["event_type"] == "goal"
        assert call_kwargs["level"] == "2016"


# ------------------------------------------------------------------
# _get_client
# ------------------------------------------------------------------


class TestGetClient:
    def test_creates_client(self, plugin_config: dict[str, Any]) -> None:
        plugin = OpenAIPlugin(plugin_config)
        client = plugin._get_client()
        assert client._api_key == "sk-test-key-12345"
        assert client._model == "gpt-4.1"

    def test_caches_client(self, plugin_config: dict[str, Any]) -> None:
        plugin = OpenAIPlugin(plugin_config)
        c1 = plugin._get_client()
        c2 = plugin._get_client()
        assert c1 is c2

    def test_no_key_source_raises(self) -> None:
        from reeln_openai_plugin.client import OpenAIError

        plugin = OpenAIPlugin({"enabled": True})
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(
                OpenAIError,
                match="No API key",
            ),
        ):
            plugin._get_client()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-env"})
    def test_env_var_fallback(self) -> None:
        plugin = OpenAIPlugin({"enabled": True})
        client = plugin._get_client()
        assert client._api_key == "sk-from-env"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-env"})
    def test_file_takes_priority_over_env(self, api_key_file: Path) -> None:
        plugin = OpenAIPlugin({"api_key_file": str(api_key_file)})
        client = plugin._get_client()
        assert client._api_key == "sk-test-key-12345"

    def test_missing_file_falls_back_to_env(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        config: dict[str, Any] = {"api_key_file": str(tmp_path / "missing.txt")}
        plugin = OpenAIPlugin(config)
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fallback"}),
            caplog.at_level(
                logging.WARNING,
            ),
        ):
            client = plugin._get_client()
        assert client._api_key == "sk-fallback"
        assert "not found" in caplog.text

    def test_missing_file_no_env_raises(self, tmp_path: Path) -> None:
        from reeln_openai_plugin.client import OpenAIError

        config: dict[str, Any] = {"api_key_file": str(tmp_path / "missing.txt")}
        plugin = OpenAIPlugin(config)
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(
                OpenAIError,
                match="No API key",
            ),
        ):
            plugin._get_client()

    def test_custom_model_and_timeout(self, api_key_file: Path) -> None:
        config: dict[str, Any] = {
            "api_key_file": str(api_key_file),
            "model": "gpt-5",
            "request_timeout_seconds": 60.0,
        }
        plugin = OpenAIPlugin(config)
        client = plugin._get_client()
        assert client._model == "gpt-5"
        assert client._timeout == 60.0


# ------------------------------------------------------------------
# on_game_init — game image
# ------------------------------------------------------------------


class TestOnGameInitGameImage:
    @patch("reeln_openai_plugin.plugin.generate_game_image")
    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_user_provided_thumbnail_skips_image_generation(
        self,
        mock_livestream: MagicMock,
        mock_game_image: MagicMock,
        plugin_config: dict[str, Any],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When game_info.thumbnail is set, skip LLM image generation."""
        from reeln_openai_plugin.livestream import LivestreamMetadata

        mock_livestream.return_value = LivestreamMetadata(title="T", description="D")

        home_info = FakeTeamInfo(team_name="A", logo_path=tmp_path / "h.png")
        away_info = FakeTeamInfo(team_name="B", logo_path=tmp_path / "a.png")

        config = {
            **plugin_config,
            "game_image_enabled": True,
            "game_image_output_dir": str(tmp_path),
        }
        plugin = OpenAIPlugin(config)
        game_info = FakeGameInfo(thumbnail="/path/to/user_thumb.png")
        context = HookContext(
            hook=Hook.ON_GAME_INIT,
            data={"game_info": game_info, "home_profile": home_info, "away_profile": away_info},
        )

        with caplog.at_level(logging.INFO):
            plugin.on_game_init(context)

        mock_game_image.assert_not_called()
        assert "game_image" not in context.shared
        assert "user-provided thumbnail" in caplog.text

    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_game_image_disabled_skips(
        self,
        mock_gen: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.livestream import LivestreamMetadata

        mock_gen.return_value = LivestreamMetadata(title="T", description="D")
        plugin = OpenAIPlugin({**plugin_config, "game_image_enabled": False})
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": FakeGameInfo()})

        plugin.on_game_init(context)

        assert "game_image" not in context.shared

    @patch("reeln_openai_plugin.plugin.generate_game_image")
    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_game_image_success(
        self,
        mock_livestream: MagicMock,
        mock_game_image: MagicMock,
        plugin_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        from reeln_openai_plugin.game_image import GameImageResult
        from reeln_openai_plugin.livestream import LivestreamMetadata

        mock_livestream.return_value = LivestreamMetadata(title="T", description="D")
        img_path = tmp_path / "game.png"
        mock_game_image.return_value = GameImageResult(image_path=img_path)

        home_info = FakeTeamInfo(team_name="Storm", logo_path=tmp_path / "h.png")
        away_info = FakeTeamInfo(team_name="Thunder", logo_path=tmp_path / "a.png")

        config = {
            **plugin_config,
            "game_image_enabled": True,
            "game_image_output_dir": str(tmp_path),
        }
        plugin = OpenAIPlugin(config)
        context = HookContext(
            hook=Hook.ON_GAME_INIT,
            data={"game_info": FakeGameInfo(), "home_profile": home_info, "away_profile": away_info},
        )

        plugin.on_game_init(context)

        assert context.shared["game_image"]["image_path"] == str(img_path)
        mock_game_image.assert_called_once()

    @patch("reeln_openai_plugin.plugin.generate_game_image")
    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_game_image_uses_model_overrides(
        self,
        mock_livestream: MagicMock,
        mock_game_image: MagicMock,
        plugin_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        from reeln_openai_plugin.game_image import GameImageResult
        from reeln_openai_plugin.livestream import LivestreamMetadata

        mock_livestream.return_value = LivestreamMetadata(title="T", description="D")
        mock_game_image.return_value = GameImageResult(image_path=tmp_path / "g.png")

        home_info = FakeTeamInfo(team_name="A", logo_path=tmp_path / "h.png")
        away_info = FakeTeamInfo(team_name="B", logo_path=tmp_path / "a.png")

        config = {
            **plugin_config,
            "game_image_enabled": True,
            "game_image_output_dir": str(tmp_path),
            "game_image_model": "custom-model",
            "game_image_renderer_model": "custom-renderer",
        }
        plugin = OpenAIPlugin(config)
        context = HookContext(
            hook=Hook.ON_GAME_INIT,
            data={"game_info": FakeGameInfo(), "home_profile": home_info, "away_profile": away_info},
        )

        plugin.on_game_init(context)

        call_kwargs = mock_game_image.call_args[1]
        assert call_kwargs["model"] == "custom-model"
        assert call_kwargs["renderer_model"] == "custom-renderer"

    @patch("reeln_openai_plugin.plugin.generate_game_image")
    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_game_image_failure_non_fatal(
        self,
        mock_livestream: MagicMock,
        mock_game_image: MagicMock,
        caplog: pytest.LogCaptureFixture,
        plugin_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        from reeln_openai_plugin.client import OpenAIError
        from reeln_openai_plugin.livestream import LivestreamMetadata

        mock_livestream.return_value = LivestreamMetadata(title="T", description="D")
        mock_game_image.side_effect = OpenAIError("Image gen failed")

        home_info = FakeTeamInfo(team_name="A", logo_path=tmp_path / "h.png")
        away_info = FakeTeamInfo(team_name="B", logo_path=tmp_path / "a.png")

        config = {
            **plugin_config,
            "game_image_enabled": True,
            "game_image_output_dir": str(tmp_path),
        }
        plugin = OpenAIPlugin(config)
        context = HookContext(
            hook=Hook.ON_GAME_INIT,
            data={"game_info": FakeGameInfo(), "home_profile": home_info, "away_profile": away_info},
        )

        with caplog.at_level(logging.WARNING):
            plugin.on_game_init(context)

        # Livestream metadata still set, game image not
        assert context.shared["livestream_metadata"]["title"] == "T"
        assert "game_image" not in context.shared
        assert "game image generation failed" in caplog.text

    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_missing_team_info_skips(
        self,
        mock_gen: MagicMock,
        caplog: pytest.LogCaptureFixture,
        plugin_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        from reeln_openai_plugin.livestream import LivestreamMetadata

        mock_gen.return_value = LivestreamMetadata(title="T", description="D")
        config = {
            **plugin_config,
            "game_image_enabled": True,
            "game_image_output_dir": str(tmp_path),
        }
        plugin = OpenAIPlugin(config)
        # No home_profile/away_profile in context.data
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": FakeGameInfo()})

        with caplog.at_level(logging.WARNING):
            plugin.on_game_init(context)

        assert "game_image" not in context.shared
        assert "missing team profiles" in caplog.text

    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_missing_logos_skips(
        self,
        mock_gen: MagicMock,
        caplog: pytest.LogCaptureFixture,
        plugin_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        from reeln_openai_plugin.livestream import LivestreamMetadata

        mock_gen.return_value = LivestreamMetadata(title="T", description="D")

        # Team profiles without logo_path
        home_info = FakeTeamInfo(team_name="A", logo_path=None)
        away_info = FakeTeamInfo(team_name="B", logo_path=None)

        config = {
            **plugin_config,
            "game_image_enabled": True,
            "game_image_output_dir": str(tmp_path),
        }
        plugin = OpenAIPlugin(config)
        context = HookContext(
            hook=Hook.ON_GAME_INIT,
            data={"game_info": FakeGameInfo(), "home_profile": home_info, "away_profile": away_info},
        )

        with caplog.at_level(logging.WARNING):
            plugin.on_game_init(context)

        assert "game_image" not in context.shared
        assert "missing team logos" in caplog.text

    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_no_output_dir_skips(
        self,
        mock_gen: MagicMock,
        caplog: pytest.LogCaptureFixture,
        plugin_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        from reeln_openai_plugin.livestream import LivestreamMetadata

        mock_gen.return_value = LivestreamMetadata(title="T", description="D")

        home_info = FakeTeamInfo(team_name="A", logo_path=tmp_path / "h.png")
        away_info = FakeTeamInfo(team_name="B", logo_path=tmp_path / "a.png")

        config = {
            **plugin_config,
            "game_image_enabled": True,
            # game_image_output_dir not set (empty string default)
        }
        plugin = OpenAIPlugin(config)
        context = HookContext(
            hook=Hook.ON_GAME_INIT,
            data={"game_info": FakeGameInfo(), "home_profile": home_info, "away_profile": away_info},
        )

        with caplog.at_level(logging.WARNING):
            plugin.on_game_init(context)

        assert "game_image" not in context.shared
        assert "game_image_output_dir not configured" in caplog.text


# ------------------------------------------------------------------
# on_frames_extracted — smart zoom
# ------------------------------------------------------------------


class TestOnFramesExtracted:
    def _make_frames(self, tmp_path: Path, count: int = 3) -> FakeExtractedFrames:
        paths = []
        timestamps = []
        for i in range(count):
            p = tmp_path / f"frame_{i:04d}.png"
            p.write_bytes(b"\x89PNG")
            paths.append(p)
            timestamps.append(float(i * 2))
        return FakeExtractedFrames(
            frame_paths=tuple(paths),
            timestamps=tuple(timestamps),
        )

    def test_disabled_skips(self, plugin_config: dict[str, Any]) -> None:
        plugin = OpenAIPlugin({**plugin_config, "smart_zoom_enabled": False})
        frames = FakeExtractedFrames()
        context = HookContext(
            hook=Hook.ON_FRAMES_EXTRACTED,
            data={"frames": frames},
        )
        plugin.on_frames_extracted(context)
        assert "smart_zoom" not in context.shared

    def test_no_frames_data_warns(
        self,
        plugin_config: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        plugin = OpenAIPlugin({**plugin_config, "smart_zoom_enabled": True})
        context = HookContext(hook=Hook.ON_FRAMES_EXTRACTED, data={})
        with caplog.at_level(logging.WARNING):
            plugin.on_frames_extracted(context)
        assert "no frames in context" in caplog.text
        assert "smart_zoom" not in context.shared

    def test_empty_frame_list_warns(
        self,
        plugin_config: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        plugin = OpenAIPlugin({**plugin_config, "smart_zoom_enabled": True})
        frames = FakeExtractedFrames(frame_paths=(), timestamps=())
        context = HookContext(
            hook=Hook.ON_FRAMES_EXTRACTED,
            data={"frames": frames},
        )
        with caplog.at_level(logging.WARNING):
            plugin.on_frames_extracted(context)
        assert "empty frame list" in caplog.text
        assert "smart_zoom" not in context.shared

    @patch.dict("os.environ", {}, clear=True)
    def test_client_failure_warns(
        self,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        config: dict[str, Any] = {
            "smart_zoom_enabled": True,
            "api_key_file": str(tmp_path / "missing.txt"),
        }
        plugin = OpenAIPlugin(config)
        frames = self._make_frames(tmp_path, count=1)
        context = HookContext(
            hook=Hook.ON_FRAMES_EXTRACTED,
            data={"frames": frames},
        )
        with caplog.at_level(logging.WARNING):
            plugin.on_frames_extracted(context)
        assert "client setup failed" in caplog.text
        assert "smart_zoom" not in context.shared

    @patch("reeln_openai_plugin.plugin.analyze_frame_for_zoom")
    def test_single_frame_success(
        self,
        mock_analyze: MagicMock,
        plugin_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        mock_analyze.return_value = (0.3, 0.7)
        plugin = OpenAIPlugin({**plugin_config, "smart_zoom_enabled": True})
        frames = self._make_frames(tmp_path, count=1)
        context = HookContext(
            hook=Hook.ON_FRAMES_EXTRACTED,
            data={"frames": frames},
        )

        plugin.on_frames_extracted(context)

        zoom_path = context.shared["smart_zoom"]["zoom_path"]
        assert len(zoom_path.points) == 1
        assert zoom_path.points[0].center_x == 0.3
        assert zoom_path.points[0].center_y == 0.7
        assert zoom_path.points[0].timestamp == 0.0
        assert zoom_path.source_width == 1920
        assert zoom_path.source_height == 1080

    @patch("reeln_openai_plugin.plugin.analyze_frame_for_zoom")
    def test_multi_frame_success(
        self,
        mock_analyze: MagicMock,
        plugin_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        mock_analyze.side_effect = [(0.3, 0.4), (0.5, 0.6), (0.7, 0.8)]
        plugin = OpenAIPlugin({**plugin_config, "smart_zoom_enabled": True})
        frames = self._make_frames(tmp_path, count=3)
        context = HookContext(
            hook=Hook.ON_FRAMES_EXTRACTED,
            data={"frames": frames},
        )

        plugin.on_frames_extracted(context)

        zoom_path = context.shared["smart_zoom"]["zoom_path"]
        assert len(zoom_path.points) == 3
        assert zoom_path.points[0].center_x == 0.3
        assert zoom_path.points[1].center_x == 0.5
        assert zoom_path.points[2].center_x == 0.7
        assert zoom_path.duration == 10.0

    @patch("reeln_openai_plugin.plugin.analyze_frame_for_zoom")
    def test_frame_error_signals_error_in_shared(
        self,
        mock_analyze: MagicMock,
        plugin_config: dict[str, Any],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Frame analysis failure after retries sets error in shared context."""
        from reeln_openai_plugin.client import OpenAIError

        mock_analyze.side_effect = OpenAIError("HTTP 500 after retries")
        plugin = OpenAIPlugin({**plugin_config, "smart_zoom_enabled": True})
        frames = self._make_frames(tmp_path, count=2)
        context = HookContext(
            hook=Hook.ON_FRAMES_EXTRACTED,
            data={"frames": frames},
        )

        with caplog.at_level(logging.ERROR):
            plugin.on_frames_extracted(context)

        assert "smart_zoom" in context.shared
        assert "error" in context.shared["smart_zoom"]
        assert "HTTP 500" in context.shared["smart_zoom"]["error"]
        assert "failed after retries" in caplog.text

    @patch("reeln_openai_plugin.plugin.analyze_frame_for_zoom")
    def test_first_frame_error_aborts_all(
        self,
        mock_analyze: MagicMock,
        plugin_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Error on first frame stops analysis — no partial zoom path."""
        from reeln_openai_plugin.client import OpenAIError

        mock_analyze.side_effect = OpenAIError("HTTP 502 Bad Gateway")
        plugin = OpenAIPlugin({**plugin_config, "smart_zoom_enabled": True})
        frames = self._make_frames(tmp_path, count=3)
        context = HookContext(
            hook=Hook.ON_FRAMES_EXTRACTED,
            data={"frames": frames},
        )

        plugin.on_frames_extracted(context)

        # Only called once — first frame fails, analysis stops
        assert mock_analyze.call_count == 1
        assert "error" in context.shared["smart_zoom"]

    @patch("reeln_openai_plugin.plugin.analyze_frame_for_zoom")
    def test_model_override(
        self,
        mock_analyze: MagicMock,
        plugin_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        mock_analyze.return_value = (0.5, 0.5)
        config = {**plugin_config, "smart_zoom_enabled": True, "smart_zoom_model": "gpt-5"}
        plugin = OpenAIPlugin(config)
        frames = self._make_frames(tmp_path, count=1)
        context = HookContext(
            hook=Hook.ON_FRAMES_EXTRACTED,
            data={"frames": frames},
        )

        plugin.on_frames_extracted(context)

        call_kwargs = mock_analyze.call_args[1]
        assert call_kwargs["model"] == "gpt-5"


# ------------------------------------------------------------------
# on_frames_extracted — frame description
# ------------------------------------------------------------------


class TestOnFramesExtractedFrameDescription:
    def _make_frames(self, tmp_path: Path, count: int = 3) -> FakeExtractedFrames:
        paths = []
        timestamps = []
        for i in range(count):
            p = tmp_path / f"frame_{i:04d}.png"
            p.write_bytes(b"\x89PNG")
            paths.append(p)
            timestamps.append(float(i * 2))
        return FakeExtractedFrames(
            frame_paths=tuple(paths),
            timestamps=tuple(timestamps),
        )

    def test_disabled_skips(self, plugin_config: dict[str, Any]) -> None:
        plugin = OpenAIPlugin({**plugin_config, "frame_description_enabled": False})
        frames = FakeExtractedFrames()
        context = HookContext(
            hook=Hook.ON_FRAMES_EXTRACTED,
            data={"frames": frames},
        )
        plugin.on_frames_extracted(context)
        assert "frame_descriptions" not in context.shared

    @patch("reeln_openai_plugin.plugin.describe_frames")
    def test_success_writes_shared_and_caches(
        self,
        mock_describe: MagicMock,
        plugin_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        from reeln_openai_plugin.frames import FrameDescriptions

        mock_describe.return_value = FrameDescriptions(
            descriptions=("Player shoots", "Goal scored"),
            summary="Quick wrist shot goal",
        )
        config = {**plugin_config, "frame_description_enabled": True}
        plugin = OpenAIPlugin(config)
        frames = self._make_frames(tmp_path, count=2)
        context = HookContext(
            hook=Hook.ON_FRAMES_EXTRACTED,
            data={"frames": frames},
        )

        plugin.on_frames_extracted(context)

        # Written to shared context
        fd = context.shared["frame_descriptions"]
        assert fd["descriptions"] == ["Player shoots", "Goal scored"]
        assert fd["summary"] == "Quick wrist shot goal"
        # Cached on instance
        assert plugin._frame_descriptions is not None
        assert plugin._frame_descriptions.summary == "Quick wrist shot goal"

    @patch("reeln_openai_plugin.plugin.describe_frames")
    def test_failure_non_fatal(
        self,
        mock_describe: MagicMock,
        caplog: pytest.LogCaptureFixture,
        plugin_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        from reeln_openai_plugin.client import OpenAIError

        mock_describe.side_effect = OpenAIError("Vision API down")
        config = {**plugin_config, "frame_description_enabled": True}
        plugin = OpenAIPlugin(config)
        frames = self._make_frames(tmp_path, count=1)
        context = HookContext(
            hook=Hook.ON_FRAMES_EXTRACTED,
            data={"frames": frames},
        )

        with caplog.at_level(logging.WARNING):
            plugin.on_frames_extracted(context)

        assert "frame description failed" in caplog.text
        assert "frame_descriptions" not in context.shared
        assert plugin._frame_descriptions is None

    @patch("reeln_openai_plugin.plugin.analyze_frame_for_zoom")
    @patch("reeln_openai_plugin.plugin.describe_frames")
    def test_failure_non_fatal_zoom_still_works(
        self,
        mock_describe: MagicMock,
        mock_analyze: MagicMock,
        caplog: pytest.LogCaptureFixture,
        plugin_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        from reeln_openai_plugin.client import OpenAIError

        mock_analyze.return_value = (0.3, 0.7)
        mock_describe.side_effect = OpenAIError("Vision API down")
        config = {
            **plugin_config,
            "smart_zoom_enabled": True,
            "frame_description_enabled": True,
        }
        plugin = OpenAIPlugin(config)
        frames = self._make_frames(tmp_path, count=1)
        context = HookContext(
            hook=Hook.ON_FRAMES_EXTRACTED,
            data={"frames": frames},
        )

        with caplog.at_level(logging.WARNING):
            plugin.on_frames_extracted(context)

        # Zoom still works
        assert "smart_zoom" in context.shared
        zoom_path = context.shared["smart_zoom"]["zoom_path"]
        assert zoom_path.points[0].center_x == 0.3
        # Frame description failed gracefully
        assert "frame_descriptions" not in context.shared
        assert "frame description failed" in caplog.text

    @patch("reeln_openai_plugin.plugin.describe_frames")
    def test_model_override(
        self,
        mock_describe: MagicMock,
        plugin_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        from reeln_openai_plugin.frames import FrameDescriptions

        mock_describe.return_value = FrameDescriptions(
            descriptions=("d",), summary="s",
        )
        config = {
            **plugin_config,
            "frame_description_enabled": True,
            "frame_description_model": "gpt-5",
        }
        plugin = OpenAIPlugin(config)
        frames = self._make_frames(tmp_path, count=1)
        context = HookContext(
            hook=Hook.ON_FRAMES_EXTRACTED,
            data={"frames": frames},
        )

        plugin.on_frames_extracted(context)

        call_kwargs = mock_describe.call_args[1]
        assert call_kwargs["model"] == "gpt-5"

    @patch("reeln_openai_plugin.plugin.describe_frames")
    def test_only_frame_description_enabled(
        self,
        mock_describe: MagicMock,
        plugin_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Frame description works even when smart zoom is disabled."""
        from reeln_openai_plugin.frames import FrameDescriptions

        mock_describe.return_value = FrameDescriptions(
            descriptions=("action",), summary="play summary",
        )
        config = {
            **plugin_config,
            "smart_zoom_enabled": False,
            "frame_description_enabled": True,
        }
        plugin = OpenAIPlugin(config)
        frames = self._make_frames(tmp_path, count=1)
        context = HookContext(
            hook=Hook.ON_FRAMES_EXTRACTED,
            data={"frames": frames},
        )

        plugin.on_frames_extracted(context)

        assert "frame_descriptions" in context.shared
        assert "smart_zoom" not in context.shared


# ------------------------------------------------------------------
# on_post_render — frame description integration
# ------------------------------------------------------------------


class TestOnPostRenderFrameDescription:
    @patch("reeln_openai_plugin.plugin.generate_render_metadata")
    def test_frame_summary_passed_to_metadata(
        self,
        mock_gen: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.frames import FrameDescriptions
        from reeln_openai_plugin.render_metadata import RenderMetadata

        mock_gen.return_value = RenderMetadata(title="T", description="D")
        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        plugin._game_info = FakeGameInfo()
        plugin._frame_descriptions = FrameDescriptions(
            descriptions=("shot", "goal"),
            summary="Wrist shot goal",
        )

        plan = MagicMock()
        plan.filter_complex = "filter"
        plan.output = MagicMock()
        plan.output.stem = "clip"
        context = HookContext(hook=Hook.POST_RENDER, data={"plan": plan, "result": MagicMock()})

        plugin.on_post_render(context)

        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs["frame_summary"] == "Wrist shot goal"

    @patch("reeln_openai_plugin.plugin.generate_render_metadata")
    def test_frame_descriptions_cleared_after_use(
        self,
        mock_gen: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.frames import FrameDescriptions
        from reeln_openai_plugin.render_metadata import RenderMetadata

        mock_gen.return_value = RenderMetadata(title="T", description="D")
        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        plugin._game_info = FakeGameInfo()
        plugin._frame_descriptions = FrameDescriptions(
            descriptions=("d",), summary="s",
        )

        plan = MagicMock()
        plan.filter_complex = "filter"
        plan.output = MagicMock()
        plan.output.stem = "clip"
        context = HookContext(hook=Hook.POST_RENDER, data={"plan": plan, "result": MagicMock()})

        plugin.on_post_render(context)

        assert plugin._frame_descriptions is None

    @patch("reeln_openai_plugin.plugin.generate_render_metadata")
    def test_no_frame_descriptions_empty_summary(
        self,
        mock_gen: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        from reeln_openai_plugin.render_metadata import RenderMetadata

        mock_gen.return_value = RenderMetadata(title="T", description="D")
        plugin = OpenAIPlugin({**plugin_config, "render_metadata_enabled": True})
        plugin._game_info = FakeGameInfo()
        assert plugin._frame_descriptions is None

        plan = MagicMock()
        plan.filter_complex = "filter"
        plan.output = MagicMock()
        plan.output.stem = "clip"
        context = HookContext(hook=Hook.POST_RENDER, data={"plan": plan, "result": MagicMock()})

        plugin.on_post_render(context)

        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs["frame_summary"] == ""


# ------------------------------------------------------------------
# Integration with registry
# ------------------------------------------------------------------


class TestIntegrationWithRegistry:
    @patch("reeln_openai_plugin.plugin.generate_livestream_metadata")
    def test_full_lifecycle(
        self,
        mock_gen: MagicMock,
        plugin_config: dict[str, Any],
    ) -> None:
        """Simulate the full plugin lifecycle: init -> register -> emit."""
        from reeln_openai_plugin.livestream import LivestreamMetadata

        mock_gen.return_value = LivestreamMetadata(title="Live!", description="Go!")
        plugin = OpenAIPlugin(plugin_config)
        registry = HookRegistry()
        plugin.register(registry)

        game_info = FakeGameInfo(home_team="Storm", away_team="Thunder")
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": game_info})
        registry.emit(Hook.ON_GAME_INIT, context)

        assert context.shared["livestream_metadata"]["title"] == "Live!"


# ------------------------------------------------------------------
# auth_check
# ------------------------------------------------------------------


class TestAuthCheckNotConfigured:
    @patch.object(OpenAIPlugin, "_resolve_api_key")
    def test_no_api_key_returns_not_configured(
        self,
        mock_resolve: MagicMock,
    ) -> None:
        from reeln_openai_plugin.client import OpenAIError

        mock_resolve.side_effect = OpenAIError("No API key: set api_key_file in config or OPENAI_API_KEY env var")
        plugin = OpenAIPlugin()
        results = plugin.auth_check()
        assert len(results) == 1
        assert results[0].service == "OpenAI API"
        assert results[0].status.value == "not_configured"
        assert "No API key" in results[0].message
        assert "api_key_file" in results[0].hint

    @patch.object(OpenAIPlugin, "_resolve_api_key")
    def test_not_configured_hint_mentions_env_var(
        self,
        mock_resolve: MagicMock,
    ) -> None:
        from reeln_openai_plugin.client import OpenAIError

        mock_resolve.side_effect = OpenAIError("missing key")
        plugin = OpenAIPlugin()
        results = plugin.auth_check()
        assert "OPENAI_API_KEY" in results[0].hint


class TestAuthCheckHttp401:
    @patch("reeln_openai_plugin.plugin.urllib.request.urlopen")
    @patch.object(OpenAIPlugin, "_resolve_api_key")
    def test_401_returns_fail(
        self,
        mock_resolve: MagicMock,
        mock_urlopen: MagicMock,
    ) -> None:
        import urllib.error

        mock_resolve.return_value = "sk-test1234567890"
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.openai.com/v1/models",
            code=401,
            msg="Unauthorized",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )
        plugin = OpenAIPlugin()
        results = plugin.auth_check()
        assert len(results) == 1
        assert results[0].service == "OpenAI API"
        assert results[0].status.value == "fail"
        assert "HTTP 401" in results[0].message
        assert results[0].identity == "sk-test..."
        assert "revoked" in results[0].hint

    @patch("reeln_openai_plugin.plugin.urllib.request.urlopen")
    @patch.object(OpenAIPlugin, "_resolve_api_key")
    def test_403_returns_fail(
        self,
        mock_resolve: MagicMock,
        mock_urlopen: MagicMock,
    ) -> None:
        import urllib.error

        mock_resolve.return_value = "sk-test1234567890"
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.openai.com/v1/models",
            code=403,
            msg="Forbidden",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )
        plugin = OpenAIPlugin()
        results = plugin.auth_check()
        assert results[0].status.value == "fail"
        assert "HTTP 403" in results[0].message


class TestAuthCheckNetworkError:
    @patch("reeln_openai_plugin.plugin.urllib.request.urlopen")
    @patch.object(OpenAIPlugin, "_resolve_api_key")
    def test_url_error_returns_warn(
        self,
        mock_resolve: MagicMock,
        mock_urlopen: MagicMock,
    ) -> None:
        import urllib.error

        mock_resolve.return_value = "sk-test1234567890"
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        plugin = OpenAIPlugin()
        results = plugin.auth_check()
        assert len(results) == 1
        assert results[0].service == "OpenAI API"
        assert results[0].status.value == "warn"
        assert "Could not validate" in results[0].message
        assert results[0].identity == "sk-test..."
        assert "Network error" in results[0].hint

    @patch("reeln_openai_plugin.plugin.urllib.request.urlopen")
    @patch.object(OpenAIPlugin, "_resolve_api_key")
    def test_timeout_returns_warn(
        self,
        mock_resolve: MagicMock,
        mock_urlopen: MagicMock,
    ) -> None:
        mock_resolve.return_value = "sk-test1234567890"
        mock_urlopen.side_effect = TimeoutError("timed out")
        plugin = OpenAIPlugin()
        results = plugin.auth_check()
        assert results[0].status.value == "warn"
        assert "key may still be valid" in results[0].hint


class TestAuthCheckSuccess:
    @patch("reeln_openai_plugin.plugin.urllib.request.urlopen")
    @patch.object(OpenAIPlugin, "_resolve_api_key")
    def test_200_returns_ok(
        self,
        mock_resolve: MagicMock,
        mock_urlopen: MagicMock,
    ) -> None:
        mock_resolve.return_value = "sk-proj-abcdefg1234567"
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"data": []}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        plugin = OpenAIPlugin()
        results = plugin.auth_check()
        assert len(results) == 1
        assert results[0].service == "OpenAI API"
        assert results[0].status.value == "ok"
        assert results[0].message == "Authenticated"
        assert results[0].identity == "sk-proj..."

    @patch("reeln_openai_plugin.plugin.urllib.request.urlopen")
    @patch.object(OpenAIPlugin, "_resolve_api_key")
    def test_redacted_key_shows_first_7_chars(
        self,
        mock_resolve: MagicMock,
        mock_urlopen: MagicMock,
    ) -> None:
        mock_resolve.return_value = "sk-live-XXXXXXXXXXX"
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"data": []}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        plugin = OpenAIPlugin()
        results = plugin.auth_check()
        assert results[0].identity == "sk-live..."

    @patch("reeln_openai_plugin.plugin.urllib.request.urlopen")
    @patch.object(OpenAIPlugin, "_resolve_api_key")
    def test_short_key_redacts_fully(
        self,
        mock_resolve: MagicMock,
        mock_urlopen: MagicMock,
    ) -> None:
        mock_resolve.return_value = "short"
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"data": []}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        plugin = OpenAIPlugin()
        results = plugin.auth_check()
        assert results[0].identity == "***"


# ------------------------------------------------------------------
# auth_refresh
# ------------------------------------------------------------------


class TestAuthRefresh:
    def test_always_returns_fail(self) -> None:
        plugin = OpenAIPlugin()
        results = plugin.auth_refresh()
        assert len(results) == 1
        assert results[0].service == "OpenAI API"
        assert results[0].status.value == "fail"
        assert "cannot be refreshed" in results[0].message

    def test_hint_mentions_platform_url(self) -> None:
        plugin = OpenAIPlugin()
        results = plugin.auth_refresh()
        assert "platform.openai.com" in results[0].hint
