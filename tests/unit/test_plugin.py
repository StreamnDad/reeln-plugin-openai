"""Tests for plugin module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from reeln.plugins.hooks import Hook, HookContext
from reeln.plugins.registry import HookRegistry

from reeln_openai_plugin.plugin import OpenAIPlugin
from tests.conftest import FakeGameInfo, FakeTeamInfo

# ------------------------------------------------------------------
# Attributes
# ------------------------------------------------------------------


class TestPluginAttributes:
    def test_name(self) -> None:
        plugin = OpenAIPlugin()
        assert plugin.name == "openai"

    def test_version(self) -> None:
        plugin = OpenAIPlugin()
        assert plugin.version == "0.4.2"

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
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path,
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
        with patch.dict("os.environ", {}, clear=True), pytest.raises(
            OpenAIError, match="No API key",
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
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        config: dict[str, Any] = {"api_key_file": str(tmp_path / "missing.txt")}
        plugin = OpenAIPlugin(config)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fallback"}), caplog.at_level(
            logging.WARNING,
        ):
            client = plugin._get_client()
        assert client._api_key == "sk-fallback"
        assert "not found" in caplog.text

    def test_missing_file_no_env_raises(self, tmp_path: Path) -> None:
        from reeln_openai_plugin.client import OpenAIError

        config: dict[str, Any] = {"api_key_file": str(tmp_path / "missing.txt")}
        plugin = OpenAIPlugin(config)
        with patch.dict("os.environ", {}, clear=True), pytest.raises(
            OpenAIError, match="No API key",
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

        home_info = FakeTeamInfo(name="A", logo_path=tmp_path / "h.png")
        away_info = FakeTeamInfo(name="B", logo_path=tmp_path / "a.png")

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

        home_info = FakeTeamInfo(name="Storm", logo_path=tmp_path / "h.png")
        away_info = FakeTeamInfo(name="Thunder", logo_path=tmp_path / "a.png")

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

        home_info = FakeTeamInfo(name="A", logo_path=tmp_path / "h.png")
        away_info = FakeTeamInfo(name="B", logo_path=tmp_path / "a.png")

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

        home_info = FakeTeamInfo(name="A", logo_path=tmp_path / "h.png")
        away_info = FakeTeamInfo(name="B", logo_path=tmp_path / "a.png")

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
        home_info = FakeTeamInfo(name="A", logo_path=None)
        away_info = FakeTeamInfo(name="B", logo_path=None)

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

        home_info = FakeTeamInfo(name="A", logo_path=tmp_path / "h.png")
        away_info = FakeTeamInfo(name="B", logo_path=tmp_path / "a.png")

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
