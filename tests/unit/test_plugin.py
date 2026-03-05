"""Tests for plugin module."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from reeln.plugins.hooks import Hook, HookContext
from reeln.plugins.registry import HookRegistry

from plugin_name.plugin import PluginName
from tests.conftest import FakeGameInfo


class TestPluginAttributes:
    def test_name(self) -> None:
        plugin = PluginName()
        assert plugin.name == "myplugin"

    def test_version(self) -> None:
        plugin = PluginName()
        assert plugin.version == "0.1.0"

    def test_api_version(self) -> None:
        plugin = PluginName()
        assert plugin.api_version == 1


class TestPluginInit:
    def test_no_config(self) -> None:
        plugin = PluginName()
        assert plugin._config == {}

    def test_empty_config(self) -> None:
        plugin = PluginName({})
        assert plugin._config == {}

    def test_with_config(self, plugin_config: dict[str, Any]) -> None:
        plugin = PluginName(plugin_config)
        assert plugin._config == plugin_config


class TestPluginRegister:
    def test_registers_on_game_init(self) -> None:
        plugin = PluginName()
        registry = HookRegistry()
        plugin.register(registry)
        assert registry.has_handlers(Hook.ON_GAME_INIT)

    def test_does_not_register_other_hooks(self) -> None:
        plugin = PluginName()
        registry = HookRegistry()
        plugin.register(registry)
        assert not registry.has_handlers(Hook.ON_GAME_FINISH)


class TestOnGameInit:
    def test_no_game_info_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        plugin = PluginName()
        context = HookContext(hook=Hook.ON_GAME_INIT, data={})

        with caplog.at_level(logging.WARNING):
            plugin.on_game_init(context)

        assert "no game_info" in caplog.text

    def test_with_game_info(self) -> None:
        plugin = PluginName()
        game_info = FakeGameInfo()
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": game_info})
        plugin.on_game_init(context)


class TestIntegrationWithRegistry:
    def test_full_lifecycle(self, plugin_config: dict[str, Any]) -> None:
        """Simulate the full plugin lifecycle: init -> register -> emit."""
        plugin = PluginName(plugin_config)
        registry = HookRegistry()
        plugin.register(registry)

        game_info = FakeGameInfo(home_team="Storm", away_team="Thunder")
        context = HookContext(hook=Hook.ON_GAME_INIT, data={"game_info": game_info})
        registry.emit(Hook.ON_GAME_INIT, context)
