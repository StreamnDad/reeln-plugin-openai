"""PluginName — reeln-cli plugin for ..."""

from __future__ import annotations

import logging
from typing import Any

from reeln.models.plugin_schema import PluginConfigSchema
from reeln.plugins.hooks import Hook, HookContext
from reeln.plugins.registry import HookRegistry

log: logging.Logger = logging.getLogger(__name__)


class PluginName:
    """Plugin that provides ... integration for reeln-cli."""

    name: str = "myplugin"
    version: str = "0.1.0"
    api_version: int = 1

    config_schema: PluginConfigSchema = PluginConfigSchema(fields=())

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config: dict[str, Any] = config or {}

    def register(self, registry: HookRegistry) -> None:
        """Register hook handlers with the reeln plugin registry."""
        registry.register(Hook.ON_GAME_INIT, self.on_game_init)

    def on_game_init(self, context: HookContext) -> None:
        """Handle ``ON_GAME_INIT``."""
        game_info = context.data.get("game_info")
        if game_info is None:
            log.warning("%s plugin: no game_info in context, skipping", self.name)
            return
