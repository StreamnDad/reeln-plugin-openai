"""Shared test fixtures for reeln-plugin-openai."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass
class FakeGameInfo:
    """Minimal stand-in for ``reeln.models.game.GameInfo``."""

    date: str = "2026-01-15"
    home_team: str = "Eagles"
    away_team: str = "Hawks"
    sport: str = "hockey"
    game_number: int = 1
    venue: str = ""
    game_time: str = ""
    description: str = ""
    thumbnail: str = ""


@dataclass
class FakeTeamProfile:
    """Minimal stand-in for a team profile object."""

    summary: str = "10-5-2 record, 3rd in division"


@dataclass
class FakeTeamInfo:
    """Minimal stand-in for a team object with logo and colors."""

    name: str = "Eagles"
    short_name: str = "EGL"
    logo_path: Path | None = None
    colors: str = "Red, White"
    game_level: str = "Varsity"


@pytest.fixture()
def game_info() -> FakeGameInfo:
    return FakeGameInfo()


@pytest.fixture()
def api_key_file(tmp_path: Path) -> Path:
    """Return a temporary API key file."""
    key_file = tmp_path / "openai_key.txt"
    key_file.write_text("sk-test-key-12345\n")
    return key_file


@pytest.fixture()
def plugin_config(api_key_file: Path) -> dict[str, Any]:
    """Return a minimal valid plugin config."""
    return {
        "enabled": True,
        "api_key_file": str(api_key_file),
    }
