"""Shared test fixtures for reeln-plugin-openai."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    level: str = ""
    tournament: str = ""


@dataclass
class FakeGameEvent:
    """Minimal stand-in for ``reeln.models.game.GameEvent``."""

    id: str = "evt-001"
    clip: str = "clip.mp4"
    segment_number: int = 1
    event_type: str = "goal"
    player: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeTeamProfile:
    """Minimal stand-in for a team profile object."""

    metadata: dict[str, Any] = field(default_factory=lambda: {"summary": "10-5-2 record, 3rd in division"})


@dataclass
class FakeTeamInfo:
    """Minimal stand-in for a team object with logo and colors."""

    team_name: str = "Eagles"
    short_name: str = "EGL"
    logo_path: Path | None = None
    colors: str = "Red, White"
    level: str = "Varsity"


@dataclass
class FakeQueueItem:
    """Minimal stand-in for ``reeln.models.queue.QueueItem``."""

    id: str = "abc123def456"
    output: str = "/games/test/shorts/clip_short.mp4"
    game_dir: str = "/games/test"
    status: str = "rendered"
    queued_at: str = "2026-04-06T18:00:00Z"
    player: str = ""
    assists: str = ""
    event_type: str = "goal"
    level: str = "11u"
    title: str = ""
    description: str = ""


@dataclass
class FakeExtractedFrames:
    """Minimal stand-in for ``reeln.models.zoom.ExtractedFrames``."""

    frame_paths: tuple[Path, ...] = ()
    timestamps: tuple[float, ...] = ()
    source_width: int = 1920
    source_height: int = 1080
    duration: float = 10.0
    fps: float = 59.94


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
