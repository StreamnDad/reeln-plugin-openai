"""Tests for game image generation."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from reeln_openai_plugin.client import OpenAIError
from reeln_openai_plugin.game_image import (
    GameImageResult,
    encode_logo,
    generate_game_image,
    resize_image,
)
from reeln_openai_plugin.prompts import PromptRegistry
from tests.conftest import FakeTeamInfo


def _make_logo(path: Path, size: tuple[int, int] = (100, 100)) -> Path:
    """Create a small PNG logo file for testing."""
    img = Image.new("RGBA", size, color=(255, 0, 0, 255))
    img.save(str(path), "PNG")
    return path


# ------------------------------------------------------------------
# GameImageResult
# ------------------------------------------------------------------


class TestGameImageResult:
    def test_frozen(self, tmp_path: Path) -> None:
        r = GameImageResult(image_path=tmp_path / "img.png")
        with pytest.raises(AttributeError):
            r.image_path = tmp_path / "other.png"  # type: ignore[misc]

    def test_fields(self, tmp_path: Path) -> None:
        p = tmp_path / "img.png"
        r = GameImageResult(image_path=p)
        assert r.image_path == p


# ------------------------------------------------------------------
# encode_logo
# ------------------------------------------------------------------


class TestEncodeLogo:
    def test_encodes_file(self, tmp_path: Path) -> None:
        logo = _make_logo(tmp_path / "logo.png")
        result = encode_logo(logo)
        # Should be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OpenAIError, match="Cannot read logo"):
            encode_logo(tmp_path / "missing.png")


# ------------------------------------------------------------------
# resize_image
# ------------------------------------------------------------------


class TestResizeImage:
    def test_resizes_to_target(self) -> None:
        img = Image.new("RGB", (1536, 1024), color=(0, 128, 255))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        raw_bytes = buf.getvalue()

        result = resize_image(raw_bytes, 1280, 720)
        # Load result and verify dimensions
        result_img = Image.open(io.BytesIO(result))
        assert result_img.size == (1280, 720)

    def test_corrupt_bytes_raises(self) -> None:
        with pytest.raises(OpenAIError, match="Cannot decode image"):
            resize_image(b"not an image at all", 100, 100)

    def test_output_is_png(self) -> None:
        import io

        img = Image.new("RGB", (100, 100))
        buf = io.BytesIO()
        img.save(buf, "PNG")

        result = resize_image(buf.getvalue(), 50, 50)
        result_img = Image.open(io.BytesIO(result))
        assert result_img.format == "PNG"


# ------------------------------------------------------------------
# generate_game_image
# ------------------------------------------------------------------


class TestGenerateGameImage:
    def test_success(self, tmp_path: Path) -> None:
        # Create logos
        home_logo = _make_logo(tmp_path / "home_logo.png")
        away_logo = _make_logo(tmp_path / "away_logo.png")

        home = FakeTeamInfo(name="Storm", short_name="STM", logo_path=home_logo, colors="Blue, White")
        away = FakeTeamInfo(name="Thunder", short_name="THN", logo_path=away_logo, colors="Gold, Black")

        # Mock client
        client = MagicMock()
        # Return a valid image (small PNG)
        img = Image.new("RGB", (1536, 1024), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        client.request_image.return_value = buf.getvalue()

        registry = PromptRegistry()
        output_dir = tmp_path / "output"

        result = generate_game_image(
            client=client,
            prompt_registry=registry,
            home=home,
            away=away,
            rink="Rink Arena",
            game_date="2026-03-10",
            game_time="19:00",
            output_dir=output_dir,
            model="gpt-5.2",
            renderer_model="gpt-image-1.5",
        )

        assert isinstance(result, GameImageResult)
        assert result.image_path.exists()
        assert result.image_path.parent == output_dir
        assert "stm" in result.image_path.name.lower()
        assert "thn" in result.image_path.name.lower()

        # Verify saved image is 1280x720
        saved = Image.open(str(result.image_path))
        assert saved.size == (1280, 720)

        # Verify client was called with correct args
        client.request_image.assert_called_once()
        call_kwargs = client.request_image.call_args[1]
        assert call_kwargs["model_override"] == "gpt-5.2"
        assert call_kwargs["renderer_model"] == "gpt-image-1.5"
        assert call_kwargs["timeout_override"] == 120.0
        assert len(call_kwargs["images"]) == 2

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        home_logo = _make_logo(tmp_path / "h.png")
        away_logo = _make_logo(tmp_path / "a.png")
        home = FakeTeamInfo(name="A", logo_path=home_logo)
        away = FakeTeamInfo(name="B", logo_path=away_logo)

        client = MagicMock()
        img = Image.new("RGB", (1536, 1024))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        client.request_image.return_value = buf.getvalue()

        output_dir = tmp_path / "nested" / "output"
        assert not output_dir.exists()

        result = generate_game_image(
            client=client,
            prompt_registry=PromptRegistry(),
            home=home,
            away=away,
            rink="Rink",
            game_date="2026-01-01",
            game_time="18:00",
            output_dir=output_dir,
            model="m",
            renderer_model="r",
        )
        assert output_dir.exists()
        assert result.image_path.exists()

    def test_missing_logo_raises(self, tmp_path: Path) -> None:
        home = FakeTeamInfo(name="A", logo_path=tmp_path / "missing.png")
        away_logo = _make_logo(tmp_path / "away.png")
        away = FakeTeamInfo(name="B", logo_path=away_logo)

        client = MagicMock()

        with pytest.raises(OpenAIError, match="Cannot read logo"):
            generate_game_image(
                client=client,
                prompt_registry=PromptRegistry(),
                home=home,
                away=away,
                rink="R",
                game_date="2026-01-01",
                game_time="18:00",
                output_dir=tmp_path,
                model="m",
                renderer_model="r",
            )

    def test_api_error_propagates(self, tmp_path: Path) -> None:
        home_logo = _make_logo(tmp_path / "h.png")
        away_logo = _make_logo(tmp_path / "a.png")
        home = FakeTeamInfo(name="A", logo_path=home_logo)
        away = FakeTeamInfo(name="B", logo_path=away_logo)

        client = MagicMock()
        client.request_image.side_effect = OpenAIError("API down")

        with pytest.raises(OpenAIError, match="API down"):
            generate_game_image(
                client=client,
                prompt_registry=PromptRegistry(),
                home=home,
                away=away,
                rink="R",
                game_date="2026-01-01",
                game_time="18:00",
                output_dir=tmp_path,
                model="m",
                renderer_model="r",
            )

    def test_output_filename_format(self, tmp_path: Path) -> None:
        home_logo = _make_logo(tmp_path / "h.png")
        away_logo = _make_logo(tmp_path / "a.png")
        home = FakeTeamInfo(name="Storm Eagles", short_name="SE", logo_path=home_logo)
        away = FakeTeamInfo(name="Thunder Hawks", short_name="TH", logo_path=away_logo)

        client = MagicMock()
        img = Image.new("RGB", (1536, 1024))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        client.request_image.return_value = buf.getvalue()

        result = generate_game_image(
            client=client,
            prompt_registry=PromptRegistry(),
            home=home,
            away=away,
            rink="Rink",
            game_date="2026/03/10",
            game_time="19:00",
            output_dir=tmp_path,
            model="m",
            renderer_model="r",
        )

        # Filename should slugify names and handle date separators
        assert result.image_path.suffix == ".png"
        name = result.image_path.stem
        # Uses short_name for slugs
        assert "se" in name
        assert "th" in name
        assert "vs" in name
        # Date slashes converted to dashes
        assert "2026-03-10" in name

    def test_no_logo_paths(self, tmp_path: Path) -> None:
        """Team objects without logo_path still work — no images sent."""
        home = FakeTeamInfo(name="A", logo_path=None)
        away = FakeTeamInfo(name="B", logo_path=None)

        client = MagicMock()
        img = Image.new("RGB", (1536, 1024))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        client.request_image.return_value = buf.getvalue()

        result = generate_game_image(
            client=client,
            prompt_registry=PromptRegistry(),
            home=home,
            away=away,
            rink="R",
            game_date="2026-01-01",
            game_time="18:00",
            output_dir=tmp_path,
            model="m",
            renderer_model="r",
        )
        assert result.image_path.exists()
        # No images sent to API
        call_kwargs = client.request_image.call_args[1]
        assert call_kwargs["images"] == []

    def test_dangerous_chars_sanitized_in_filename(self, tmp_path: Path) -> None:
        """Team names or dates with path-traversal chars produce safe filenames."""
        home_logo = _make_logo(tmp_path / "h.png")
        away_logo = _make_logo(tmp_path / "a.png")
        home = FakeTeamInfo(name="../evil", short_name="../evil", logo_path=home_logo)
        away = FakeTeamInfo(name="B/bad", short_name="B/bad", logo_path=away_logo)

        client = MagicMock()
        img = Image.new("RGB", (1536, 1024))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        client.request_image.return_value = buf.getvalue()

        result = generate_game_image(
            client=client,
            prompt_registry=PromptRegistry(),
            home=home,
            away=away,
            rink="R",
            game_date="2026/../../../etc",
            game_time="18:00",
            output_dir=tmp_path / "output",
            model="m",
            renderer_model="r",
        )

        # File stays inside output_dir and has no dangerous chars
        assert result.image_path.parent == tmp_path / "output"
        name = result.image_path.name
        assert "/" not in name
        assert ".." not in name

    def test_prompt_rendered_with_variables(self, tmp_path: Path) -> None:
        home_logo = _make_logo(tmp_path / "h.png")
        away_logo = _make_logo(tmp_path / "a.png")
        home = FakeTeamInfo(name="Storm", logo_path=home_logo, colors="Blue", game_level="U14")
        away = FakeTeamInfo(name="Thunder", logo_path=away_logo, colors="Gold", game_level="U14")

        client = MagicMock()
        img = Image.new("RGB", (1536, 1024))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        client.request_image.return_value = buf.getvalue()

        generate_game_image(
            client=client,
            prompt_registry=PromptRegistry(),
            home=home,
            away=away,
            rink="Big Arena",
            game_date="2026-03-10",
            game_time="19:00",
            output_dir=tmp_path,
            model="m",
            renderer_model="r",
        )

        call_kwargs = client.request_image.call_args[1]
        prompt = call_kwargs["prompt"]
        assert "Storm" in prompt
        assert "Thunder" in prompt
        assert "Blue" in prompt
        assert "Gold" in prompt
        assert "Big Arena" in prompt
        assert "U14" in prompt
