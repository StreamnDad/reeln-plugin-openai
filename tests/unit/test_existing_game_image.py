"""Tests for adding a sponsor to finished game artwork."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image, ImageChops, ImageCms, ImageDraw
from reeln.plugins.hooks import Hook, HookContext

from reeln_openai_plugin.broadcast_paint import SPONSOR_FOOTER_HEIGHT
from reeln_openai_plugin.client import OpenAIError
from reeln_openai_plugin.game_image import SPONSOR_MAX_FILE_BYTES, add_sponsor_to_existing_game_image
from reeln_openai_plugin.plugin import OpenAIPlugin


@pytest.fixture
def game_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "game"
    directory.mkdir()
    Image.new("RGBA", (300, 80), "#f0a019").save(directory / "game-sponsor.png")
    return directory


def _save_artwork(path: Path, *, image_format: str = "PNG") -> None:
    artwork = Image.new("RGB", (1280, 720), "#293762")
    draw = ImageDraw.Draw(artwork)
    draw.rectangle((0, 0, 640, 360), fill="#ab111d")
    draw.rectangle((641, 0, 1279, 360), fill="#04a0ae")
    draw.rectangle((0, 361, 640, 719), fill="#b1a001")
    draw.rectangle((641, 361, 1279, 719), fill="#1d4e11")
    artwork.save(path, image_format)


@pytest.mark.parametrize(("suffix", "image_format"), [(".png", "PNG"), (".jpg", "JPEG"), (".webp", "WEBP")])
def test_finished_art_is_kept_in_full_above_sponsor(
    tmp_path: Path,
    game_dir: Path,
    suffix: str,
    image_format: str,
) -> None:
    source = tmp_path / f"finished{suffix}"
    _save_artwork(source, image_format=image_format)
    original = source.read_bytes()
    sponsor = game_dir / "game-sponsor.png"
    original_sponsor = sponsor.read_bytes()
    output = add_sponsor_to_existing_game_image(source, game_dir, tmp_path / "images")
    assert output.name == "sponsored-game-image.png"
    assert source.read_bytes() == original
    assert sponsor.read_bytes() == original_sponsor
    assert not output.with_name("game-image.png").exists()

    artwork_height = 720 - SPONSOR_FOOTER_HEIGHT
    expected_width = round(1280 * artwork_height / 720)
    image_left = (1280 - expected_width) // 2
    with Image.open(output) as rendered, Image.open(source) as original_image:
        assert rendered.size == (1280, 720)
        displayed_artwork = rendered.crop((image_left, 0, image_left + expected_width, artwork_height))
        expected = original_image.convert("RGBA").resize((expected_width, artwork_height), Image.Resampling.LANCZOS)
        assert ImageChops.difference(displayed_artwork, expected).getbbox(alpha_only=False) is None
        assert rendered.getpixel((930, 646)) == (240, 160, 25, 255)


def test_will_not_add_a_second_footer_to_the_saved_result(tmp_path: Path, game_dir: Path) -> None:
    source = tmp_path / "finished.png"
    _save_artwork(source)
    output = add_sponsor_to_existing_game_image(source, game_dir, tmp_path)
    original = output.read_bytes()

    with pytest.raises(OpenAIError, match="already sponsored"):
        add_sponsor_to_existing_game_image(output, game_dir, tmp_path)

    assert output.read_bytes() == original


def test_rejects_nonimage_and_requires_a_sponsor(tmp_path: Path, game_dir: Path) -> None:
    source = tmp_path / "bad.png"
    source.write_text("This is not an image.")
    with pytest.raises(OpenAIError, match="Cannot read the existing game image"):
        add_sponsor_to_existing_game_image(source, game_dir, tmp_path)
    assert not list(tmp_path.glob("games/**/sponsored-game-image.png"))

    (game_dir / "game-sponsor.png").unlink()
    with pytest.raises(OpenAIError, match="Choose a sponsor logo"):
        add_sponsor_to_existing_game_image(source, game_dir, tmp_path)


@pytest.mark.parametrize(
    ("source_type", "error"),
    [
        ("oversize_file", "no larger than 32 MiB"),
        ("oversize_canvas", "at most 8192 pixels per edge"),
        ("animated", "Choose a still game image"),
        ("bad_icc", "Export the existing game image in sRGB"),
    ],
)
def test_rejects_unsupported_finished_artwork(
    tmp_path: Path, game_dir: Path, source_type: str, error: str,
) -> None:
    source = tmp_path / "finished.png"
    if source_type == "oversize_file":
        with source.open("wb") as file:
            file.truncate(SPONSOR_MAX_FILE_BYTES + 1)
    elif source_type == "oversize_canvas":
        Image.new("RGB", (8193, 1), "#93ed6d").save(source)
    elif source_type == "animated":
        first = Image.new("RGB", (30, 40), "red")
        second = Image.new("RGB", (30, 40), "blue")
        first.save(source, save_all=True, append_images=[second])
    else:
        Image.new("RGB", (30, 40), "#93ed6d").save(source, icc_profile=b"invalid ICC")

    with pytest.raises(OpenAIError, match=error):
        add_sponsor_to_existing_game_image(source, game_dir, tmp_path)
    assert not list(tmp_path.glob("games/**/sponsored-game-image.png"))


@pytest.mark.parametrize("mode", ["RGB", "P"])
def test_srgb_color_managed_finished_artwork(tmp_path: Path, game_dir: Path, mode: str) -> None:
    source = tmp_path / "finished.png"
    image = Image.new("RGB", (40, 30), "#93ed6d")
    if mode == "P":
        image = image.quantize(colors=1)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    image.save(source, icc_profile=profile.tobytes())

    output = add_sponsor_to_existing_game_image(source, game_dir, tmp_path)
    with Image.open(output) as rendered:
        assert rendered.getpixel((640, 200)) == (147, 237, 109, 255)


def test_existing_art_hook_starts_no_client_and_needs_no_game_info(tmp_path: Path, game_dir: Path) -> None:
    source = tmp_path / "finished.png"
    _save_artwork(source)
    plugin = OpenAIPlugin({"enabled": True, "game_image_output_dir": str(tmp_path)})
    context = HookContext(
        hook=Hook.ON_GAME_INIT,
        data={"regenerate_image_only": True, "existing_game_image_path": str(source), "game_dir": str(game_dir)},
    )
    with patch.object(plugin, "_get_client") as client:
        plugin.on_game_init(context)
    client.assert_not_called()
    result = context.shared["game_image"]
    assert result["existing_artwork_preserved"] is True
    assert result["sponsor_logo_applied"] is True
    assert Path(result["image_path"]).name == "sponsored-game-image.png"
    assert Path(result["image_path"]).is_file()

    context.data["existing_game_image_path"] = ""
    with patch.object(plugin, "_get_client") as client:
        plugin.on_game_init(context)
    client.assert_not_called()
    assert "game_image" not in context.shared
    assert "Choose an existing game image" in context.shared["game_image_error"]


@pytest.mark.parametrize(
    ("game_dir_arg", "output_dir_arg", "error"),
    [(None, "/images", "A game directory is required"), ("/game", "", "Set the game image output directory")],
)
def test_existing_art_requires_game_and_output_directory(
    game_dir_arg: str | None, output_dir_arg: str, error: str,
) -> None:
    plugin = OpenAIPlugin({"enabled": True, "game_image_output_dir": output_dir_arg})
    context = HookContext(
        hook=Hook.ON_GAME_INIT,
        data={"regenerate_image_only": True, "existing_game_image_path": "/image.png", "game_dir": game_dir_arg},
    )
    plugin.on_game_init(context)
    assert error in context.shared["game_image_error"]
    assert "game_image" not in context.shared
