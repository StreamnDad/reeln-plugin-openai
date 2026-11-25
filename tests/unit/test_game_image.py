"""Tests for game image generation."""

from __future__ import annotations

import io
import shutil
import struct
import zlib
from hashlib import sha256
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageCms, ImageDraw, PngImagePlugin

from reeln_openai_plugin.client import OpenAIError
from reeln_openai_plugin.game_image import (
    BACKGROUND_INSTRUCTIONS,
    SPONSOR_MAX_FILE_BYTES,
    GameImageResult,
    _fit_background,
    _game_path_identity,
    _per_game_output_dir,
    _write_game_png,
    add_sponsor_to_existing_game_image,
    generate_game_image,
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
# fit_background
# ------------------------------------------------------------------


class TestFitBackground:
    def test_center_crops_without_squashing(self) -> None:
        img = Image.new("RGBA", (1280, 1024), color=(0, 128, 255, 255))
        img.putpixel((562, 182), (200, 80, 10, 255))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        background = _fit_background(buf.getvalue())
        assert background.size == (1280, 720)
        assert background.getpixel((562, 30)) == (200, 80, 10, 255)

    def test_corrupt_bytes_raises(self) -> None:
        with pytest.raises(OpenAIError, match="Cannot decode image"):
            _fit_background(b"not an image at all")


def test_per_game_output_matches_windows_path_identity(tmp_path: Path) -> None:
    assert _game_path_identity(r"\\?\C:\Games\one") == r"C:\Games\one"
    assert _game_path_identity(r"\\?\UNC\server\Games\one") == r"\\server\Games\one"
    with pytest.raises(OpenAIError, match="A game directory with a name"):
        _per_game_output_dir(tmp_path, Path(tmp_path.anchor))


def test_cannot_write_png_below_existing_file(tmp_path: Path) -> None:
    not_a_directory = tmp_path / "images"
    not_a_directory.write_bytes(b"retain me")
    with pytest.raises(OpenAIError, match="Cannot save game image"):
        _write_game_png(not_a_directory / "game.png", Image.new("RGBA", (20, 20)), sponsored=False)
    assert not_a_directory.read_bytes() == b"retain me"


# ------------------------------------------------------------------
# generate_game_image
# ------------------------------------------------------------------


class TestGenerateGameImage:
    def test_success(self, tmp_path: Path) -> None:
        # Create logos
        home_logo = _make_logo(tmp_path / "home_logo.png")
        away_logo = _make_logo(tmp_path / "away_logo.png")

        home = FakeTeamInfo(team_name="Storm", short_name="STM", logo_path=home_logo, colors="Blue, White")
        away = FakeTeamInfo(team_name="Thunder", short_name="THN", logo_path=away_logo, colors="Gold, Black")

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
        assert saved.format == "PNG"
        assert result.team_logos_composited is True
        assert result.sponsor_logo_applied is False

        # Verify client was called with correct args
        client.request_image.assert_called_once()
        call_kwargs = client.request_image.call_args[1]
        assert call_kwargs["model_override"] == "gpt-5.2"
        assert call_kwargs["renderer_model"] == "gpt-image-1.5"
        assert call_kwargs["timeout_override"] == 120.0
        assert call_kwargs["images"] == []

    def test_sparse_metadata_keeps_long_team_name_and_profile_accent(self, tmp_path: Path) -> None:
        home = FakeTeamInfo(
            team_name=" ".join(["Northern", "Valley", "Community", "Recreation", "Hockey", "Association"]),
            level="",
            colors="#777777, #030f60",
        )
        away = FakeTeamInfo(team_name="", level="")
        client = MagicMock()
        background = io.BytesIO()
        Image.new("RGB", (1280, 720), "#101010").save(background, "PNG")
        client.request_image.return_value = background.getvalue()

        result = generate_game_image(
            client=client,
            prompt_registry=PromptRegistry(),
            home=home,
            away=away,
            rink="",
            game_date="",
            game_time="",
            output_dir=tmp_path,
            model="m",
            renderer_model="r",
        )

        with Image.open(result.image_path) as image:
            assert image.getpixel((1160, 670)) == (16, 16, 16, 255)
            home_title = image.crop((70, 472, 580, 500)).convert("L").histogram()
            away_title = image.crop((703, 472, 1215, 500)).convert("L").histogram()
            assert sum(home_title[190:]) > 300
            assert sum(away_title[190:]) == 0

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        home_logo = _make_logo(tmp_path / "h.png")
        away_logo = _make_logo(tmp_path / "a.png")
        home = FakeTeamInfo(team_name="A", logo_path=home_logo)
        away = FakeTeamInfo(team_name="B", logo_path=away_logo)

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
        home = FakeTeamInfo(team_name="A", logo_path=tmp_path / "missing.png")
        away_logo = _make_logo(tmp_path / "away.png")
        away = FakeTeamInfo(team_name="B", logo_path=away_logo)

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
        home = FakeTeamInfo(team_name="A", logo_path=home_logo)
        away = FakeTeamInfo(team_name="B", logo_path=away_logo)

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
        home = FakeTeamInfo(team_name="Storm Eagles", short_name="SE", logo_path=home_logo)
        away = FakeTeamInfo(team_name="Thunder Hawks", short_name="TH", logo_path=away_logo)

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
        home = FakeTeamInfo(team_name="A", logo_path=None)
        away = FakeTeamInfo(team_name="B", logo_path=None)

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
        assert result.team_logos_composited is False
        call_kwargs = client.request_image.call_args[1]
        assert call_kwargs["images"] == []

    def test_dangerous_chars_sanitized_in_filename(self, tmp_path: Path) -> None:
        """Team names or dates with path-traversal chars produce safe filenames."""
        home_logo = _make_logo(tmp_path / "h.png")
        away_logo = _make_logo(tmp_path / "a.png")
        home = FakeTeamInfo(team_name="../evil", short_name="../evil", logo_path=home_logo)
        away = FakeTeamInfo(team_name="B/bad", short_name="B/bad", logo_path=away_logo)

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
        home = FakeTeamInfo(team_name="Storm", logo_path=home_logo, colors="Blue", level="U14")
        away = FakeTeamInfo(team_name="Thunder", logo_path=away_logo, colors="Gold", level="U14")

        client = MagicMock()
        img = Image.new("RGB", (1536, 1024))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        client.request_image.return_value = buf.getvalue()

        custom_prompt = tmp_path / "custom.txt"
        custom_prompt.write_text(
            "{{home_team}} {{away_team}} {{home_colors}} {{away_colors}} {{rink}} "
            "{{game_level}} {{level}} {{description}} {{tournament}}. Draw logos!",
        )
        generate_game_image(
            client=client,
            prompt_registry=PromptRegistry(overrides={"game_image": str(custom_prompt)}),
            home=home,
            away=away,
            rink="Big Arena",
            game_date="2026-03-10",
            game_time="19:00",
            output_dir=tmp_path,
            model="m",
            renderer_model="r",
            level="2016",
            description="Semifinal game",
            tournament="State Championship",
        )

        call_kwargs = client.request_image.call_args[1]
        prompt = call_kwargs["prompt"]
        assert "Storm" in prompt
        assert "Thunder" in prompt
        assert "Blue" in prompt
        assert "Gold" in prompt
        assert "Big Arena" in prompt
        assert "U14" in prompt
        assert "2016" in prompt
        assert "Semifinal game" in prompt
        assert "State Championship" in prompt
        assert prompt.endswith(BACKGROUND_INSTRUCTIONS)
        assert call_kwargs["images"] == []


class TestSponsorComposition:
    def setup_method(self) -> None:
        self.client = MagicMock()
        background = io.BytesIO()
        Image.new("RGBA", (1536, 1024), "#a18cc5").save(background, "PNG")
        self.client.request_image.return_value = background.getvalue()

    def generate(self, tmp_path: Path, game_dir: Path, sponsor_path: Path | None = None) -> GameImageResult:
        return generate_game_image(
            client=self.client,
            prompt_registry=PromptRegistry(),
            home=FakeTeamInfo(team_name="Roseville", logo_path=game_dir / "roseville.png"),
            away=FakeTeamInfo(team_name="Hawks", logo_path=game_dir / "hawks.png"),
            rink="Roseville Ice Arena",
            game_date="2026-11-24",
            game_time="7:15 PM",
            output_dir=tmp_path / "output",
            model="m",
            renderer_model="r",
            game_dir=game_dir,
            sponsor_logo_path=sponsor_path,
        )

    @pytest.fixture
    def game_dir(self, tmp_path: Path) -> Path:
        game_dir = tmp_path / "2026-11-24_hawks_at_roseville"
        game_dir.mkdir()
        home = Image.new("RGBA", (64, 32), "#efdb28")
        home.putpixel((12, 4), (2, 7, 18, 255))
        home.save(game_dir / "roseville.png")
        away = Image.new("RGBA", (30, 80), "#248fe3")
        away.putpixel((4, 22), (7, 18, 32, 255))
        away.save(game_dir / "hawks.png")
        return game_dir

    def test_exact_assigned_assets_and_proportional_sponsor(self, tmp_path: Path, game_dir: Path) -> None:
        sponsor_path = game_dir / "game-sponsor.png"
        logo = Image.new("RGBA", (1200, 450), (0, 0, 0, 0))
        draw = ImageDraw.Draw(logo)
        draw.rectangle((10, 20, 200, 100), fill=(243, 17, 36, 255))
        draw.rectangle((310, 120, 510, 290), fill=(0, 0, 0, 75))
        draw.rectangle((950, 255, 1150, 440), fill=(14, 247, 74, 255))
        logo.save(sponsor_path)
        originals = {path: path.read_bytes() for path in game_dir.iterdir()}

        result = self.generate(tmp_path, game_dir)
        scope = sha256(str(game_dir.resolve()).encode("utf-8")).hexdigest()[:16]
        assert result.image_path == tmp_path / "output" / "games" / game_dir.name / scope / "game-image.png"
        assert result.team_logos_composited is True
        assert result.sponsor_logo_applied is True
        assert self.client.request_image.call_args.kwargs["images"] == []
        for path, data in originals.items():
            assert path.read_bytes() == data

        with (
            Image.open(result.image_path) as image,
            Image.open(game_dir / "roseville.png") as roseville,
            Image.open(game_dir / "hawks.png") as hawks,
        ):
            assert image.crop((256, 212, 320, 244)).tobytes() == roseville.tobytes()
            assert image.crop((977, 188, 1007, 268)).tobytes() == hawks.tobytes()
            fitted = logo.resize((309, 116), Image.Resampling.LANCZOS)
            mat = Image.new("RGBA", fitted.size, "#22272e")
            mat.alpha_composite(fitted)
            assert image.crop((776, 588, 1085, 704)).tobytes() == mat.tobytes()

    def test_matchup_keeps_generated_art_visible_around_logos(self, tmp_path: Path, game_dir: Path) -> None:
        background = Image.new("RGBA", (1280, 720), "#e24739")
        ImageDraw.Draw(background).rectangle((640, 0, 1279, 719), fill="#2452a1")
        buffer = io.BytesIO()
        background.save(buffer, "PNG")
        self.client.request_image.return_value = buffer.getvalue()

        for filename, color in (("roseville.png", "#22262d"), ("hawks.png", "#f7f8ff")):
            logo = Image.new("RGBA", (100, 100))
            ImageDraw.Draw(logo).ellipse((10, 10, 89, 89), fill=color)
            logo.save(game_dir / filename)
        _make_logo(game_dir / "game-sponsor.png", (160, 56))

        result = self.generate(tmp_path, game_dir)
        with Image.open(result.image_path) as image:
            assert image.getpixel((288, 228)) == (34, 38, 45, 255)
            assert image.getpixel((992, 228)) == (247, 248, 255, 255)
            # Visible color in the portrait frames responds to each half of the supplied backdrop.
            red, _, blue, _ = image.getpixel((125, 140))
            assert red - blue > 25
            red, _, blue, _ = image.getpixel((1155, 140))
            assert blue - red > 25
            assert image.getpixel((8, 8)) == (226, 71, 57, 255)
            assert image.getpixel((1270, 8)) == (36, 82, 161, 255)
            # The broad metallic VS stays readily distinguishable from both team marks.
            versus_luminance = image.crop((548, 196, 735, 292)).convert("L").histogram()
            assert sum(versus_luminance[211:]) > 1000

    @pytest.mark.parametrize(
        ("size", "fitted_size", "color", "backing"),
        [
            ((600, 600), (116, 116), "#ffffff", "#22272e"),
            ((80, 20), (328, 82), "#000000", "#e9ebee"),
            ((8192, 1), (328, 1), "#d4131a", "#e9ebee"),
        ],
    )
    def test_sponsor_scales_to_fit_and_has_contrasting_backing(
        self,
        tmp_path: Path,
        game_dir: Path,
        size: tuple[int, int],
        fitted_size: tuple[int, int],
        color: str,
        backing: str,
    ) -> None:
        Image.new("RGBA", size, color).save(game_dir / "game-sponsor.png")
        result = self.generate(tmp_path, game_dir)
        width, height = fitted_size
        left, top = 930 - width // 2, 646 - height // 2
        with Image.open(result.image_path) as image:
            found = image.crop((left, top, left + width, top + height))
            assert found.tobytes() == Image.new("RGBA", fitted_size, color).tobytes()
            assert image.getpixel((left - 5, 646)) == Image.new("RGBA", (1, 1), backing).getpixel((0, 0))

    def test_unassigned_images_never_become_sponsor(self, tmp_path: Path, game_dir: Path) -> None:
        _make_logo(game_dir / "downloads-sponsor.png")
        result = self.generate(tmp_path, game_dir)
        assert result.sponsor_logo_applied is False
        with Image.open(result.image_path) as image:
            assert image.getpixel((1160, 670)) == (161, 140, 197, 255)

    def test_same_game_directory_name_in_different_roots_stays_isolated(
        self,
        tmp_path: Path,
        game_dir: Path,
    ) -> None:
        other_game = tmp_path / "other-season" / game_dir.name
        shutil.copytree(game_dir, other_game)
        _make_logo(other_game / "game-sponsor.png", (51, 24))

        unsponsored = self.generate(tmp_path, game_dir)
        sponsored = self.generate(tmp_path, other_game)

        assert sponsored.image_path != unsponsored.image_path
        assert unsponsored.image_path.is_file() and sponsored.image_path.is_file()
        assert sponsored.sponsor_logo_applied is True
        assert unsponsored.sponsor_logo_applied is False
        with Image.open(unsponsored.image_path) as image:
            assert image.getpixel((1160, 670)) == (161, 140, 197, 255)

    def test_cli_sponsor_file_persisted_verbatim(self, tmp_path: Path, game_dir: Path) -> None:
        source = _make_logo(tmp_path / "provided.png", (51, 24))
        result = self.generate(tmp_path, game_dir, source)
        assert result.sponsor_logo_applied is True
        assert (game_dir / "game-sponsor.png").read_bytes() == source.read_bytes()
        source.unlink()
        assert self.generate(tmp_path, game_dir).sponsor_logo_applied is True

    def test_generated_sponsored_image_cannot_receive_a_second_footer(self, tmp_path: Path, game_dir: Path) -> None:
        _make_logo(game_dir / "game-sponsor.png")
        result = self.generate(tmp_path, game_dir)
        copied = tmp_path / "renamed.png"
        shutil.copyfile(result.image_path, copied)
        with pytest.raises(OpenAIError, match="already sponsored"):
            add_sponsor_to_existing_game_image(copied, game_dir, tmp_path / "output")
        assert not result.image_path.with_name("sponsored-game-image.png").exists()

    @pytest.mark.parametrize("asset_type", ["directory", "symlink"])
    def test_invalid_sponsor_file_type(self, tmp_path: Path, game_dir: Path, asset_type: str) -> None:
        sponsor = game_dir / "game-sponsor.png"
        if asset_type == "directory":
            sponsor.mkdir()
        else:
            sponsor.symlink_to(_make_logo(tmp_path / "outside.png"))
        with pytest.raises(OpenAIError, match="must be a regular PNG file"):
            self.generate(tmp_path, game_dir)
        self.client.request_image.assert_not_called()

    @pytest.mark.parametrize("trailing_bytes", [False, True])
    def test_rejects_incomplete_sponsor_png(self, tmp_path: Path, game_dir: Path, trailing_bytes: bool) -> None:
        sponsor = _make_logo(game_dir / "game-sponsor.png")
        data = sponsor.read_bytes()
        sponsor.write_bytes(data + b"other content" if trailing_bytes else data[:-12])
        with pytest.raises(OpenAIError, match="complete PNG"):
            self.generate(tmp_path, game_dir)
        self.client.request_image.assert_not_called()

    def test_cannot_read_sponsor_from_filesystem(self, tmp_path: Path, game_dir: Path) -> None:
        source = _make_logo(tmp_path / "sponsor.png")
        with pytest.MonkeyPatch.context() as patcher:
            real_open = Path.open

            def open_file(path: Path, *args: object, **kwargs: object) -> object:
                if path == source:
                    raise PermissionError("permission denied")
                return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

            patcher.setattr(Path, "open", open_file)
            with pytest.raises(OpenAIError, match="Cannot read sponsor logo"):
                self.generate(tmp_path, game_dir, source)
        self.client.request_image.assert_not_called()

    @pytest.mark.parametrize("size", [(8193, 1), (1, 8193), (4000, 4001)])
    def test_oversize_sponsor_dimensions_fail_before_request(
        self,
        tmp_path: Path,
        game_dir: Path,
        size: tuple[int, int],
    ) -> None:
        _make_logo(game_dir / "game-sponsor.png", size)
        with pytest.raises(OpenAIError, match="at most 8192 pixels per edge and 16 million pixels"):
            self.generate(tmp_path, game_dir)
        self.client.request_image.assert_not_called()

    def test_decompression_bomb_fails_before_request(self, tmp_path: Path, game_dir: Path) -> None:
        sponsor = _make_logo(game_dir / "game-sponsor.png")
        data = bytearray(sponsor.read_bytes())
        struct.pack_into(">II", data, 16, 20_000, 20_000)
        struct.pack_into(">I", data, 29, zlib.crc32(data[12:29]))
        sponsor.write_bytes(data)
        with pytest.raises(OpenAIError, match="at most 8192 pixels per edge and 16 million pixels"):
            self.generate(tmp_path, game_dir)
        self.client.request_image.assert_not_called()

    def test_oversize_sponsor_file_fails_before_request(self, tmp_path: Path, game_dir: Path) -> None:
        with (game_dir / "game-sponsor.png").open("wb") as oversized:
            oversized.truncate(SPONSOR_MAX_FILE_BYTES + 1)
        with pytest.raises(OpenAIError, match="no larger than 32 MiB"):
            self.generate(tmp_path, game_dir)
        self.client.request_image.assert_not_called()

    def test_fully_transparent_sponsor_fails_before_request(self, tmp_path: Path, game_dir: Path) -> None:
        Image.new("RGBA", (200, 100)).save(game_dir / "game-sponsor.png")
        with pytest.raises(OpenAIError, match="must contain visible artwork"):
            self.generate(tmp_path, game_dir)
        self.client.request_image.assert_not_called()

    def test_real_png_content_required(self, tmp_path: Path, game_dir: Path) -> None:
        Image.new("RGB", (40, 30)).save(game_dir / "game-sponsor.png", "JPEG")
        with pytest.raises(OpenAIError, match="must be a static PNG"):
            self.generate(tmp_path, game_dir)
        self.client.request_image.assert_not_called()

    def test_must_be_8_bit_before_compositing(self, tmp_path: Path, game_dir: Path) -> None:
        Image.new("I;16", (40, 30)).save(game_dir / "game-sponsor.png", "PNG")
        with pytest.raises(OpenAIError, match="must be an 8-bit PNG"):
            self.generate(tmp_path, game_dir)
        self.client.request_image.assert_not_called()

    def test_must_not_be_animated(self, tmp_path: Path, game_dir: Path) -> None:
        first = Image.new("RGBA", (40, 30), "red")
        second = Image.new("RGBA", (40, 30), "yellow")
        first.save(game_dir / "game-sponsor.png", "PNG", save_all=True, append_images=[second])
        with pytest.raises(OpenAIError, match="must be a static PNG"):
            self.generate(tmp_path, game_dir)
        self.client.request_image.assert_not_called()

    def test_single_frame_apng_is_also_rejected(self, tmp_path: Path, game_dir: Path) -> None:
        metadata = PngImagePlugin.PngInfo()
        metadata.add(b"acTL", struct.pack(">II", 1, 0))
        Image.new("RGBA", (40, 30), "red").save(game_dir / "game-sponsor.png", pnginfo=metadata)
        with pytest.raises(OpenAIError, match="must be a static PNG"):
            self.generate(tmp_path, game_dir)
        self.client.request_image.assert_not_called()

    def test_exif_rotation_rejected_to_keep_source_orientation(self, tmp_path: Path, game_dir: Path) -> None:
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGBA", (40, 30), "red").save(game_dir / "game-sponsor.png", exif=exif)
        with pytest.raises(OpenAIError, match="original display orientation"):
            self.generate(tmp_path, game_dir)
        self.client.request_image.assert_not_called()

    def test_srgb_profile_preserves_pixel_colors(self, tmp_path: Path, game_dir: Path) -> None:
        source = Image.new("RGBA", (30, 20), "#a02b6a")
        srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        source.save(game_dir / "game-sponsor.png", "PNG", icc_profile=srgb.tobytes())

        result = self.generate(tmp_path, game_dir)
        with Image.open(result.image_path) as image:
            assert image.info.get("srgb") == 0
            assert image.crop((843, 588, 1017, 704)).tobytes() == Image.new("RGBA", (174, 116), "#a02b6a").tobytes()

    @pytest.mark.parametrize("valid_profile", [False, True])
    def test_rejects_icc_profile_that_changes_sponsor_colors(
        self, tmp_path: Path, game_dir: Path, valid_profile: bool,
    ) -> None:
        profile_bytes = b"invalid ICC data"
        if valid_profile:
            srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
            profile = bytearray(srgb.tobytes())
            (count,) = struct.unpack_from(">I", profile, 128)
            # Set the RGB tone response curves to linear light to get a real non-sRGB RGB profile.
            for tag in range(132, 132 + 12 * count, 12):
                name, offset, _ = struct.unpack_from(">4sII", profile, tag)
                if name in (b"rTRC", b"gTRC", b"bTRC"):
                    profile[offset:offset + 14] = struct.pack(">4sIIH", b"curv", 0, 1, 256)
                    struct.pack_into(">I", profile, tag + 8, 14)
            profile_bytes = bytes(profile)
        Image.new("RGB", (30, 20), "#7870e2").save(game_dir / "game-sponsor.png", icc_profile=profile_bytes)
        with pytest.raises(OpenAIError, match="must use sRGB colors"):
            self.generate(tmp_path, game_dir)
        self.client.request_image.assert_not_called()

    @pytest.mark.parametrize("color_chunk", [b"cICP", b"cHRM", b"sRGB"])
    def test_sponsor_color_metadata(self, tmp_path: Path, game_dir: Path, color_chunk: bytes) -> None:
        metadata = PngImagePlugin.PngInfo()
        metadata.add(
            color_chunk,
            {
                b"cICP": bytes((1, 13, 0, 1)),
                b"cHRM": struct.pack(">8I", 31270, 32900, 64000, 33000, 30000, 60000, 15000, 5000),
                b"sRGB": b"\x00",
            }[color_chunk],
        )
        Image.new("RGB", (30, 20), "#a02b6a").save(game_dir / "game-sponsor.png", pnginfo=metadata)
        if color_chunk == b"sRGB":
            assert self.generate(tmp_path, game_dir).sponsor_logo_applied
        else:
            with pytest.raises(OpenAIError, match="must use sRGB colors"):
                self.generate(tmp_path, game_dir)
            self.client.request_image.assert_not_called()

    def test_non_srgb_gamma_rejected_to_keep_colors_exact(self, tmp_path: Path, game_dir: Path) -> None:
        metadata = PngImagePlugin.PngInfo()
        metadata.add(b"gAMA", struct.pack(">I", 100000))
        Image.new("RGB", (40, 30), "#a02b6a").save(game_dir / "game-sponsor.png", pnginfo=metadata)
        with pytest.raises(OpenAIError, match="must use sRGB colors"):
            self.generate(tmp_path, game_dir)
        self.client.request_image.assert_not_called()
