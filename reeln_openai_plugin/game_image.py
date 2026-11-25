"""Game image thumbnail generation via OpenAI image generation tool."""

from __future__ import annotations

import io
import os
import re
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from PIL import (
    Image,
    ImageChops,
    ImageCms,
    ImageFilter,
    ImageOps,
    PngImagePlugin,
    UnidentifiedImageError,
)

from reeln_openai_plugin.broadcast_paint import SPONSOR_FOOTER_HEIGHT, draw_broadcast_matchup, draw_sponsor_footer
from reeln_openai_plugin.client import OpenAIClient, OpenAIError
from reeln_openai_plugin.prompts import PromptRegistry

IMAGE_TIMEOUT_SECONDS: float = 120.0
TARGET_WIDTH: int = 1280
TARGET_HEIGHT: int = 720
SPONSOR_FILENAME = "game-sponsor.png"
SPONSOR_FOOTER_MARKER = "reeln_sponsor_footer"
SPONSOR_LOGO_FIT_SIZE = (328, 116)
SPONSOR_MAX_EDGE = 8192
SPONSOR_MAX_PIXELS = 16_000_000
SPONSOR_MAX_FILE_BYTES = 32 * 1024 * 1024
TEAM_LOGO_MAX_SIZE = (420, 245)
PNG_IHDR_BIT_DEPTH_OFFSET = 24
PNG_SRGB_GAMMA = 45455
PNG_SRGB_CHROMATICITY = (31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000)
SPONSOR_COLOR_ERROR = "Sponsor logo must use sRGB colors. Supply an sRGB PNG from the sponsor."

BACKGROUND_INSTRUCTIONS = (
    "Generate the background artwork only. The game details, typography, and official "
    "logos are added locally afterward. The background must contain no text, numerals, "
    "logos, team marks, sponsor marks, crests, badges, or logo placeholders. "
    "These artwork-only requirements apply to all preceding directions."
)


@dataclass(frozen=True)
class GameImageResult:
    """Path to the generated game thumbnail image."""

    image_path: Path
    team_logos_composited: bool = False
    sponsor_logo_applied: bool = False


def _fit_background(raw_bytes: bytes) -> Image.Image:
    """Center-crop the generated artwork to 16:9 before adding the logo layers."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as image:
            return ImageOps.fit(
                image.convert("RGBA"),
                (TARGET_WIDTH, TARGET_HEIGHT),
                Image.Resampling.LANCZOS,
            )
    except (OSError, Image.DecompressionBombError) as exc:
        raise OpenAIError(f"Cannot decode image from API response: {exc}") from exc


def _load_team_logo(path: Path) -> Image.Image:
    try:
        with Image.open(path.expanduser()) as logo:
            artwork = logo.convert("RGBA")
            artwork.thumbnail(TEAM_LOGO_MAX_SIZE, Image.Resampling.LANCZOS)
            return artwork
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise OpenAIError(f"Cannot read logo file {path}: {exc}") from exc


def _validate_sponsor_colors(logo: Image.Image, artwork: Image.Image) -> None:
    """Check that the source pixels can keep their appearance on the sRGB canvas."""
    if profile_bytes := logo.info.get("icc_profile"):
        try:
            profile = ImageCms.ImageCmsProfile(io.BytesIO(profile_bytes))
            converted = ImageCms.profileToProfile(
                artwork,
                profile,
                ImageCms.createProfile("sRGB"),
                outputMode="RGBA",
            )
        except (OSError, ValueError, ImageCms.PyCMSError) as exc:
            raise OpenAIError(SPONSOR_COLOR_ERROR) from exc
        assert converted is not None  # ImageCms returns None only when transforming in place.
        if ImageChops.difference(converted, artwork).getbbox(alpha_only=False) is not None:
            raise OpenAIError(SPONSOR_COLOR_ERROR)
        return

    if "srgb" in logo.info:
        return
    gamma = logo.info.get("gamma")
    if gamma is not None and abs(round(gamma * 100_000) - PNG_SRGB_GAMMA) > 1:
        raise OpenAIError(SPONSOR_COLOR_ERROR)
    chromaticity = logo.info.get("chromaticity")
    if chromaticity is not None and (
        len(chromaticity) != len(PNG_SRGB_CHROMATICITY)
        or any(
            abs(round(actual * 100_000) - standard) > 1
            for actual, standard in zip(chromaticity, PNG_SRGB_CHROMATICITY, strict=True)
        )
    ):
        raise OpenAIError(SPONSOR_COLOR_ERROR)


def _png_chunk_types(raw_bytes: bytes) -> set[bytes]:
    """Read chunk names so unsupported display metadata cannot be silently ignored."""
    offset = 8  # PNG signature
    types = set()
    while offset + 12 <= len(raw_bytes):
        size = int.from_bytes(raw_bytes[offset : offset + 4], "big")
        kind = raw_bytes[offset + 4 : offset + 8]
        types.add(kind)
        offset += size + 12  # length, type, payload, CRC
        if kind == b"IEND" and offset == len(raw_bytes):
            return types
    raise OpenAIError("Choose a complete PNG file for the sponsor logo.")


def _load_sponsor(game_dir: Path | None, sponsor_logo_path: Path | None) -> Image.Image | None:
    game_sponsor_path = game_dir / SPONSOR_FILENAME if game_dir is not None else None
    if game_sponsor_path is not None and game_sponsor_path.is_symlink():
        raise OpenAIError(f"Sponsor logo must be a regular PNG file: {game_sponsor_path}")
    if game_sponsor_path is not None and game_sponsor_path.exists():
        path = game_sponsor_path
    elif sponsor_logo_path is not None:
        path = sponsor_logo_path.expanduser()
    else:
        return None

    try:
        if not path.is_file():
            raise OpenAIError(f"Sponsor logo must be a regular PNG file: {path}")
        with path.open("rb") as source:
            raw_bytes = source.read(SPONSOR_MAX_FILE_BYTES + 1)
        if len(raw_bytes) > SPONSOR_MAX_FILE_BYTES:
            raise OpenAIError("Sponsor logo must be no larger than 32 MiB.")
        with Image.open(io.BytesIO(raw_bytes), formats=("PNG",)) as logo:
            if max(logo.size) > SPONSOR_MAX_EDGE or logo.width * logo.height > SPONSOR_MAX_PIXELS:
                raise OpenAIError("Sponsor logo must be at most 8192 pixels per edge and 16 million pixels total.")
            chunks = _png_chunk_types(raw_bytes)
            if b"acTL" in chunks:
                raise OpenAIError("Sponsor logo must be a static PNG")
            if b"cICP" in chunks:
                raise OpenAIError(SPONSOR_COLOR_ERROR)
            if raw_bytes[PNG_IHDR_BIT_DEPTH_OFFSET] == 16:
                raise OpenAIError("Sponsor logo must be an 8-bit PNG")
            sponsor = logo.convert("RGBA")
            if logo.getexif().get(274, 1) != 1:
                raise OpenAIError("Sponsor logo must be stored in its original display orientation")
            _validate_sponsor_colors(logo, sponsor)
            sponsor = _fit_sponsor(sponsor)
            if sponsor.getbbox() is None:
                raise OpenAIError("Sponsor logo must contain visible artwork at thumbnail size.")
        if game_sponsor_path is not None and path != game_sponsor_path:
            _write_atomic(game_sponsor_path, raw_bytes)
        return sponsor
    except UnidentifiedImageError as exc:
        raise OpenAIError("Sponsor logo must be a static PNG") from exc
    except Image.DecompressionBombError as exc:
        raise OpenAIError("Sponsor logo must be at most 8192 pixels per edge and 16 million pixels total.") from exc
    except (OSError, ValueError) as exc:
        raise OpenAIError(f"Cannot read sponsor logo {path}: {exc}") from exc


def _fit_sponsor(artwork: Image.Image) -> Image.Image:
    """Fit the entire source canvas proportionally into the footer."""
    return _contain_artwork(artwork, SPONSOR_LOGO_FIT_SIZE)


def _contain_artwork(artwork: Image.Image, bounds: tuple[int, int]) -> Image.Image:
    scale = min(bounds[0] / artwork.width, bounds[1] / artwork.height)
    size = (max(1, round(artwork.width * scale)), max(1, round(artwork.height * scale)))
    return artwork.resize(size, Image.Resampling.LANCZOS)


def _compose_game_image(
    canvas: Image.Image,
    home_logo: Image.Image | None,
    away_logo: Image.Image | None,
    sponsor: Image.Image | None,
    variables: dict[str, str],
) -> None:
    draw_broadcast_matchup(canvas, home_logo, away_logo, variables, sponsored=sponsor is not None)
    if sponsor is not None:
        draw_sponsor_footer(canvas, sponsor)


def _build_prompt_variables(
    home: object,
    away: object,
    rink: str,
    game_date: str,
    game_time: str,
    level: str,
    description: str,
    tournament: str,
    sport: str,
) -> dict[str, str]:
    """Extract template variables from team objects and game info."""
    return {
        "home_team": str(getattr(home, "team_name", "")),
        "away_team": str(getattr(away, "team_name", "")),
        "home_colors": str(getattr(home, "colors", "")),
        "away_colors": str(getattr(away, "colors", "")),
        "game_level": str(getattr(home, "level", "")),
        "rink": rink,
        "venue": rink,
        "sport": sport,
        "game_date": game_date,
        "date": game_date,
        "game_time": game_time,
        "level": level,
        "description": description,
        "tournament": tournament,
    }


def _slugify(name: str) -> str:
    """Convert a name to a filesystem-safe slug."""
    return re.sub(r"[^\w-]", "_", name).lower()


def _game_path_identity(canonical: str) -> str:
    """Strip Windows verbatim path prefixes to match Dock's canonical path hash."""
    if canonical.startswith("\\\\?\\UNC\\"):
        return "\\\\" + canonical[8:]
    return canonical.removeprefix("\\\\?\\")


def _per_game_output_dir(output_dir: Path, game_dir: Path) -> Path:
    canonical = game_dir.resolve()
    if not canonical.name:
        raise OpenAIError("A game directory with a name is required")
    identity = _game_path_identity(str(canonical))
    scope = sha256(identity.encode("utf-8")).hexdigest()[:16]
    return output_dir.resolve() / "games" / canonical.name / scope


def _output_path(output_dir: Path, game_dir: Path | None, home: object, away: object, game_date: str) -> Path:
    if game_dir is not None:
        return _per_game_output_dir(output_dir, game_dir) / "game-image.png"

    home_slug = _slugify(str(getattr(home, "short_name", getattr(home, "team_name", "home"))))
    away_slug = _slugify(str(getattr(away, "short_name", getattr(away, "team_name", "away"))))
    date_slug = _slugify(game_date.replace("/", "-"))
    return output_dir / f"{date_slug}_{home_slug}_vs_{away_slug}.png"


def _write_atomic(output_path: Path, contents: bytes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=output_path.parent)
    temp_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as candidate:
            candidate.write(contents)
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def generate_game_image(
    client: OpenAIClient,
    prompt_registry: PromptRegistry,
    home: object,
    away: object,
    rink: str,
    game_date: str,
    game_time: str,
    output_dir: Path,
    model: str,
    renderer_model: str,
    level: str = "",
    description: str = "",
    tournament: str = "",
    game_dir: Path | None = None,
    sponsor_logo_path: Path | None = None,
    *,
    sport: str = "",
    prompt_override: str | None = None,
) -> GameImageResult:
    """Generate a game thumbnail image and save it as a 1280x720 PNG.

    OpenAI renders the background. The app adds game details and the explicitly
    assigned team and sponsor artwork locally.
    """
    home_logo_path = getattr(home, "logo_path", None)
    away_logo_path = getattr(away, "logo_path", None)
    home_logo = _load_team_logo(Path(home_logo_path)) if home_logo_path else None
    away_logo = _load_team_logo(Path(away_logo_path)) if away_logo_path else None
    out_path = _output_path(output_dir.expanduser(), game_dir, home, away, game_date)
    sponsor = _load_sponsor(game_dir, sponsor_logo_path)

    variables = _build_prompt_variables(
        home,
        away,
        rink,
        game_date,
        game_time,
        level,
        description,
        tournament,
        sport,
    )
    prompt = prompt_registry.render("game_image", variables, template_text=prompt_override)

    raw_bytes = client.request_image(
        prompt=f"{prompt}\n\n{BACKGROUND_INSTRUCTIONS}",
        images=[],
        model_override=model,
        renderer_model=renderer_model,
        renderer_size="1536x1024",
        output_format="png",
        timeout_override=IMAGE_TIMEOUT_SECONDS,
    )

    canvas = _fit_background(raw_bytes)
    _compose_game_image(canvas, home_logo, away_logo, sponsor, variables)
    _write_game_png(out_path, canvas, sponsored=sponsor is not None)
    return GameImageResult(
        image_path=out_path,
        team_logos_composited=home_logo is not None and away_logo is not None,
        sponsor_logo_applied=sponsor is not None,
    )


def _write_game_png(out_path: Path, canvas: Image.Image, *, sponsored: bool) -> None:
    encoded = io.BytesIO()
    png_info = PngImagePlugin.PngInfo()
    png_info.add(b"sRGB", b"\x00")
    if sponsored:
        png_info.add_text(SPONSOR_FOOTER_MARKER, "1")
    canvas.save(encoded, "PNG", pnginfo=png_info, icc_profile=None)
    try:
        _write_atomic(out_path, encoded.getvalue())
    except OSError as exc:
        raise OpenAIError(f"Cannot save game image to {out_path}: {exc}") from exc


def _load_existing_artwork(path: Path) -> Image.Image:
    """Decode a finished graphic using its original framing and orientation."""
    try:
        with path.open("rb") as file:
            data = file.read(SPONSOR_MAX_FILE_BYTES + 1)
        if len(data) > SPONSOR_MAX_FILE_BYTES:
            raise OpenAIError("The existing game image must be no larger than 32 MiB.")
        with Image.open(io.BytesIO(data), formats=("PNG", "JPEG", "WEBP")) as image:
            if image.info.get(SPONSOR_FOOTER_MARKER) == "1":
                raise OpenAIError("Choose the original game graphic instead of the already sponsored copy.")
            if getattr(image, "n_frames", 1) != 1:
                raise OpenAIError("Choose a still game image in PNG, JPG, or WebP format.")
            if max(image.size) > SPONSOR_MAX_EDGE or image.width * image.height > SPONSOR_MAX_PIXELS:
                raise OpenAIError("The existing game image must be at most 8192 pixels per edge and 16 megapixels.")
            oriented = ImageOps.exif_transpose(image)
            artwork = oriented.convert("RGBA")
            if profile := oriented.info.get("icc_profile"):
                source = oriented if oriented.mode in {"RGB", "RGBA", "L", "CMYK", "LAB"} else artwork
                try:
                    converted = ImageCms.profileToProfile(
                        source,
                        ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                        ImageCms.createProfile("sRGB"),
                        outputMode="RGBA",
                    )
                except (OSError, ValueError, ImageCms.PyCMSError) as exc:
                    raise OpenAIError("Export the existing game image in sRGB and try again.") from exc
                assert converted is not None  # ImageCms returns None only when transforming in place.
                converted.putalpha(artwork.getchannel("A"))
                return converted
            return artwork
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise OpenAIError(f"Cannot read the existing game image {path}: {exc}") from exc


def add_sponsor_to_existing_game_image(source: Path, game_dir: Path, output_dir: Path) -> Path:
    """Place the whole finished graphic above a sponsor footer without an API call."""
    out_path = _per_game_output_dir(output_dir.expanduser(), game_dir) / "sponsored-game-image.png"
    source = source.expanduser().resolve()
    if source == out_path.resolve():
        raise OpenAIError("Choose the original game graphic instead of the already sponsored copy.")

    sponsor = _load_sponsor(game_dir, None)
    if sponsor is None:
        raise OpenAIError("Choose a sponsor logo for this game first.")

    artwork = _load_existing_artwork(source)
    artwork_area = (TARGET_WIDTH, TARGET_HEIGHT - SPONSOR_FOOTER_HEIGHT)
    fitted = _contain_artwork(artwork, artwork_area)

    canvas = Image.new("RGBA", (TARGET_WIDTH, TARGET_HEIGHT), "#090e1a")
    # Surround the complete, sharp graphic with a soft continuation of its own colors.
    backdrop = ImageOps.fit(artwork, artwork_area, Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(50))
    canvas.alpha_composite(backdrop)
    canvas.alpha_composite(Image.new("RGBA", artwork_area, (5, 9, 20, 168)))
    canvas.alpha_composite(fitted, ((artwork_area[0] - fitted.width) // 2, (artwork_area[1] - fitted.height) // 2))
    draw_sponsor_footer(canvas, sponsor)
    _write_game_png(out_path, canvas, sponsored=True)
    return out_path
