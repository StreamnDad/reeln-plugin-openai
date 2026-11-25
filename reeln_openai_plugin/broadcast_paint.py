"""Broadcast styling painted above the generated scene, below the official marks."""

from __future__ import annotations

import re
from itertools import pairwise
from math import ceil

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

RGB = tuple[int, int, int]
Point = tuple[int, int]
SPONSOR_FOOTER_HEIGHT = 148
METAL_STOPS: tuple[tuple[float, RGB], ...] = (
    (0.0, (252, 254, 255)),
    (0.32, (236, 243, 252)),
    (0.46, (171, 187, 207)),
    (0.50, (247, 251, 255)),
    (1.0, (121, 139, 164)),
)


def _gradient(size: tuple[int, int], stops: tuple[tuple[float, RGB], ...]) -> Image.Image:
    """Interpolate metallic highlights down a one-pixel strip and stretch it."""
    width, height = size
    strip = Image.new("RGB", (1, height))
    draw = ImageDraw.Draw(strip)
    for (start, light), (end, dark) in pairwise(stops):
        first, last = round(start * (height - 1)), round(end * (height - 1))
        for y in range(first, last + 1):
            blend = (y - first) / max(1, last - first)
            red, green, blue = (round(a + (b - a) * blend) for a, b in zip(light, dark, strict=True))
            draw.point((0, y), fill=(red, green, blue))
    return strip.resize((width, height), Image.Resampling.NEAREST)


def _plate(canvas: Image.Image, points: tuple[Point, ...], accent: RGB, opacity: int = 245) -> None:
    """Draw an angular glass-and-metal plate with a highlight and a recessed edge."""
    left, right = min(x for x, _ in points), max(x for x, _ in points)
    top, bottom = min(y for _, y in points), max(y for _, y in points)
    size = (right - left + 1, bottom - top + 1)
    shape = tuple((x - left, y - top) for x, y in points)
    tint = tuple(18 + round(value * 0.22) for value in accent)
    color = (tint[0], tint[1], tint[2])
    fill = _gradient(size, ((0.0, color), (0.30, (12, 19, 31)), (0.80, (7, 12, 20)), (1.0, color)))
    mask = Image.new("L", size)
    ImageDraw.Draw(mask).polygon(shape, fill=opacity)
    fill.putalpha(mask)
    canvas.alpha_composite(fill, (left, top))

    closed = (*points, points[0])
    draw = ImageDraw.Draw(canvas)
    draw.line(closed, fill="#030910", width=8, joint="curve")
    draw.line(closed, fill=accent, width=4, joint="curve")
    draw.line(closed, fill="#c0cfe2", width=1, joint="curve")
    draw.line((points[0], points[1]), fill="#eff6ff", width=2)


def _strap(left: int, right: int, top: int, height: int, bevel: int = 28) -> tuple[Point, ...]:
    return (left, top), (right, top), (right - bevel, top + height), (left + bevel, top + height)


def _text_mask(text: str, width: int, size: int, slant: float) -> Image.Image:
    font = ImageFont.load_default(size)
    stroke = 2 if size >= 36 else 1
    left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
    mask = Image.new("L", (max(1, ceil(right - left)) + 12, max(1, ceil(bottom - top)) + 12))
    ImageDraw.Draw(mask).text((6 - left, 6 - top), text, font=font, fill=255, stroke_width=stroke)
    shear = ceil(mask.height * slant)
    target_width = min(width - shear, max(1, round(mask.width * 0.84)))
    mask = mask.resize((target_width, mask.height), Image.Resampling.LANCZOS)
    return mask.transform(
        (mask.width + shear, mask.height),
        Image.Transform.AFFINE,
        (1, slant, -shear, 0, 1, 0),
        Image.Resampling.BICUBIC,
    )


def _metal_text(
    canvas: Image.Image,
    text: str,
    center: Point,
    width: int,
    size: int,
    *,
    slant: float = 0.17,
) -> None:
    if not text:
        return
    # Fit long team and venue names by decreasing the font before condensing glyphs.
    font = ImageFont.load_default(size)
    while size > 16 and font.getlength(text) * 0.84 + size * slant > width:
        size -= 2
        font = ImageFont.load_default(size)
    mask = _text_mask(text, width, size, slant)
    x, y = center[0] - mask.width // 2, center[1] - mask.height // 2
    shadow_box = (x + 3, y + 5, x + 3 + mask.width, y + 5 + mask.height)
    canvas.paste("#020509", shadow_box, mask.filter(ImageFilter.MaxFilter(9)))
    canvas.paste("#26364c", (x, y, x + mask.width, y + mask.height), mask.filter(ImageFilter.MaxFilter(5)))
    canvas.paste(_gradient(mask.size, METAL_STOPS), (x, y), mask)


def _artwork_luminance(artwork: Image.Image, alpha: Image.Image) -> float:
    mostly_opaque = alpha.point(lambda opacity: 255 if opacity >= 128 else 0)
    mask = alpha if mostly_opaque.getbbox() is None else mostly_opaque
    return ImageStat.Stat(artwork.convert("L"), mask).median[0]


def _team_logo(canvas: Image.Image, logo: Image.Image | None, center: Point) -> None:
    if logo is None:
        return
    # A bloom behind dark transparent marks provides contrast without changing their pixels.
    alpha = logo.getchannel("A")
    x, y = center
    if alpha.histogram()[255] != logo.width * logo.height and alpha.getbbox() and _artwork_luminance(logo, alpha) < 95:
        mask = Image.new("L", canvas.size)
        ImageDraw.Draw(mask).ellipse((x - 172, y - 121, x + 172, y + 121), fill=156)
        glow = Image.new("RGBA", canvas.size, "#b0c7e0")
        glow.putalpha(mask.filter(ImageFilter.GaussianBlur(38)))
        canvas.alpha_composite(glow)
    canvas.alpha_composite(logo, (x - logo.width // 2, y - logo.height // 2))


def _team_accent(colors: str, fallback: RGB) -> RGB:
    for code in re.findall(r"#[\da-fA-F]{6}\b", colors):
        red, green, blue = (int(code[index : index + 2], 16) for index in (1, 3, 5))
        if max(red, green, blue) - min(red, green, blue) > 30:
            boost = 1 + max(0, 162 - max(red, green, blue)) / max(1, max(red, green, blue))
            return min(255, round(red * boost)), min(255, round(green * boost)), min(255, round(blue * boost))
    return fallback


def _versus_beam(canvas: Image.Image, name_top: int) -> None:
    # A bright diagonal seam links both team colors behind the center matchup mark.
    seam = Image.new("RGBA", canvas.size)
    draw = ImageDraw.Draw(seam)
    draw.line(((812, 49), (522, name_top + 93)), fill=(115, 178, 252, 215), width=16)
    draw.line(((801, 56), (509, name_top + 65)), fill=(255, 110, 111, 166), width=8)
    canvas.alpha_composite(seam.filter(ImageFilter.GaussianBlur(13)))
    ImageDraw.Draw(canvas).line(((805, 67), (528, name_top + 81)), fill="#a9d1ff", width=2)


def draw_broadcast_matchup(
    canvas: Image.Image,
    home_logo: Image.Image | None,
    away_logo: Image.Image | None,
    variables: dict[str, str],
    *,
    sponsored: bool,
) -> None:
    """Frame the matchup, then compose the official art and exact metadata above it."""
    content_bottom = canvas.height - SPONSOR_FOOTER_HEIGHT if sponsored else canvas.height - 12
    name_top = round(content_bottom * 0.64)
    logo_y = (name_top + 91) // 2
    home = _team_accent(variables["home_colors"], (221, 43, 53))
    away = _team_accent(variables["away_colors"], (43, 126, 227))
    steel = (78, 105, 140)

    left = ((-37, 75), (483, 116), (588, logo_y + 27), (494, name_top - 4), (-37, name_top - 39))
    right = tuple((1280 - x, y) for x, y in reversed(left))
    _plate(canvas, left, home, 172)
    _plate(canvas, right, away, 172)
    _versus_beam(canvas, name_top)

    _team_logo(canvas, home_logo, (288, logo_y))
    _team_logo(canvas, away_logo, (992, logo_y))
    _metal_text(canvas, "VS", (640, logo_y + 16), 224, 143, slant=0.29)

    _plate(canvas, _strap(372, 908, 22, 56, 34), steel)
    header = variables["sport"].replace("_", " ").replace("-", " ").upper() or "GAME DAY"
    _metal_text(canvas, header, (640, 50), 410, 40)

    for center, label, accent in (
        (323, variables["home_team"], home),
        (957, variables["away_team"], away),
    ):
        _plate(canvas, _strap(center - 300, center + 300, name_top, 62), accent)
        _metal_text(canvas, label.upper(), (center, name_top + 31), 501, 46, slant=0.10)

    match_top = name_top + 68
    level = variables["level"] or variables["game_level"]
    event = "  /  ".join(dict.fromkeys(text.upper() for text in (level, variables["tournament"]) if text))
    if event:
        _plate(canvas, _strap(73, 1207, match_top, 52), steel)
        _metal_text(canvas, event, (640, match_top + 26), 1000, 36)
        match_top += 58

    timing = (variables["rink"], variables["game_date"], variables["game_time"])
    details = "   /   ".join(text for text in timing if text)
    if details:
        _plate(canvas, _strap(134, 1146, match_top, 42, 17), steel)
        _metal_text(canvas, details.upper(), (640, match_top + 21), 922, 27, slant=0.08)


def draw_sponsor_footer(canvas: Image.Image, sponsor: Image.Image) -> None:
    """Add the natural-reading sponsor credit and the approved mark in a protected footer."""
    top = canvas.height - SPONSOR_FOOTER_HEIGHT
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, top, canvas.width, canvas.height), fill="#0a141f")
    _plate(canvas, _strap(38, canvas.width - 38, top + 6, SPONSOR_FOOTER_HEIGHT - 14, 43), (69, 99, 133))

    center_y = top + SPONSOR_FOOTER_HEIGHT // 2
    font = ImageFont.load_default(38)
    for line, offset in (("This game is brought", -22), ("to you in part by", 22)):
        ImageDraw.Draw(canvas).text((477, center_y + offset), line, font=font, fill="#f1f6ff", anchor="mm")

    x, y = 930 - sponsor.width // 2, center_y - sponsor.height // 2
    luminance = _artwork_luminance(sponsor, sponsor.getchannel("A"))
    backing = "#22272e" if luminance >= 127 else "#e9ebee"
    ImageDraw.Draw(canvas).rounded_rectangle(
        (x - 11, y - 7, x + sponsor.width + 11, y + sponsor.height + 7), radius=7, fill=backing
    )
    canvas.alpha_composite(sponsor, (x, y))
