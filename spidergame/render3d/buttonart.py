"""Rounded-plate button artwork for the Panda3D menus.

The look comes from the supplied reference art: a red plate with a heavy blue
outline, generously rounded corners, and white uppercase text centred in both
axes.

Two sources, in priority order:

1. A PNG in ``assets/ui/buttons``. Those files have their label baked into the
   artwork, so a button backed by one draws no text of its own — overlaying
   would double it up.
2. A generated plate in the same style, with Panda3D drawing the label live.

The generated path is what makes the text alignment exact. A baked label is
only ever as centred as the exporter left it, whereas live text is positioned
by :data:`TEXT_VCENTER` against the plate's true middle.

Nothing here imports Panda3D at module scope, so the geometry helpers stay
importable (and testable) on machines without it.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "ui" / "buttons"

# Sampled from the reference art, which is the *selected* state. The idle plate
# is pushed well darker rather than slightly darker: with four buttons stacked,
# a subtle difference leaves the player unable to tell what is highlighted.
FILL_IDLE = (132, 24, 28)
FILL_ACTIVE = (255, 51, 51)
BORDER_IDLE = (18, 0, 96)
BORDER_ACTIVE = (46, 12, 232)
TEXT_IDLE = (0.78, 0.82, 0.92, 1.0)
TEXT_ACTIVE = (1.0, 1.0, 1.0, 1.0)

# Panda3D places a text node's *baseline* at its position, so text asked to sit
# at y=0 renders visibly above the centre of its plate. Uppercase cap height in
# the default font is close to 0.72 of the text scale, so shifting down by half
# of that centres the glyph body. Multiply by the text scale in the same
# coordinate space as the offset.
TEXT_VCENTER = -0.36

# Corner radius and outline as fractions of the plate's short edge, chosen to
# match the reference art at any plate size.
RADIUS_RATIO = 0.30
BORDER_RATIO = 0.11


def slug(label: str) -> str:
    """``"BACK TO MENU"`` -> ``"back-to-menu"``, the PNG naming convention."""
    cleaned = [c.lower() if c.isalnum() else "-" for c in label.strip()]
    parts = "".join(cleaned).split("-")
    return "-".join(part for part in parts if part)


def image_path(label: str) -> Path | None:
    """Bespoke artwork for this label, if it has been supplied."""
    candidate = ASSET_DIR / f"{slug(label)}.png"
    return candidate if candidate.is_file() else None


def plate_coverage(x: float, y: float, width: float, height: float,
                   radius: float, border: float) -> tuple[float, float]:
    """Return ``(alpha, fill_weight)`` for one pixel of a rounded plate.

    Uses a signed distance field rather than supersampling: it gives smooth
    edges from a single sample per pixel, which keeps generating a plate cheap
    enough to do at startup.

    ``alpha`` fades across the outer edge; ``fill_weight`` is 1 in the red
    centre and 0 in the blue outline, fading across that boundary too.
    """
    half_w = width / 2.0
    half_h = height / 2.0
    qx = abs(x) - (half_w - radius)
    qy = abs(y) - (half_h - radius)
    outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
    distance = outside + min(max(qx, qy), 0.0) - radius

    alpha = min(max(0.5 - distance, 0.0), 1.0)
    fill = min(max(0.5 - (distance + border), 0.0), 1.0)
    return alpha, fill


def _plate_image(width: int, height: int, fill: tuple[int, int, int],
                 border: tuple[int, int, int]):
    from panda3d.core import PNMImage

    image = PNMImage(width, height, 4)
    image.fill(0.0, 0.0, 0.0)
    image.alphaFill(0.0)

    short_edge = min(width, height)
    radius = short_edge * RADIUS_RATIO
    stroke = short_edge * BORDER_RATIO
    half_w, half_h = width / 2.0, height / 2.0

    for py in range(height):
        y = py + 0.5 - half_h
        for px in range(width):
            x = px + 0.5 - half_w
            alpha, weight = plate_coverage(x, y, width, height, radius, stroke)
            if alpha <= 0.0:
                continue
            r = (fill[0] * weight + border[0] * (1.0 - weight)) / 255.0
            g = (fill[1] * weight + border[1] * (1.0 - weight)) / 255.0
            b = (fill[2] * weight + border[2] * (1.0 - weight)) / 255.0
            image.setXelA(px, py, r, g, b, alpha)
    return image


@lru_cache(maxsize=16)
def plate_texture(width: int, height: int, active: bool):
    """A generated plate, cached so repeated buttons share one texture."""
    from panda3d.core import Texture

    image = _plate_image(
        width,
        height,
        FILL_ACTIVE if active else FILL_IDLE,
        BORDER_ACTIVE if active else BORDER_IDLE,
    )
    texture = Texture(f"button-plate-{width}x{height}-{int(active)}")
    texture.load(image)
    texture.setMagfilter(Texture.FTLinear)
    texture.setMinfilter(Texture.FTLinearMipmapLinear)
    # Clamp so the transparent margin never tiles back over the plate edge.
    texture.setWrapU(Texture.WMClamp)
    texture.setWrapV(Texture.WMClamp)
    return texture


@lru_cache(maxsize=16)
def label_texture(label: str):
    """Supplied artwork for ``label``, or ``None`` when there is none."""
    path = image_path(label)
    if path is None:
        return None
    from panda3d.core import Texture, TexturePool

    texture = TexturePool.loadTexture(str(path))
    if texture is None:
        return None
    texture.setMagfilter(Texture.FTLinear)
    texture.setMinfilter(Texture.FTLinearMipmapLinear)
    return texture


def button_art(label: str, width: int, height: int, active: bool):
    """``(texture, draws_own_text)`` for a menu button.

    ``draws_own_text`` is False when supplied artwork already carries the
    label, so the caller knows to suppress its text node.
    """
    supplied = label_texture(label)
    if supplied is not None:
        return supplied, False
    return plate_texture(width, height, active), True


__all__ = [
    "ASSET_DIR",
    "TEXT_VCENTER",
    "TEXT_IDLE",
    "TEXT_ACTIVE",
    "button_art",
    "image_path",
    "label_texture",
    "plate_coverage",
    "plate_texture",
    "slug",
]
