"""The player figure and the web line.

Deliberately a handful of flat-shaded boxes. At the speeds involved you read the
silhouette and the colour, never the detail, and a chunky red-and-blue shape
against grey concrete is exactly the PS1 register the rest of the renderer is in.
"""

from __future__ import annotations

import math

from .palette import WEB
from .projection import box_faces, project_segment

SUIT_RED = (196, 40, 46)
SUIT_RED_DARK = (138, 26, 32)
SUIT_BLUE = (44, 62, 148)
SUIT_BLUE_DARK = (28, 40, 104)

# Shades per face, so the figure reads as solid rather than as a flat blob.
_FACE_SHADE = {
    "front": 1.0,
    "back": 0.55,
    "left": 0.75,
    "right": 0.75,
    "top": 0.88,
    "bottom": 0.45,
}


def _box(cx, cy, cz, w, h, d, colour, dark):
    faces = box_faces(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2,
                      cz - d / 2, cz + d / 2)
    out = []
    for name, pts in faces.items():
        s = _FACE_SHADE[name]
        base = colour if s > 0.6 else dark
        out.append((pts, (int(base[0] * s), int(base[1] * s), int(base[2] * s))))
    return out


def player_faces(sim) -> list:
    """Boxes for the figure, in world space.

    Limbs swing back when moving fast and tuck when attached, which costs four
    lines and does most of the work of making the figure look like it is
    actually being flung around rather than sliding through the air.
    """
    x, y, z = sim.x, sim.y, sim.z

    speed_t = min(sim.speed / 120.0, 1.0)
    tuck = 0.55 if sim.attached else 0.0
    arm_up = 1.1 * tuck
    leg_back = 0.8 * speed_t

    out = []
    # torso
    out += _box(x, y, z, 1.5, 2.2, 1.0, SUIT_RED, SUIT_RED_DARK)
    # head
    out += _box(x, y + 1.7, z + 0.1, 1.0, 1.0, 1.0, SUIT_RED, SUIT_RED_DARK)
    # arms — raised towards the web when attached
    for sx in (-1, 1):
        out += _box(x + sx * 1.15, y + 0.5 + arm_up, z - 0.1,
                    0.55, 1.7, 0.55, SUIT_RED, SUIT_RED_DARK)
    # legs — trail behind with speed
    for sx in (-1, 1):
        out += _box(x + sx * 0.45, y - 1.9, z - leg_back,
                    0.6, 1.9, 0.6, SUIT_BLUE, SUIT_BLUE_DARK)
    return out


def draw_web(surface, sim, cam, half_w, half_h, focal, cos_r, sin_r) -> None:
    """The line from hand to anchor.

    Width scales with proximity so it does not vanish into a single pixel at
    distance, and the anchor gets a small flare so you can see *what* you caught
    — which is the feedback that teaches anchor placement.
    """
    if sim.anchor is None:
        return
    import pygame

    a = sim.anchor
    hand = (sim.x, sim.y + 1.2, sim.z)
    seg = project_segment(hand, (a.x, a.y, a.z), cam,
                          half_w, half_h, focal, cos_r, sin_r)
    if seg is None:
        return

    dist = math.dist((a.x, a.y, a.z), (sim.x, sim.y, sim.z))
    width = max(1, int(4.0 - dist / 34.0))
    pygame.draw.line(surface, WEB, seg[0], seg[1], width)

    # Anchor flare, sized by distance so it stays visible but never dominates.
    r = max(2, int(7.0 - dist / 18.0))
    pygame.draw.circle(surface, WEB, (int(seg[1][0]), int(seg[1][1])), r)
