"""Dusk-over-Manhattan palette, and the fog that hides the draw distance.

The colour scheme is doing double duty. It sets the PS1 look, and it carries the
verb: anything the player must PUNCH is hot orange and glowing, anything they
must SWING AROUND is dead grey concrete. That split has to stay absolute — at
200km/h the player reads colour long before they read shape, and a moment spent
deciding which verb applies is a moment they are already dead.
"""

from __future__ import annotations

SKY_TOP = (14, 12, 38)
SKY_HORIZON = (92, 54, 74)
FOG = SKY_HORIZON

STREET = (26, 22, 34)
STREET_BAND = (38, 33, 48)

# Muted, desaturated — bright buildings would fight the bombs for attention.
BUILDING_COLORS = (
    (58, 52, 66),
    (48, 44, 60),
    (66, 56, 62),
    (42, 46, 62),
    (60, 50, 54),
)

WINDOW_LIT = (226, 178, 96)
WINDOW_LIT_COOL = (150, 176, 214)

# Per-face brightness. Cheap directional lighting: one warm key from the front.
SHADE_FRONT = 1.00
SHADE_SIDE = 0.72
SHADE_BACK = 0.55
SHADE_TOP = 0.42

# Reserved for the objects that carry a verb.
BOMB = (255, 140, 32)
BOMB_GLOW = (255, 214, 120)
WEB = (238, 242, 250)

# Fog envelope, in world units.
FOG_START = 140.0
FOG_END = 420.0


def shade(color, factor: float):
    return (
        int(color[0] * factor),
        int(color[1] * factor),
        int(color[2] * factor),
    )


def fogged(color, distance: float):
    """Blend towards fog with distance.

    Fog is not decoration here — it is what lets the world generator recycle
    geometry a few hundred units out without the player ever seeing anything
    pop in.
    """
    if distance <= FOG_START:
        return color
    if distance >= FOG_END:
        return FOG
    t = (distance - FOG_START) / (FOG_END - FOG_START)
    inv = 1.0 - t
    return (
        int(color[0] * inv + FOG[0] * t),
        int(color[1] * inv + FOG[1] * t),
        int(color[2] * inv + FOG[2] * t),
    )


def make_sky(width: int, height: int):
    """Vertical gradient, precomputed once.

    Valid only while the camera has no pitch — the horizon is pinned to the
    screen midpoint. Adding pitch or large roll later means this has to become
    real geometry.
    """
    import pygame

    surf = pygame.Surface((width, height)).convert()
    horizon = height // 2
    for y in range(horizon):
        t = y / max(horizon - 1, 1)
        surf.fill(
            (
                int(SKY_TOP[0] * (1 - t) + SKY_HORIZON[0] * t),
                int(SKY_TOP[1] * (1 - t) + SKY_HORIZON[1] * t),
                int(SKY_TOP[2] * (1 - t) + SKY_HORIZON[2] * t),
            ),
            (0, y, width, 1),
        )
    surf.fill(FOG, (0, horizon, width, height - horizon))
    return surf
