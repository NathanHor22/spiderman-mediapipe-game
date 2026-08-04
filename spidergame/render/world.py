"""Procedural street canyon that generates ahead and retires behind.

Two independent rows of buildings, one per side, each with its own z cursor and
its own rhythm of widths and gaps. Keeping the rows uncorrelated matters — if
both sides stepped together the canyon would read as a repeating tunnel, and the
gaps between buildings are where anchor points and sight lines come from.

Everything a building needs for rendering is baked once at spawn. Rebuilding
face tuples every frame would dominate the frame time for no reason: the
geometry is static, only the camera moves.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .palette import (
    BUILDING_COLORS,
    SHADE_BACK,
    SHADE_FRONT,
    SHADE_SIDE,
    SHADE_TOP,
    WINDOW_LIT,
    WINDOW_LIT_COOL,
    shade,
)
from .projection import box_faces

STREET_HALF = 14.0  # half-width of the canyon the player flies down

# Only the inner faces get windows, and only up close. Windows are the single
# biggest polygon-count risk in the renderer, so they are budgeted twice: by
# distance here, and by a hard per-frame cap in the renderer.
WINDOW_DISTANCE = 170.0
WINDOW_ROW_STEP = 6.0
WINDOW_COL_STEP = 7.0
WINDOW_W = 2.4
WINDOW_H = 3.2
WINDOW_LIT_CHANCE = 0.34


@dataclass
class Building:
    x0: float
    x1: float
    z0: float
    z1: float
    height: float
    side: int  # -1 left of the street, +1 right

    faces: tuple = field(default_factory=tuple)  # ((pts, colour), ...)
    windows: tuple = field(default_factory=tuple)

    @property
    def inner_x(self) -> float:
        """The face that looks onto the street — where webs will anchor."""
        return self.x1 if self.side < 0 else self.x0


def _build_faces(b: Building, base) -> tuple:
    f = box_faces(b.x0, b.x1, 0.0, b.height, b.z0, b.z1)
    inner = "right" if b.side < 0 else "left"
    # The outer side face and the bottom can never be seen from inside the
    # canyon, so they are never built. Back-face culling would reject them
    # anyway; not creating them saves the test.
    return (
        (f["front"], shade(base, SHADE_FRONT)),
        (f["back"], shade(base, SHADE_BACK)),
        (f[inner], shade(base, SHADE_SIDE)),
        (f["top"], shade(base, SHADE_TOP)),
    )


def _build_windows(b: Building, rng: random.Random) -> tuple:
    # Nudged off the wall towards the street so the painter's algorithm has an
    # unambiguous order: wall first, then windows on top of it.
    x = b.inner_x + (0.06 if b.side < 0 else -0.06)
    out = []

    y = 8.0
    while y + WINDOW_H < b.height - 4.0:
        z = b.z0 + 3.0
        while z + WINDOW_W < b.z1 - 3.0:
            if rng.random() < WINDOW_LIT_CHANCE:
                y0, y1 = y, y + WINDOW_H
                z0, z1 = z, z + WINDOW_W
                if b.side < 0:  # normal points +x, towards the street
                    pts = ((x, y0, z1), (x, y0, z0), (x, y1, z0), (x, y1, z1))
                else:  # normal points -x
                    pts = ((x, y0, z0), (x, y0, z1), (x, y1, z1), (x, y1, z0))
                colour = WINDOW_LIT if rng.random() < 0.8 else WINDOW_LIT_COOL
                out.append((pts, colour))
            z += WINDOW_COL_STEP
        y += WINDOW_ROW_STEP
    return tuple(out)


class WorldStrip:
    def __init__(self, seed: int = 7, far: float = 460.0) -> None:
        self.rng = random.Random(seed)
        self.far = far
        self.buildings: list[Building] = []
        self._cursor = {-1: 0.0, 1: 0.0}
        self._next_id = 0

    def _spawn(self, side: int) -> None:
        # Small gaps and deep footprints, so each row reads as a near-continuous
        # wall. Wide gaps expose raw sky, and with nothing behind the row that
        # looks like a hole in the world rather than an alley.
        z0 = self._cursor[side] + self.rng.uniform(0.5, 3.5)
        depth_z = self.rng.uniform(20.0, 40.0)
        z1 = z0 + depth_z

        thickness = self.rng.uniform(18.0, 34.0)
        if side < 0:
            x1 = -STREET_HALF
            x0 = x1 - thickness
        else:
            x0 = STREET_HALF
            x1 = x0 + thickness

        # Tall enough that anchors sit well above the player. Short buildings
        # give a nearly horizontal web, which reads as a speed boost rather
        # than a swing — and the canyon walls towering over you is the look.
        height = self.rng.uniform(72.0, 132.0)
        base = self.rng.choice(BUILDING_COLORS)

        b = Building(x0=x0, x1=x1, z0=z0, z1=z1, height=height, side=side)
        b.faces = _build_faces(b, base)
        b.windows = _build_windows(b, self.rng)

        self.buildings.append(b)
        self._cursor[side] = z1

    def update(self, cam_z: float) -> None:
        for side in (-1, 1):
            while self._cursor[side] < cam_z + self.far:
                self._spawn(side)
        # Keep a little behind the camera so a backward nudge does not tear a
        # hole in the world.
        cutoff = cam_z - 40.0
        if self.buildings and self.buildings[0].z1 < cutoff:
            self.buildings = [b for b in self.buildings if b.z1 >= cutoff]

    def building_near(self, side: int, z: float,
                      window: float = 45.0) -> Building | None:
        """Building on `side` closest to `z` — where a web will anchor.

        Prefers one whose footprint actually contains `z`, and otherwise takes
        the nearest within `window`. The fallback matters: the rows have gaps,
        and aiming into a gap should still find you something to swing from
        rather than silently failing to attach.
        """
        best = None
        best_d = window
        for b in self.buildings:
            if b.side != side:
                continue
            if b.z0 <= z <= b.z1:
                return b
            d = b.z0 - z if z < b.z0 else z - b.z1
            if d < best_d:
                best_d = d
                best = b
        return best

    def nearest_ahead(self, cam_z: float, side: int) -> Building | None:
        """First building on `side` in front of the camera.

        Unused by the renderer — this is the hook the web-anchor system will
        use to decide what a thwip actually attaches to.
        """
        best = None
        for b in self.buildings:
            if b.side != side or b.z1 < cam_z:
                continue
            if best is None or b.z0 < best.z0:
                best = b
        return best
