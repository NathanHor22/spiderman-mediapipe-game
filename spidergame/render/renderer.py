"""Painter's-algorithm scene renderer.

The whole thing is flat-shaded quads sorted back-to-front with distance fog.
That is not a compromise on the way to "real" 3D — flat shading, no per-pixel
lighting, chunky geometry and an aggressive fog wall *is* the PS1 look. The
cheap renderer is what gets the aesthetic, so it is worth keeping it cheap.
"""

from __future__ import annotations

import math

import pygame

from .palette import (
    FOG_END,
    STREET,
    STREET_BAND,
    fogged,
    make_sky,
)
from .projection import Camera3D, face_visible, project_face
from .world import STREET_HALF, WINDOW_DISTANCE, WorldStrip

STREET_SLABS = 16  # split so fog varies along the road instead of flat-filling
BAND_SPACING = 30.0
BAND_DEPTH = 2.2

# Hard ceiling on window quads per frame. The distance cutoff usually keeps us
# well under it; this is the backstop for the case where the camera drops low
# and stares down a long row of near facades.
WINDOW_BUDGET = 320


class Renderer:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.sky = make_sky(width, height)
        self.half_w = width * 0.5
        self.half_h = height * 0.5
        self.stats = {"faces": 0, "windows": 0}

    def resize(self, width: int, height: int) -> None:
        self.__init__(width, height)

    def _quad(self, surface, pts_world, colour, cam, focal, cos_r, sin_r,
              cull: bool = True) -> bool:
        if cull and not face_visible(pts_world, cam):
            return False
        pts, depth = project_face(
            pts_world, cam, self.half_w, self.half_h, focal, cos_r, sin_r
        )
        if pts is None:
            return False
        pygame.draw.polygon(surface, fogged(colour, depth), pts)
        return True

    def render(self, surface, cam: Camera3D, world: WorldStrip,
               show_windows: bool = True, actors=None) -> dict:
        """`actors` is [(faces, sort_z), ...] — figures merged into the sort.

        Merged rather than drawn afterwards because with no z-buffer, painting
        the player last would put them in front of a building they are actually
        behind. That happens constantly: the player flies within a few units of
        the facades they are swinging from.
        """
        focal = cam.focal(self.width)
        cos_r = math.cos(cam.roll)
        sin_r = math.sin(cam.roll)
        faces = windows = 0

        surface.blit(self.sky, (0, 0))

        # --- street ------------------------------------------------------
        # Drawn as slabs rather than one quad so each gets its own fog depth;
        # a single quad would fog uniformly and flatten the road into a stripe.
        near_z = cam.z
        far_z = cam.z + FOG_END
        step = (far_z - near_z) / STREET_SLABS
        for i in reversed(range(STREET_SLABS)):  # far to near, like everything else
            z0 = near_z + i * step
            z1 = z0 + step
            quad = (
                (-STREET_HALF, 0.0, z0),
                (-STREET_HALF, 0.0, z1),
                (STREET_HALF, 0.0, z1),
                (STREET_HALF, 0.0, z0),
            )
            if self._quad(surface, quad, STREET, cam, focal, cos_r, sin_r):
                faces += 1

        # Cross-street bands. Pure speed feedback — without something ticking
        # past underneath, forward motion down a uniform road reads as static.
        first = math.floor(near_z / BAND_SPACING) * BAND_SPACING
        z = first
        while z < far_z:
            if z > near_z:
                quad = (
                    (-STREET_HALF, 0.02, z),
                    (-STREET_HALF, 0.02, z + BAND_DEPTH),
                    (STREET_HALF, 0.02, z + BAND_DEPTH),
                    (STREET_HALF, 0.02, z),
                )
                if self._quad(surface, quad, STREET_BAND, cam, focal, cos_r, sin_r):
                    faces += 1
            z += BAND_SPACING

        # --- buildings and actors, far to near -----------------------------
        items = [(b.z0, b) for b in world.buildings if b.z1 > cam.z and b.z0 < far_z]
        for actor_faces, sort_z in (actors or ()):
            items.append((sort_z, actor_faces))
        items.sort(key=lambda it: it[0], reverse=True)

        for _, item in items:
            if isinstance(item, list):  # actor: a plain list of (pts, colour)
                for pts_world, colour in item:
                    if self._quad(surface, pts_world, colour, cam,
                                  focal, cos_r, sin_r):
                        faces += 1
                continue

            b = item
            for pts_world, colour in b.faces:
                if self._quad(surface, pts_world, colour, cam, focal, cos_r, sin_r):
                    faces += 1

            if not show_windows or windows >= WINDOW_BUDGET:
                continue
            if b.z0 - cam.z > WINDOW_DISTANCE:
                continue
            for pts_world, colour in b.windows:
                if windows >= WINDOW_BUDGET:
                    break
                # Windows must be culled like any other face. A building being
                # inside the z range does not mean its inner wall still faces
                # us — once the camera draws level with a facade the wall turns
                # away, and unculled windows carried on drawing as lit rectangles
                # floating in mid-air with nothing behind them.
                if not face_visible(pts_world, cam):
                    continue
                pts, depth = project_face(
                    pts_world, cam, self.half_w, self.half_h, focal, cos_r, sin_r
                )
                if pts is None or depth > WINDOW_DISTANCE:
                    continue
                pygame.draw.polygon(surface, fogged(colour, depth), pts)
                windows += 1

        self.stats = {"faces": faces, "windows": windows}
        return self.stats
