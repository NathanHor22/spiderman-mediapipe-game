"""Perspective projection, near-plane clipping and back-face culling.

World axes: +x right, +y up, +z forward down the street canyon. The camera looks
straight down +z with no pitch, which is what lets the sky be a static gradient.

There is no z-buffer. Everything is sorted back-to-front and painted over —
which is fine, because the scene is a corridor of convex boxes that never
interpenetrate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Anything closer than this is clipped away. Without the clip, a face straddling
# the camera plane divides by a near-zero depth and smears across the screen —
# which you would hit constantly, since the whole game is flying past buildings.
NEAR = 0.6


@dataclass
class Camera3D:
    x: float = 0.0
    y: float = 34.0
    z: float = 0.0
    fov_deg: float = 78.0
    roll: float = 0.0  # radians, banks the world against lateral velocity

    def focal(self, screen_w: int) -> float:
        return (screen_w * 0.5) / math.tan(math.radians(self.fov_deg) * 0.5)


def face_visible(face, cam: Camera3D) -> bool:
    """Standard back-face cull, done in world space.

    Deliberately not a screen-space winding test — that would depend on the
    y-down flip and the roll, and get subtly wrong at the edges. Comparing the
    face normal against the direction to the camera has no such dependency.

    Every face is wound counter-clockwise viewed from outside, so the
    right-handed cross product points outward.
    """
    p0, p1, p2 = face[0], face[1], face[2]
    ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    vx, vy, vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    return (
        nx * (cam.x - p0[0]) + ny * (cam.y - p0[1]) + nz * (cam.z - p0[2])
    ) > 0.0


def _clip_near(poly):
    """Sutherland-Hodgman against the single plane z = NEAR."""
    out = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        a_in = a[2] >= NEAR
        b_in = b[2] >= NEAR
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = (NEAR - a[2]) / (b[2] - a[2])
            out.append(
                (
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    NEAR,
                )
            )
    return out


def project_face(face, cam: Camera3D, half_w: float, half_h: float,
                 focal: float, cos_r: float, sin_r: float):
    """World-space polygon -> (screen points, mean depth), or (None, 0)."""
    cs = []
    for px, py, pz in face:
        ex = px - cam.x
        ey = py - cam.y
        ez = pz - cam.z
        if sin_r:
            ex, ey = ex * cos_r - ey * sin_r, ex * sin_r + ey * cos_r
        cs.append((ex, ey, ez))

    cs = _clip_near(cs)
    if len(cs) < 3:
        return None, 0.0

    pts = []
    depth = 0.0
    for ex, ey, ez in cs:
        s = focal / ez
        pts.append((half_w + ex * s, half_h - ey * s))
        depth += ez
    return pts, depth / len(cs)


def project_point(p, cam: Camera3D, half_w: float, half_h: float,
                  focal: float, cos_r: float, sin_r: float):
    """Single world point -> ((sx, sy), depth), or None if behind the camera."""
    ex = p[0] - cam.x
    ey = p[1] - cam.y
    ez = p[2] - cam.z
    if sin_r:
        ex, ey = ex * cos_r - ey * sin_r, ex * sin_r + ey * cos_r
    if ez < NEAR:
        return None
    s = focal / ez
    return (half_w + ex * s, half_h - ey * s), ez


def project_segment(a, b, cam: Camera3D, half_w: float, half_h: float,
                    focal: float, cos_r: float, sin_r: float):
    """Line segment -> two screen points, clipped to the near plane.

    The web line runs from the player to an anchor that is often behind the
    camera, so without clipping it would flip to the wrong side of the screen
    exactly when it matters most.
    """
    pts = []
    for p in (a, b):
        ex = p[0] - cam.x
        ey = p[1] - cam.y
        ez = p[2] - cam.z
        if sin_r:
            ex, ey = ex * cos_r - ey * sin_r, ex * sin_r + ey * cos_r
        pts.append([ex, ey, ez])

    p0, p1 = pts
    if p0[2] < NEAR and p1[2] < NEAR:
        return None
    if p0[2] < NEAR or p1[2] < NEAR:
        near_p, far_p = (p0, p1) if p0[2] < NEAR else (p1, p0)
        t = (NEAR - near_p[2]) / (far_p[2] - near_p[2])
        near_p[0] += (far_p[0] - near_p[0]) * t
        near_p[1] += (far_p[1] - near_p[1]) * t
        near_p[2] = NEAR

    out = []
    for ex, ey, ez in pts:
        s = focal / ez
        out.append((half_w + ex * s, half_h - ey * s))
    return out


def box_faces(x0, x1, y0, y1, z0, z1):
    """The six faces of an axis-aligned box, wound CCW as seen from outside.

    Ordering here is load-bearing: get one backwards and its normal inverts, so
    face_visible() culls exactly the faces it should be keeping.
    """
    return {
        "front": ((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)),
        "back": ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)),
        "left": ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)),
        "right": ((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1)),
        "top": ((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)),
        "bottom": ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)),
    }
