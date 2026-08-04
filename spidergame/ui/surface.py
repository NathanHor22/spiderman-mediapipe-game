"""Camera frames into pygame surfaces."""

from __future__ import annotations

import cv2
import pygame


def bgr_to_surface(frame) -> pygame.Surface:
    """OpenCV BGR array -> pygame Surface.

    Via `frombuffer` rather than `surfarray.make_surface`, which wants
    (width, height, 3) and so needs a transpose of every frame. A colour convert
    plus a buffer wrap is markedly cheaper than transposing 900KB sixty times a
    second.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")


def fit_rect(src_w: int, src_h: int, box: pygame.Rect) -> pygame.Rect:
    """Largest rect with the source aspect ratio that fits in `box`, centred."""
    if src_w <= 0 or src_h <= 0:
        return box
    scale = min(box.width / src_w, box.height / src_h)
    w = int(src_w * scale)
    h = int(src_h * scale)
    return pygame.Rect(box.x + (box.width - w) // 2,
                       box.y + (box.height - h) // 2, w, h)
