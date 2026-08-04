"""Title, tutorial, countdown and death screens.

All of these draw *over* a live corridor render rather than over a flat colour.
It costs nothing — the renderer is already there — and it means the first thing
you see is the game world moving, not a menu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pygame

from ..control import ControlState
from . import hud

TITLE_RED = (206, 44, 50)
TITLE_BLUE = (92, 122, 220)


def dim_backdrop(surface, alpha: int = 150) -> None:
    veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    veil.fill((6, 5, 12, alpha))
    surface.blit(veil, (0, 0))


def panel(surface, rect, alpha: int = 226, edge=hud.PANEL_EDGE) -> None:
    box = pygame.Surface(rect.size, pygame.SRCALPHA)
    box.fill((12, 10, 20, alpha))
    surface.blit(box, rect.topleft)
    pygame.draw.rect(surface, edge, rect, 1)


def draw_thwip_glyph(surface, rect: pygame.Rect, pulse: float = 0.0) -> None:
    """A hand making the web-shooter sign.

    Worth the forty lines: "index and pinky extended, middle and ring folded" is
    a sentence people have to decode, but the shape is instant. Extended fingers
    are lit, folded ones are stubs.
    """
    cx = rect.centerx
    base_y = rect.bottom - rect.height * 0.22
    palm_w = rect.width * 0.46
    palm_h = rect.height * 0.30

    glow = int(40 * (0.5 + 0.5 * pulse))
    palm = pygame.Rect(0, 0, int(palm_w), int(palm_h))
    palm.center = (cx, int(base_y))
    pygame.draw.rect(surface, (150 + glow // 3, 40, 46), palm, border_radius=8)

    finger_w = palm_w * 0.19
    spread = palm_w * 0.26
    # (offset from centre, extended?)
    layout = ((-1.5, True), (-0.5, False), (0.5, False), (1.5, True))
    for slot, extended in layout:
        x = cx + slot * spread
        if extended:
            h = rect.height * 0.40
            colour = (108 + glow, 226, 138)
        else:
            h = rect.height * 0.11
            colour = (86, 80, 96)
        bar = pygame.Rect(0, 0, int(finger_w), int(h))
        bar.midbottom = (int(x), int(palm.top + 4))
        pygame.draw.rect(surface, colour, bar, border_radius=int(finger_w / 2))

    # thumb, off to the side and unimportant — the classifier ignores it
    thumb = pygame.Rect(0, 0, int(finger_w * 0.9), int(rect.height * 0.20))
    thumb.center = (int(cx - palm_w * 0.72), int(base_y - palm_h * 0.15))
    pygame.draw.rect(surface, (150, 60, 64), thumb, border_radius=6)


@dataclass
class TutorialStep:
    title: str
    hint: str
    check: Callable[[ControlState], bool]
    hold_s: float = 0.0
    show_glyph: bool = False
    progress: float = 0.0
    _held: float = 0.0

    def update(self, ctrl: ControlState, dt: float) -> bool:
        if self.check(ctrl):
            self._held += dt
        else:
            self._held = max(0.0, self._held - dt * 1.6)
        if self.hold_s <= 0.0:
            return self._held > 0.0
        self.progress = min(1.0, self._held / self.hold_s)
        return self.progress >= 1.0


class TutorialFlow:
    """Teaches the one thing the game is made of: hold, then let go.

    Gated on the player actually performing each action rather than on a timer,
    so nobody reaches the street still guessing at the gesture. The aim step
    exists because hand position choosing the anchor side is the least
    guessable part of the scheme.
    """

    def __init__(self, vision: bool) -> None:
        self.vision = vision
        self._seen_left = False
        self._seen_right = False

        def aim_check(ctrl: ControlState) -> bool:
            if ctrl.tracking_lost:
                return False
            if ctrl.hand_x < 0.35:
                self._seen_left = True
            elif ctrl.hand_x > 0.65:
                self._seen_right = True
            return self._seen_left and self._seen_right

        if vision:
            self.steps = [
                TutorialStep(
                    "MAKE THE WEB-SHOOTER SIGN",
                    "index and pinky out, middle and ring folded into your palm",
                    lambda c: c.thwip_held, hold_s=1.2, show_glyph=True),
                TutorialStep(
                    "NOW LET GO",
                    "dropping the sign releases the web — that is your jump",
                    lambda c: not c.thwip_held and not c.tracking_lost,
                    hold_s=0.5),
                TutorialStep(
                    "MOVE YOUR HAND LEFT AND RIGHT",
                    "your hand picks which building you web — left hand, left wall",
                    aim_check, hold_s=0.3),
            ]
        else:
            self.steps = [
                TutorialStep("HOLD SPACE", "this fires a web at a building",
                             lambda c: c.thwip_held, hold_s=1.0, show_glyph=True),
                TutorialStep("NOW RELEASE IT",
                             "letting go releases the web — that is your jump",
                             lambda c: not c.thwip_held, hold_s=0.4),
                TutorialStep("MOVE THE MOUSE LEFT AND RIGHT",
                             "the mouse picks which building you web",
                             aim_check, hold_s=0.3),
            ]
        self.index = 0

    @property
    def done(self) -> bool:
        return self.index >= len(self.steps)

    @property
    def current(self) -> TutorialStep | None:
        return None if self.done else self.steps[self.index]

    def update(self, ctrl: ControlState, dt: float) -> None:
        step = self.current
        if step is not None and step.update(ctrl, dt):
            self.index += 1


def draw_tutorial(surface, fonts, flow: TutorialFlow, ctrl: ControlState,
                  cam_surface, pulse: float) -> None:
    font, mid, big = fonts
    w, h = surface.get_size()
    dim_backdrop(surface, 165)

    box = pygame.Rect(w // 2 - 430, h // 2 - 210, 860, 420)
    panel(surface, box)

    hud.draw_text(surface, font, f"TUTORIAL  {flow.index + 1} / {len(flow.steps)}",
                  (box.x + 30, box.y + 22), hud.DIM)

    step = flow.current
    if step is None:
        return

    hud.draw_text(surface, big, step.title, (box.x + 30, box.y + 58), hud.INK)
    hud.draw_text(surface, mid, step.hint, (box.x + 30, box.y + 118), hud.DIM)

    if step.show_glyph:
        draw_thwip_glyph(surface, pygame.Rect(box.x + 40, box.y + 165, 180, 210),
                         pulse)

    # Live camera inset, so you can see what the classifier is seeing while you
    # try to match the pose.
    if cam_surface is not None:
        inset = pygame.Rect(box.right - 330, box.y + 165, 290, 218)
        scaled = pygame.transform.smoothscale(cam_surface, inset.size)
        surface.blit(scaled, inset.topleft)
        pygame.draw.rect(surface, hud.PANEL_EDGE, inset, 1)
        state = ("NO HAND" if ctrl.tracking_lost else
                 "THWIP" if ctrl.thwip_held else "idle")
        colour = (hud.BAD if ctrl.tracking_lost else
                  hud.GOOD if ctrl.thwip_held else hud.DIM)
        hud.draw_text(surface, font, state, (inset.x + 8, inset.y + 6), colour)

    bar = pygame.Rect(box.x + 260, box.bottom - 66, 300, 16)
    hud.draw_bar(surface, bar, step.progress, 1.0, hud.GOOD)
    hud.draw_text(surface, font, "hold it...", (bar.x, bar.y - 22), hud.DIM)
    hud.draw_text(surface, font, "ESC to skip", (box.right - 130, box.bottom - 32),
                  hud.DIM)


def draw_title(surface, fonts, *, vision: bool, camera_index: int,
               camera_info: str, names: list[str], pulse: float,
               cam_surface=None) -> None:
    font, mid, big = fonts
    w, h = surface.get_size()
    dim_backdrop(surface, 140)

    title_font = big
    hud.draw_text(surface, title_font, "SPIDER", (w // 2 - 300, 96), TITLE_RED)
    hud.draw_text(surface, title_font, "SWING", (w // 2 - 60, 96), TITLE_BLUE)
    hud.draw_text(surface, mid, "webs, buildings, and a webcam",
                  (w // 2 - 296, 158), hud.DIM)

    # Sized to its contents — the camera list only exists in vision mode.
    box = pygame.Rect(w // 2 - 300, 210, 600, 214 if vision else 152)
    panel(surface, box)

    y = box.y + 24
    if vision:
        hud.draw_text(surface, font, "CAMERA", (box.x + 26, y), hud.DIM)
        y += 26
        hud.draw_text(surface, mid, camera_info, (box.x + 26, y), hud.INK)
        y += 34
        for n in names[:3]:
            hud.draw_text(surface, font, n, (box.x + 26, y), hud.DIM)
            y += 20
        y += 8
        hud.draw_text(surface, font,
                      "[ and ] switch camera  —  pick the one you can see yourself in",
                      (box.x + 26, y), hud.WARN)
        y += 30
    else:
        hud.draw_text(surface, font, "INPUT", (box.x + 26, y), hud.DIM)
        y += 26
        hud.draw_text(surface, mid, "keyboard + mouse", (box.x + 26, y), hud.INK)
        y += 34
        hud.draw_text(surface, font, "run with --vision to use the webcam",
                      (box.x + 26, y), hud.DIM)
        y += 30

    hud.draw_text(surface, font, "hold SPACE / the web-shooter sign to swing",
                  (box.x + 26, y), hud.DIM)
    y += 20
    hud.draw_text(surface, font, "release to let go — that is the whole game",
                  (box.x + 26, y), hud.DIM)

    glow = hud.GOOD if pulse > 0 else (70, 150, 96)
    hud.draw_text(surface, mid, "PRESS ENTER TO START",
                  (w // 2 - 140, box.bottom + 34), glow)
    hud.draw_text(surface, font, "T tutorial     ESC quit",
                  (w // 2 - 90, box.bottom + 68), hud.DIM)

    if vision and cam_surface is not None:
        inset = pygame.Rect(w - 300, h - 250, 280, 210)
        surface.blit(pygame.transform.smoothscale(cam_surface, inset.size),
                     inset.topleft)
        pygame.draw.rect(surface, hud.PANEL_EDGE, inset, 1)
        hud.draw_text(surface, font, f"index {camera_index}",
                      (inset.x + 8, inset.y + 6), hud.INK)


def draw_countdown(surface, fonts, value: str) -> None:
    font, mid, big = fonts
    w, h = surface.get_size()
    dim_backdrop(surface, 90)
    img = big.render(value, True, hud.INK)
    surface.blit(img, (w // 2 - img.get_width() // 2, h // 2 - img.get_height() // 2))
