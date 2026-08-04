"""Title, tutorial, countdown and death screens.

All of these draw *over* a live corridor render rather than over a flat colour.
It costs nothing — the renderer is already there — and it means the first thing
you see is the game world moving, not a menu.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pygame

from ..control import ControlState
from . import hud
from .surface import fit_rect

TITLE_RED = (206, 44, 50)
TITLE_BLUE = (92, 122, 220)
TITLE_MENU_ITEMS = ("START", "TRAINING", "QUIT")


@dataclass
class TitleMenu:
    """Small input-agnostic controller for the three title-screen actions.

    ``draw_title`` only needs ``menu.selected``.  Keeping navigation here avoids
    teaching the renderer about the game state machine, while giving callers a
    single, testable place to handle arrows, W/S and the existing hotkeys.
    """

    selected: int = 0

    def __post_init__(self) -> None:
        self.selected %= len(TITLE_MENU_ITEMS)

    @property
    def action(self) -> str:
        return TITLE_MENU_ITEMS[self.selected].lower()

    def move(self, delta: int) -> None:
        self.selected = (self.selected + delta) % len(TITLE_MENU_ITEMS)

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Update selection and return an activated action, if any."""
        if event.type != pygame.KEYDOWN:
            return None
        if event.key in (pygame.K_UP, pygame.K_w):
            self.move(-1)
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            self.move(1)
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return self.action
        elif event.key == pygame.K_t:
            self.selected = 1
            return "training"
        elif event.key == pygame.K_ESCAPE:
            self.selected = 2
            return "quit"
        return None

    def select_at(self, position, hit_rects) -> bool:
        """Select the menu slab under ``position`` without activating it."""
        for index, rect in enumerate(hit_rects):
            if rect.collidepoint(position):
                self.selected = index % len(TITLE_MENU_ITEMS)
                return True
        return False

    def activate_at(self, position, hit_rects) -> str | None:
        """Return the clicked action, or ``None`` outside every menu slab."""
        return self.action if self.select_at(position, hit_rects) else None


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
                    "move your hand left and right to choose the next wall",
                    aim_check, hold_s=0.3),
            ]
        else:
            self.steps = [
                TutorialStep("HOLD SPACE", "this fires a web at a building",
                             lambda c: c.thwip_held, hold_s=1.0),
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

    if not flow.vision:
        key_box = pygame.Rect(box.x + 55, box.y + 205, 210, 62)
        if flow.index == 0:
            key_text = "SPACE"
            key_active = ctrl.thwip_held
        elif flow.index == 1:
            key_text = "LET GO"
            key_active = not ctrl.thwip_held
        else:
            key_text = "MOVE MOUSE"
            key_active = abs(ctrl.hand_x - 0.5) > 0.15
        _draw_keycap(surface, mid, key_text, key_box, active=key_active)

    # Live camera inset, so you can see what the classifier is seeing while you
    # try to match the pose.
    if flow.vision:
        inset = pygame.Rect(box.right - 330, box.y + 165, 290, 218)
        draw_camera_preview(surface, cam_surface, inset)
        pygame.draw.rect(surface, hud.PANEL_EDGE, inset, 1)
        if cam_surface is None:
            state, colour = "CAMERA STARTING", hud.WARN
        elif ctrl.tracking_lost:
            state, colour = "NO HAND - MOVE INTO FRAME", hud.BAD
        elif flow.index == 0:
            state = "WEB-SHOOTER DETECTED" if ctrl.thwip_held else "MAKE THE SIGN"
            colour = hud.GOOD if ctrl.thwip_held else hud.WARN
        elif flow.index == 1:
            state = "RELEASE DETECTED" if not ctrl.thwip_held else "RELAX THE SIGN"
            colour = hud.GOOD if not ctrl.thwip_held else hud.WARN
        else:
            state, colour = "HAND AIM TRACKING", hud.GOOD
        hud.draw_text(surface, font, state, (inset.x + 8, inset.y + 6), colour)

    bar = pygame.Rect(box.x + 260, box.bottom - 66, 240, 16)
    hud.draw_bar(surface, bar, step.progress, 1.0, hud.GOOD)
    prompts = (
        "hold it...",
        "release it...",
        "sweep both sides...",
    )
    hud.draw_text(surface, font, prompts[min(flow.index, 2)],
                  (bar.x, bar.y - 22), hud.DIM)
    hud.draw_text(surface, font, "ESC back to title",
                  (box.right - 176, box.bottom - 32), hud.DIM)


def _ellipsize(font, text: str, max_width: int) -> str:
    """Fit one line without allowing camera errors to escape their card."""
    if font.size(text)[0] <= max_width:
        return text
    suffix = "..."
    while text and font.size(text + suffix)[0] > max_width:
        text = text[:-1]
    return text + suffix


def _comic_panel(surface, rect: pygame.Rect, *, accent=TITLE_BLUE,
                 alpha: int = 238) -> None:
    """A hard-edged, offset panel matching the low-poly PS1 presentation."""
    pygame.draw.rect(surface, (2, 2, 7), rect.move(7, 7))
    box = pygame.Surface(rect.size, pygame.SRCALPHA)
    box.fill((10, 8, 18, alpha))
    surface.blit(box, rect.topleft)
    pygame.draw.rect(surface, (1, 1, 4), rect, 5)
    pygame.draw.rect(surface, accent, rect, 2)
    pygame.draw.line(surface, TITLE_RED,
                     (rect.x + 3, rect.y + 3),
                     (rect.x + rect.width // 3, rect.y + 3), 2)


def _draw_web_fan(surface, origin: tuple[int, int], direction: tuple[int, int],
                  size: int) -> None:
    """Cheap straight-line webbing; deliberately angular rather than smooth."""
    ox, oy = origin
    dx, dy = direction
    rays = ((size, 0), (size, size // 3), (size, size),
            (size // 3, size), (0, size))
    points: list[tuple[int, int]] = []
    for rx, ry in rays:
        point = (ox + dx * rx, oy + dy * ry)
        points.append(point)
        pygame.draw.line(surface, (74, 80, 110), origin, point, 1)
    for fraction in (0.30, 0.58, 0.84):
        ring = [
            (int(ox + (px - ox) * fraction),
             int(oy + (py - oy) * fraction))
            for px, py in points
        ]
        pygame.draw.lines(surface, (48, 52, 76), False, ring, 1)


def _scaled_word(font, text: str, colour, scale: float = 1.35) -> pygame.Surface:
    image = font.render(text, True, colour)
    size = (max(1, int(image.get_width() * scale)),
            max(1, int(image.get_height() * scale)))
    return pygame.transform.scale(image, size)


def _draw_logo(surface, big, mid, centre_x: int, y: int) -> None:
    words = (("SPIDER", TITLE_RED), ("SWING", TITLE_BLUE))
    rendered = [(_scaled_word(big, word, colour), word, colour)
                for word, colour in words]
    gap = 24
    total_w = sum(item[0].get_width() for item in rendered) + gap
    x = centre_x - total_w // 2

    for face, word, colour in rendered:
        black = _scaled_word(big, word, (2, 2, 5))
        shadow_colour = TITLE_BLUE if colour == TITLE_RED else TITLE_RED
        shadow = _scaled_word(big, word, shadow_colour)
        surface.blit(shadow, (x + 6, y + 6))
        for off_x, off_y in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            surface.blit(black, (x + off_x, y + off_y))
        surface.blit(face, (x, y))
        x += face.get_width() + gap

    subtitle = mid.render("ENDLESS CITY SWINGER", True, hud.INK)
    surface.blit(subtitle, (centre_x - subtitle.get_width() // 2,
                            y + rendered[0][0].get_height() + 2))


def _draw_keycap(surface, font, text: str, rect: pygame.Rect,
                 *, active: bool = False) -> None:
    pygame.draw.rect(surface, (2, 2, 6), rect.move(4, 4))
    pygame.draw.rect(surface, (36, 32, 52), rect)
    pygame.draw.rect(surface, TITLE_RED if active else TITLE_BLUE, rect, 2)
    image = font.render(text, True, hud.INK)
    surface.blit(image, (rect.centerx - image.get_width() // 2,
                         rect.centery - image.get_height() // 2))


def vision_status(ctrl: ControlState | None, has_frame: bool) -> tuple[str, tuple]:
    """Return the title camera's user-facing tracking state and colour."""
    if not has_frame:
        return "CAMERA STARTING", hud.WARN
    if ctrl is None:
        return "CAMERA READY", hud.GOOD
    if ctrl.tracking_lost:
        return "NO HAND - MOVE INTO FRAME", hud.BAD
    if ctrl.thwip_held:
        return "WEB-SHOOTER DETECTED", hud.GOOD
    return "HAND FOUND - MAKE THE SIGN", hud.WARN


def draw_camera_preview(surface, cam_surface, box: pygame.Rect) -> pygame.Rect:
    """Draw a letterboxed camera image and return its aspect-correct rect."""
    pygame.draw.rect(surface, (3, 3, 8), box)
    if cam_surface is None:
        return pygame.Rect(box.centerx, box.centery, 0, 0)
    fitted = fit_rect(cam_surface.get_width(), cam_surface.get_height(), box)
    scaled = pygame.transform.smoothscale(cam_surface, fitted.size)
    surface.blit(scaled, fitted.topleft)
    return fitted


def _draw_preview_overlay(surface, box: pygame.Rect) -> None:
    scanlines = pygame.Surface(box.size, pygame.SRCALPHA)
    for y in range(2, box.height, 5):
        pygame.draw.line(scanlines, (2, 2, 8, 42), (0, y), (box.width, y))
    surface.blit(scanlines, box.topleft)

    length = 18
    colour = hud.INK
    corners = (
        ((box.left, box.top + length), (box.left, box.top),
         (box.left + length, box.top)),
        ((box.right - length, box.top), (box.right, box.top),
         (box.right, box.top + length)),
        ((box.left, box.bottom - length), (box.left, box.bottom),
         (box.left + length, box.bottom)),
        ((box.right - length, box.bottom), (box.right, box.bottom),
         (box.right, box.bottom - length)),
    )
    for a, b, c in corners:
        pygame.draw.lines(surface, colour, False, (a, b, c), 2)


def _draw_status_badge(surface, font, rect: pygame.Rect, text: str, colour) -> None:
    pygame.draw.rect(surface, (24, 21, 34), rect)
    pygame.draw.rect(surface, colour, rect, 2)
    pygame.draw.circle(surface, colour, (rect.x + 16, rect.centery), 5)
    hud.draw_text(surface, font, text, (rect.x + 30, rect.y + 8), colour)


def _draw_instruction_row(surface, font, mid, rect: pygame.Rect,
                          label: str, instruction: str, number: int) -> None:
    pygame.draw.rect(surface, (22, 19, 32), rect)
    pygame.draw.line(surface, TITLE_BLUE, rect.topleft, rect.bottomleft, 3)
    hud.draw_text(surface, mid, f"0{number}", (rect.x + 12, rect.y + 9), TITLE_RED)
    hud.draw_text(surface, font, label, (rect.x + 52, rect.y + 7), hud.INK)
    hud.draw_text(surface, font, instruction,
                  (rect.x + 52, rect.y + 27), hud.DIM)


def _draw_menu_item(surface, font, mid, rect: pygame.Rect, label: str,
                    shortcut: str, selected: bool, pulse: float) -> None:
    if selected:
        points = (rect.topleft, (rect.right - 13, rect.top),
                  (rect.right, rect.centery),
                  (rect.right - 13, rect.bottom), rect.bottomleft)
        shadow = [(x + 4, y + 4) for x, y in points]
        pygame.draw.polygon(surface, TITLE_BLUE, shadow)
        red = max(155, min(228, int(194 + pulse * 18)))
        pygame.draw.polygon(surface, (red, 38, 46), points)
        colour = (246, 242, 242)
    else:
        pygame.draw.rect(surface, (23, 20, 32), rect)
        pygame.draw.rect(surface, hud.PANEL_EDGE, rect, 1)
        colour = hud.DIM
    hud.draw_text(surface, mid, label, (rect.x + 18, rect.y + 7), colour)
    key_image = font.render(shortcut, True, colour)
    surface.blit(key_image, (rect.right - key_image.get_width() - 18,
                             rect.centery - key_image.get_height() // 2))


def draw_title(surface, fonts, *, vision: bool, camera_index: int = 0,
               camera_info: str = "", names: list[str] | tuple[str, ...] = (),
               pulse: float = 0.0, cam_surface=None,
               ctrl: ControlState | None = None,
               selected: int | TitleMenu = 0,
               sound_available: bool = True) -> tuple[pygame.Rect, ...]:
    """Draw the title and return the START/TRAINING/QUIT hit rectangles.

    Existing callers may keep using the old keyword arguments. New callers can
    pass a ``TitleMenu`` (or its integer selection) plus the latest
    ``ControlState`` for live vision feedback.
    """
    font, mid, big = fonts
    w, h = surface.get_size()
    dim_backdrop(surface, 150)

    _draw_web_fan(surface, (24, 20), (1, 1), 150)
    _draw_web_fan(surface, (w - 24, 20), (-1, 1), 150)
    _draw_logo(surface, big, mid, w // 2, 30)

    margin = max(28, int(w * 0.04))
    plate_y = max(148, int(h * 0.21))
    plate = pygame.Rect(margin, plate_y, w - margin * 2,
                        h - plate_y - max(28, int(h * 0.05)))
    _comic_panel(surface, plate)

    inner = plate.inflate(-48, -42)
    gap = 24
    left_w = int((inner.width - gap) * 0.44)
    left = pygame.Rect(inner.x, inner.y, left_w, inner.height)
    right = pygame.Rect(left.right + gap, inner.y,
                        inner.right - left.right - gap, inner.height)

    badge_text = "VISION CONTROL" if vision else "KEYBOARD + MOUSE"
    badge_w = font.size(badge_text)[0] + 28
    badge = pygame.Rect(left.x, left.y, badge_w, 27)
    pygame.draw.rect(surface, TITLE_BLUE, badge)
    hud.draw_text(surface, font, badge_text, (badge.x + 14, badge.y + 5),
                  (248, 248, 252), shadow=False)
    sound_text = "SOUND READY" if sound_available else "NO SOUND"
    sound_colour = hud.GOOD if sound_available else hud.WARN
    sound_img = font.render(sound_text, True, sound_colour)
    sound_x = left.right - sound_img.get_width()
    pygame.draw.circle(surface, sound_colour,
                       (sound_x - 10, left.y + badge.height // 2), 4)
    surface.blit(sound_img, (sound_x, left.y + 5))
    hud.draw_text(surface, mid, "MASTER THE SWING",
                  (left.x, left.y + 39), hud.INK)

    if vision:
        instructions = (
            ("AIM", "move your hand left and right"),
            ("THWIP", "index + pinky out, middle + ring curled"),
            ("RELEASE", "relax the sign at the top of the arc"),
        )
    else:
        instructions = (
            ("AIM", "move the mouse left and right"),
            ("THWIP", "hold SPACE or the left mouse button"),
            ("RELEASE", "let go at the top of the arc"),
        )
    row_y = left.y + 76
    row_h = 49
    for index, (label, instruction) in enumerate(instructions, start=1):
        row = pygame.Rect(left.x, row_y + (index - 1) * (row_h + 5),
                          left.width, row_h)
        _draw_instruction_row(surface, font, mid, row, label, instruction, index)

    hud.draw_text(surface, font, "SELECT MISSION", (left.x, left.y + 246), hud.DIM)
    selection = selected.selected if isinstance(selected, TitleMenu) else int(selected)
    selection %= len(TITLE_MENU_ITEMS)
    menu_rects: list[pygame.Rect] = []
    direct_shortcuts = ("", "T", "ESC")
    menu_y = left.y + 269
    for index, (label, direct) in enumerate(
        zip(TITLE_MENU_ITEMS, direct_shortcuts)
    ):
        item = pygame.Rect(left.x, menu_y + index * 48, left.width, 40)
        menu_rects.append(item)
        if index == selection:
            shortcut = f"ENTER / {direct}" if direct else "ENTER"
        else:
            shortcut = direct
        _draw_menu_item(surface, font, mid, item, label, shortcut,
                        index == selection, pulse)
    hud.draw_text(surface, font, "UP / DOWN select    ENTER confirm",
                  (left.x, left.bottom - 18), hud.DIM)

    _comic_panel(surface, right, accent=TITLE_RED, alpha=224)
    card_pad = 18
    if vision:
        hud.draw_text(surface, font, "LIVE CAMERA", (right.x + card_pad, right.y + 15),
                      hud.DIM)
        index_text = f"CAMERA {camera_index}"
        index_img = font.render(index_text, True, hud.INK)
        surface.blit(index_img, (right.right - card_pad - index_img.get_width(),
                                 right.y + 15))

        preview = pygame.Rect(right.x + card_pad, right.y + 45,
                              right.width - card_pad * 2,
                              max(120, right.height - 164))
        fitted = draw_camera_preview(surface, cam_surface, preview)
        if cam_surface is None:
            waiting = mid.render("WAITING FOR CAMERA...", True, hud.DIM)
            surface.blit(waiting, (preview.centerx - waiting.get_width() // 2,
                                   preview.centery - waiting.get_height() // 2))
        _draw_preview_overlay(surface, preview)
        status, status_colour = vision_status(ctrl, cam_surface is not None)
        status_rect = pygame.Rect(preview.x, preview.bottom + 12,
                                  preview.width, 34)
        _draw_status_badge(surface, font, status_rect, status, status_colour)

        info = camera_info.strip() or "[ / ] SWITCH CAMERA"
        info = _ellipsize(font, info, preview.width)
        hud.draw_text(surface, font, info, (preview.x, status_rect.bottom + 10),
                      hud.WARN if "(" in info and "auto" not in info else hud.DIM)
        hint = "Keep your full hand inside the frame"
        hud.draw_text(surface, font, hint, (preview.x, status_rect.bottom + 31),
                      hud.DIM)
        _ = fitted, names
    else:
        hud.draw_text(surface, font, "CONTROL DECK",
                      (right.x + card_pad, right.y + 15), hud.DIM)
        glyph = pygame.Rect(right.x + 28, right.y + 55,
                            min(190, right.width // 3), 226)
        draw_thwip_glyph(surface, glyph, pulse)
        hud.draw_text(surface, font, "THE WEB-SHOOTER",
                      (glyph.x + 6, glyph.bottom + 8), hud.DIM)

        key_x = right.x + right.width // 2
        hud.draw_text(surface, mid, "HOLD TO SWING",
                      (key_x, right.y + 74), hud.INK)
        _draw_keycap(surface, mid, "SPACE",
                     pygame.Rect(key_x, right.y + 112,
                                 right.right - key_x - 28, 54), active=True)
        hud.draw_text(surface, font, "or LEFT MOUSE",
                      (key_x, right.y + 181), hud.DIM)
        pygame.draw.line(surface, TITLE_BLUE,
                         (key_x, right.y + 220),
                         (right.right - 28, right.y + 220), 2)
        hud.draw_text(surface, mid, "MOVE TO AIM",
                      (key_x, right.y + 239), hud.INK)
        _draw_keycap(surface, font, "MOUSE LEFT / RIGHT",
                     pygame.Rect(key_x, right.y + 278,
                                 right.right - key_x - 28, 44))
        status_rect = pygame.Rect(right.x + card_pad, right.bottom - 70,
                                  right.width - card_pad * 2, 34)
        _draw_status_badge(surface, font, status_rect, "INPUT READY", hud.GOOD)
        hud.draw_text(surface, font, "Add --vision to play with hand gestures",
                      (right.x + card_pad, right.bottom - 25), hud.DIM)

    return tuple(menu_rects)


def draw_countdown(surface, fonts, value: str) -> None:
    font, mid, big = fonts
    w, h = surface.get_size()
    dim_backdrop(surface, 90)
    img = big.render(value, True, hud.INK)
    surface.blit(img, (w // 2 - img.get_width() // 2, h // 2 - img.get_height() // 2))
