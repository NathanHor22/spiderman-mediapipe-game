"""Small HUD widgets, shared by the tools and the game."""

from __future__ import annotations

import pygame

INK = (206, 210, 224)
DIM = (126, 132, 150)
GOOD = (108, 226, 138)
WARN = (255, 186, 74)
BAD = (240, 88, 88)
PANEL_BG = (16, 14, 24)
PANEL_EDGE = (44, 40, 60)


def draw_text(surface, font, text, pos, colour=INK, shadow=True):
    """Text with a 1px drop shadow, so the HUD stays readable over the sky."""
    x, y = pos
    if shadow:
        surface.blit(font.render(text, True, (0, 0, 0)), (x + 1, y + 1))
    img = font.render(text, True, colour)
    surface.blit(img, (x, y))
    return img.get_height()


def draw_bar(surface, rect, value, maximum, colour, *, marker=None,
             bg=(38, 34, 48)):
    """Horizontal meter with an optional threshold marker.

    The marker is the point of the widget — a raw number cannot tell you whether
    a threshold has headroom over your idle motion, but a bar sitting next to a
    line can, at a glance.
    """
    pygame.draw.rect(surface, bg, rect)
    if maximum > 0:
        frac = max(0.0, min(value / maximum, 1.0))
        if frac > 0:
            fill = pygame.Rect(rect.x, rect.y, int(rect.width * frac), rect.height)
            pygame.draw.rect(surface, colour, fill)
    if marker is not None and maximum > 0:
        mx = rect.x + int(rect.width * max(0.0, min(marker / maximum, 1.0)))
        pygame.draw.line(surface, INK, (mx, rect.y - 4),
                         (mx, rect.bottom + 4), 2)
    pygame.draw.rect(surface, PANEL_EDGE, rect, 1)


class Panel:
    """Stacked label/value rows in a bordered box."""

    def __init__(self, rect: pygame.Rect, font, title: str = "") -> None:
        self.rect = rect
        self.font = font
        self.title = title
        self._y = 0

    def begin(self, surface) -> None:
        pygame.draw.rect(surface, PANEL_BG, self.rect)
        pygame.draw.rect(surface, PANEL_EDGE, self.rect, 1)
        self._y = self.rect.y + 10
        if self.title:
            draw_text(surface, self.font, self.title,
                      (self.rect.x + 12, self._y), DIM)
            self._y += self.font.get_height() + 8

    def row(self, surface, label, value, colour=INK) -> None:
        draw_text(surface, self.font, label, (self.rect.x + 12, self._y), DIM)
        img = self.font.render(str(value), True, colour)
        surface.blit(img, (self.rect.right - 12 - img.get_width(), self._y))
        self._y += self.font.get_height() + 4

    def gap(self, px: int = 8) -> None:
        self._y += px

    @property
    def y(self) -> int:
        return self._y

    def advance(self, px: int) -> None:
        self._y += px
