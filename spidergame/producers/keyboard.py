"""Keyboard/mouse stand-in for the webcam.

This is not a fallback, it is the tuning rig. Swing physics get dialled in here
first, where input is exact and instantaneous, so that when something feels wrong
it is unambiguously the physics. Only once swinging is fun on a spacebar does the
vision producer get plugged in.

  hold SPACE / left mouse ... thwip
  mouse position ........... hand_x / hand_y
  F / right mouse .......... punch
  SHIFT .................... pretend a second hand is up
"""

from __future__ import annotations

import pygame

from ..control import ControlState


class KeyboardProducer:
    def __init__(self) -> None:
        self._punch_latched = False

    def handle_event(self, event: pygame.event.Event) -> None:
        """Feed pygame events in so transients are edge-detected, not sampled.

        Polling get_pressed() for the punch key would miss a fast tap that
        started and ended inside one frame.
        """
        if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
            self._punch_latched = True
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            self._punch_latched = True

    def poll(self) -> ControlState:
        keys = pygame.key.get_pressed()
        buttons = pygame.mouse.get_pressed()
        w, h = pygame.display.get_surface().get_size()
        mx, my = pygame.mouse.get_pos()

        fired = self._punch_latched
        self._punch_latched = False

        return ControlState(
            thwip_held=keys[pygame.K_SPACE] or buttons[0],
            hand_x=min(max(mx / max(w, 1), 0.0), 1.0),
            hand_y=min(max(my / max(h, 1), 0.0), 1.0),
            num_hands=2 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1,
            punch_fired=fired,
            tracking_lost=False,
        )

    def close(self) -> None:
        pass
