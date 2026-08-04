"""The seam between input and game.

Everything downstream of ControlState is input-agnostic: the physics, the world
and the game loop never learn whether a webcam or a keyboard filled these fields
in. That is deliberate — swing feel gets tuned on the keyboard producer, where a
bad number is unambiguously a bad number and not a dropped landmark frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ControlState:
    """One frame of player intent."""

    # Sustained pose. Web is attached for as long as this is true.
    thwip_held: bool = False

    # Palm centre of the primary hand, normalised to the camera frame.
    # x: 0 = far left, 1 = far right.  y: 0 = top, 1 = bottom.
    # Picks which building we anchor to (x) and how high the anchor sits (y).
    hand_x: float = 0.5
    hand_y: float = 0.5

    # 2 unlocks the double-web burst.
    num_hands: int = 0

    # Transient. True for exactly one poll, on the rising edge of a punch.
    punch_fired: bool = False

    # No hand in frame. The game should warn and grant grace, not kill.
    tracking_lost: bool = True


class ControlProducer(Protocol):
    """Anything that can fill in a ControlState once per game frame."""

    def poll(self) -> ControlState:
        """Return the latest intent. Consumes any latched transient events."""
        ...

    def close(self) -> None:
        ...
