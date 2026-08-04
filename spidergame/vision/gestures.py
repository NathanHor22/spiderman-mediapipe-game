"""Rule-based gesture classification on MediaPipe hand landmarks.

Two gestures, deliberately very different in shape so they can never be
confused for one another:

  THWIP  index + pinky extended, middle + ring folded.
  PUNCH  a fist that is rapidly growing in apparent size.

The thumb is ignored everywhere. It is the least reliably tracked digit, it
folds sideways rather than inward so the extension test is weaker on it, and
neither gesture needs it to be unambiguous. Dropping it buys accuracy for free.

Finger extension is measured from each finger's own MCP-PIP-DIP-tip chain.  A
straight finger has an end-to-end distance close to its full bone-chain length;
a curled finger does not.  This is independent of which way the palm faces and
does not penalise a short or splayed pinky the way a wrist-to-tip ratio does.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

# MediaPipe Hands landmark indices.
WRIST = 0
THUMB_MCP, THUMB_TIP = 2, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Four fingers as (mcp, pip, dip, tip). Thumb excluded, see module docstring.
FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

# dist(mcp, tip) / sum(each bone length). A perfectly straight finger is 1.0;
# curling lowers the ratio. The dead zone absorbs tracking jitter and natural
# bends at the joints. Thwip handles a folded-side ambiguous result separately
# because curled middle/ring fingertips are often hidden by the palm.
EXTENDED_STRAIGHTNESS = 0.86
FOLDED_STRAIGHTNESS = 0.80

EXTENDED, AMBIGUOUS, FOLDED = 1, 0, -1


def _dist3(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _dist2(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def palm_scale(lm) -> float:
    """Apparent size of the palm, as a fraction of frame width.

    Strictly 2D: this is our cheap proxy for distance-to-camera, so it has to
    measure how big the hand *looks*, not how big it is. Wrist-to-middle-MCP is
    the right span to use because it is pose invariant — unlike a bounding box,
    it does not shrink when the fingers curl, so closing a fist cannot by itself
    look like a punch.
    """
    return _dist2(lm[WRIST], lm[MIDDLE_MCP])


def finger_states(lm) -> dict[str, int]:
    """EXTENDED / FOLDED / AMBIGUOUS per finger.

    Uses the ratio of each finger's end-to-end distance to the length of its
    three-bone chain. This survives palm/back views, wrist rotation and finger
    splay. When available, callers should pass MediaPipe's metric world
    landmarks rather than image-normalised landmarks.
    """
    out = {}
    for name, joints in FINGERS.items():
        chain = sum(_dist3(lm[a], lm[b]) for a, b in zip(joints, joints[1:]))
        if chain < 1e-6:
            out[name] = AMBIGUOUS
            continue
        straightness = _dist3(lm[joints[0]], lm[joints[-1]]) / chain
        if straightness >= EXTENDED_STRAIGHTNESS:
            out[name] = EXTENDED
        elif straightness <= FOLDED_STRAIGHTNESS:
            out[name] = FOLDED
        else:
            out[name] = AMBIGUOUS
    return out


def is_thwip(states: dict[str, int]) -> bool:
    """Index and pinky out, middle and ring curled towards the palm.

    A curled fingertip is frequently occluded in back-of-hand views. Accepting
    AMBIGUOUS for the two inward fingers avoids dropping a genuine pose while
    still rejecting an open hand, whose middle/ring fingers are EXTENDED.
    """
    return (
        states["index"] == EXTENDED
        and states["pinky"] == EXTENDED
        and states["middle"] != EXTENDED
        and states["ring"] != EXTENDED
    )


def is_fist(states: dict[str, int]) -> bool:
    return all(states[f] == FOLDED for f in FINGERS)


def palm_centre(lm) -> tuple[float, float]:
    """Normalised (x, y) of the palm — steadier than the wrist point alone."""
    pts = (lm[WRIST], lm[INDEX_MCP], lm[PINKY_MCP])
    return (
        sum(p[0] for p in pts) / 3.0,
        sum(p[1] for p in pts) / 3.0,
    )


class PoseLatch:
    """Time-based hysteresis for a sustained pose.

    Asymmetric on purpose: quick to engage, slow to release. A web that takes
    a moment to attach feels responsive; a web that drops because tracking
    blinked briefly feels broken, and mid-swing is the worst possible moment
    for it. Durations, rather than frame counts, keep release/re-arm behaviour
    consistent when MediaPipe runs at 9 fps on one launch and 30 fps on another.
    """

    def __init__(self, on_s: float = 0.04, off_s: float = 0.14) -> None:
        self.on_s = on_s
        self.off_s = off_s
        self._state = False
        self._pending_since: float | None = None

    def update(self, raw: bool, now: float) -> bool:
        if raw == self._state:
            self._pending_since = None
            return self._state

        if self._pending_since is None or now < self._pending_since:
            self._pending_since = now
            return self._state

        needed = self.on_s if raw else self.off_s
        if now - self._pending_since >= needed:
            self._state = raw
            self._pending_since = None
        return self._state

    @property
    def state(self) -> bool:
        return self._state


@dataclass
class PunchDetector:
    """Fires on a fist that is accelerating towards the camera.

    The fist pose alone is not enough — a hand resting in a loose fist would
    machine-gun. What makes a punch a punch is the forward motion, which we read
    as the growth rate of the apparent palm size. Rate is relative (per second,
    as a fraction of current size) so it does not care how close you sit to the
    camera or how big your hands are.
    """

    growth_threshold: float = 2.5  # relative palm growth per second
    refractory_s: float = 0.40
    window_s: float = 0.16

    _history: deque = field(default_factory=lambda: deque(maxlen=12))
    _last_fire: float = -999.0
    rate: float = 0.0  # exposed for the calibration harness

    def reset(self) -> None:
        self._history.clear()
        self.rate = 0.0

    def update(self, fist: bool, scale: float, now: float) -> bool:
        self._history.append((now, scale))

        # Compare against the oldest sample still inside the window.
        oldest = None
        for t, s in self._history:
            if now - t <= self.window_s:
                oldest = (t, s)
                break
        if oldest is None or len(self._history) < 3:
            self.rate = 0.0
            return False

        t0, s0 = oldest
        dt = now - t0
        if dt < 1e-3 or s0 < 1e-6:
            self.rate = 0.0
            return False

        self.rate = ((scale - s0) / s0) / dt

        if not fist:
            return False
        if now - self._last_fire < self.refractory_s:
            return False
        if self.rate < self.growth_threshold:
            return False

        self._last_fire = now
        self._history.clear()
        return True
