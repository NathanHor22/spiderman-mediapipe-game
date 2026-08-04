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

# Confidence ramps are deliberately wider than the debug-state dead zone. A
# real pinky is commonly a little bent even in a good web-shooter pose, while a
# partly hidden curled finger can be reconstructed a little too straight. The
# continuous score lets the temporal detector be strict while acquiring and
# forgiving only after a web has genuinely been established.
EXTENSION_CONFIDENCE_LOW = 0.72
EXTENSION_CONFIDENCE_HIGH = 0.88
FOLD_CONFIDENCE_LOW = 0.78
FOLD_CONFIDENCE_HIGH = 0.90

THWIP_ACQUIRE_CONFIDENCE = 0.55
THWIP_HOLD_CONFIDENCE = 0.25

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


def finger_straightness(lm) -> dict[str, float | None]:
    """Continuous straightness ratio for each non-thumb finger.

    Uses the ratio of each finger's end-to-end distance to the length of its
    three-bone chain. This survives palm/back views, wrist rotation and finger
    splay. When available, callers should pass MediaPipe's metric world
    landmarks rather than image-normalised landmarks.

    ``None`` means the landmark chain was degenerate. Keeping that distinct
    from 0.0 prevents a tracking failure from looking like a confidently
    folded finger.
    """
    out: dict[str, float | None] = {}
    for name, joints in FINGERS.items():
        chain = sum(_dist3(lm[a], lm[b]) for a, b in zip(joints, joints[1:]))
        if chain < 1e-6:
            out[name] = None
            continue
        out[name] = _dist3(lm[joints[0]], lm[joints[-1]]) / chain
    return out


def states_from_straightness(
    straightness: dict[str, float | None],
) -> dict[str, int]:
    """Quantise continuous ratios for old callers and the debug harness."""
    out = {}
    for name in FINGERS:
        value = straightness.get(name)
        if value is None:
            out[name] = AMBIGUOUS
            continue
        if value >= EXTENDED_STRAIGHTNESS:
            out[name] = EXTENDED
        elif value <= FOLDED_STRAIGHTNESS:
            out[name] = FOLDED
        else:
            out[name] = AMBIGUOUS
    return out


def finger_states(lm) -> dict[str, int]:
    """EXTENDED / FOLDED / AMBIGUOUS per finger.

    Compatibility wrapper around :func:`finger_straightness`. New gesture
    decisions should retain the continuous values instead of quantising them.
    """
    return states_from_straightness(finger_straightness(lm))


def _ramp(value: float | None, low: float, high: float) -> float:
    if value is None or not math.isfinite(value) or high <= low:
        return 0.0
    return min(1.0, max(0.0, (value - low) / (high - low)))


def extension_confidence(value: float | None) -> float:
    """0..1 confidence that a finger is extended."""
    return _ramp(value, EXTENSION_CONFIDENCE_LOW, EXTENSION_CONFIDENCE_HIGH)


def fold_confidence(value: float | None) -> float:
    """0..1 confidence that a finger is folded."""
    if value is None or not math.isfinite(value):
        return 0.0
    return 1.0 - _ramp(value, FOLD_CONFIDENCE_LOW, FOLD_CONFIDENCE_HIGH)


def thwip_confidence(straightness: dict[str, float | None]) -> float:
    """Continuous confidence for index+pinky out, middle+ring folded.

    The weakest required finger controls the result. This is intentionally
    conservative: three correct fingers must not hide an open ring finger or a
    lost pinky landmark.
    """
    return min(
        extension_confidence(straightness.get("index")),
        extension_confidence(straightness.get("pinky")),
        fold_confidence(straightness.get("middle")),
        fold_confidence(straightness.get("ring")),
    )


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


class ThwipLatch:
    """Confidence-aware latch with stricter acquisition than maintenance.

    Ambiguous curled fingertips are useful evidence while maintaining a web,
    because those fingertips are often occluded by the palm. They are not
    enough to *start* a web: acquisition requires every required finger to
    clear the higher confidence threshold for a short, time-based interval.
    """

    def __init__(
        self,
        on_s: float = 0.06,
        off_s: float = 0.16,
        acquire_confidence: float = THWIP_ACQUIRE_CONFIDENCE,
        hold_confidence: float = THWIP_HOLD_CONFIDENCE,
    ) -> None:
        if not 0.0 <= hold_confidence <= acquire_confidence <= 1.0:
            raise ValueError("expected 0 <= hold confidence <= acquire confidence <= 1")
        self.acquire_confidence = acquire_confidence
        self.hold_confidence = hold_confidence
        self._latch = PoseLatch(on_s=on_s, off_s=off_s)
        self.raw = False

    def update(self, confidence: float, now: float) -> bool:
        threshold = (
            self.hold_confidence if self._latch.state else self.acquire_confidence
        )
        self.raw = confidence >= threshold
        return self._latch.update(self.raw, now)

    @property
    def state(self) -> bool:
        return self._latch.state

    @property
    def threshold(self) -> float:
        return self.hold_confidence if self.state else self.acquire_confidence


class AimSmoother:
    """Time-based EMA for the palm centre used to choose an anchor.

    Time-based alpha keeps the feel consistent across fast and slow cameras.
    After a longer tracking gap the new position is accepted immediately so a
    stale palm location cannot drag the next shot across the screen.
    """

    def __init__(
        self,
        time_constant_s: float = 0.08,
        reset_gap_s: float = 0.35,
    ) -> None:
        if time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be positive")
        self.time_constant_s = time_constant_s
        self.reset_gap_s = reset_gap_s
        self._value: tuple[float, float] | None = None
        self._last_time: float | None = None

    def update(self, x: float, y: float, now: float) -> tuple[float, float]:
        sample = (x, y)
        if self._value is None or self._last_time is None:
            self._value = sample
        else:
            dt = now - self._last_time
            if dt <= 0.0 or dt >= self.reset_gap_s:
                self._value = sample
            else:
                alpha = 1.0 - math.exp(-dt / self.time_constant_s)
                self._value = tuple(
                    old + alpha * (new - old)
                    for old, new in zip(self._value, sample)
                )
        self._last_time = now
        return self._value

    @property
    def value(self) -> tuple[float, float] | None:
        return self._value


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
