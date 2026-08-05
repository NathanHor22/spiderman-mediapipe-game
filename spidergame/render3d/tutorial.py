"""Input- and renderer-agnostic state for the 3D game's front end.

The Panda window should only be responsible for drawing the returned text and
forwarding normalized key names.  This module owns menu intent, the three
gesture-training gates, and countdown timing, so none of those rules need to
be hidden in GUI callbacks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class MenuIntent(str, Enum):
    """Actions a title-menu selection can request from the game."""

    START = "start"
    TRAINING = "training"
    SETTINGS = "settings"
    QUIT = "quit"


@dataclass
class TitleMenu:
    """Tiny menu model that understands normalized keyboard key names.

    Panda emits names such as ``"arrow_up"`` while tests and other front ends
    often use ``"up"``.  Both forms are accepted; no window-system constants
    are imported here.
    """

    selected: int = 0
    options: tuple[MenuIntent, ...] = (
        MenuIntent.START,
        MenuIntent.TRAINING,
        MenuIntent.SETTINGS,
        MenuIntent.QUIT,
    )

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("a title menu needs at least one option")
        if len(set(self.options)) != len(self.options):
            raise ValueError("title menu options must be unique")
        self.selected %= len(self.options)

    @property
    def intent(self) -> MenuIntent:
        """The intent represented by the current selection."""

        return self.options[self.selected]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(option.value.upper() for option in self.options)

    def move(self, delta: int) -> None:
        self.selected = (self.selected + int(delta)) % len(self.options)

    def select(self, intent: MenuIntent | str) -> None:
        requested = MenuIntent(intent)
        try:
            self.selected = self.options.index(requested)
        except ValueError as exc:
            raise ValueError(f"{requested.value!r} is not in this menu") from exc

    def activate(self) -> MenuIntent:
        return self.intent

    def handle_key(self, key: str) -> MenuIntent | None:
        """Handle one key edge and return an activated intent, if any.

        Navigation only changes ``selected``.  Enter activates it.  ``T`` is a
        direct training shortcut and Escape requests quit when that option is
        present, matching the existing keyboard front end.
        """

        normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"up", "arrow_up", "w"}:
            self.move(-1)
        elif normalized in {"down", "arrow_down", "s"}:
            self.move(1)
        elif normalized in {"enter", "return", "kp_enter"}:
            return self.activate()
        elif normalized in {"t", "training"} and MenuIntent.TRAINING in self.options:
            self.select(MenuIntent.TRAINING)
            return MenuIntent.TRAINING
        elif normalized in {"settings", "options"} and MenuIntent.SETTINGS in self.options:
            self.select(MenuIntent.SETTINGS)
            return MenuIntent.SETTINGS
        elif normalized in {"start"} and MenuIntent.START in self.options:
            self.select(MenuIntent.START)
            return MenuIntent.START
        elif normalized in {"escape", "esc", "quit"} and MenuIntent.QUIT in self.options:
            self.select(MenuIntent.QUIT)
            return MenuIntent.QUIT
        return None


class TutorialInputLike(Protocol):
    """Minimum input seam needed by :class:`TutorialController`."""

    thwip_held: bool
    tracking_lost: bool
    hand_x: float


@dataclass(frozen=True)
class TutorialInput:
    """One normalized tutorial sample.

    ``tracking_lost`` defaults to false so keyboard callers need only provide
    their held state and pointer X position.
    """

    thwip_held: bool = False
    tracking_lost: bool = False
    hand_x: float = 0.5

    @classmethod
    def from_control(cls, control: TutorialInputLike) -> "TutorialInput":
        return cls(
            thwip_held=bool(control.thwip_held),
            tracking_lost=bool(control.tracking_lost),
            hand_x=float(control.hand_x),
        )


class TutorialStep(str, Enum):
    HOLD = "hold"
    RELEASE = "release"
    SWEEP = "sweep"
    COMPLETE = "complete"


@dataclass(frozen=True)
class TutorialTimings:
    """Required stable time for each training gate, in seconds."""

    hold: float
    release: float
    sweep: float = 0.30
    decay_rate: float = 1.60

    def __post_init__(self) -> None:
        for name in ("hold", "release", "sweep"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite duration")
        if not math.isfinite(self.decay_rate) or self.decay_rate < 0.0:
            raise ValueError("decay_rate must be finite and non-negative")

    @classmethod
    def for_mode(cls, vision: bool) -> "TutorialTimings":
        return cls(
            hold=1.20 if vision else 1.00,
            release=0.50 if vision else 0.40,
        )


@dataclass(frozen=True)
class TutorialView:
    """Everything a GUI needs to draw the current training card."""

    step: TutorialStep
    step_number: int
    step_count: int
    title: str
    hint: str
    status: str
    prompt: str
    progress: float
    done: bool
    seen_left: bool = False
    seen_right: bool = False


_VISION_COPY = {
    TutorialStep.HOLD: (
        "MAKE THE WEB-SHOOTER SIGN",
        "Index and pinky out; curl the middle and ring fingers.",
        "hold it...",
    ),
    TutorialStep.RELEASE: (
        "NOW LET GO",
        "Relax the sign while keeping your hand in the camera frame.",
        "release it...",
    ),
    TutorialStep.SWEEP: (
        "AIM LEFT AND RIGHT",
        "Sweep your hand to both sides to choose a web anchor.",
        "sweep both sides...",
    ),
}

_KEYBOARD_COPY = {
    TutorialStep.HOLD: (
        "HOLD SPACE",
        "Space or the left mouse button fires and holds a web.",
        "hold it...",
    ),
    TutorialStep.RELEASE: (
        "NOW RELEASE IT",
        "Let go near the top of the arc to launch forward.",
        "release it...",
    ),
    TutorialStep.SWEEP: (
        "AIM LEFT AND RIGHT",
        "Move the mouse to both sides to choose a web anchor.",
        "move across both sides...",
    ),
}


class TutorialController:
    """Sequential hold, release and left/right training gates.

    Vision mode treats lost tracking as invalid input.  Keyboard mode ignores
    that field, which means the same ``ControlState`` object can be passed in
    even though its conservative default is ``tracking_lost=True``.
    """

    STEP_COUNT = 3

    def __init__(
        self,
        vision: bool,
        *,
        timings: TutorialTimings | None = None,
        left_threshold: float = 0.35,
        right_threshold: float = 0.65,
    ) -> None:
        if not (0.0 <= left_threshold < right_threshold <= 1.0):
            raise ValueError("aim thresholds must satisfy 0 <= left < right <= 1")
        self.vision = bool(vision)
        self.timings = timings or TutorialTimings.for_mode(self.vision)
        self.left_threshold = float(left_threshold)
        self.right_threshold = float(right_threshold)
        self.reset()

    def reset(self) -> None:
        self.step = TutorialStep.HOLD
        self._elapsed = 0.0
        self._seen_left = False
        self._seen_right = False
        self._sample = TutorialInput()

    @property
    def done(self) -> bool:
        return self.step is TutorialStep.COMPLETE

    @property
    def index(self) -> int:
        return {
            TutorialStep.HOLD: 0,
            TutorialStep.RELEASE: 1,
            TutorialStep.SWEEP: 2,
            TutorialStep.COMPLETE: self.STEP_COUNT,
        }[self.step]

    @staticmethod
    def _validated_dt(dt: float) -> float:
        value = float(dt)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("dt must be finite and non-negative")
        return value

    def _tracked(self, sample: TutorialInput) -> bool:
        return not self.vision or not sample.tracking_lost

    def _advance_gate(self, valid: bool, dt: float, duration: float) -> bool:
        if valid:
            self._elapsed += dt
        else:
            self._elapsed = max(
                0.0,
                self._elapsed - dt * self.timings.decay_rate,
            )
        return self._elapsed >= duration

    def update(self, control: TutorialInputLike | TutorialInput, dt: float) -> bool:
        """Advance one frame and return whether training is now complete."""

        dt = self._validated_dt(dt)
        sample = (
            control
            if isinstance(control, TutorialInput)
            else TutorialInput.from_control(control)
        )
        if not math.isfinite(sample.hand_x):
            sample = TutorialInput(
                thwip_held=sample.thwip_held,
                tracking_lost=sample.tracking_lost,
                hand_x=0.5,
            )
        self._sample = sample
        if self.done:
            return True

        tracked = self._tracked(sample)
        if self.step is TutorialStep.HOLD:
            if self._advance_gate(
                tracked and sample.thwip_held,
                dt,
                self.timings.hold,
            ):
                self.step = TutorialStep.RELEASE
                self._elapsed = 0.0
        elif self.step is TutorialStep.RELEASE:
            if self._advance_gate(
                tracked and not sample.thwip_held,
                dt,
                self.timings.release,
            ):
                self.step = TutorialStep.SWEEP
                self._elapsed = 0.0
                self._seen_left = False
                self._seen_right = False
        else:
            if tracked:
                if sample.hand_x <= self.left_threshold:
                    self._seen_left = True
                if sample.hand_x >= self.right_threshold:
                    self._seen_right = True
            both_sides = tracked and self._seen_left and self._seen_right
            if self._advance_gate(both_sides, dt, self.timings.sweep):
                self.step = TutorialStep.COMPLETE
                self._elapsed = 0.0
        return self.done

    def _status(self) -> str:
        tracked = self._tracked(self._sample)
        if self.done:
            return "TRAINING COMPLETE"
        if self.vision and not tracked:
            if self.step is TutorialStep.RELEASE:
                return "KEEP YOUR HAND IN FRAME"
            return "NO HAND - MOVE INTO FRAME"
        if self.step is TutorialStep.HOLD:
            if self._sample.thwip_held:
                return (
                    "WEB-SHOOTER DETECTED"
                    if self.vision
                    else "SPACE HELD"
                )
            return "MAKE THE SIGN" if self.vision else "HOLD SPACE"
        if self.step is TutorialStep.RELEASE:
            if self._sample.thwip_held:
                return "RELAX THE SIGN" if self.vision else "LET GO OF SPACE"
            return "RELEASE DETECTED"
        if not self._seen_left and not self._seen_right:
            return "MOVE LEFT, THEN RIGHT" if self.vision else "MOVE MOUSE LEFT"
        if self._seen_left and not self._seen_right:
            return "NOW MOVE RIGHT"
        if self._seen_right and not self._seen_left:
            return "NOW MOVE LEFT"
        return "HAND AIM TRACKING" if self.vision else "AIM RANGE DETECTED"

    def _progress(self) -> float:
        if self.done:
            return 1.0
        if self.step is TutorialStep.HOLD:
            return min(1.0, self._elapsed / self.timings.hold)
        if self.step is TutorialStep.RELEASE:
            return min(1.0, self._elapsed / self.timings.release)
        sides = int(self._seen_left) + int(self._seen_right)
        if sides < 2:
            return sides * 0.5
        settle = min(1.0, self._elapsed / self.timings.sweep)
        return 0.5 + settle * 0.5

    @property
    def view(self) -> TutorialView:
        if self.done:
            return TutorialView(
                step=self.step,
                step_number=self.STEP_COUNT,
                step_count=self.STEP_COUNT,
                title="READY TO SWING",
                hint="Training complete. The run is about to begin.",
                status=self._status(),
                prompt="get ready...",
                progress=1.0,
                done=True,
                seen_left=True,
                seen_right=True,
            )
        copy = _VISION_COPY if self.vision else _KEYBOARD_COPY
        title, hint, prompt = copy[self.step]
        return TutorialView(
            step=self.step,
            step_number=self.index + 1,
            step_count=self.STEP_COUNT,
            title=title,
            hint=hint,
            status=self._status(),
            prompt=prompt,
            progress=self._progress(),
            done=False,
            seen_left=self._seen_left,
            seen_right=self._seen_right,
        )


class Countdown:
    """Fixed-step ``3, 2, 1, GO`` countdown driven only by ``dt``."""

    def __init__(
        self,
        *,
        labels: tuple[str, ...] = ("3", "2", "1", "GO"),
        step_seconds: float = 0.55,
    ) -> None:
        if not labels:
            raise ValueError("countdown needs at least one label")
        if not math.isfinite(step_seconds) or step_seconds <= 0.0:
            raise ValueError("step_seconds must be a positive finite duration")
        self.labels = tuple(str(label) for label in labels)
        self.step_seconds = float(step_seconds)
        self.reset()

    def reset(self) -> None:
        self.elapsed = 0.0

    @property
    def duration(self) -> float:
        return len(self.labels) * self.step_seconds

    @property
    def done(self) -> bool:
        return self.elapsed >= self.duration

    @property
    def index(self) -> int:
        if self.done:
            return len(self.labels)
        return min(int(self.elapsed / self.step_seconds), len(self.labels) - 1)

    @property
    def text(self) -> str:
        return "" if self.done else self.labels[self.index]

    @property
    def status(self) -> str:
        if self.done:
            return ""
        return "SWING!" if self.index == len(self.labels) - 1 else "GET READY"

    @property
    def progress(self) -> float:
        """Overall countdown progress in the inclusive range zero to one."""

        return min(1.0, self.elapsed / self.duration)

    @property
    def step_progress(self) -> float:
        if self.done:
            return 1.0
        return (self.elapsed % self.step_seconds) / self.step_seconds

    def update(self, dt: float) -> bool:
        """Advance the timer and return whether play may begin."""

        dt = TutorialController._validated_dt(dt)
        self.elapsed = min(self.duration, self.elapsed + dt)
        return self.done


__all__ = [
    "Countdown",
    "MenuIntent",
    "TitleMenu",
    "TutorialController",
    "TutorialInput",
    "TutorialInputLike",
    "TutorialStep",
    "TutorialTimings",
    "TutorialView",
]
