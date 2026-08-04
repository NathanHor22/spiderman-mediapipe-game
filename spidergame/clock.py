"""Two clocks, because spider-sense slows the world but not the webcam.

Nothing uses time dilation yet. It goes in now anyway: retrofitting scaled dt
into physics that has already been tuned against raw frame time means retuning
every constant in the game, and the Goblin encounter is built on this.

The rule is simple and absolute — physics, world scroll and animation read
`game_dt`; UI, the camera thread and the gesture pipeline read `real_dt`. The
vision side must never be slowed, because running detection at full speed while
the world crawls is the entire reason bullet-time makes the punch gesture
viable in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GameClock:
    time_scale: float = 1.0
    target_scale: float = 1.0

    # Easing in is quick and sharp (the world drops away), easing out is slower
    # with a kick on the far side (speed comes rushing back).
    ease_in_s: float = 0.12
    ease_out_s: float = 0.20

    real_dt: float = 0.0
    game_dt: float = 0.0
    real_time: float = 0.0
    game_time: float = 0.0

    # Guards against a hitch — a 2s stall must not teleport the player through
    # a building because one frame integrated two seconds of gravity.
    max_dt: float = 0.05

    def tick(self, real_dt: float) -> None:
        real_dt = min(real_dt, self.max_dt)
        self.real_dt = real_dt
        self.real_time += real_dt

        if self.time_scale != self.target_scale:
            duration = (
                self.ease_in_s
                if self.target_scale < self.time_scale
                else self.ease_out_s
            )
            step = real_dt / max(duration, 1e-4)
            delta = self.target_scale - self.time_scale
            if abs(delta) <= step:
                self.time_scale = self.target_scale
            else:
                self.time_scale += step * (1.0 if delta > 0 else -1.0)

        self.game_dt = real_dt * self.time_scale
        self.game_time += self.game_dt

    def enter_slowmo(self, scale: float = 0.25) -> None:
        self.target_scale = scale

    def exit_slowmo(self) -> None:
        self.target_scale = 1.0
