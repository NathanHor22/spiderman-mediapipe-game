"""Small procedural soundscape for web swinging.

The game deliberately has no external audio assets.  These effects are built
once, in memory, as signed 16-bit stereo PCM and handed to pygame's mixer.  A
missing or broken audio device is never fatal: :class:`SoundSystem` simply
becomes a silent object with the same public API.

Call :func:`prepare_mixer` before ``pygame.init()`` so SDL uses the intended
low-latency format, then construct ``SoundSystem`` after pygame is initialised.
"""

from __future__ import annotations

import math
import random
from array import array
from collections.abc import Iterable
from typing import Any, Callable

import pygame

from .game.swing import SwingEvent


SAMPLE_RATE = 44_100
SAMPLE_SIZE = -16
OUTPUT_CHANNELS = 2
MIXER_BUFFER = 512

SHOT_CHANNEL = 0
CABLE_CHANNEL = 1
MOTION_CHANNEL = 2
RESERVED_CHANNELS = 3


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


StereoVoice = Callable[[int, float, float], float | tuple[float, float]]


def _render_pcm(duration: float, voice: StereoVoice) -> bytes:
    """Render a mono-or-stereo voice to mixer-ready stereo PCM."""
    frame_count = max(1, round(duration * SAMPLE_RATE))
    samples = array("h")
    append = samples.append
    scale = 32_767

    for frame in range(frame_count):
        t = frame / SAMPLE_RATE
        progress = frame / max(1, frame_count - 1)
        value = voice(frame, t, progress)
        if isinstance(value, tuple):
            left, right = value
        else:
            left = right = value
        append(round(_clamp(left, -1.0, 1.0) * scale))
        append(round(_clamp(right, -1.0, 1.0) * scale))

    return samples.tobytes()


def _shot_pcm() -> bytes:
    rng = random.Random(0x51D3)
    noise = [rng.uniform(-1.0, 1.0) for _ in range(round(0.155 * SAMPLE_RATE))]

    def voice(frame: int, t: float, progress: float) -> float:
        # A steep chirp gives the comic-book "thwip"; a short noisy attack
        # prevents it from reading as a clean electronic beep.
        chirp_phase = math.tau * (1_550.0 * t - 0.5 * 1_280.0 * t * t / 0.155)
        attack = 1.0 - math.exp(-180.0 * t)
        envelope = attack * (1.0 - progress) ** 2.7
        snap = noise[frame] * math.exp(-42.0 * t)
        return envelope * (0.68 * math.sin(chirp_phase) + 0.27 * snap)

    return _render_pcm(0.155, voice)


def _attach_pcm() -> bytes:
    rng = random.Random(0xA77A)
    noise = [rng.uniform(-1.0, 1.0) for _ in range(round(0.130 * SAMPLE_RATE))]

    def voice(frame: int, t: float, progress: float) -> float:
        twang = math.sin(math.tau * (176.0 * t + 31.0 * t * t))
        overtone = math.sin(math.tau * (535.0 * t - 95.0 * t * t))
        body = math.exp(-19.0 * t) * (0.56 * twang + 0.22 * overtone)
        click = 0.30 * noise[frame] * math.exp(-75.0 * t)
        return (1.0 - progress) * (body + click)

    return _render_pcm(0.130, voice)


def _miss_pcm() -> bytes:
    rng = random.Random(0xB1FF)
    raw = [rng.uniform(-1.0, 1.0) for _ in range(round(0.190 * SAMPLE_RATE))]
    filtered: list[float] = []
    fast = slow = 0.0
    for sample in raw:
        fast += 0.24 * (sample - fast)
        slow += 0.035 * (sample - slow)
        filtered.append(fast - slow)

    def voice(frame: int, t: float, progress: float) -> tuple[float, float]:
        envelope = math.sin(math.pi * progress) * (1.0 - progress) ** 0.35
        air = filtered[frame] * 1.7
        # Slight stereo offset makes a missed line feel like it passed by.
        pan = progress * 1.4 - 0.7
        return (
            air * envelope * (0.62 - 0.20 * pan),
            air * envelope * (0.62 + 0.20 * pan),
        )

    return _render_pcm(0.190, voice)


def _release_pcm() -> bytes:
    def voice(_frame: int, t: float, progress: float) -> float:
        phase = math.tau * (255.0 * t + 0.5 * 760.0 * t * t / 0.105)
        envelope = (1.0 - math.exp(-150.0 * t)) * (1.0 - progress) ** 3
        return 0.52 * envelope * math.sin(phase)

    return _render_pcm(0.105, voice)


def _whoosh_pcm() -> bytes:
    """Build a seamless, noise-like loop from periodic partials."""
    duration = 0.750
    frame_count = round(duration * SAMPLE_RATE)
    rng = random.Random(0x5A17)
    # Integer cycle counts make every partial meet itself at the loop seam.
    harmonics = [rng.randint(95, 1_850) for _ in range(18)]
    phases_left = [rng.uniform(0.0, math.tau) for _ in harmonics]
    phases_right = [phase + rng.uniform(-0.8, 0.8) for phase in phases_left]
    weights = [1.0 / math.sqrt(harmonic) for harmonic in harmonics]
    normaliser = 0.48 / sum(weights)

    def voice(frame: int, _t: float, _progress: float) -> tuple[float, float]:
        turn = math.tau * frame / frame_count
        left = right = 0.0
        for harmonic, phase_l, phase_r, weight in zip(
            harmonics, phases_left, phases_right, weights
        ):
            left += math.sin(turn * harmonic + phase_l) * weight
            right += math.sin(turn * harmonic + phase_r) * weight
        return left * normaliser, right * normaliser

    return _render_pcm(duration, voice)


def _build_pcm_bank() -> dict[str, bytes]:
    """Return a deterministic bank; kept separate for mixer-free tests."""
    return {
        "shot": _shot_pcm(),
        "attach": _attach_pcm(),
        "miss": _miss_pcm(),
        "release": _release_pcm(),
        "whoosh": _whoosh_pcm(),
    }


def prepare_mixer() -> None:
    """Request the format used by the procedural buffers before pygame init."""
    try:
        pygame.mixer.pre_init(
            frequency=SAMPLE_RATE,
            size=SAMPLE_SIZE,
            channels=OUTPUT_CHANNELS,
            buffer=MIXER_BUFFER,
        )
    except pygame.error:
        # SoundSystem will make a second, guarded attempt after pygame.init().
        pass


class SoundSystem:
    """Play swing events while degrading cleanly to silence.

    ``mixer`` is injectable for tests. Game code should leave it unset.
    Public methods are safe to call even when ``enabled`` is false.
    """

    def __init__(self, master_volume: float = 0.75, *, mixer=None) -> None:
        self.master_volume = _clamp(_safe_number(master_volume, 0.75))
        self.enabled = False
        self.error = ""
        self._mixer = pygame.mixer if mixer is None else mixer
        self._sounds: dict[str, Any] = {}
        self._shot_channel = None
        self._cable_channel = None
        self._motion_channel = None

        try:
            if self._mixer.get_init() is None:
                self._mixer.init(
                    frequency=SAMPLE_RATE,
                    size=SAMPLE_SIZE,
                    channels=OUTPUT_CHANNELS,
                    buffer=MIXER_BUFFER,
                )

            current_channels = self._mixer.get_num_channels()
            if current_channels < RESERVED_CHANNELS:
                self._mixer.set_num_channels(RESERVED_CHANNELS)
            self._mixer.set_reserved(RESERVED_CHANNELS)

            self._shot_channel = self._mixer.Channel(SHOT_CHANNEL)
            self._cable_channel = self._mixer.Channel(CABLE_CHANNEL)
            self._motion_channel = self._mixer.Channel(MOTION_CHANNEL)
            self._sounds = {
                name: self._mixer.Sound(buffer=pcm)
                for name, pcm in _build_pcm_bank().items()
            }

            self._shot_channel.set_volume(self.master_volume * 0.82)
            self._cable_channel.set_volume(self.master_volume * 0.72)
            self._motion_channel.set_volume(0.0)
            self.enabled = True
        except pygame.error as exc:
            self._disable(exc)

    def _disable(self, exc: BaseException) -> None:
        self.enabled = False
        self.error = str(exc)
        self._sounds.clear()
        self._shot_channel = None
        self._cable_channel = None
        self._motion_channel = None

    def _start_motion(self) -> None:
        if self._motion_channel is None or self._motion_channel.get_busy():
            return
        self._motion_channel.play(self._sounds["whoosh"], loops=-1, fade_ms=80)

    def handle(self, events: Iterable[SwingEvent], sim=None) -> None:
        """Consume the events returned by ``SwingSim.update``."""
        if not self.enabled:
            return
        try:
            for event in events:
                if event is SwingEvent.SHOT:
                    self._shot_channel.play(self._sounds["shot"])
                elif event is SwingEvent.ATTACH:
                    self._cable_channel.play(self._sounds["attach"])
                    self._start_motion()
                elif event is SwingEvent.MISS:
                    self._cable_channel.play(self._sounds["miss"])
                elif event is SwingEvent.RELEASE:
                    self._cable_channel.play(self._sounds["release"])
                    self._motion_channel.fadeout(100)
            if sim is not None:
                self.update(sim)
        except pygame.error as exc:
            self._disable(exc)

    def update(self, sim) -> None:
        """Match the continuous whoosh to current swing speed and tension."""
        if not self.enabled:
            return
        try:
            if not bool(getattr(sim, "attached", False)):
                if self._motion_channel.get_busy():
                    self._motion_channel.fadeout(90)
                return

            self._start_motion()
            speed = max(0.0, _safe_number(getattr(sim, "speed", 0.0)))
            angular = max(0.0, _safe_number(getattr(sim, "angular_speed", 0.0)))
            tension = max(0.0, _safe_number(getattr(sim, "web_tension", 0.0)))
            speed_gain = _clamp((speed - 18.0) / 105.0)
            angular_gain = _clamp(angular / 2.4)
            tension_gain = 0.72 + 0.28 * _clamp(tension / 90.0)
            taut_gain = 1.0 if bool(getattr(sim, "rope_taut", False)) else 0.22
            motion = max(0.06, 0.68 * speed_gain + 0.32 * angular_gain)
            volume = self.master_volume * 0.42 * motion * tension_gain * taut_gain

            # Lateral velocity provides a restrained stereo cue without
            # changing pitch (which pygame's mixer does not support).
            pan = _clamp(_safe_number(getattr(sim, "vx", 0.0)) / 75.0,
                         -0.32, 0.32)
            self._motion_channel.set_volume(
                _clamp(volume * (1.0 - pan)),
                _clamp(volume * (1.0 + pan)),
            )
        except pygame.error as exc:
            self._disable(exc)

    def stop(self, fade_ms: int = 80) -> None:
        """Stop every owned channel, for title/death/reset transitions."""
        if not self.enabled:
            return
        try:
            fade_ms = max(0, int(fade_ms))
            for channel in (
                self._shot_channel,
                self._cable_channel,
                self._motion_channel,
            ):
                if fade_ms:
                    channel.fadeout(fade_ms)
                else:
                    channel.stop()
        except (pygame.error, TypeError, ValueError) as exc:
            self._disable(exc)

    def close(self) -> None:
        """Release this system's channels without quitting pygame's mixer."""
        self.stop(fade_ms=0)


__all__ = [
    "SoundSystem",
    "prepare_mixer",
    "SAMPLE_RATE",
    "SHOT_CHANNEL",
    "CABLE_CHANNEL",
    "MOTION_CHANNEL",
]
