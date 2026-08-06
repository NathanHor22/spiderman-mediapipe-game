"""Soundscape for web swinging: recorded cues over a procedural bed.

Every effect exists twice. Recorded files in ``assets/audio`` are preferred, and
a synthesised equivalent is built in memory as a fallback — so the game still
has a full set of cues if an asset is missing, and the test suite can run
against a fake mixer with no files and no audio device at all. A missing or
broken device is never fatal: :class:`SoundSystem` becomes a silent object with
the same public API.

The long menu theme is streamed through ``pygame.mixer.music`` rather than
loaded as a ``Sound``. Music is a single always-looping stream, which is exactly
what that API is for, and it keeps a 1.4MB decode off the channel mixer.

Call :func:`prepare_mixer` before ``pygame.init()`` so SDL uses the intended
low-latency format, then construct ``SoundSystem`` after pygame is initialised.
"""

from __future__ import annotations

import math
import random
from array import array
from collections.abc import Iterable
from pathlib import Path
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
IMPACT_CHANNEL = 3
RESERVED_CHANNELS = 4

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "audio"

# Recorded cues, by the bank name they replace. Anything absent falls back to
# the synthesised version of the same name.
SOUND_FILES = {
    "attach": "web-swing.mp3",
    "impact": "fall-impact.mp3",
}
MUSIC_MENU = "menu-theme.mp3"

# Recorded cues are trimmed to their audible span on load. Exported effects
# routinely carry seconds of leading silence — the supplied web-swing has 1.7s
# of it before the whoosh — and a cue that starts late is a cue you never hear,
# because the next swing replaces it on the channel first.
SILENCE_FLOOR = 0.06       # fraction of the file's own peak
TRIM_LEAD_IN = 0.015       # keep a sliver before the first audible sample
TRIM_TAIL = 0.120          # and let the decay breathe
TRIM_FADE_IN = 0.004       # de-click the new edges
TRIM_FADE_OUT = 0.060


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


def _impact_pcm() -> bytes:
    """Fallback for hitting the street, if the recorded thud is missing.

    Death is the one moment the player most needs audible confirmation of, so it
    gets a synthesised backstop rather than silently having no cue.
    """
    rng = random.Random(0x0DEAD)
    raw = [rng.uniform(-1.0, 1.0) for _ in range(round(0.420 * SAMPLE_RATE))]
    rumble: list[float] = []
    low = 0.0
    for sample in raw:
        low += 0.045 * (sample - low)
        rumble.append(low)

    def voice(frame: int, t: float, progress: float) -> float:
        thud = math.sin(math.tau * (74.0 * t - 26.0 * t * t)) * math.exp(-7.0 * t)
        crack = rumble[frame] * 5.2 * math.exp(-11.0 * t)
        return (1.0 - progress) ** 1.4 * (0.62 * thud + 0.5 * crack)

    return _render_pcm(0.420, voice)


def trim_silence(raw: bytes) -> bytes:
    """Crop stereo 16-bit PCM to its audible span, with de-clicking fades.

    Returns the input unchanged when it is already tight or entirely silent, so
    a cue that needs no trimming is never degraded by round-tripping.
    """
    samples = array("h")
    samples.frombytes(raw)
    frames = len(samples) // OUTPUT_CHANNELS
    if frames < 2:
        return raw

    magnitude = [
        max(abs(samples[i * OUTPUT_CHANNELS]), abs(samples[i * OUTPUT_CHANNELS + 1]))
        for i in range(frames)
    ]
    peak = max(magnitude)
    if peak <= 0:
        return raw

    floor = peak * SILENCE_FLOOR
    first = next((i for i, m in enumerate(magnitude) if m >= floor), None)
    if first is None:
        return raw
    last = next(
        i for i in range(frames - 1, -1, -1) if magnitude[i] >= floor
    )

    start = max(0, first - round(TRIM_LEAD_IN * SAMPLE_RATE))
    end = min(frames, last + 1 + round(TRIM_TAIL * SAMPLE_RATE))
    if start == 0 and end == frames:
        return raw

    cropped = samples[start * OUTPUT_CHANNELS:end * OUTPUT_CHANNELS]
    cropped_frames = len(cropped) // OUTPUT_CHANNELS

    fade_in = min(round(TRIM_FADE_IN * SAMPLE_RATE), cropped_frames)
    fade_out = min(round(TRIM_FADE_OUT * SAMPLE_RATE), cropped_frames)
    for i in range(fade_in):
        gain = i / fade_in
        for c in range(OUTPUT_CHANNELS):
            cropped[i * OUTPUT_CHANNELS + c] = round(
                cropped[i * OUTPUT_CHANNELS + c] * gain)
    for i in range(fade_out):
        gain = i / fade_out
        frame = cropped_frames - 1 - i
        for c in range(OUTPUT_CHANNELS):
            cropped[frame * OUTPUT_CHANNELS + c] = round(
                cropped[frame * OUTPUT_CHANNELS + c] * gain)

    return cropped.tobytes()


def _build_pcm_bank() -> dict[str, bytes]:
    """Return a deterministic bank; kept separate for mixer-free tests."""
    return {
        "shot": _shot_pcm(),
        "attach": _attach_pcm(),
        "miss": _miss_pcm(),
        "release": _release_pcm(),
        "whoosh": _whoosh_pcm(),
        "impact": _impact_pcm(),
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

    def __init__(self, master_volume: float = 0.75, *, mixer=None,
                 use_assets: bool | None = None, music_volume: float = 0.5,
                 asset_dir: Path | None = None) -> None:
        self.master_volume = _clamp(_safe_number(master_volume, 0.75))
        self.music_volume = _clamp(_safe_number(music_volume, 0.5))
        self.enabled = False
        self.error = ""
        self._mixer = pygame.mixer if mixer is None else mixer
        self._asset_dir = ASSET_DIR if asset_dir is None else Path(asset_dir)
        # Recorded assets need the real mixer: an injected fake has no decoder
        # and no filesystem expectations. Tests therefore always exercise the
        # synthesised bank, which keeps their assertions deterministic.
        self._use_assets = (mixer is None) if use_assets is None else use_assets
        self.loaded_assets: list[str] = []
        self._sounds: dict[str, Any] = {}
        self._shot_channel = None
        self._cable_channel = None
        self._motion_channel = None
        self._impact_channel = None
        self._music_ready = False

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
            self._impact_channel = self._mixer.Channel(IMPACT_CHANNEL)
            self._sounds = {
                name: self._mixer.Sound(buffer=pcm)
                for name, pcm in _build_pcm_bank().items()
            }
            if self._use_assets:
                self._load_asset_sounds()

            self._shot_channel.set_volume(self.master_volume * 0.82)
            self._cable_channel.set_volume(self.master_volume * 0.72)
            self._motion_channel.set_volume(0.0)
            self._impact_channel.set_volume(self.master_volume)
            self.enabled = True
        except pygame.error as exc:
            self._disable(exc)

    def _load_asset_sounds(self) -> None:
        """Swap recorded cues in over their synthesised counterparts.

        Each file is handled independently: one unreadable asset costs you that
        one cue, not the whole bank.
        """
        for name, filename in SOUND_FILES.items():
            path = self._asset_dir / filename
            if not path.is_file():
                continue
            try:
                sound = self._mixer.Sound(str(path))
                trimmed = trim_silence(sound.get_raw())
                if trimmed:
                    sound = self._mixer.Sound(buffer=trimmed)
                self._sounds[name] = sound
                self.loaded_assets.append(name)
            except (pygame.error, AttributeError, ValueError):
                # Keep the synthesised version already in the bank.
                pass

    def _disable(self, exc: BaseException) -> None:
        self.enabled = False
        self.error = str(exc)
        self._sounds.clear()
        self._shot_channel = None
        self._cable_channel = None
        self._motion_channel = None
        self._impact_channel = None
        self.stop_music(fade_ms=0)

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

    def play_fall(self) -> None:
        """The impact of hitting the street.

        On its own channel precisely because the death transition calls
        :meth:`stop`, which fades the swing cues out. Sharing a channel would
        mean the one sound the player most needs to hear got cut off by the
        cleanup for the event that triggered it.
        """
        if not self.enabled or self._impact_channel is None:
            return
        try:
            self._impact_channel.play(self._sounds["impact"])
        except (pygame.error, KeyError) as exc:
            if isinstance(exc, pygame.error):
                self._disable(exc)

    # ------------------------------------------------------------------ music

    def play_menu_music(self, fade_ms: int = 900) -> None:
        """Loop the menu theme. Safe to call repeatedly — restarts nothing."""
        if not self._use_assets or self._music_ready:
            return
        path = self._asset_dir / MUSIC_MENU
        if not path.is_file():
            return
        try:
            pygame.mixer.music.load(str(path))
            pygame.mixer.music.set_volume(self.music_volume)
            pygame.mixer.music.play(loops=-1, fade_ms=max(0, int(fade_ms)))
            self._music_ready = True
        except pygame.error:
            # No music is survivable; the rest of the mix carries on.
            self._music_ready = False

    def stop_music(self, fade_ms: int = 600) -> None:
        # Idempotent, so the game loop can call this every frame it is not in a
        # menu without restarting a fadeout sixty times a second.
        if not self._use_assets or not self._music_ready:
            return
        self._music_ready = False
        try:
            if fade_ms > 0:
                pygame.mixer.music.fadeout(int(fade_ms))
            else:
                pygame.mixer.music.stop()
        except pygame.error:
            pass

    @property
    def music_playing(self) -> bool:
        return self._music_ready

    def stop(self, fade_ms: int = 80) -> None:
        """Stop the swing cues, for title/death/reset transitions.

        Deliberately excludes the impact channel and the music stream, both of
        which outlive the moment that triggers this.
        """
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
        self.stop_music(fade_ms=0)
        if self._impact_channel is not None:
            try:
                self._impact_channel.stop()
            except (pygame.error, TypeError, ValueError):
                pass


__all__ = [
    "SoundSystem",
    "prepare_mixer",
    "SAMPLE_RATE",
    "SHOT_CHANNEL",
    "CABLE_CHANNEL",
    "MOTION_CHANNEL",
    "IMPACT_CHANNEL",
    "ASSET_DIR",
    "SOUND_FILES",
    "MUSIC_MENU",
    "trim_silence",
]
