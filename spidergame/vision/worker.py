"""Threaded capture + inference.

Three threads, and the split between them is the whole point.

  capture thread    does nothing but cap.read() into a slot
  inference thread  takes the *newest* frame and runs MediaPipe on it
  game thread       reads the latest conclusion, never blocks

Capture and inference are separated because `cap.read()` blocks for a full frame
interval — 32ms on this camera. Running it inline with inference serialises the
two, so 32ms of waiting plus 13ms of work becomes a 22fps pipeline when the
camera could sustain 30.

More importantly, the inference thread always takes the *newest* captured frame
and drops anything it missed. Without that, a slow inference pass leaves frames
queued in the driver buffer and every one you process is already stale — the
pipeline keeps its framerate while silently accumulating latency, which is the
worst possible failure mode for a game read off a webcam.

Gesture classification lives in here too, not in the producer. The pose latch
consumes each new vision result with its real timestamp; if the game updated it
at 60fps while frames arrived at 25fps, the same conclusion would be applied two
or three times and the hysteresis window would mean nothing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2

from .calibration import Calibration
from .gestures import (
    PoseLatch,
    PunchDetector,
    finger_states,
    is_fist,
    is_thwip,
    palm_centre,
    palm_scale,
)


@dataclass
class GestureSnapshot:
    """Latest conclusion from the vision thread."""

    thwip_held: bool = False
    hand_x: float = 0.5
    hand_y: float = 0.5
    num_hands: int = 0
    tracking_lost: bool = True

    # Debug surface for the harness — raw, pre-hysteresis values.
    raw_thwip: bool = False
    raw_fist: bool = False
    states: dict = field(default_factory=dict)
    scale: float = 0.0
    growth_rate: float = 0.0

    # Timing, split so the harness can say *which* stage is the bottleneck
    # rather than just reporting one number that could mean either.
    vision_fps: float = 0.0
    capture_fps: float = 0.0
    inference_ms: float = 0.0
    dropped_frames: int = 0


class _CaptureThread(threading.Thread):
    """Reads frames as fast as the device allows, keeps only the newest."""

    def __init__(self, index: int, width: int, height: int) -> None:
        super().__init__(daemon=True)
        self.index = index
        self.width = width
        self.height = height

        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._fps = 0.0
        self._stop_event = threading.Event()
        self._opened = threading.Event()
        self.error: str | None = None

    def _open(self):
        # DSHOW opens in well under a second on Windows; the default MSMF
        # backend can stall for five or more.
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = cv2.VideoCapture(self.index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                # Smallest buffer the driver will accept, so a slow consumer
                # cannot build up a backlog of stale frames.
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
        return None

    def run(self) -> None:
        cap = self._open()
        if cap is None:
            self.error = f"could not open camera {self.index}"
            self._opened.set()
            return

        fps_ema = 0.0
        last = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue
                now = time.perf_counter()
                dt = now - last
                last = now
                if dt > 0:
                    inst = 1.0 / dt
                    fps_ema = inst if fps_ema == 0.0 else fps_ema * 0.9 + inst * 0.1
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                    self._fps = fps_ema
                self._opened.set()
        finally:
            cap.release()

    def latest(self):
        with self._lock:
            return self._frame, self._seq, self._fps

    def wait_open(self, timeout: float) -> bool:
        return self._opened.wait(timeout)

    def stop(self) -> None:
        self._stop_event.set()


class VisionWorker(threading.Thread):
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        calibration: Calibration | None = None,
        keep_frame: bool = False,
        num_hands: int = 1,
        infer_scale: float = 1.0,
        warmup_max_s: float = 2.0,
        warmup_target_ms: float = 30.0,
    ) -> None:
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.calibration = calibration or Calibration()
        self.keep_frame = keep_frame
        # One hand by default. The double-web burst wants two, but that is an
        # unbuilt feature and detection cost scales with the cap, so it stays
        # off until something needs it.
        self.num_hands = num_hands
        # Downscale before inference. MediaPipe rescales internally so this
        # buys less than you would expect, but it is here as a lever if a
        # slower machine needs one.
        self.infer_scale = infer_scale

        # See _warmup(). Bounded so a machine that never reaches the target
        # still starts, just slower.
        self.warmup_max_s = warmup_max_s
        self.warmup_target_ms = warmup_target_ms
        self.warmup_result_ms = 0.0
        self.warmup_passes = 0

        self._lock = threading.Lock()
        self._snapshot = GestureSnapshot()
        self._punch_pending = False
        self._frame = None
        self._landmarks: list = []
        # Not `_stop`: threading.Thread already uses that name for an internal
        # method, and shadowing it breaks is_alive() once the thread exits.
        self._stop_event = threading.Event()
        self._ready = threading.Event()
        self._error: str | None = None
        self._dropped = 0

        self._capture = _CaptureThread(camera_index, width, height)
        self._thwip_latch = PoseLatch()
        self._punch = PunchDetector(
            growth_threshold=self.calibration.punch_growth_threshold
        )

    # ---------------------------------------------------------------- reading

    def snapshot(self) -> GestureSnapshot:
        with self._lock:
            return self._snapshot

    def consume_punch(self) -> bool:
        """Read-and-clear. Latched in the worker so a punch is never missed
        when the game polls slower than inference, and never double-counted
        when it polls faster."""
        with self._lock:
            fired = self._punch_pending
            self._punch_pending = False
            return fired

    def debug_frame(self):
        """(bgr_frame, landmark_lists) for the harness. None until first frame."""
        with self._lock:
            return self._frame, self._landmarks

    def wait_ready(self, timeout: float = 15.0) -> bool:
        return self._ready.wait(timeout)

    @property
    def error(self) -> str | None:
        return self._error

    def set_punch_threshold(self, value: float) -> None:
        self._punch.growth_threshold = value
        self.calibration.punch_growth_threshold = value

    # ---------------------------------------------------------------- running

    def run(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision

            from .models import ensure_hand_model

            model_path = ensure_hand_model()
        except Exception as exc:  # pragma: no cover - setup/network failure
            self._error = f"mediapipe setup failed: {exc}"
            self._ready.set()
            return

        self._capture.start()
        if not self._capture.wait_open(20.0) or self._capture.error:
            self._error = self._capture.error or "camera did not deliver frames"
            self._ready.set()
            return

        landmarker = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
                # VIDEO rather than IMAGE: it carries tracking between frames,
                # so a hand already found is refined rather than re-detected.
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=self.num_hands,
                # Keep acquisition at MediaPipe's balanced default. Gesture
                # geometry below is selective; a stricter detector only makes
                # rotated/back-of-hand poses unnecessarily hard to reacquire.
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        stamp_ms = self._warmup(landmarker, mp)

        fps_ema = 0.0
        infer_ema = 0.0
        last_t = time.perf_counter()
        last_seq = -1

        try:
            while not self._stop_event.is_set():
                frame, seq, cap_fps = self._capture.latest()
                if frame is None or seq == last_seq:
                    # Nothing new yet — yield rather than re-running inference
                    # on a frame we have already seen.
                    time.sleep(0.002)
                    continue

                # Everything between the last processed frame and this one was
                # captured and thrown away. Surfacing the count makes it obvious
                # when inference is falling behind the camera.
                if last_seq >= 0:
                    self._dropped += max(0, seq - last_seq - 1)
                last_seq = seq

                # Mirror, so moving your hand right moves things right on screen.
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.infer_scale != 1.0:
                    rgb = cv2.resize(
                        rgb, None, fx=self.infer_scale, fy=self.infer_scale
                    )

                # VIDEO tracking needs the real elapsed time between frames.
                # Clamp to last+1 because it also rejects two timestamps that
                # happen to land inside the same clock millisecond.
                stamp_ms = max(stamp_ms + 1, int(time.perf_counter() * 1000.0))
                t_infer = time.perf_counter()
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamp_ms
                )
                infer_ms = (time.perf_counter() - t_infer) * 1000.0
                infer_ema = (
                    infer_ms if infer_ema == 0.0 else infer_ema * 0.9 + infer_ms * 0.1
                )

                now = time.perf_counter()
                dt = now - last_t
                last_t = now
                if dt > 0:
                    inst = 1.0 / dt
                    fps_ema = inst if fps_ema == 0.0 else fps_ema * 0.9 + inst * 0.1

                self._classify(result, now, fps_ema, cap_fps, infer_ema, frame)
                self._ready.set()
        finally:
            landmarker.close()
            self._capture.stop()

    def _warmup(self, landmarker, mp) -> int:
        """Run a few passes before play starts, so the first frames are not cold.

        Worth ~2 seconds for allocator and model warmup, and no more. It was
        originally 12s, on the theory that inference times on this machine were
        a CPU clock-ramp effect that sustained load would fix. That turned out
        to be wrong — see the note in the README. Inference here is bimodal
        (~15ms or ~100ms) for reasons outside this process, and a long warmup
        just spent fifteen seconds failing to influence which mode we land in.
        """
        deadline = time.perf_counter() + self.warmup_max_s
        recent: list[float] = []
        stamp = int(time.perf_counter() * 1000.0)

        frame, _, _ = self._capture.latest()
        if frame is not None:
            rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        else:
            import numpy as np

            rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        while time.perf_counter() < deadline and not self._stop_event.is_set():
            stamp = max(stamp + 1, int(time.perf_counter() * 1000.0))
            t = time.perf_counter()
            landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamp
            )
            recent.append((time.perf_counter() - t) * 1000.0)
            if len(recent) > 5:
                recent.pop(0)
            if len(recent) == 5 and max(recent) < self.warmup_target_ms:
                break

        self.warmup_result_ms = recent[-1] if recent else 0.0
        self.warmup_passes = stamp
        return stamp

    def _classify(self, result, now, fps, cap_fps, infer_ms, frame) -> None:
        hand_lists = []
        for hand in getattr(result, "hand_landmarks", None) or ():
            hand_lists.append([(p.x, p.y, p.z) for p in hand])
        world_hand_lists = []
        for hand in getattr(result, "hand_world_landmarks", None) or ():
            world_hand_lists.append([(p.x, p.y, p.z) for p in hand])

        snap = GestureSnapshot(
            vision_fps=fps,
            capture_fps=cap_fps,
            inference_ms=infer_ms,
            dropped_frames=self._dropped,
            num_hands=len(hand_lists),
        )

        if not hand_lists:
            snap.tracking_lost = True
            snap.thwip_held = self._thwip_latch.update(False, now)
            self._punch.reset()
            # Hold the last known hand position rather than snapping to centre —
            # a hand leaving frame should not yank the anchor point across the
            # street on the way out.
            with self._lock:
                snap.hand_x = self._snapshot.hand_x
                snap.hand_y = self._snapshot.hand_y
                self._snapshot = snap
                self._store_debug(frame, hand_lists)
            return

        # Primary hand = the one that looks biggest, i.e. closest to the camera.
        primary_index = max(
            range(len(hand_lists)), key=lambda i: palm_scale(hand_lists[i])
        )
        primary = hand_lists[primary_index]

        # World landmarks are metric 3D coordinates, so finger straightness is
        # stable when the palm or fingertips point towards the camera. Fall
        # back to normalised landmarks for compatibility with mocked/older
        # MediaPipe results that do not supply the parallel world list.
        geometry = primary
        if (
            primary_index < len(world_hand_lists)
            and len(world_hand_lists[primary_index]) == len(primary)
        ):
            geometry = world_hand_lists[primary_index]
        states = finger_states(geometry)
        scale = palm_scale(primary)
        cx, cy = palm_centre(primary)

        raw_thwip = is_thwip(states)
        raw_fist = is_fist(states)
        fired = self._punch.update(raw_fist, scale, now)

        snap.tracking_lost = False
        snap.thwip_held = self._thwip_latch.update(raw_thwip, now)
        snap.hand_x = min(max(cx, 0.0), 1.0)
        snap.hand_y = min(max(cy, 0.0), 1.0)
        snap.raw_thwip = raw_thwip
        snap.raw_fist = raw_fist
        snap.states = states
        snap.scale = scale
        snap.growth_rate = self._punch.rate

        with self._lock:
            self._snapshot = snap
            if fired:
                self._punch_pending = True
            self._store_debug(frame, hand_lists)

    def _store_debug(self, frame, hand_lists) -> None:
        # Caller holds the lock.
        self._landmarks = hand_lists
        if self.keep_frame:
            self._frame = frame

    def close(self) -> None:
        self._stop_event.set()
        self._capture.stop()
        if self.is_alive():
            self.join(timeout=2.0)
        if self._capture.is_alive():
            self._capture.join(timeout=2.0)
