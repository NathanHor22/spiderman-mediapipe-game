"""Step 1 — gesture harness (pygame).

Standalone tuning rig for the two gestures. No game, no physics: just the
webcam, the landmarks, and a very loud readout of what the classifier currently
believes. Run this before anything else, because the Goblin encounter rests
entirely on punch detection being reliable in *your* room, at *your* distance,
under *your* lighting.

  C     calibrate the punch threshold (throw three punches)
  [ ]   nudge the punch threshold by hand
  0-3   switch capture index (indices move when you plug a USB camera in)
  S     save calibration
  L     toggle landmark overlay
  ESC   quit

  --camera N   start on a specific index; omit to auto-pick a working one
"""

from __future__ import annotations

import argparse
import sys
import time

import pygame

from spidergame.ui import bgr_to_surface, fit_rect, hud
from spidergame.vision.calibration import Calibration
from spidergame.vision.gestures import EXTENDED, FOLDED
from spidergame.vision.worker import VisionWorker

WIDTH, HEIGHT = 1180, 700
GROWTH_SPAN = 8.0  # top of the growth meter, in relative-growth per second

CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
TIPS = (4, 8, 12, 16, 20)


class PunchCalibrator:
    """Collects peak growth rates from a few real punches.

    Threshold lands at 55% of the median peak: high enough that ordinary hand
    motion never reaches it, low enough that a tired third punch still counts.
    Median rather than mean because the first punch of three is usually an
    over-eager outlier.
    """

    SAMPLES = 3

    def __init__(self) -> None:
        self.active = False
        self.peaks: list[float] = []
        self._peak = 0.0
        self._armed = False
        self._last = 0.0

    def start(self) -> None:
        self.active = True
        self.peaks = []
        self._peak = 0.0
        self._armed = False

    def update(self, is_fist: bool, rate: float, now: float) -> None:
        if not self.active or now - self._last < 0.6:
            return
        if is_fist and rate > 0.8:
            self._armed = True
            self._peak = max(self._peak, rate)
        elif self._armed and rate < 0.4:
            self.peaks.append(self._peak)
            self._peak = 0.0
            self._armed = False
            self._last = now

    @property
    def done(self) -> bool:
        return len(self.peaks) >= self.SAMPLES

    def threshold(self) -> float:
        ordered = sorted(self.peaks)
        return max(1.2, ordered[len(ordered) // 2] * 0.55)


def draw_landmarks(surface, hands, rect) -> None:
    for lm in hands:
        pts = [(rect.x + p[0] * rect.width, rect.y + p[1] * rect.height)
               for p in lm]
        for a, b in CONNECTIONS:
            pygame.draw.line(surface, (120, 124, 140), pts[a], pts[b], 2)
        for i, p in enumerate(pts):
            colour = (120, 190, 255) if i in TIPS else (235, 238, 246)
            pygame.draw.circle(surface, colour, (int(p[0]), int(p[1])), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None,
                    help="capture index; omit to auto-pick a working camera")
    args = ap.parse_args()

    pygame.init()
    pygame.display.set_caption("spider harness — gestures")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas", 15)
    label_font = pygame.font.SysFont("consolas", 46, bold=True)
    clock = pygame.time.Clock()

    calib = Calibration.load()

    def status(msg: str) -> None:
        screen.fill((10, 9, 16))
        hud.draw_text(screen, font, msg, (24, 24), hud.INK)
        pygame.display.flip()
        pygame.event.pump()

    camera_index = args.camera
    if camera_index is None:
        from spidergame.vision import devices

        status("looking for a working camera...")
        camera_index = devices.pick_default()
        if camera_index is None:
            print("no camera delivering usable frames — check the privacy "
                  "shutter and Windows camera permissions", file=sys.stderr)
            pygame.quit()
            return 1

    def start_worker(index: int):
        status(f"opening camera {index} / downloading model...")
        w = VisionWorker(camera_index=index, calibration=calib, keep_frame=True)
        w.start()
        if not w.wait_ready(45.0) or w.error:
            w.close()
            return None, w.error or "timeout"
        return w, None

    worker, err = start_worker(camera_index)
    if worker is None:
        print(f"vision failed to start: {err}", file=sys.stderr)
        pygame.quit()
        return 1

    calibrator = PunchCalibrator()
    show_landmarks = True
    punch_flash = -10.0

    feed_box = pygame.Rect(16, 16, 780, 668)
    panel = hud.Panel(pygame.Rect(812, 16, 352, 668), font, "GESTURE HARNESS")

    running = True
    while running:
        clock.tick(60)
        now = time.perf_counter()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_c:
                    calibrator.start()
                    print("calibration: throw three punches at the camera")
                elif event.key == pygame.K_l:
                    show_landmarks = not show_landmarks
                elif event.key == pygame.K_LEFTBRACKET:
                    calib.punch_growth_threshold = max(
                        0.5, calib.punch_growth_threshold - 0.1)
                    worker.set_punch_threshold(calib.punch_growth_threshold)
                elif event.key == pygame.K_RIGHTBRACKET:
                    calib.punch_growth_threshold += 0.1
                    worker.set_punch_threshold(calib.punch_growth_threshold)
                elif event.key == pygame.K_s:
                    print(f"saved {calib.save()}")
                elif pygame.K_0 <= event.key <= pygame.K_3:
                    want = event.key - pygame.K_0
                    if want != camera_index:
                        worker.close()
                        new_worker, new_err = start_worker(want)
                        if new_worker is None:
                            # Fall back rather than leaving the harness with no
                            # camera at all; a dead tool teaches you nothing.
                            print(f"camera {want}: {new_err} — staying on "
                                  f"{camera_index}", file=sys.stderr)
                            worker, _ = start_worker(camera_index)
                        else:
                            worker, camera_index = new_worker, want

        frame, hands = worker.debug_frame()
        snap = worker.snapshot()

        if worker.consume_punch():
            punch_flash = now
        calibrator.update(snap.raw_fist, snap.growth_rate, now)

        if calibrator.active and calibrator.done:
            new_threshold = calibrator.threshold()
            worker.set_punch_threshold(new_threshold)
            calib.punch_growth_threshold = new_threshold
            calib.neutral_palm_scale = snap.scale or calib.neutral_palm_scale
            calibrator.active = False
            print(f"calibrated punch threshold -> {new_threshold:.2f} "
                  f"(peaks {', '.join(f'{p:.2f}' for p in calibrator.peaks)})")

        screen.fill((10, 9, 16))

        rect = feed_box
        if frame is not None:
            surf = bgr_to_surface(frame)
            rect = fit_rect(surf.get_width(), surf.get_height(), feed_box)
            screen.blit(pygame.transform.smoothscale(surf, rect.size), rect.topleft)
            if show_landmarks and hands:
                draw_landmarks(screen, hands, rect)
        pygame.draw.rect(screen, hud.PANEL_EDGE, rect, 1)

        if calibrator.active:
            text, colour = f"PUNCH!  {len(calibrator.peaks)}/{calibrator.SAMPLES}", (120, 190, 255)
        elif snap.tracking_lost:
            text, colour = "NO HAND", hud.BAD
        elif snap.thwip_held:
            text, colour = "THWIP", hud.GOOD
        elif snap.raw_fist:
            text, colour = "FIST", hud.WARN
        else:
            text, colour = "IDLE", hud.DIM
        hud.draw_text(screen, label_font, text, (rect.x + 18, rect.y + 14), colour)

        if now - punch_flash < 0.22:
            pygame.draw.rect(screen, hud.WARN, rect, 10)
            hud.draw_text(screen, label_font, "PUNCH",
                          (rect.centerx - 90, rect.centery - 30), hud.WARN)

        # --- panel --------------------------------------------------------
        panel.begin(screen)
        infer_colour = (hud.GOOD if snap.inference_ms < 25 else
                        hud.WARN if snap.inference_ms < 45 else hud.BAD)
        panel.row(screen, "camera", f"index {camera_index}", hud.INK)
        panel.row(screen, "pipeline", f"{snap.vision_fps:5.1f} fps")
        panel.row(screen, "capture", f"{snap.capture_fps:5.1f} fps")
        panel.row(screen, "inference", f"{snap.inference_ms:6.1f} ms", infer_colour)
        panel.row(screen, "dropped", snap.dropped_frames,
                  hud.BAD if snap.dropped_frames > 30 else hud.DIM)
        panel.gap()
        panel.row(screen, "hands", snap.num_hands)
        panel.row(screen, "palm scale", f"{snap.scale:.3f}")
        panel.row(screen, "hand x/y", f"{snap.hand_x:.2f} / {snap.hand_y:.2f}")
        panel.gap(14)

        hud.draw_text(screen, font, "punch growth", (panel.rect.x + 12, panel.y),
                      hud.DIM)
        hud.draw_text(screen, font, f"{snap.growth_rate:5.2f}/s",
                      (panel.rect.right - 80, panel.y), hud.INK)
        panel.advance(22)
        meter = pygame.Rect(panel.rect.x + 12, panel.y, panel.rect.width - 24, 18)
        over = snap.growth_rate >= calib.punch_growth_threshold
        hud.draw_bar(screen, meter, snap.growth_rate, GROWTH_SPAN,
                     hud.WARN if over else (74, 96, 104),
                     marker=calib.punch_growth_threshold)
        panel.advance(30)
        panel.row(screen, "threshold", f"{calib.punch_growth_threshold:.2f}")
        panel.gap(10)

        for name in ("index", "middle", "ring", "pinky"):
            s = snap.states.get(name, 0)
            txt = "EXTENDED" if s == EXTENDED else "folded" if s == FOLDED else "?"
            col = hud.GOOD if s == EXTENDED else hud.DIM if s == FOLDED else hud.WARN
            panel.row(screen, name, txt, col)

        hud.draw_text(screen, font,
                      "C calibrate   [ ] threshold   0-3 camera   S save   "
                      "L landmarks   ESC",
                      (16, HEIGHT - 14), hud.DIM)

        pygame.display.flip()

    worker.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
