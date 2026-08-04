"""Camera preflight — is there a webcam, does it deliver frames, is the lens open?

Worth having as its own tool. "The game isn't reacting to me" has three very
different causes: no camera, a camera delivering black frames because a privacy
shutter or Windows permission is blocking it, or a camera that is fine while the
gesture thresholds are wrong. This separates the first two from the third before
you go looking in the wrong place.

  python run_camera_check.py            probe indices 0-3, no window
  python run_camera_check.py --preview  also open a live preview
  python run_camera_check.py --index 1  probe one specific device
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

# MSMF is the Windows default and can stall for several seconds opening a
# device; DSHOW opens in well under one. Both are probed because a handful of
# devices enumerate under only one of them.
BACKENDS = (
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("ANY", cv2.CAP_ANY),
)

# Mean pixel value below this means we are getting frames but seeing nothing —
# almost always a physical shutter, a taped lens, or a blocked permission
# rather than a broken camera.
DARK_THRESHOLD = 8.0


def probe(index: int, backend_name: str, backend_id: int, frames: int = 30):
    cap = cv2.VideoCapture(index, backend_id)
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # First reads on a cold device are slow and sometimes black while exposure
    # settles, so warm up before timing or measuring brightness.
    for _ in range(5):
        cap.read()

    ok_count = 0
    brightness = 0.0
    shape = None
    t0 = time.perf_counter()
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        ok_count += 1
        shape = frame.shape
        brightness += float(frame.mean())
    elapsed = time.perf_counter() - t0
    cap.release()

    if ok_count == 0:
        return {"backend": backend_name, "opened": True, "frames": 0}

    return {
        "backend": backend_name,
        "opened": True,
        "frames": ok_count,
        "fps": ok_count / elapsed if elapsed > 0 else 0.0,
        "width": shape[1],
        "height": shape[0],
        "brightness": brightness / ok_count,
    }


def preview(index: int, backend_id: int) -> None:
    """Live preview in pygame, same as everything else in the project."""
    import pygame

    from spidergame.ui import bgr_to_surface, fit_rect, hud

    cap = cv2.VideoCapture(index, backend_id)
    if not cap.isOpened():
        print(f"could not open camera {index} for preview", file=sys.stderr)
        return

    pygame.init()
    pygame.display.set_caption(f"camera check — index {index}")
    screen = pygame.display.set_mode((960, 620))
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()
    box = pygame.Rect(16, 16, 928, 540)

    print("preview open — press ESC or Q to close")
    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_ESCAPE, pygame.K_q):
                running = False

        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirrored, same as the game sees it
        mean = float(frame.mean())

        screen.fill((10, 9, 16))
        surf = bgr_to_surface(frame)
        rect = fit_rect(surf.get_width(), surf.get_height(), box)
        screen.blit(pygame.transform.smoothscale(surf, rect.size), rect.topleft)
        pygame.draw.rect(screen, hud.PANEL_EDGE, rect, 1)

        colour = hud.GOOD if mean > DARK_THRESHOLD else hud.BAD
        hud.draw_text(screen, font, f"mean brightness {mean:5.1f}",
                      (24, rect.bottom + 14), colour)
        hud.draw_text(screen, font,
                      f"{frame.shape[1]}x{frame.shape[0]}   "
                      f"{clock.get_fps():4.1f} fps   wave at the camera   ESC to close",
                      (24, rect.bottom + 36), hud.DIM)
        pygame.display.flip()

    cap.release()
    pygame.quit()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None, help="probe one index only")
    ap.add_argument("--max-index", type=int, default=3)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    indices = [args.index] if args.index is not None else range(args.max_index + 1)
    working: list[tuple[int, int, dict]] = []

    for index in indices:
        found = False
        for name, backend_id in BACKENDS:
            result = probe(index, name, backend_id)
            if result is None:
                continue
            found = True
            if result["frames"] == 0:
                print(f"camera {index} [{name}]: opens but delivers no frames "
                      f"— likely in use by another app")
                continue

            dark = result["brightness"] < DARK_THRESHOLD
            flag = "  <-- BLACK FRAMES" if dark else ""
            print(f"camera {index} [{name}]: {result['width']}x{result['height']} "
                  f"@ {result['fps']:.1f} fps, brightness {result['brightness']:.1f}{flag}")
            if not dark:
                working.append((index, backend_id, result))
            break
        if not found and args.index is not None:
            print(f"camera {index}: no device")

    print()
    if not working:
        print("No usable camera found.")
        print("  - check the privacy shutter / tape on the lens")
        print("  - Windows Settings > Privacy & security > Camera:")
        print("      'Camera access' and 'Let desktop apps access your camera' both On")
        print("  - close Teams / Zoom / OBS, which hold the device exclusively")
        return 1

    index, backend_id, result = working[0]
    print(f"Use camera index {index}. Both entry points default to 0:")
    print(f"    VisionWorker(camera_index={index})")
    if result["fps"] < 20:
        print(f"NOTE: {result['fps']:.1f} fps capture is low — gesture response "
              f"will suffer regardless of how fast inference runs.")

    if args.preview:
        preview(index, backend_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
