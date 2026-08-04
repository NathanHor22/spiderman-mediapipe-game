"""Finding and identifying cameras.

Windows device enumeration and OpenCV's capture indices are two separate
orderings with no dependable mapping between them — WMI can tell you a device is
called "HD Webcam" and sits on VID_5986, but not that OpenCV will open it as
index 1. Rather than guess from vendor IDs and be wrong on someone else's
machine, the game shows a live preview and lets you pick the one you can see
yourself in.

The names are still worth collecting: they turn "index 0 or index 1?" into a
recognisable list.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

import cv2

# Integrated laptop webcams cluster on a handful of vendor IDs. Used only to
# annotate a guess in the UI, never to choose for the player.
INTEGRATED_VENDORS = {
    "5986": "Bison / Acer",
    "04F2": "Chicony",
    "0BDA": "Realtek",
    "13D3": "IMC Networks",
    "30C9": "Luxvisions",
    "0C45": "Microdia",
}

DARK_THRESHOLD = 8.0


@dataclass
class CameraInfo:
    index: int
    width: int
    height: int
    fps: float
    brightness: float

    @property
    def dark(self) -> bool:
        return self.brightness < DARK_THRESHOLD

    def label(self) -> str:
        return f"index {self.index}  {self.width}x{self.height}  {self.fps:.0f}fps"


def probe(index: int, frames: int = 12) -> CameraInfo | None:
    """Open one index and see whether it actually delivers pixels."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Cold reads are slow and sometimes black while exposure settles.
    for _ in range(4):
        cap.read()

    import time

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

    if ok_count == 0 or shape is None:
        return None
    return CameraInfo(
        index=index,
        width=shape[1],
        height=shape[0],
        fps=ok_count / elapsed if elapsed > 0 else 0.0,
        brightness=brightness / ok_count,
    )


def available(max_index: int = 3) -> list[CameraInfo]:
    found = []
    for i in range(max_index + 1):
        info = probe(i)
        if info is not None:
            found.append(info)
    return found


def pick_default(max_index: int = 3) -> int | None:
    """First index that delivers real pixels, best framerate wins ties.

    Plugging in a USB camera reshuffles capture indices, and a laptop's built-in
    camera can sit at index 0 while returning black frames because a privacy
    shutter is closed. Defaulting to 0 hands the player a black screen and no
    explanation, so instead pick something that is demonstrably working.

    Cameras returning black frames are skipped, not preferred-against: a camera
    you cannot see out of is useless regardless of its framerate.
    """
    candidates = [c for c in available(max_index) if not c.dark]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.fps).index


def system_names() -> list[str]:
    """Camera names from Windows, annotated with a built-in/USB guess.

    Best-effort and purely informational — the order here is NOT the order of
    OpenCV's indices, which is exactly why the player picks visually.
    """
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_PnPEntity | "
             "Where-Object { $_.PNPClass -eq 'Camera' } | "
             "ForEach-Object { $_.Name + '|' + $_.DeviceID }"],
            capture_output=True, text=True, timeout=12,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    names = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        name, device_id = line.split("|", 1)
        tag = ""
        upper = device_id.upper()
        if "VID_" in upper:
            vid = upper.split("VID_", 1)[1][:4]
            if vid in INTEGRATED_VENDORS:
                tag = f"  (likely built-in — {INTEGRATED_VENDORS[vid]})"
            else:
                tag = "  (likely external USB)"
        names.append(name.strip() + tag)
    return names
