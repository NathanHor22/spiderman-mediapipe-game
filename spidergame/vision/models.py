"""Fetches the HandLandmarker model bundle on first run.

Current MediaPipe releases ship the Tasks API only — the old bundled
`mp.solutions.hands` is gone, and Tasks loads its weights from a `.task` file
you supply rather than from anything inside the wheel. So the model is a real
asset with a real download, and the repo stays free of a 7MB binary.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"


def ensure_hand_model(path: Path | None = None) -> Path:
    path = path or MODEL_PATH
    if path.exists() and path.stat().st_size > 1024:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading hand landmarker model -> {path}", file=sys.stderr)

    # Via a temp file so an interrupted download cannot leave a truncated
    # bundle behind that then fails to load on every subsequent run.
    tmp = path.with_suffix(".part")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as resp, \
                open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 16):
                fh.write(chunk)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    print(f"model ready ({path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    return path
