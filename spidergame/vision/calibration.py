"""Per-player, per-camera thresholds.

Only the punch threshold really needs calibrating. Finger-extension ratios are
close to universal across hands, but how fast your palm appears to grow when you
punch depends entirely on how far you sit from the camera and how much room you
have to throw — those vary hugely between setups.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "calibration.json"


@dataclass
class Calibration:
    punch_growth_threshold: float = 2.5
    neutral_palm_scale: float = 0.16

    @classmethod
    def load(cls, path: Path | None = None) -> "Calibration":
        path = path or DEFAULT_PATH
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: Path | None = None) -> Path:
        path = path or DEFAULT_PATH
        path.write_text(json.dumps(asdict(self), indent=2))
        return path
