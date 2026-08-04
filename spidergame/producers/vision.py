"""Adapts the vision thread's output into a ControlState.

Thin by design. All the real work — classification, hysteresis, punch latching —
happens on the vision thread at vision framerate. This just copies the latest
conclusion across.
"""

from __future__ import annotations

from ..control import ControlState
from ..vision.calibration import Calibration
from ..vision.worker import VisionWorker


class VisionProducer:
    def __init__(
        self,
        camera_index: int = 0,
        calibration: Calibration | None = None,
        keep_frame: bool = False,
        num_hands: int = 1,
        infer_scale: float = 1.0,
    ) -> None:
        self.worker = VisionWorker(
            camera_index=camera_index,
            calibration=calibration or Calibration.load(),
            keep_frame=keep_frame,
            num_hands=num_hands,
            infer_scale=infer_scale,
        )
        self.worker.start()

    def wait_ready(self, timeout: float = 15.0) -> bool:
        ok = self.worker.wait_ready(timeout)
        return ok and self.worker.error is None

    @property
    def error(self) -> str | None:
        return self.worker.error

    def poll(self) -> ControlState:
        snap = self.worker.snapshot()
        return ControlState(
            thwip_held=snap.thwip_held,
            hand_x=snap.hand_x,
            hand_y=snap.hand_y,
            num_hands=snap.num_hands,
            punch_fired=self.worker.consume_punch(),
            tracking_lost=snap.tracking_lost,
        )

    def close(self) -> None:
        self.worker.close()
