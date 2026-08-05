"""Renderer-independent settings-menu state for the Panda3D front end.

The menu deliberately knows nothing about Panda3D or camera discovery.  The
caller supplies the camera sources it has found, draws :attr:`SettingsMenu.view`,
and performs the requested apply/back action.  Camera changes are pending until
the caller confirms a successful switch with :meth:`SettingsMenu.commit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping


class SettingsIntent(str, Enum):
    """Actions that leave (or attempt to leave) the settings screen."""

    APPLY = "apply"
    BACK = "back"


class SettingsRow(str, Enum):
    """Focusable rows in their visual/navigation order."""

    CAMERA = "camera"
    APPLY = "apply"
    BACK = "back"


@dataclass(frozen=True)
class CameraSource:
    """One camera choice, with display copy ready for a menu widget."""

    index: int
    label: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError("camera index must be an integer")
        if self.index < 0:
            raise ValueError("camera index must be non-negative")
        cleaned = str(self.label).strip()
        object.__setattr__(self, "label", cleaned or f"Camera {self.index}")


@dataclass(frozen=True)
class SettingsView:
    """All dynamic copy and selection state needed to draw the screen."""

    title: str
    focused_row: SettingsRow
    camera_heading: str
    camera_value: str
    camera_position: str
    camera_hint: str
    apply_label: str
    back_label: str
    can_change_camera: bool
    dirty: bool


class SettingsMenu:
    """Camera settings and keyboard navigation without renderer dependencies.

    Camera cycling wraps by default: Previous on the first source selects the
    last source, and Next on the last source selects the first.  Pass
    ``wrap_cameras=False`` to clamp at the ends instead.  Up/Down menu-row
    navigation always wraps between Camera, Apply, and Back.
    """

    rows = (SettingsRow.CAMERA, SettingsRow.APPLY, SettingsRow.BACK)

    def __init__(
        self,
        camera_sources: Iterable[int | CameraSource],
        *,
        active_camera: int | None = None,
        labels: Mapping[int, str] | None = None,
        wrap_cameras: bool = True,
    ) -> None:
        self.wrap_cameras = bool(wrap_cameras)
        self.sources = self._coerce_sources(camera_sources, labels or {})
        self.focused = 0
        self._camera_cursor = self._initial_cursor(active_camera)
        self._active_camera = self.pending_camera

    @staticmethod
    def _coerce_sources(
        camera_sources: Iterable[int | CameraSource],
        labels: Mapping[int, str],
    ) -> tuple[CameraSource, ...]:
        result: list[CameraSource] = []
        seen: set[int] = set()
        for value in camera_sources:
            source = (
                value
                if isinstance(value, CameraSource)
                else CameraSource(value, labels.get(value, ""))
            )
            if source.index in seen:
                raise ValueError(f"duplicate camera index: {source.index}")
            seen.add(source.index)
            result.append(source)
        return tuple(result)

    def _initial_cursor(self, active_camera: int | None) -> int | None:
        if not self.sources:
            if active_camera is not None:
                CameraSource(active_camera)  # validate even when list is empty
            return None
        if active_camera is None:
            return 0
        CameraSource(active_camera)  # validate before comparing
        for cursor, source in enumerate(self.sources):
            if source.index == active_camera:
                return cursor
        raise ValueError("active camera must be one of the supplied sources")

    @property
    def focused_row(self) -> SettingsRow:
        return self.rows[self.focused]

    @property
    def pending_source(self) -> CameraSource | None:
        if self._camera_cursor is None:
            return None
        return self.sources[self._camera_cursor]

    @property
    def pending_camera(self) -> int | None:
        source = self.pending_source
        return None if source is None else source.index

    @property
    def active_camera(self) -> int | None:
        """Camera last confirmed by :meth:`commit`."""

        return self._active_camera

    @property
    def dirty(self) -> bool:
        return self.pending_camera != self.active_camera

    @property
    def row_labels(self) -> tuple[str, str, str]:
        """Stable, UI-ready labels in navigation order."""

        return ("CAMERA SOURCE", "APPLY", "BACK")

    @property
    def view(self) -> SettingsView:
        source = self.pending_source
        if source is None:
            camera_value = "NO CAMERAS FOUND"
            camera_position = "0 OF 0"
            camera_hint = "Connect a camera, then reopen Settings."
        else:
            camera_value = source.label.upper()
            camera_position = f"{self._camera_cursor + 1} OF {len(self.sources)}"
            camera_hint = "Use Left / Right to choose a camera."
        return SettingsView(
            title="SETTINGS",
            focused_row=self.focused_row,
            camera_heading=self.row_labels[0],
            camera_value=camera_value,
            camera_position=camera_position,
            camera_hint=camera_hint,
            apply_label=self.row_labels[1],
            back_label=self.row_labels[2],
            can_change_camera=len(self.sources) > 1,
            dirty=self.dirty,
        )

    def move_focus(self, delta: int) -> SettingsRow:
        self.focused = (self.focused + int(delta)) % len(self.rows)
        return self.focused_row

    def select_row(self, row: SettingsRow | str) -> None:
        requested = SettingsRow(row)
        self.focused = self.rows.index(requested)

    def cycle_camera(self, delta: int) -> int | None:
        """Move the pending camera selection and return its index.

        Empty and single-source menus are safe no-ops.  A zero delta also
        leaves the current source unchanged.
        """

        if self._camera_cursor is None or len(self.sources) < 2 or not delta:
            return self.pending_camera
        target = self._camera_cursor + int(delta)
        if self.wrap_cameras:
            target %= len(self.sources)
        else:
            target = max(0, min(len(self.sources) - 1, target))
        self._camera_cursor = target
        return self.pending_camera

    def previous_camera(self) -> int | None:
        return self.cycle_camera(-1)

    def next_camera(self) -> int | None:
        return self.cycle_camera(1)

    def commit(self) -> int | None:
        """Confirm that the caller successfully applied the pending source."""

        self._active_camera = self.pending_camera
        return self._active_camera

    def discard(self) -> int | None:
        """Restore the pending selection to the last committed source."""

        if self._active_camera is None:
            self._camera_cursor = None if not self.sources else 0
            return self.pending_camera
        for cursor, source in enumerate(self.sources):
            if source.index == self._active_camera:
                self._camera_cursor = cursor
                break
        return self.pending_camera

    @staticmethod
    def _normalize_key(key: str) -> str:
        return str(key).strip().lower().replace("-", "_").replace(" ", "_")

    def handle_key(self, key: str) -> SettingsIntent | None:
        """Handle a normalized key edge and return an action, if activated."""

        normalized = self._normalize_key(key)
        if normalized in {"up", "arrow_up", "w", "shift_tab"}:
            self.move_focus(-1)
        elif normalized in {"down", "arrow_down", "s", "tab"}:
            self.move_focus(1)
        elif normalized in {"left", "arrow_left", "a", "[", "bracketleft"}:
            if self.focused_row is SettingsRow.CAMERA:
                self.previous_camera()
        elif normalized in {"right", "arrow_right", "d", "]", "bracketright"}:
            if self.focused_row is SettingsRow.CAMERA:
                self.next_camera()
        elif normalized in {"enter", "return", "kp_enter", "space"}:
            if self.focused_row is SettingsRow.APPLY:
                return SettingsIntent.APPLY
            if self.focused_row is SettingsRow.BACK:
                return SettingsIntent.BACK
        elif normalized in {"escape", "esc", "back"}:
            return SettingsIntent.BACK
        elif normalized == "apply":
            return SettingsIntent.APPLY
        return None


__all__ = [
    "CameraSource",
    "SettingsIntent",
    "SettingsMenu",
    "SettingsRow",
    "SettingsView",
]
