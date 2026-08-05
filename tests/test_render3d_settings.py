"""Focused tests for renderer-independent 3D settings state."""

from __future__ import annotations

import unittest

from spidergame.render3d.settings import (
    CameraSource,
    SettingsIntent,
    SettingsMenu,
    SettingsRow,
)


class CameraSourceTests(unittest.TestCase):
    def test_default_and_custom_labels_are_ui_ready(self) -> None:
        self.assertEqual(CameraSource(2).label, "Camera 2")
        self.assertEqual(CameraSource(4, "  Front camera  ").label, "Front camera")

    def test_indices_must_be_non_negative_integers(self) -> None:
        for invalid in (-1, -100):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                CameraSource(invalid)
        for invalid in (True, 1.5, "1"):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                CameraSource(invalid)  # type: ignore[arg-type]


class SettingsMenuTests(unittest.TestCase):
    def test_default_camera_cycle_wraps_both_directions(self) -> None:
        menu = SettingsMenu([0, 2, 5], active_camera=0)

        self.assertEqual(menu.previous_camera(), 5)
        self.assertTrue(menu.dirty)
        self.assertEqual(menu.next_camera(), 0)
        self.assertFalse(menu.dirty)

    def test_bounded_camera_cycle_clamps_at_ends(self) -> None:
        menu = SettingsMenu([1, 3], active_camera=1, wrap_cameras=False)

        self.assertEqual(menu.previous_camera(), 1)
        self.assertEqual(menu.cycle_camera(20), 3)
        self.assertEqual(menu.next_camera(), 3)

    def test_keyboard_navigation_is_normalized_and_rows_wrap(self) -> None:
        menu = SettingsMenu([0, 1])

        self.assertIsNone(menu.handle_key(" ARROW-UP "))
        self.assertEqual(menu.focused_row, SettingsRow.BACK)
        self.assertEqual(menu.handle_key("Return"), SettingsIntent.BACK)

        menu.handle_key("TAB")
        self.assertEqual(menu.focused_row, SettingsRow.CAMERA)
        menu.handle_key("Arrow Right")
        self.assertEqual(menu.pending_camera, 1)
        self.assertTrue(menu.dirty)

        menu.handle_key("down")
        self.assertEqual(menu.focused_row, SettingsRow.APPLY)
        self.assertEqual(menu.handle_key("space"), SettingsIntent.APPLY)

    def test_camera_keys_only_change_camera_while_camera_row_has_focus(self) -> None:
        menu = SettingsMenu([3, 7])
        menu.select_row(SettingsRow.APPLY)
        menu.handle_key("]")
        self.assertEqual(menu.pending_camera, 3)

        menu.select_row("camera")
        menu.handle_key("bracketright")
        self.assertEqual(menu.pending_camera, 7)

    def test_commit_and_discard_keep_failed_apply_recoverable(self) -> None:
        menu = SettingsMenu([0, 4], active_camera=0)
        menu.next_camera()
        self.assertEqual(menu.pending_camera, 4)
        self.assertEqual(menu.active_camera, 0)

        self.assertEqual(menu.discard(), 0)
        menu.next_camera()
        self.assertEqual(menu.commit(), 4)
        self.assertEqual(menu.active_camera, 4)
        self.assertFalse(menu.dirty)

    def test_empty_sources_have_safe_copy_and_back_navigation(self) -> None:
        menu = SettingsMenu([])

        self.assertIsNone(menu.pending_camera)
        self.assertIsNone(menu.next_camera())
        self.assertEqual(menu.view.camera_value, "NO CAMERAS FOUND")
        self.assertEqual(menu.view.camera_position, "0 OF 0")
        self.assertFalse(menu.view.can_change_camera)
        self.assertEqual(menu.handle_key("escape"), SettingsIntent.BACK)

    def test_labels_mapping_and_view_copy(self) -> None:
        menu = SettingsMenu([0, 2], labels={0: "Built-in webcam", 2: "USB Camera"})

        self.assertEqual(menu.view.title, "SETTINGS")
        self.assertEqual(menu.view.camera_heading, "CAMERA SOURCE")
        self.assertEqual(menu.view.camera_value, "BUILT-IN WEBCAM")
        self.assertEqual(menu.view.camera_position, "1 OF 2")
        self.assertEqual(menu.row_labels, ("CAMERA SOURCE", "APPLY", "BACK"))

    def test_duplicate_or_unknown_active_source_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SettingsMenu([0, 0])
        with self.assertRaises(ValueError):
            SettingsMenu([0, 1], active_camera=3)


if __name__ == "__main__":
    unittest.main()
