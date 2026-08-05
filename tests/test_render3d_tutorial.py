"""Focused tests for the renderer-independent 3D front-end state."""

from __future__ import annotations

import unittest

from spidergame.control import ControlState
from spidergame.render3d.tutorial import (
    Countdown,
    MenuIntent,
    TitleMenu,
    TutorialController,
    TutorialInput,
    TutorialStep,
    TutorialTimings,
)


FAST = TutorialTimings(hold=0.20, release=0.10, sweep=0.10)


class TitleMenuTests(unittest.TestCase):
    def test_navigation_wraps_and_enter_returns_selected_intent(self) -> None:
        menu = TitleMenu()

        self.assertIsNone(menu.handle_key("arrow_up"))
        self.assertEqual(menu.intent, MenuIntent.QUIT)
        self.assertEqual(menu.handle_key("enter"), MenuIntent.QUIT)
        self.assertIsNone(menu.handle_key("down"))
        self.assertEqual(menu.intent, MenuIntent.START)

    def test_direct_start_training_and_quit_keys_are_keyboard_agnostic(self) -> None:
        menu = TitleMenu(selected=2)

        self.assertEqual(menu.handle_key("start"), MenuIntent.START)
        self.assertEqual(menu.handle_key("T"), MenuIntent.TRAINING)
        self.assertEqual(menu.handle_key("settings"), MenuIntent.SETTINGS)
        self.assertEqual(menu.handle_key("escape"), MenuIntent.QUIT)
        self.assertEqual(
            menu.labels,
            ("START", "TRAINING", "SETTINGS", "QUIT"),
        )


class VisionTutorialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.flow = TutorialController(vision=True, timings=FAST)

    def test_hold_release_and_sweep_must_complete_in_order(self) -> None:
        held = TutorialInput(thwip_held=True, hand_x=0.5)
        self.assertFalse(self.flow.update(held, 0.10))
        self.assertEqual(self.flow.step, TutorialStep.HOLD)
        self.assertAlmostEqual(self.flow.view.progress, 0.5)
        self.assertEqual(self.flow.view.status, "WEB-SHOOTER DETECTED")

        self.flow.update(held, 0.10)
        self.assertEqual(self.flow.step, TutorialStep.RELEASE)
        self.assertEqual(self.flow.view.status, "RELAX THE SIGN")

        self.flow.update(TutorialInput(thwip_held=False), 0.10)
        self.assertEqual(self.flow.step, TutorialStep.SWEEP)
        self.assertEqual(self.flow.view.step_number, 3)

        self.flow.update(TutorialInput(hand_x=0.20), 0.01)
        self.assertFalse(self.flow.done)
        self.assertTrue(self.flow.view.seen_left)
        self.assertEqual(self.flow.view.progress, 0.5)
        self.assertEqual(self.flow.view.status, "NOW MOVE RIGHT")

        self.flow.update(TutorialInput(hand_x=0.80), 0.05)
        self.assertFalse(self.flow.done)
        self.assertTrue(self.flow.view.seen_right)
        self.assertAlmostEqual(self.flow.view.progress, 0.75)
        self.assertTrue(self.flow.update(TutorialInput(hand_x=0.50), 0.05))
        self.assertTrue(self.flow.done)
        self.assertEqual(self.flow.view.status, "TRAINING COMPLETE")
        self.assertEqual(self.flow.view.progress, 1.0)

    def test_tracking_loss_blocks_release_even_when_pose_is_relaxed(self) -> None:
        self.flow.update(TutorialInput(thwip_held=True), 0.20)
        self.assertEqual(self.flow.step, TutorialStep.RELEASE)

        lost = TutorialInput(thwip_held=False, tracking_lost=True)
        self.flow.update(lost, 1.0)
        self.assertEqual(self.flow.step, TutorialStep.RELEASE)
        self.assertEqual(self.flow.view.progress, 0.0)
        self.assertEqual(self.flow.view.status, "KEEP YOUR HAND IN FRAME")

        self.flow.update(TutorialInput(thwip_held=False), 0.10)
        self.assertEqual(self.flow.step, TutorialStep.SWEEP)

    def test_interrupted_hold_decays_instead_of_passing_on_noisy_frames(self) -> None:
        self.flow.update(TutorialInput(thwip_held=True), 0.15)
        self.flow.update(TutorialInput(thwip_held=False), 0.05)
        self.assertEqual(self.flow.step, TutorialStep.HOLD)
        self.assertAlmostEqual(self.flow.view.progress, 0.35)

    def test_status_copy_explains_missing_and_idle_gestures(self) -> None:
        self.flow.update(TutorialInput(tracking_lost=True), 0.0)
        self.assertIn("NO HAND", self.flow.view.status)
        self.flow.update(TutorialInput(tracking_lost=False), 0.0)
        self.assertEqual(self.flow.view.status, "MAKE THE SIGN")


class KeyboardTutorialTests(unittest.TestCase):
    def test_keyboard_mode_ignores_tracking_lost_on_control_state(self) -> None:
        flow = TutorialController(vision=False, timings=FAST)

        # ControlState's default tracking_lost=True is meaningful for cameras,
        # but must not prevent Space/mouse training.
        flow.update(ControlState(thwip_held=True), 0.20)
        self.assertEqual(flow.step, TutorialStep.RELEASE)
        self.assertEqual(flow.view.status, "LET GO OF SPACE")
        flow.update(ControlState(thwip_held=False), 0.10)
        self.assertEqual(flow.step, TutorialStep.SWEEP)

        flow.update(ControlState(hand_x=0.20), 0.01)
        flow.update(ControlState(hand_x=0.80), 0.10)
        self.assertTrue(flow.done)

    def test_reset_restores_first_card_and_progress(self) -> None:
        flow = TutorialController(vision=False, timings=FAST)
        flow.update(TutorialInput(thwip_held=True), 0.20)
        flow.reset()
        self.assertEqual(flow.step, TutorialStep.HOLD)
        self.assertEqual(flow.view.title, "HOLD SPACE")
        self.assertEqual(flow.view.progress, 0.0)


class CountdownTests(unittest.TestCase):
    def test_labels_advance_at_exact_boundaries_and_finish_after_go(self) -> None:
        countdown = Countdown(step_seconds=0.50)
        self.assertEqual((countdown.text, countdown.status), ("3", "GET READY"))

        countdown.update(0.50)
        self.assertEqual(countdown.text, "2")
        countdown.update(1.00)
        self.assertEqual((countdown.text, countdown.status), ("GO", "SWING!"))
        self.assertFalse(countdown.done)
        self.assertTrue(countdown.update(0.50))
        self.assertEqual(countdown.text, "")
        self.assertEqual(countdown.progress, 1.0)

    def test_large_frame_finishes_and_reset_is_reusable(self) -> None:
        countdown = Countdown()
        self.assertTrue(countdown.update(100.0))
        countdown.reset()
        self.assertFalse(countdown.done)
        self.assertEqual(countdown.text, "3")
        self.assertEqual(countdown.progress, 0.0)

    def test_negative_or_non_finite_dt_is_rejected(self) -> None:
        countdown = Countdown()
        with self.assertRaises(ValueError):
            countdown.update(-0.01)
        with self.assertRaises(ValueError):
            countdown.update(float("nan"))


if __name__ == "__main__":
    unittest.main()
