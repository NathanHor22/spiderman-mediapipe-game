import unittest
from types import SimpleNamespace

from spidergame.render3d.game import (
    DEFAULT_THWIP_IMAGE,
    GameStartupError,
    GameState,
    PandaKeyboardProducer,
    _camera_switch_allowed,
    _short_camera_message,
    build_parser,
    config_from_args,
    simulation_to_render,
    velocity_to_render_hpr,
)


class Render3DGameHelperTests(unittest.TestCase):
    def test_simulation_axes_map_to_panda_z_up(self):
        self.assertEqual(simulation_to_render(3, 7, 11), (3.0, 11.0, 7.0))

    def test_rising_velocity_pitches_character_up(self):
        _heading, pitch, roll = velocity_to_render_hpr(0.0, 8.0, 20.0)

        self.assertGreater(pitch, 0.0)
        self.assertEqual(roll, 0.0)

    def test_headless_keyboard_poll_does_not_require_pointer_api(self):
        producer = PandaKeyboardProducer(
            SimpleNamespace(mouseWatcherNode=None, win=SimpleNamespace())
        )

        state = producer.poll()

        self.assertFalse(state.thwip_held)
        self.assertEqual((state.hand_x, state.hand_y), (0.5, 0.5))
        self.assertFalse(state.tracking_lost)

    def test_cli_defaults_to_mediapipe_vision(self):
        args = build_parser().parse_args([])

        config = config_from_args(args)

        self.assertTrue(config.vision)
        self.assertIsNone(config.camera_index)

    def test_keyboard_flag_is_an_explicit_camera_free_fallback(self):
        args = build_parser().parse_args(["--keyboard"])

        config = config_from_args(args)

        self.assertFalse(config.vision)

    def test_headless_default_stays_camera_free_for_smoke_tests(self):
        args = build_parser().parse_args(["--headless"])

        config = config_from_args(args)

        self.assertFalse(config.vision)

    def test_camera_index_cannot_be_combined_with_keyboard_mode(self):
        args = build_parser().parse_args(["--keyboard", "--camera", "1"])

        with self.assertRaisesRegex(GameStartupError, "keyboard"):
            config_from_args(args)

    def test_vision_and_keyboard_flags_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--vision", "--keyboard"])

    def test_camera_switching_is_available_on_every_preview_screen(self):
        self.assertTrue(_camera_switch_allowed(GameState.TITLE, True))
        self.assertTrue(_camera_switch_allowed(GameState.TUTORIAL, True))
        self.assertTrue(_camera_switch_allowed(GameState.SETTINGS, True))
        self.assertFalse(_camera_switch_allowed(GameState.PLAYING, True))
        self.assertFalse(_camera_switch_allowed(GameState.TITLE, False))

    def test_supplied_thwip_reference_is_a_bundled_png(self):
        self.assertTrue(DEFAULT_THWIP_IMAGE.is_file())
        self.assertEqual(DEFAULT_THWIP_IMAGE.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_vision_flag_remains_compatible_with_old_launch_commands(self):
        args = build_parser().parse_args(
            ["--vision", "--camera", "2", "--no-character", "--no-audio"]
        )

        config = config_from_args(args)

        self.assertTrue(config.vision)
        self.assertEqual(config.camera_index, 2)
        self.assertIsNone(config.character_asset)
        self.assertIsNone(config.character_manifest)
        self.assertFalse(config.audio)

    def test_cli_resolves_character_and_manifest_together(self):
        args = build_parser().parse_args(
            [
                "--character",
                "models/hero.glb",
                "--character-manifest",
                "models/hero.json",
            ]
        )

        config = config_from_args(args)

        self.assertEqual(config.character_asset.name, "hero.glb")
        self.assertEqual(config.character_manifest.name, "hero.json")

    def test_negative_camera_index_is_rejected(self):
        args = build_parser().parse_args(["--camera", "-1"])

        with self.assertRaisesRegex(GameStartupError, "camera"):
            config_from_args(args)

    def test_camera_is_auto_detected_when_index_is_omitted(self):
        args = build_parser().parse_args([])

        config = config_from_args(args)

        self.assertIsNone(config.camera_index)

    def test_skip_tutorial_alias_enters_the_same_frontend_path(self):
        args = build_parser().parse_args(["--skip-tutorial"])

        config = config_from_args(args)

        self.assertTrue(config.skip_title)

    def test_long_camera_diagnostic_is_normalized_and_bounded(self):
        message = _short_camera_message("driver\n  failure " * 20, limit=40)

        self.assertLessEqual(len(message), 40)
        self.assertNotIn("\n", message)
        self.assertTrue(message.endswith("..."))


if __name__ == "__main__":
    unittest.main()
