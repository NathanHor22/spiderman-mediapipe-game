import unittest
from types import SimpleNamespace

from spidergame.render3d.game import (
    GameStartupError,
    PandaKeyboardProducer,
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

    def test_cli_enables_vision_camera_without_character(self):
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


if __name__ == "__main__":
    unittest.main()
