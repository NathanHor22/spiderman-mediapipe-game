import unittest
from pathlib import Path
from unittest import mock

from spidergame.render3d import buttonart


class SlugTests(unittest.TestCase):
    def test_labels_become_png_names(self):
        self.assertEqual(buttonart.slug("START"), "start")
        self.assertEqual(buttonart.slug("BACK TO MENU"), "back-to-menu")
        self.assertEqual(buttonart.slug("APPLY CAMERA"), "apply-camera")
        self.assertEqual(
            buttonart.slug("THWIP AT THE CAMERA TO TEST IT OUT!"),
            "thwip-at-the-camera-to-test-it-out",
        )

    def test_slug_collapses_runs_and_trims(self):
        self.assertEqual(buttonart.slug("  A -- B  "), "a-b")


class PlateGeometryTests(unittest.TestCase):
    """The SDF drives both the silhouette and the outline, so it is worth
    pinning independently of Panda3D being installed."""

    W, H = 448.0, 64.0
    R, B = H * 0.30, H * 0.11

    def coverage(self, x, y):
        return buttonart.plate_coverage(x, y, self.W, self.H, self.R, self.B)

    def test_centre_is_opaque_fill(self):
        alpha, fill = self.coverage(0.0, 0.0)
        self.assertEqual(alpha, 1.0)
        self.assertEqual(fill, 1.0)

    def test_edge_is_opaque_border_not_fill(self):
        alpha, fill = self.coverage(self.W / 2 - 1.0, 0.0)
        self.assertEqual(alpha, 1.0)
        self.assertEqual(fill, 0.0, "outline must not be filled red")

    def test_outside_is_transparent(self):
        alpha, _ = self.coverage(self.W / 2 + 4.0, 0.0)
        self.assertEqual(alpha, 0.0)

    def test_corners_are_rounded_away(self):
        alpha, _ = self.coverage(self.W / 2 - 0.5, self.H / 2 - 0.5)
        self.assertEqual(alpha, 0.0, "square corner survived the radius")

    def test_edge_is_antialiased_rather_than_hard(self):
        # A band of partial coverage must exist across the boundary, otherwise
        # the plate has jagged edges.
        samples = [self.coverage(self.W / 2 - 0.5 + i * 0.25, 0.0)[0]
                   for i in range(8)]
        self.assertTrue(any(0.0 < a < 1.0 for a in samples), samples)


class ArtSelectionTests(unittest.TestCase):
    def test_supplied_png_wins_and_suppresses_live_text(self):
        sentinel = object()
        with mock.patch.object(buttonart, "label_texture", return_value=sentinel):
            texture, draws_text = buttonart.button_art("START", 448, 64, True)
        self.assertIs(texture, sentinel)
        self.assertFalse(draws_text, "baked label would be drawn twice")

    def test_generated_plate_draws_its_own_text(self):
        plate = object()
        with mock.patch.object(buttonart, "label_texture", return_value=None), \
             mock.patch.object(buttonart, "plate_texture", return_value=plate):
            texture, draws_text = buttonart.button_art("START", 448, 64, False)
        self.assertIs(texture, plate)
        self.assertTrue(draws_text)

    def test_image_path_only_resolves_real_files(self):
        self.assertIsNone(buttonart.image_path("definitely-not-a-button"))
        with mock.patch.object(Path, "is_file", return_value=True):
            found = buttonart.image_path("START")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "start.png")


class TextCentringTests(unittest.TestCase):
    def test_vertical_offset_is_negative_and_sane(self):
        # Panda3D puts the text baseline at the node position, so centring the
        # glyph body needs a downward shift of about half the cap height.
        self.assertLess(buttonart.TEXT_VCENTER, 0.0)
        self.assertAlmostEqual(buttonart.TEXT_VCENTER, -0.36, places=2)

    def test_every_menu_button_uses_the_shared_offset(self):
        # Guards against a stray hand-tuned magic number reappearing: the
        # original title buttons had no offset at all and floated high.
        source = (Path(__file__).resolve().parents[1]
                  / "spidergame" / "render3d" / "game.py").read_text()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("text_pos="):
                self.assertIn(
                    "buttonart.TEXT_VCENTER", stripped,
                    f"hand-tuned text offset: {stripped}",
                )


if __name__ == "__main__":
    unittest.main()
