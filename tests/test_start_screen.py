"""Focused render and input checks for the title screen."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from spidergame.control import ControlState
from spidergame.ui import hud, screens


class TitleMenuTests(unittest.TestCase):
    def test_navigation_wraps_and_activates_selected_action(self) -> None:
        menu = screens.TitleMenu()

        menu.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        self.assertEqual(menu.selected, 2)
        self.assertEqual(
            menu.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)
            ),
            "quit",
        )

        menu.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
        self.assertEqual(menu.action, "start")

    def test_existing_hotkeys_map_to_menu_actions(self) -> None:
        menu = screens.TitleMenu()
        self.assertEqual(
            menu.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t)),
            "training",
        )
        self.assertEqual(menu.selected, 1)
        self.assertEqual(
            menu.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)
            ),
            "quit",
        )
        self.assertEqual(menu.selected, 2)

    def test_mouse_hover_and_click_use_the_rendered_hit_rects(self) -> None:
        menu = screens.TitleMenu()
        hits = (
            pygame.Rect(10, 10, 100, 30),
            pygame.Rect(10, 50, 100, 30),
            pygame.Rect(10, 90, 100, 30),
        )

        self.assertTrue(menu.select_at((40, 60), hits))
        self.assertEqual(menu.selected, 1)
        self.assertEqual(menu.activate_at((40, 100), hits), "quit")
        self.assertEqual(menu.selected, 2)
        self.assertIsNone(menu.activate_at((300, 300), hits))
        self.assertEqual(menu.selected, 2)


class StartScreenRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        cls.fonts = (
            pygame.font.Font(None, 18),
            pygame.font.Font(None, 25),
            pygame.font.Font(None, 48),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def make_surface(self) -> pygame.Surface:
        surface = pygame.Surface((1280, 720))
        surface.fill((90, 54, 74))
        return surface

    def test_old_draw_title_call_remains_valid_and_returns_menu_hits(self) -> None:
        surface = self.make_surface()
        hits = screens.draw_title(
            surface,
            self.fonts,
            vision=False,
            camera_index=0,
            camera_info="",
            names=[],
            pulse=0.25,
        )

        self.assertEqual(len(hits), 3)
        self.assertTrue(all(surface.get_rect().contains(rect) for rect in hits))
        self.assertTrue(all(not hits[i].colliderect(hits[i + 1]) for i in range(2)))

    def test_selected_menu_item_has_comic_red_highlight(self) -> None:
        surface = self.make_surface()
        menu = screens.TitleMenu(selected=2)
        hits = screens.draw_title(
            surface, self.fonts, vision=False, selected=menu, pulse=0.0
        )

        selected_pixel = surface.get_at((hits[2].x + 5, hits[2].centery))[:3]
        idle_pixel = surface.get_at((hits[0].x + 5, hits[0].centery))[:3]
        self.assertGreater(selected_pixel[0], selected_pixel[2])
        self.assertLess(sum(idle_pixel), sum(selected_pixel))

    def test_camera_preview_preserves_source_aspect_ratio(self) -> None:
        surface = pygame.Surface((500, 260))
        source = pygame.Surface((640, 480))
        source.fill((210, 30, 40))
        box = pygame.Rect(50, 30, 400, 200)

        fitted = screens.draw_camera_preview(surface, source, box)

        self.assertAlmostEqual(fitted.width / fitted.height, 4 / 3, places=2)
        self.assertEqual(fitted.height, box.height)
        centre = surface.get_at(box.center)[:3]
        self.assertTrue(all(abs(actual - expected) <= 3
                            for actual, expected in zip(centre, (210, 30, 40))))
        self.assertEqual(surface.get_at((box.x + 4, box.centery))[:3], (3, 3, 8))

    def test_camera_preview_handles_missing_frame(self) -> None:
        surface = pygame.Surface((320, 200))
        fitted = screens.draw_camera_preview(
            surface, None, pygame.Rect(20, 20, 280, 160)
        )
        self.assertEqual(fitted.size, (0, 0))

    def test_vision_title_renders_live_state_and_long_camera_error(self) -> None:
        surface = self.make_surface()
        camera = pygame.Surface((1280, 720))
        camera.fill((24, 70, 110))
        ctrl = ControlState(thwip_held=True, tracking_lost=False)

        hits = screens.draw_title(
            surface,
            self.fonts,
            vision=True,
            camera_index=12,
            camera_info="camera failure: " + "device unavailable " * 20,
            names=["a device name that deliberately does not map to the index"],
            pulse=1.0,
            cam_surface=camera,
            ctrl=ctrl,
            selected=1,
            sound_available=False,
        )
        self.assertEqual(len(hits), 3)

    def test_live_tracking_statuses_are_unambiguous(self) -> None:
        self.assertEqual(screens.vision_status(None, False),
                         ("CAMERA STARTING", hud.WARN))
        self.assertEqual(screens.vision_status(None, True),
                         ("CAMERA READY", hud.GOOD))

        lost = ControlState(tracking_lost=True)
        self.assertEqual(screens.vision_status(lost, True)[1], hud.BAD)
        self.assertIn("NO HAND", screens.vision_status(lost, True)[0])

        idle = ControlState(tracking_lost=False, thwip_held=False)
        self.assertEqual(screens.vision_status(idle, True)[1], hud.WARN)
        self.assertIn("HAND FOUND", screens.vision_status(idle, True)[0])

        thwip = ControlState(tracking_lost=False, thwip_held=True)
        self.assertEqual(screens.vision_status(thwip, True)[1], hud.GOOD)
        self.assertIn("DETECTED", screens.vision_status(thwip, True)[0])


if __name__ == "__main__":
    unittest.main()
