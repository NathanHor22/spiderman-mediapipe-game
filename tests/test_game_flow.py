import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

import run_game
from spidergame.control import ControlState


class FakeKeyboardProducer:
    instances = []

    def __init__(self):
        self.polls = 0
        self.closed = 0
        self.events = []
        self.__class__.instances.append(self)

    def handle_event(self, event):
        self.events.append(event)

    def poll(self):
        self.polls += 1
        return ControlState(tracking_lost=False, num_hands=1)

    def close(self):
        self.closed += 1


class FakeAudio:
    instances = []

    def __init__(self):
        self.enabled = True
        self.handles = 0
        self.stops = 0
        self.closed = 0
        self.__class__.instances.append(self)

    def handle(self, events, sim):
        self.handles += 1

    def stop(self, *args, **kwargs):
        self.stops += 1

    def close(self):
        self.closed += 1


class EventFeed:
    def __init__(self, *batches):
        self._batches = iter(batches)

    def __call__(self):
        return next(
            self._batches,
            [pygame.event.Event(pygame.QUIT)],
        )


class CameraStub:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


class CameraSwitchTests(unittest.TestCase):
    def test_failed_candidate_keeps_the_working_camera_alive(self):
        current = CameraStub()

        producer, index, error, switched = run_game._try_camera_switch(
            current,
            1,
            2,
            lambda _index: (None, "timed out"),
        )

        self.assertIs(producer, current)
        self.assertEqual(index, 1)
        self.assertEqual(error, "camera 2: timed out")
        self.assertFalse(switched)
        self.assertEqual(current.closed, 0)

    def test_ready_candidate_replaces_and_closes_the_old_camera(self):
        current = CameraStub()
        candidate = CameraStub()

        producer, index, error, switched = run_game._try_camera_switch(
            current,
            1,
            2,
            lambda _index: (candidate, None),
        )

        self.assertIs(producer, candidate)
        self.assertEqual(index, 2)
        self.assertIsNone(error)
        self.assertTrue(switched)
        self.assertEqual(current.closed, 1)
        self.assertEqual(candidate.closed, 0)


class GameFlowIntegrationTests(unittest.TestCase):
    def setUp(self):
        FakeKeyboardProducer.instances.clear()
        FakeAudio.instances.clear()

    def run_main(self, event_feed, *arguments):
        with (
            patch.object(sys, "argv", ["run_game.py", *arguments]),
            patch.object(run_game, "KeyboardProducer", FakeKeyboardProducer),
            patch.object(run_game, "SoundSystem", FakeAudio),
            patch.object(run_game, "prepare_mixer"),
            patch.object(pygame.event, "get", event_feed),
        ):
            result = run_game.main()
        return (
            result,
            FakeKeyboardProducer.instances[-1],
            FakeAudio.instances[-1],
        )

    def test_start_goes_directly_to_countdown_not_training(self):
        feed = EventFeed(
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)],
            [pygame.event.Event(pygame.QUIT)],
        )
        with (
            patch.object(run_game.screens, "draw_countdown", wraps=run_game.screens.draw_countdown) as countdown,
            patch.object(run_game.screens, "draw_tutorial", wraps=run_game.screens.draw_tutorial) as tutorial,
        ):
            result, producer, audio = self.run_main(feed)

        self.assertEqual(result, 0)
        self.assertEqual(countdown.call_count, 1)
        self.assertEqual(tutorial.call_count, 0)
        self.assertEqual(producer.polls, 1)
        self.assertEqual(producer.closed, 1)
        self.assertEqual(audio.closed, 1)

    def test_training_is_a_distinct_title_action(self):
        feed = EventFeed(
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t)],
            [pygame.event.Event(pygame.QUIT)],
        )
        with patch.object(
            run_game.screens,
            "draw_tutorial",
            wraps=run_game.screens.draw_tutorial,
        ) as tutorial:
            self.run_main(feed)

        self.assertEqual(tutorial.call_count, 1)

    def test_title_menu_can_be_started_with_the_mouse(self):
        feed = EventFeed(
            [],
            [
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN,
                    button=1,
                    pos=(100, 460),
                )
            ],
            [pygame.event.Event(pygame.QUIT)],
        )
        with patch.object(
            run_game.screens,
            "draw_countdown",
            wraps=run_game.screens.draw_countdown,
        ) as countdown:
            self.run_main(feed)

        self.assertEqual(countdown.call_count, 1)

    def test_escape_during_countdown_returns_to_title(self):
        feed = EventFeed(
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)],
            [pygame.event.Event(pygame.QUIT)],
        )
        with patch.object(
            run_game.screens,
            "draw_title",
            wraps=run_game.screens.draw_title,
        ) as title:
            _, producer, _ = self.run_main(feed)

        self.assertEqual(title.call_count, 1)
        self.assertEqual(producer.polls, 2)

    def test_skip_tutorial_bypasses_title_and_starts_countdown(self):
        feed = EventFeed([], [pygame.event.Event(pygame.QUIT)])
        with (
            patch.object(run_game.screens, "draw_countdown", wraps=run_game.screens.draw_countdown) as countdown,
            patch.object(run_game.screens, "draw_title", wraps=run_game.screens.draw_title) as title,
        ):
            self.run_main(feed, "--skip-tutorial")

        self.assertEqual(countdown.call_count, 1)
        self.assertEqual(title.call_count, 0)

    def test_quit_does_not_poll_or_advance_one_last_frame(self):
        feed = EventFeed([pygame.event.Event(pygame.QUIT)])

        _, producer, audio = self.run_main(feed)

        self.assertEqual(producer.polls, 0)
        self.assertEqual(audio.handles, 0)
        self.assertEqual(producer.closed, 1)
        self.assertEqual(audio.closed, 1)


if __name__ == "__main__":
    unittest.main()
