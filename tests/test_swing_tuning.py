import math
import unittest

from spidergame.control import ControlState
from spidergame.game import tuning as T
from spidergame.game.swing import SwingEvent, SwingSim
from spidergame.render.world import WorldStrip


class SwingHeightTests(unittest.TestCase):
    def test_start_has_more_fall_recovery_room(self):
        self.assertEqual(T.START_Y, 50.0)
        self.assertGreater(T.START_Y - T.STREET_DEATH_Y, 45.0)

    def test_lowest_anchor_still_has_required_rise(self):
        self.assertGreater(
            T.ANCHOR_HEIGHT_MIN,
            T.START_Y + T.ANCHOR_MIN_RISE,
        )

    def test_reset_uses_tuned_start_height(self):
        sim = SwingSim()
        self.assertEqual(sim.y, T.START_Y)
        self.assertTrue(sim.alive)

    def test_player_survives_three_quarters_second_of_input_delay(self):
        sim = SwingSim()
        world = WorldStrip(seed=11)
        no_input = ControlState()

        for _ in range(45):
            world.update(sim.z + 60.0)
            sim.update(1.0 / 60.0, no_input, world)

        self.assertTrue(sim.alive)
        self.assertGreater(sim.y, T.STREET_DEATH_Y)

    def test_default_web_catch_lifts_player(self):
        sim = SwingSim()
        world = WorldStrip(seed=11)
        hold = ControlState(
            thwip_held=True,
            hand_x=0.35,
            hand_y=0.5,
            tracking_lost=False,
        )

        for _ in range(15):
            world.update(sim.z + 60.0)
            sim.update(1.0 / 60.0, hold, world)

        self.assertEqual(sim.attaches, 1)
        self.assertTrue(sim.rope_taut)
        self.assertGreater(sim.y, T.START_Y + 2.0)
        self.assertGreater(sim.angular_speed, 0.0)

    def test_default_swing_releases_before_orbit_with_upward_momentum(self):
        sim = SwingSim()
        world = WorldStrip(seed=11)
        hold = ControlState(
            thwip_held=True,
            hand_x=0.35,
            hand_y=0.5,
            tracking_lost=False,
        )
        releases = 0

        for _ in range(240):
            world.update(sim.z + 60.0)
            events = sim.update(1.0 / 120.0, hold, world)
            releases += events.count(SwingEvent.RELEASE)
            if not sim.attached and sim.attaches:
                break

        self.assertEqual(releases, 1)
        self.assertEqual(sim.auto_releases, 1)
        self.assertEqual(sim.release_reason, "rise limit")
        self.assertGreater(sim.y, T.START_Y + 10.0)
        self.assertGreater(sim.vy, 0.0)
        self.assertLess(sim.web_arc, math.tau)

        # The held sign is still the same shot; release/re-form is required.
        self.assertEqual(sim.update(1.0 / 60.0, hold, world), ())
        self.assertEqual(sim.attaches, 1)

    def test_high_anchor_uses_swept_angle_before_momentum_reverses(self):
        sim = SwingSim()
        world = WorldStrip(seed=5)
        high_reaching_hold = ControlState(
            thwip_held=True,
            hand_x=0.1,
            hand_y=0.15,
            tracking_lost=False,
        )

        for _ in range(360):
            world.update(sim.z + 60.0)
            events = sim.update(1.0 / 240.0, high_reaching_hold, world)
            if SwingEvent.RELEASE in events:
                break

        self.assertEqual(sim.release_reason, "rise limit")
        self.assertGreaterEqual(
            math.degrees(sim.web_arc),
            T.AUTO_RELEASE_SWEEP_DEG,
        )
        self.assertGreater(sim.y, T.START_Y + 10.0)
        self.assertGreater(sim.vy, 0.0)
        self.assertGreater(sim.vz, 0.0)


if __name__ == "__main__":
    unittest.main()
