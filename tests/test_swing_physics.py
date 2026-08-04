import math
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from spidergame.control import ControlState
from spidergame.game import tuning as T
from spidergame.game.swing import Anchor, SwingEvent, SwingSim
from spidergame.render.world import Building, WorldStrip


HOLD = ControlState(thwip_held=True, tracking_lost=False)


class SwingPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.world = WorldStrip(seed=11)
        self.tuning = ExitStack()
        self.addCleanup(self.tuning.close)
        for name, value in {
            "SPEED_RESTORE": 0.0,
            "AIR_DRAG": 0.0,
            "REEL_RATE": 0.0,
            "WEB_PULL_ACCEL": 0.0,
            "LATERAL_LIMIT": 1_000_000.0,
            "CEILING_Y": 1_000_000.0,
            "MAX_SPEED_TOTAL": 1_000_000.0,
            "MIN_REST": 1.0,
        }.items():
            self.tuning.enter_context(patch.object(T, name, value))

    @staticmethod
    def attached_sim(radius, velocity, length):
        sim = SwingSim()
        sim.x, sim.y, sim.z = radius
        sim.vx, sim.vy, sim.vz = velocity
        sim.anchor = Anchor(0.0, 50.0, 0.0, length, taut=True)
        sim._armed = False
        return sim

    def test_vertical_taut_web_supports_weight(self):
        sim = self.attached_sim((0.0, 30.0, 0.0), (0.0, 0.0, 0.0), 20.0)

        for _ in range(60):
            sim.update(1.0 / 120.0, HOLD, self.world)

        self.assertAlmostEqual(sim.y, 30.0, places=5)
        self.assertAlmostEqual(sim.vy, 0.0, places=5)
        self.assertAlmostEqual(
            math.dist((sim.x, sim.y, sim.z), (0.0, 50.0, 0.0)),
            20.0,
            places=5,
        )
        self.assertAlmostEqual(sim.web_tension, T.GRAVITY, places=4)

    def test_gravity_is_projected_onto_the_swing_tangent(self):
        offset = 20.0 / math.sqrt(2.0)
        sim = self.attached_sim((offset, 50.0 - offset, 0.0), (0, 0, 0), 20.0)
        dt = 1.0e-4

        sim.update(dt, HOLD, self.world)

        expected = -T.GRAVITY * 0.5 * dt
        self.assertAlmostEqual(sim.vx, expected, delta=2.0e-5)
        self.assertAlmostEqual(sim.vy, expected, delta=2.0e-5)

    def test_tangential_motion_curves_without_stretching_web(self):
        self.tuning.enter_context(patch.object(T, "GRAVITY", 0.0))
        sim = self.attached_sim((10.0, 50.0, 0.0), (0.0, 0.0, 20.0), 10.0)

        for _ in range(120):
            sim.update(1.0 / 120.0, HOLD, self.world)

        distance = math.dist((sim.x, sim.y, sim.z), (0.0, 50.0, 0.0))
        self.assertAlmostEqual(distance, 10.0, delta=1.0e-4)
        self.assertAlmostEqual(sim.speed, 20.0, delta=0.2)
        self.assertAlmostEqual(sim.angular_speed, 2.0, delta=0.03)

    def test_reeling_shortens_web_and_raises_player(self):
        self.tuning.enter_context(patch.object(T, "REEL_RATE", 12.0))
        sim = self.attached_sim((0.0, 30.0, 0.0), (0.0, 0.0, 0.0), 20.0)

        for _ in range(30):
            sim.update(1.0 / 120.0, HOLD, self.world)

        self.assertAlmostEqual(sim.anchor.rest_length, 17.0, delta=1.0e-4)
        self.assertAlmostEqual(sim.y, 33.0, delta=0.03)
        self.assertGreater(sim.vy, 0.0)

    def test_reeling_stops_at_the_per_shot_target(self):
        self.tuning.enter_context(patch.object(T, "REEL_RATE", 12.0))
        sim = self.attached_sim((0.0, 30.0, 0.0), (0.0, 0.0, 0.0), 20.0)
        sim.anchor.reel_target = 14.0

        for _ in range(240):
            sim.update(1.0 / 120.0, HOLD, self.world)

        self.assertAlmostEqual(sim.anchor.rest_length, 14.0, delta=1.0e-4)
        self.assertAlmostEqual(sim.y, 36.0, delta=0.03)

    def test_inward_motion_makes_an_unpowered_web_slack(self):
        self.tuning.enter_context(patch.object(T, "GRAVITY", 0.0))
        sim = self.attached_sim((10.0, 50.0, 0.0), (-5.0, 0.0, 0.0), 10.0)

        sim.update(0.05, HOLD, self.world)

        self.assertFalse(sim.rope_taut)
        self.assertAlmostEqual(sim.vx, -5.0, places=6)
        self.assertLess(
            math.dist((sim.x, sim.y, sim.z), (0.0, 50.0, 0.0)),
            sim.anchor.rest_length,
        )

    def test_outward_motion_is_caught_without_losing_tangent_speed(self):
        self.tuning.enter_context(patch.object(T, "GRAVITY", 0.0))
        sim = self.attached_sim((9.9, 50.0, 0.0), (5.0, 0.0, 7.0), 10.0)

        sim.update(0.05, HOLD, self.world)

        distance = math.dist((sim.x, sim.y, sim.z), (0.0, 50.0, 0.0))
        radius = (sim.x / distance, (sim.y - 50.0) / distance, sim.z / distance)
        radial_speed = sim.vx * radius[0] + sim.vy * radius[1] + sim.vz * radius[2]
        self.assertLessEqual(distance, 10.0 + 1e-6)
        self.assertLessEqual(radial_speed, 1e-5)
        self.assertGreater(abs(sim.vz), 6.5)

    def test_catch_impulse_has_requested_upward_component(self):
        sim = SwingSim()
        anchor = Anchor(14.0, 74.0, 12.0, 30.0)
        before = sim.vy

        sim._catch_web(anchor)

        self.assertGreaterEqual(sim.vy - before, T.WEB_CATCH_UP_SPEED - 1e-6)

    def test_anchor_search_skips_a_low_roof_for_a_taller_neighbour(self):
        world = WorldStrip(seed=1)
        world.buildings = [
            Building(-30.0, -14.0, 0.0, 30.0, 72.0, -1),
            Building(-30.0, -14.0, 32.0, 62.0, 132.0, -1),
        ]
        sim = SwingSim()
        sim.y = 80.0
        aim_left = ControlState(
            thwip_held=True,
            hand_x=0.1,
            hand_y=0.5,
            tracking_lost=False,
        )

        anchor = sim._pick_anchor(world, aim_left)

        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.z, 32.0)
        self.assertGreater(anchor.y, sim.y)
        self.assertEqual(sum(sim.whiff_reason.values()), 0)

    def test_passing_anchor_does_not_auto_detach(self):
        self.tuning.enter_context(patch.object(T, "GRAVITY", 0.0))
        sim = self.attached_sim((0.0, 50.0, 10.0), (0.0, 0.0, 5.0), 10.0)

        sim.update(1.0 / 60.0, HOLD, self.world)

        self.assertTrue(sim.attached)

    def test_swept_arc_cap_releases_once_and_requires_a_new_pose(self):
        self.tuning.enter_context(patch.object(T, "GRAVITY", 0.0))
        self.tuning.enter_context(patch.object(T, "MAX_SWING_ARC_DEG", 20.0))
        sim = self.attached_sim(
            (10.0, 50.0, 0.0),
            (0.0, 0.0, 20.0),
            10.0,
        )

        seen = []
        for _ in range(60):
            seen.extend(sim.update(1.0 / 120.0, HOLD, self.world))
            if not sim.attached:
                break

        self.assertFalse(sim.attached)
        self.assertEqual(seen.count(SwingEvent.RELEASE), 1)
        self.assertEqual(sim.release_reason, "arc limit")
        self.assertEqual(sim.auto_releases, 1)

        # Keeping the same pose cannot immediately fire and attach again.
        self.assertEqual(sim.update(1.0 / 60.0, HOLD, self.world), ())
        self.assertFalse(sim.attached)

    def test_web_events_are_edge_triggered(self):
        sim = SwingSim()
        with patch.object(sim, "_pick_anchor", return_value=None):
            first = sim.update(1.0 / 60.0, HOLD, self.world)
            second = sim.update(1.0 / 60.0, HOLD, self.world)

        self.assertEqual(first, (SwingEvent.SHOT, SwingEvent.MISS))
        self.assertEqual(second, ())

    def test_manual_release_preserves_velocity_and_emits_once(self):
        sim = self.attached_sim(
            (0.0, 30.0, 0.0),
            (3.0, 4.0, 5.0),
            20.0,
        )
        before = (sim.vx, sim.vy, sim.vz)
        release = ControlState(thwip_held=False, tracking_lost=False)

        events = sim.update(1.0e-9, release, self.world)

        self.assertEqual(events, (SwingEvent.RELEASE,))
        self.assertAlmostEqual(sim.vx, before[0], places=6)
        self.assertAlmostEqual(sim.vy, before[1], places=6)
        self.assertAlmostEqual(sim.vz, before[2], places=6)
        self.assertEqual(sim.update(1.0e-9, release, self.world), ())


if __name__ == "__main__":
    unittest.main()
