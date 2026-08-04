import math
import unittest
from types import SimpleNamespace

from spidergame.vision.gestures import (
    AMBIGUOUS,
    EXTENDED,
    FOLDED,
    FINGERS,
    PoseLatch,
    finger_states,
    is_fist,
    is_thwip,
)
from spidergame.vision.worker import VisionWorker


def _hand(finger_shapes):
    """Build a small, anatomically ordered synthetic 21-landmark hand."""
    points = [(0.0, 0.0, 0.0) for _ in range(21)]
    bases = {
        "index": (-0.45, 0.75, 0.0),
        "middle": (-0.15, 0.85, 0.0),
        "ring": (0.15, 0.82, 0.0),
        "pinky": (0.45, 0.70, 0.0),
    }
    for name, shape in finger_shapes.items():
        joints = FINGERS[name]
        base = bases[name]
        if shape == "extended":
            direction = (-0.16, 0.99, 0.0) if name == "pinky" else (0.0, 1.0, 0.0)
            chain = [base]
            for length in (0.55, 0.42, 0.32):
                previous = chain[-1]
                chain.append(tuple(
                    previous[i] + direction[i] * length for i in range(3)
                ))
        elif shape == "folded":
            # Three clear bends: up from the palm, towards the camera, then
            # back down. End-to-end distance is much shorter than the chain.
            chain = [
                base,
                (base[0], base[1] + 0.55, base[2]),
                (base[0], base[1] + 0.55, base[2] + 0.42),
                (base[0], base[1] + 0.23, base[2] + 0.42),
            ]
        elif shape == "ambiguous":
            # Equal segments turning 40 degrees at each joint produce a
            # straightness of about .84: deliberately inside the dead zone.
            chain = [base]
            for angle in (0.0, 40.0, 80.0):
                radians = math.radians(angle)
                previous = chain[-1]
                chain.append((
                    previous[0] + math.sin(radians),
                    previous[1] + math.cos(radians),
                    previous[2],
                ))
        else:
            raise ValueError(shape)
        for landmark, point in zip(joints, chain):
            points[landmark] = point
    return points


def _transform(points, angle_degrees, mirror=False):
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    transformed = []
    for x, y, z in points:
        x = -x if mirror else x
        transformed.append((
            2.4 * (x * cosine - y * sine) + 3.0,
            2.4 * (x * sine + y * cosine) - 1.0,
            2.4 * z + 0.5,
        ))
    return transformed


class FingerGeometryTests(unittest.TestCase):
    def test_web_shooter_pose_matches_with_rotation_and_either_hand(self):
        pose = _hand({
            "index": "extended",
            "middle": "folded",
            "ring": "folded",
            "pinky": "extended",
        })

        for angle in (0.0, 55.0, 130.0, 225.0):
            for mirror in (False, True):
                with self.subTest(angle=angle, mirror=mirror):
                    states = finger_states(_transform(pose, angle, mirror))
                    self.assertEqual(states["index"], EXTENDED)
                    self.assertEqual(states["middle"], FOLDED)
                    self.assertEqual(states["ring"], FOLDED)
                    self.assertEqual(states["pinky"], EXTENDED)
                    self.assertTrue(is_thwip(states))

    def test_thumb_position_does_not_affect_web_shooter_pose(self):
        pose = _hand({
            "index": "extended",
            "middle": "folded",
            "ring": "folded",
            "pinky": "extended",
        })

        for thumb_tip in ((-2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 1.0, -1.0)):
            with self.subTest(thumb_tip=thumb_tip):
                pose[4] = thumb_tip
                self.assertTrue(is_thwip(finger_states(pose)))

    def test_occluded_curled_fingers_may_be_ambiguous(self):
        pose = _hand({
            "index": "extended",
            "middle": "ambiguous",
            "ring": "ambiguous",
            "pinky": "extended",
        })
        states = finger_states(pose)

        self.assertEqual(states["middle"], AMBIGUOUS)
        self.assertEqual(states["ring"], AMBIGUOUS)
        self.assertTrue(is_thwip(states))

    def test_open_hand_and_fist_are_not_web_shooter_pose(self):
        open_hand = _hand({name: "extended" for name in FINGERS})
        fist = _hand({name: "folded" for name in FINGERS})

        self.assertFalse(is_thwip(finger_states(open_hand)))
        self.assertFalse(is_thwip(finger_states(fist)))
        self.assertTrue(is_fist(finger_states(fist)))


class PoseLatchTests(unittest.TestCase):
    def test_brief_tracking_blink_does_not_release_web(self):
        latch = PoseLatch(on_s=0.04, off_s=0.14)
        self.assertFalse(latch.update(True, 1.00))
        self.assertTrue(latch.update(True, 1.05))

        self.assertTrue(latch.update(False, 1.10))
        self.assertTrue(latch.update(True, 1.20))

    def test_release_delay_is_time_based_at_low_frame_rate(self):
        latch = PoseLatch(on_s=0.04, off_s=0.14)
        latch.update(True, 2.000)
        self.assertTrue(latch.update(True, 2.111))

        self.assertTrue(latch.update(False, 2.222))
        self.assertTrue(latch.update(False, 2.333))
        self.assertFalse(latch.update(False, 2.444))


class VisionWorkerClassificationTests(unittest.TestCase):
    @staticmethod
    def _mediapipe_hand(points):
        return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]

    def test_worker_uses_world_landmarks_for_finger_geometry(self):
        # Make image-space geometry deliberately look like an open hand. The
        # parallel metric world geometry contains the actual thwip pose.
        image_hand = _hand({name: "extended" for name in FINGERS})
        world_hand = _hand({
            "index": "extended",
            "middle": "folded",
            "ring": "folded",
            "pinky": "extended",
        })
        result = SimpleNamespace(
            hand_landmarks=[self._mediapipe_hand(image_hand)],
            hand_world_landmarks=[self._mediapipe_hand(world_hand)],
        )
        worker = VisionWorker()
        try:
            worker._classify(
                result,
                now=10.0,
                fps=30.0,
                cap_fps=30.0,
                infer_ms=12.0,
                frame=None,
            )
            snapshot = worker.snapshot()
            self.assertTrue(snapshot.raw_thwip)
            self.assertEqual(snapshot.states["middle"], FOLDED)
            self.assertEqual(snapshot.states["ring"], FOLDED)
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
