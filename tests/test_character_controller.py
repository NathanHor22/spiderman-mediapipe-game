import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from spidergame.render3d.character import (
    CharacterAnimState,
    CharacterController,
    base_animation_for_sim,
    event_animation_requests,
    parse_character_manifest,
    resolve_clip_names,
    select_hand_joint,
    swing_frame,
    swing_phase,
)


class _FakeNode:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.pos = (0.0, 0.0, 0.0)
        self.removed = False
        if parent is not None:
            parent.children.append(self)

    def attach_new_node(self, name):
        return _FakeNode(name, self)

    def set_pos(self, *values):
        self.pos = tuple(float(value) for value in values)

    def get_top(self):
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    def get_pos(self, _other=None):
        position = list(self.pos)
        node = self.parent
        while node is not None:
            position = [a + b for a, b in zip(position, node.pos)]
            node = node.parent
        return tuple(position)

    def get_x(self, other=None):
        return self.get_pos(other)[0]

    def remove_node(self):
        self.removed = True
        if self.parent is not None and self in self.parent.children:
            self.parent.children.remove(self)


class _FakeJoint:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class _FakeActor:
    def __init__(self):
        self.parent = None
        self.calls = []
        self.scale = None
        self.pos = None
        self.cleaned = False

    def reparent_to(self, parent):
        self.parent = parent

    def get_anim_names(self):
        return ["idle", "fall", "shoot", "attach", "swing", "release"]

    def pose(self, clip, frame):
        self.calls.append(("pose", clip, frame))

    def get_tight_bounds(self):
        return ((-1.0, -0.5, 0.0), (1.0, 0.5, 6.0))

    def set_scale(self, value):
        self.scale = value

    def set_pos(self, *values):
        self.pos = values

    def get_joints(self, _part, _pattern):
        return [_FakeJoint("DEF-hand.R"), _FakeJoint("DEF-hand.L")]

    def expose_joint(self, _node, _part, name):
        side = 1.0 if name.endswith(".R") else -1.0
        node = self.parent.attach_new_node(name)
        node.set_pos(side, 0.0, 3.0)
        return node

    def stop(self, *_args):
        self.calls.append(("stop",))

    def set_play_rate(self, rate, clip):
        self.calls.append(("rate", clip, rate))

    def loop(self, clip, restart=1):
        self.calls.append(("loop", clip, restart))

    def play(self, clip):
        self.calls.append(("play", clip))

    def get_duration(self, _clip):
        return 1.0

    def get_num_frames(self, _clip):
        return 9

    def update(self, force=False):
        self.calls.append(("update", force))

    def cleanup(self):
        self.cleaned = True


class CharacterStateHelperTests(unittest.TestCase):
    def test_events_map_to_ordered_one_shots_and_ignore_miss(self):
        events = [
            SimpleNamespace(name="SHOT"),
            SimpleNamespace(name="ATTACH"),
            SimpleNamespace(name="MISS"),
            SimpleNamespace(name="RELEASE"),
        ]

        self.assertEqual(
            event_animation_requests(events),
            (
                CharacterAnimState.SHOOT,
                CharacterAnimState.ATTACH,
                CharacterAnimState.RELEASE,
            ),
        )

    def test_base_state_is_idle_then_fall_or_swing(self):
        self.assertEqual(
            base_animation_for_sim(
                SimpleNamespace(attached=False, alive=True, elapsed=0.0)
            ),
            CharacterAnimState.IDLE,
        )
        self.assertEqual(
            base_animation_for_sim(
                SimpleNamespace(attached=False, alive=True, elapsed=0.1)
            ),
            CharacterAnimState.FALL,
        )
        self.assertEqual(
            base_animation_for_sim(
                SimpleNamespace(attached=True, alive=True, elapsed=0.1)
            ),
            CharacterAnimState.SWING,
        )

    def test_swing_phase_and_frame_are_clamped(self):
        self.assertEqual(swing_phase(-1.0), 0.0)
        self.assertAlmostEqual(swing_phase(math.radians(60.0)), 0.5)
        self.assertEqual(swing_phase(math.radians(240.0)), 1.0)
        self.assertEqual(swing_frame(-1.0, 9), 0)
        self.assertEqual(swing_frame(0.5, 9), 4)
        self.assertEqual(swing_frame(2.0, 9), 8)


class CharacterManifestHelperTests(unittest.TestCase):
    def test_manifest_converts_y_up_bounds_and_reads_socket_names(self):
        manifest = parse_character_manifest(
            {
                "local_bounds_y_up": {
                    "min": [-1, 0, -0.5],
                    "max": [1, 6, 0.75],
                },
                "animations": [
                    "idle",
                    "fall",
                    "shoot",
                    "attach",
                    "swing",
                    "release",
                ],
                "joints": {
                    "right_hand": "DEF-hand.R",
                    "left_hand": "DEF-hand.L",
                },
                "physics_pivot_y_up": [0, 3, 0.25],
                "heading_offset_degrees": 180,
            }
        )

        self.assertEqual(manifest.bounds.minimum, (-1.0, -0.75, 0.0))
        self.assertEqual(manifest.bounds.maximum, (1.0, 0.5, 6.0))
        self.assertEqual(manifest.joints["right"], "DEF-hand.R")
        self.assertEqual(manifest.clips["swing"], "swing")
        self.assertEqual(manifest.physics_pivot, (0.0, -0.25, 3.0))
        self.assertEqual(manifest.heading_offset_degrees, 180.0)

    def test_clip_resolution_handles_armature_prefixes(self):
        available = [
            "Rig|idle",
            "Rig|fall",
            "Rig|shoot",
            "Rig|attach",
            "Rig|swing",
            "Rig|release",
        ]

        clips = resolve_clip_names(available)

        self.assertEqual(clips["attach"], "Rig|attach")
        self.assertEqual(set(clips), {
            "idle", "fall", "shoot", "attach", "swing", "release"
        })

    def test_manifest_converts_blender_z_up_pivot_to_panda(self):
        manifest = parse_character_manifest(
            {"physics_pivot_blender_z_up": [1.0, -2.0, 10.0]}
        )

        self.assertEqual(manifest.physics_pivot, (1.0, 2.0, 10.0))

    def test_joint_selection_prefers_manifest_then_deform_alias(self):
        joints = ["upper_arm.R", "DEF-hand.R", "hand_ik.R", "CustomLeftHand"]

        self.assertEqual(
            select_hand_joint(joints, "left", "CustomLeftHand"),
            "CustomLeftHand",
        )
        self.assertEqual(select_hand_joint(joints, "right"), "DEF-hand.R")


class CharacterControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.asset = Path(self.temp_dir.name) / "spider_man.glb"
        self.asset.touch()
        self.parent = _FakeNode("player")
        self.parent.set_pos(7.0, 11.0, 13.0)
        self.actor = _FakeActor()
        loader = patch(
            "spidergame.render3d.character._load_actor",
            return_value=self.actor,
        )
        loader.start()
        self.addCleanup(loader.stop)

    def test_controller_scales_grounds_and_does_not_move_parent(self):
        before = self.parent.pos
        controller = CharacterController(self.parent, self.asset, target_height=4.8)

        self.assertAlmostEqual(controller.model_scale, 0.8)
        self.assertAlmostEqual(self.actor.pos[0], 0.0)
        self.assertAlmostEqual(self.actor.pos[1], 0.0)
        self.assertAlmostEqual(self.actor.pos[2], -2.4)
        self.assertEqual(self.parent.pos, before)
        self.assertIn(("loop", "idle", 1), self.actor.calls)

    def test_reset_clears_one_shots_and_enters_requested_loop(self):
        controller = CharacterController(self.parent, self.asset)
        sim = SimpleNamespace(attached=True, alive=True, elapsed=1.0, web_arc=0.0)
        controller.update(
            sim,
            [SimpleNamespace(name="SHOT"), SimpleNamespace(name="ATTACH")],
            0.0,
        )

        controller.reset(CharacterAnimState.FALL)

        self.assertEqual(controller.state, CharacterAnimState.FALL)
        self.assertEqual(controller.current_clip, "fall")
        self.assertEqual(len(controller._one_shot_queue), 0)

    def test_shoot_attach_queue_then_physics_scrubs_swing(self):
        controller = CharacterController(self.parent, self.asset)
        sim = SimpleNamespace(
            attached=True,
            alive=True,
            elapsed=1.0,
            web_arc=math.radians(60.0),
        )

        controller.update(
            sim,
            [SimpleNamespace(name="SHOT"), SimpleNamespace(name="ATTACH")],
            0.0,
        )
        self.assertEqual(controller.state, CharacterAnimState.SHOOT)

        controller.update(sim, (), 0.24)
        self.assertEqual(controller.state, CharacterAnimState.ATTACH)

        controller.update(sim, (), 0.30)
        self.assertEqual(controller.state, CharacterAnimState.SWING)
        self.assertIn(("pose", "swing", 4), self.actor.calls)

    def test_release_interrupts_pending_animation_and_hands_follow_parent(self):
        controller = CharacterController(self.parent, self.asset)
        sim = SimpleNamespace(attached=False, alive=True, elapsed=1.0, web_arc=0.0)
        controller.update(sim, [SimpleNamespace(name="SHOT")], 0.0)
        controller.update(sim, [SimpleNamespace(name="RELEASE")], 0.0)

        self.assertEqual(controller.state, CharacterAnimState.RELEASE)
        self.assertEqual(
            controller.hand_world_position(SimpleNamespace(x=-20.0)),
            (6.0, 11.0, 16.0),
        )
        self.assertEqual(
            controller.hand_world_position(SimpleNamespace(x=20.0)),
            (8.0, 11.0, 16.0),
        )

        controller.cleanup()
        self.assertTrue(self.actor.cleaned)
        self.assertFalse(self.parent.removed)


if __name__ == "__main__":
    unittest.main()
