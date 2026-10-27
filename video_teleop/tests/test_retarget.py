"""Actual GMR IK checks, model limits, streaming state and bounded capture."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from video_teleop.robot_retarget import StreamingRetargeter, UDPPublisher
from video_teleop.stream_capture import StreamCapture


def standing_pose():
    # H36M in robot axes, then converted back to VideoPose3D camera axes.
    world = np.array([
        [0, 0, 0], [0, -.1, -.1], [0, -.1, -.5], [0, -.1, -.9],
        [0, .1, -.1], [0, .1, -.5], [0, .1, -.9], [0, 0, .2],
        [0, 0, .4], [0, 0, .55], [0, 0, .7], [0, .22, .4],
        [0, .24, .1], [0, .24, -.15], [0, -.22, .4],
        [0, -.24, .1], [0, -.24, -.15],
    ])
    return world[:, [1, 2, 0]] * [-1, -1, 1]


class RetargetTests(unittest.TestCase):
    def test_raise_arm_tracks_with_limits_and_speed_bound(self):
        robot = StreamingRetargeter(max_joint_speed=3.)
        pose = standing_pose()
        for _ in range(50):
            packet = robot.update(pose, 1, 1 / 30)
        initial_z = robot.gmr.configuration.data.body("left_wrist_yaw_link").xpos[2]
        raised = pose.copy()
        raised[12, 1], raised[13, 1] = -.7, -1.0
        previous = robot.previous[robot.qpos_addresses].copy()
        for _ in range(70):
            packet = robot.update(raised, 1, 1 / 30)
            joints = robot.previous[robot.qpos_addresses].copy()
            self.assertLessEqual(np.max(np.abs(joints - previous)), 0.100001)
            self.assertTrue(np.all(joints >= robot.ranges[:, 0] - 1e-6))
            self.assertTrue(np.all(joints <= robot.ranges[:, 1] + 1e-6))
            self.assertAlmostEqual(np.linalg.norm(packet["qpos"][3:7]), 1)
            previous = joints
        final_z = robot.gmr.configuration.data.body("left_wrist_yaw_link").xpos[2]
        self.assertGreater(final_z - initial_z, .25)
        self.assertEqual(len(packet["joint_names"]), 29)
        self.assertEqual(len(packet["qpos"]), 36)

    def test_loss_and_person_switch_do_not_publish_stale_pose(self):
        robot = StreamingRetargeter(max_joint_speed=3.)
        pose = standing_pose()
        first = robot.update(pose, 1, 1 / 30)
        for _ in range(4):
            robot.update(pose, 1, 1 / 30)
        before = robot.previous[robot.qpos_addresses].copy()
        switched = robot.update(pose, 2, 1 / 30)
        self.assertEqual(switched["person_id"], 2)
        self.assertLessEqual(np.max(np.abs(robot.previous[robot.qpos_addresses] - before)), .100001)
        lost = robot.update(None, None, 1 / 30)
        self.assertEqual(lost["status"], "no_pose")
        self.assertIsNone(lost["joint_positions"])
        self.assertIsNone(lost["qpos"])
        invalid = robot.update(np.zeros((17, 3)), 1, 1 / 30)
        self.assertEqual(invalid["status"], "invalid_pose")
        self.assertIsNone(invalid["qpos"])

    def test_adapter_uses_robot_segment_lengths(self):
        robot = StreamingRetargeter(max_joint_speed=3.)
        targets = robot.adapter.convert(standing_pose())
        for side in ("left", "right"):
            for a, b in (("hip", "knee"), ("knee", "foot"), ("shoulder", "elbow"), ("elbow", "wrist")):
                a, b = f"{side}_{a}", f"{side}_{b}"
                actual = np.linalg.norm(targets[a][0] - targets[b][0])
                expected = np.linalg.norm(robot.adapter.rest[a] - robot.adapter.rest[b])
                self.assertAlmostEqual(actual, expected)

    def test_brief_loss_and_bad_frame_keep_internal_ik_warm_without_exporting_joints(self):
        robot = StreamingRetargeter(max_joint_speed=3.)
        pose = standing_pose()
        pose[12, 1], pose[13, 1] = -.7, -1.
        for _ in range(35):
            robot.update(pose, 7, 1 / 30)
        before = robot.previous.copy()
        self.assertGreater(np.linalg.norm(before[robot.qpos_addresses]), .3)
        lost = robot.update(None, None, 1 / 30)
        self.assertIsNone(lost["joint_positions"])
        self.assertFalse(any(lost["active_joint_mask"]))
        np.testing.assert_array_equal(robot.previous, before)
        bad = robot.update(np.zeros((17, 3)), 7, 1 / 30)
        self.assertEqual(bad["status"], "invalid_pose")
        self.assertIsNone(bad["qpos"])
        np.testing.assert_array_equal(robot.previous, before)
        robot.update(pose, 7, 1 / 30)
        self.assertLessEqual(np.max(np.abs(robot.previous[robot.qpos_addresses] - before[robot.qpos_addresses])), .100001)

    def test_udp_packet_schema_and_nonblocking_socket(self):
        with patch("video_teleop.teleop_stream.socket.socket") as factory:
            publisher = UDPPublisher("127.0.0.1:9870")
            packet = {"schema": "gmr-g1-v1", "joint_positions": [0.1, 0.2]}
            publisher.publish(packet)
            payload, destination = factory.return_value.sendto.call_args.args
            self.assertEqual(json.loads(payload), packet)
            self.assertEqual(destination, ("127.0.0.1", 9870))
            factory.return_value.setblocking.assert_called_once_with(False)
            publisher.close()

    def test_hidden_legs_are_not_targets_and_are_held_while_arm_moves(self):
        robot = StreamingRetargeter(max_joint_speed=3.)
        pose = standing_pose()
        for _ in range(5):
            robot.update(pose, 1, 1 / 30)
        before = robot.previous.copy()
        mask = np.ones(17, dtype=bool)
        mask[:8] = False  # Both legs, pelvis and spine absent from the camera.
        partial = pose.copy()
        partial[~mask] = np.nan
        partial[12, 1], partial[13, 1] = -.7, -1.0
        leg_slots = [i for i, name in enumerate(robot.joint_names) if any(part in name for part in ("hip_", "knee_", "ankle_"))]
        leg_qpos = robot.qpos_addresses[leg_slots]
        for _ in range(50):
            packet = robot.update(partial, 1, 1 / 30, observed=mask)
            self.assertEqual(packet["status"], "partial_tracking")
            self.assertTrue(all(packet["joint_positions"][i] is None for i in leg_slots))
            np.testing.assert_array_equal(robot.previous[leg_qpos], before[leg_qpos])
            np.testing.assert_array_equal(robot.previous[:7], before[:7])
        self.assertTrue(all("hip" not in part and "knee" not in part and "foot" not in part
                            for part in packet["retargeted_body_parts"]))
        self.assertGreater(robot.gmr.configuration.data.body("left_wrist_yaw_link").xpos[2], 1.0)

    def test_hidden_arm_predictions_cannot_change_retargeting(self):
        robots = [StreamingRetargeter(max_joint_speed=3.), StreamingRetargeter(max_joint_speed=3.)]
        pose = standing_pose()
        mask = np.ones(17, dtype=bool)
        mask[[11, 12, 13]] = False
        variants = [pose.copy(), pose.copy()]
        variants[0][~mask] = np.nan
        variants[1][~mask] = 1e6  # Hallucinated but finite predictions must be ignored.
        for _ in range(5):
            packets = [robot.update(points, 1, 1 / 30, observed=mask) for robot, points in zip(robots, variants)]
            np.testing.assert_allclose(packets[0]["qpos"], packets[1]["qpos"], atol=1e-9)
        for name, value in zip(packets[0]["joint_names"], packets[0]["joint_positions"]):
            if name.startswith("left_") and any(part in name for part in ("shoulder_", "elbow_", "wrist_")):
                self.assertIsNone(value)
        restored = robots[0].update(pose, 1, 1 / 30, observed=np.ones(17, dtype=bool))
        self.assertEqual(restored["status"], "partial_tracking")
        for name, active in zip(restored["joint_names"], restored["active_joint_mask"]):
            self.assertEqual(active, "wrist_pitch_joint" not in name)

    def test_any_missing_leg_joint_disables_the_entire_leg(self):
        for side, chain, other_side in (("left", (4, 5, 6), "right"), ("right", (1, 2, 3), "left")):
            for missing in chain:
                with self.subTest(side=side, missing=missing):
                    robot = StreamingRetargeter(max_joint_speed=3.)
                    pose = standing_pose()
                    robot.update(pose, 1, 1 / 30)
                    before = robot.previous.copy()
                    mask = np.ones(17, dtype=bool)
                    mask[missing] = False
                    # Finite hallucinations and moving visible leg points must
                    # not bypass the whole-leg gate or move its held joints.
                    partial = pose.copy()
                    partial[list(chain), 2] += .25
                    partial[missing] = 1e6
                    partial[12, 1], partial[13, 1] = -.7, -1.0
                    slots = [i for i, name in enumerate(robot.joint_names)
                             if name.startswith(side + "_") and any(p in name for p in ("hip_", "knee_", "ankle_"))]
                    addresses = robot.qpos_addresses[slots]
                    for _ in range(4):
                        packet = robot.update(partial, 1, 1 / 30, observed=mask)
                        np.testing.assert_array_equal(robot.previous[addresses], before[addresses])
                        np.testing.assert_array_equal(robot.previous[:7], before[:7])
                        self.assertTrue(all(packet["joint_positions"][i] is None for i in slots))
                        self.assertFalse(any(packet["active_joint_mask"][i] for i in slots))
                        for part in ("hip", "knee", "foot"):
                            self.assertNotIn(f"{side}_{part}", packet["retargeted_body_parts"])
                            self.assertIn(f"{other_side}_{part}", packet["retargeted_body_parts"])
                        self.assertIn("left_elbow", packet["retargeted_body_parts"])
                    restored = robot.update(pose, 1, 1 / 30)
                    self.assertTrue(all(restored["active_joint_mask"][i] for i in slots))
                    self.assertLessEqual(np.max(np.abs(robot.previous[addresses] - before[addresses])), .100001)

    def test_palm_changes_wrist_pitch_and_loss_holds_it(self):
        robot = StreamingRetargeter(max_joint_speed=3.)
        pose = standing_pose()
        pitch = robot.model.joint("left_wrist_pitch_joint").qposadr[0]
        for _ in range(60):
            packet = robot.update(pose, 1, 1 / 30, palms={"left": np.array([0, .07, 0])})
        initial = robot.previous[pitch]
        for _ in range(60):
            packet = robot.update(pose, 1, 1 / 30, palms={"left": np.array([0, .04, .06])})
        self.assertGreater(abs(robot.previous[pitch] - initial), .1)
        self.assertTrue(packet["palms_tracked"]["left"])
        self.assertFalse(packet["palms_tracked"]["right"])
        self.assertIn("left_palm", packet["retargeted_body_parts"])
        held = robot.previous[pitch]
        for _ in range(5):
            packet = robot.update(pose, 1, 1 / 30, palms={})
            self.assertEqual(robot.previous[pitch], held)
            self.assertIsNone(packet["joint_positions"][robot.joint_names.index("left_wrist_pitch_joint")])
        self.assertNotIn("left_palm", packet["retargeted_body_parts"])

    def test_palm_without_body_wrist_is_not_retargeted(self):
        robot = StreamingRetargeter(max_joint_speed=3.)
        mask = np.ones(17, dtype=bool)
        mask[13] = False
        packet = robot.update(standing_pose(), 1, 1 / 30, observed=mask, palms={"left": np.array([0, .07, 0])})
        self.assertFalse(packet["palms_tracked"]["left"])
        self.assertNotIn("left_palm", packet["retargeted_body_parts"])

    def test_no_visible_parts_have_no_generated_commands(self):
        robot = StreamingRetargeter(max_joint_speed=3.)
        packet = robot.update(standing_pose(), 1, 1 / 30, observed=np.zeros(17, dtype=bool))
        self.assertEqual(packet["status"], "no_pose")
        self.assertIsNone(packet["joint_positions"])
        self.assertIsNone(packet["qpos"])
        self.assertEqual(packet["retargeted_body_parts"], [])


class FakeCapture:
    def __init__(self):
        self.items = iter([(True, np.full((2, 2, 3), i, dtype=np.uint8)) for i in range(3)])
        self.closed = False

    def read(self):
        return next(self.items, (False, None))

    def release(self):
        self.closed = True


class CaptureTests(unittest.TestCase):
    def test_latest_frame_has_no_backlog(self):
        capture = FakeCapture()
        reader = StreamCapture(capture)
        reader.latest = True
        reader._collect()
        ok, image, index, *_ = reader.read()
        self.assertTrue(ok)
        self.assertEqual(index, 2)
        self.assertTrue(np.all(image == 2))
        self.assertFalse(reader.read()[0])
        self.assertTrue(capture.closed)

    def test_video_file_keeps_every_frame(self):
        capture = FakeCapture()
        reader = StreamCapture(capture)
        self.assertEqual([reader.read()[2] for _ in range(3)], [0, 1, 2])
        self.assertFalse(reader.read()[0])
        reader.close()
        self.assertTrue(capture.closed)


if __name__ == "__main__":
    unittest.main()
