from types import SimpleNamespace as NS
import unittest

import numpy as np

from video_teleop.tracking_health import tracking_health


class TrackingHealthTests(unittest.TestCase):
    def test_detector_loss_and_tracker_warmup_are_distinguished(self):
        for detections, reason in ((0, "no_person"), (1, "tracker_warmup")):
            result = tracking_health(NS(detection_count=detections), None, None, None, None)
            self.assertEqual(result["reason"], reason)

    def test_current_body_failure_is_not_reported_as_a_valid_pose(self):
        lifter = NS(failure_reason="pose_association_rejected", observed=np.zeros(17, bool),
                    visible_body_2d=8, candidate_count=1)
        result = tracking_health(NS(detection_count=1), lifter, 5, None, {"status": "no_pose"})
        self.assertEqual(result["reason"], "pose_association_rejected")
        self.assertEqual(result["visible_body_2d"], 8)
        self.assertEqual(result["observed_3d"], 0)

    def test_valid_legs_do_not_claim_that_arms_are_tracked(self):
        packet = dict(status="partial_tracking", joint_names=["left_knee_joint", "left_elbow_joint", "right_wrist_pitch_joint"],
                      active_joint_mask=[True, False, False])
        lifter = NS(observed=np.ones(17, bool))
        result = tracking_health(NS(detection_count=1), lifter, 1, np.ones((17, 3)), packet)
        self.assertEqual(result["reason"], "tracking")
        self.assertEqual(result["active_robot_joints"], 1)
        self.assertEqual(result["arm_joints"], {"left": 0, "right": 0})
        packet["active_joint_mask"] = [False] * 3
        self.assertEqual(tracking_health(NS(detection_count=1), lifter, 1, np.ones((17, 3)), packet)["reason"], "no_retargetable_parts")

