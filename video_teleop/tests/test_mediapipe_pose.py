"""Depth, visibility and current-crop association checks without weights."""

import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from video_teleop.mediapipe_pose import MP_TO_COCO, MediaPipePose, h36m_pose, select_pose
from video_teleop.pose_visibility import COCO_NAMES


def sample():
    points = np.array([
        [70, 320, 1], [65, 330, 1], [65, 310, 1], [70, 345, 1], [70, 295, 1],
        [140, 380, 1], [140, 260, 1], [210, 420, 1], [210, 220, 1],
        [250, 450, 1], [250, 190, 1], [280, 360, 1], [280, 280, 1],
        [350, 370, 1], [350, 270, 1], [440, 380, 1], [440, 260, 1],
    ], dtype=float)
    image = [NS(x=.5, y=.5, z=0., visibility=1., presence=1.) for _ in range(33)]
    world = [NS(x=0., y=0., z=0.) for _ in range(33)]
    for (y, x, _), index in zip(points, MP_TO_COCO):
        image[index] = NS(x=x / 640, y=y / 480, z=0., visibility=1., presence=1.)
        world[index] = NS(x=(x - 320) / 400, y=(y - 280) / 400, z=.2)
    return points, image, world


def result(image, world):
    return NS(pose_landmarks=[image], pose_world_landmarks=[world])


class MediaPipePoseTests(unittest.TestCase):
    def test_parallel_result_still_requires_current_visible_body_and_matching_person(self):
        points, image, world = sample()
        tracker = MediaPipePose.__new__(MediaPipePose)
        tracker.indices = np.arange(17)
        tracker.threshold = .5
        tracker.person_id = None
        tracker.observed = np.zeros(17, dtype=bool)
        tracker.detect = Mock(side_effect=AssertionError("Already inferred this frame"))
        frame = np.zeros((480, 640, 3), np.uint8)
        detections = result(image, world)
        self.assertIsNotNone(tracker.update(frame, points, 1, result=detections))
        # Prefetched 3D predictions cannot enable limbs missing from current 2D.
        points[11:, 2] = 0.
        xyz = tracker.update(frame, points, 1, result=detections)
        self.assertFalse(tracker.observed[:8].any())
        self.assertTrue(np.isnan(xyz[:8]).all())
        self.assertIsNone(tracker.update(frame, points * [1, 1, 0], 2, result=detections))
        self.assertFalse(tracker.observed.any())
        self.assertEqual(tracker.failure_reason, "insufficient_visible_2d")
        points, _, _ = sample()
        for landmark in image:
            landmark.x += .5
        self.assertIsNone(tracker.update(frame, points, 3, result=detections))
        self.assertFalse(tracker.observed.any())
        self.assertEqual(tracker.failure_reason, "pose_association_rejected")
        self.assertEqual(tracker.candidate_count, 1)
        detections.pose_landmarks = []
        self.assertIsNone(tracker.update(frame, points, 3, result=detections))
        self.assertEqual(tracker.failure_reason, "no_mediapipe_pose")
        tracker.detect.assert_not_called()

    def test_depth_is_taken_from_world_landmarks_with_camera_axes(self):
        points, image, world = sample()
        world[15].z = -.4  # Left hand extends toward camera, hips at +.2.
        matched = select_pose(result(image, world), points, 640, 480, .5)
        xyz, observed = h36m_pose(*matched)
        np.testing.assert_allclose(xyz[0], [0., 0., 0.])
        np.testing.assert_allclose(xyz[13], [.325, -.075, -.6])
        world[15].z = .8  # Identical 2D projection, opposite depth motion.
        farther, _ = h36m_pose(world, matched[1])
        self.assertAlmostEqual(farther[13, 2], .6)
        self.assertTrue(observed[11:17].all())
        self.assertFalse(observed[[9, 10]].any())

    def test_cropped_parts_and_low_mediapipe_visibility_never_enable_targets(self):
        points, image, world = sample()
        points[11:, 0] = 600  # Hips/legs absent from current input.
        image[16].visibility = .1  # Other arm's wrist is occluded.
        matched = select_pose(result(image, world), points, 640, 480, .5)
        xyz, observed = h36m_pose(*matched)
        self.assertFalse(observed[:8].any())
        self.assertTrue(np.isnan(xyz[:8]).all())
        self.assertFalse(observed[16])
        self.assertTrue(np.isnan(xyz[16]).all())
        self.assertTrue(np.isfinite(xyz[11:14]).all())
        image[15].presence = .1
        matched = select_pose(result(image, world), points, 640, 480, .5)
        self.assertFalse(matched[1][9])

    def test_person_association_ignores_faces_and_rejects_ambiguity(self):
        points, image, world = sample()
        points[:5] = np.nan
        duplicate = result(image, world)
        self.assertIsNotNone(select_pose(duplicate, points, 640, 480, .5))
        duplicate.pose_landmarks *= 2
        duplicate.pose_world_landmarks *= 2
        self.assertIsNone(select_pose(duplicate, points, 640, 480, .5))
        for landmark in image:
            landmark.x += .5  # Another person far from selected 2D skeleton.
        self.assertIsNone(select_pose(result(image, world), points, 640, 480, .5))

    def test_disagreeing_or_nonfinite_wrist_is_not_exported(self):
        points, image, world = sample()
        image[15].x = .05
        matched = select_pose(result(image, world), points, 640, 480, .5)
        xyz, observed = h36m_pose(*matched)
        self.assertFalse(observed[13])
        self.assertTrue(np.isnan(xyz[13]).all())
        world[16].z = np.nan
        xyz, observed = h36m_pose(world, np.ones(17, dtype=bool))
        self.assertFalse(observed[16])

    def test_identity_switch_and_loss_never_reuse_previous_estimates(self):
        points, image, world = sample()
        detections = result(image, world)
        detectors = []

        class Detector:
            def __init__(self):
                self.calls, self.closed = 0, False

            def detect(self, _image):
                self.calls += 1
                return detections

            def close(self):
                self.closed = True

        def open_detector(tracker):
            tracker.detector = Detector()
            detectors.append(tracker.detector)

        mp = NS(Image=lambda **kw: kw["data"], ImageFormat=NS(SRGB="SRGB"))
        with patch.dict(sys.modules, {"mediapipe": mp}), patch.object(MediaPipePose, "_open", open_detector):
            tracker = MediaPipePose("unused.task", dict(enumerate(COCO_NAMES)))
            frame = np.zeros((480, 640, 3), np.uint8)
            self.assertIsNotNone(tracker.update(frame, points, 1))
            self.assertIsNotNone(tracker.update(frame, points, 1))
            tracker.update(frame, points, 2)
            self.assertEqual(detectors[0].calls, 3)
            self.assertEqual(len(detectors), 1)  # IMAGE mode has no tracking state.
            self.assertIsNone(tracker.update(frame, None, None))
            self.assertFalse(tracker.observed.any())
            tracker.update(frame, points, 2)
            detections.pose_landmarks = []
            detections.pose_world_landmarks = []
            self.assertIsNone(tracker.update(frame, points, 2))
            self.assertFalse(tracker.observed.any())
            tracker.close()
            self.assertTrue(detectors[0].closed)


if __name__ == "__main__":
    unittest.main()
