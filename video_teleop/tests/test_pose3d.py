"""Geometry and temporal-state checks; no downloaded weights are required."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib"))

import cv2
import numpy as np
import torch

from video_teleop.pose3d import COCO_NAMES, Pose3DViewer, PoseLifter
from video_teleop.palm_tracking import PalmObservation
from video_teleop.third_party.videopose3d.model import TemporalModel, TemporalModelOptimized1f


def tiny_model(*_args, **_kwargs):
    return TemporalModelOptimized1f(17, 2, 17, [3] * 5, channels=8)


class Pose3DTests(unittest.TestCase):
    def test_optimized_model_matches_reference_center_frame(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        reference = TemporalModel(17, 2, 17, [3] * 5, channels=8).eval()
        optimized = tiny_model().eval()
        optimized.load_state_dict(reference.state_dict())
        inputs = torch.randn(2, 243, 17, 2)
        with torch.inference_mode():
            torch.testing.assert_close(optimized(inputs), reference(inputs), atol=1e-6, rtol=1e-5)

    def test_person_switch_and_partial_pose_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tiny.bin"
            torch.save({"model_pos": tiny_model().state_dict()}, checkpoint)
            names = dict(enumerate(COCO_NAMES))
            with patch("video_teleop.pose3d.TemporalModelOptimized1f", side_effect=tiny_model):
                lifter = PoseLifter(checkpoint, "cpu", names)
                fresh = PoseLifter(checkpoint, "cpu", names)
            points = np.full((17, 3), [100, 200, 1], dtype=np.float32)
            previous = points.copy()
            previous[:, 1] += 100
            lifter.update(previous, 1, 640, 480)
            actual = lifter.update(points, 2, 640, 480)
            expected = fresh.update(points, 2, 640, 480)
            np.testing.assert_allclose(actual, expected)
            np.testing.assert_array_equal(actual[0], [0, 0, 0])
            # The centre sample is the current frame; future padding repeats it.
            np.testing.assert_allclose(lifter.history[-1][0], [-0.375, -0.4375])
            points[5, 2] = 0
            partial = lifter.update(points, 2, 640, 480)
            self.assertIsNotNone(partial)
            self.assertFalse(lifter.observed[11])
            self.assertTrue(np.isnan(partial[11]).all())
            self.assertTrue(np.isfinite(partial[14]).all())
            points[:, 2] = 0
            self.assertIsNone(lifter.update(points, 2, 640, 480))
            self.assertEqual(len(lifter.history), 0)
            self.assertIsNone(lifter.previous_xyz)
            self.assertIsNone(lifter.update(None, None, 640, 480))

    def test_viewer_rotation_zoom_and_empty_pose(self):
        viewer = Pose3DViewer()
        points = np.zeros((17, 3))
        points[1] = [0.2, 0.3, 0.4]
        viewer.render(points, 1)
        # Camera (right, down, forward) -> view (right, depth, up).
        x, y, z = viewer.lines[0].get_data_3d()
        np.testing.assert_allclose([x[1], y[1], z[1]], [0.2, 0.4, -0.3])
        viewer.mouse(cv2.EVENT_LBUTTONDOWN, 100, 100, 0, None)
        viewer.mouse(cv2.EVENT_MOUSEMOVE, 140, 120, 0, None)
        viewer.mouse(cv2.EVENT_LBUTTONUP, 140, 120, 0, None)
        self.assertEqual((viewer.azimuth, viewer.elevation), (-45, 25))
        viewer.mouse(cv2.EVENT_MOUSEWHEEL, 0, 0, 120 << 16, None)
        self.assertAlmostEqual(viewer.radius, 0.9)
        points[1:7] = np.nan
        viewer.render(points, 1)
        self.assertTrue(all(len(line.get_data_3d()[0]) == 0 for line in viewer.lines[:6]))
        image = viewer.render(None, None)
        self.assertEqual(image.shape, (500, 500, 3))
        self.assertTrue(all(len(line.get_data_3d()[0]) == 0 for line in viewer.lines))

    def test_cropped_upper_body_keeps_arms_without_exporting_legs(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tiny.bin"
            torch.save({"model_pos": tiny_model().state_dict()}, checkpoint)
            with patch("video_teleop.pose3d.TemporalModelOptimized1f", side_effect=tiny_model):
                lifter = PoseLifter(checkpoint, "cpu", dict(enumerate(COCO_NAMES)))
            points = np.full((17, 3), [100, 200, 1], dtype=np.float32)
            points[11:, 0] = 700
            partial = lifter.update(points, 1, 640, 480)
            self.assertIsNotNone(partial)
            self.assertTrue(np.isnan(partial[:8]).all())
            self.assertTrue(np.isfinite(partial[11:]).all())
            points[11:, 0] = 200
            full = lifter.update(points, 1, 640, 480)
            self.assertTrue(np.isfinite(full[np.arange(17) != 10]).all())
            self.assertTrue(np.isnan(full[10]).all())
            self.assertEqual(len(lifter.history), 1)

    def test_palm_extends_wrist_and_disappears_without_observation(self):
        viewer = Pose3DViewer()
        points = np.zeros((17, 3))
        hand = np.zeros((21, 3))
        hand[[5, 9, 13, 17], 1] = .07
        palm = PalmObservation(np.zeros((21, 2)), hand)
        viewer.render(points, 1, palms={"left": palm})
        x, y, z = viewer.palm_lines["left"].get_data_3d()
        np.testing.assert_allclose([x[1], y[1], z[1]], [0, 0, -.07])
        viewer.render(points, 1, palms={})
        self.assertEqual(len(viewer.palm_lines["left"].get_data_3d()[0]), 0)


if __name__ == "__main__":
    unittest.main()
