import unittest

import numpy as np

from video_teleop.pose_visibility import coco_visibility, h36m_visibility, serializable_pose


class VisibilityTests(unittest.TestCase):
    def test_confidence_and_frame_bounds_override_generated_coordinates(self):
        points = np.full((17, 3), [100, 200, 1], dtype=float)
        points[13] = [600, 200, 1]  # Confident knee outside a cropped frame.
        points[14] = [479, 200, 1]  # At the crop boundary, not reliably visible.
        points[15, 2] = .1
        points[16, 0] = np.nan
        mask = coco_visibility(points, 640, 480, .5)
        self.assertTrue(mask[:13].all())
        self.assertFalse(mask[13:].any())
        observed = h36m_visibility(mask)
        self.assertFalse(observed[[2, 3, 5, 6]].any())

    def test_virtual_torso_is_not_observed_without_hips(self):
        mask = np.ones(17, dtype=bool)
        mask[11:17] = False
        observed = h36m_visibility(mask)
        self.assertFalse(observed[:8].any())
        self.assertTrue(observed[11:].all())

    def test_missing_3d_points_serialize_as_null_not_nan(self):
        xyz = np.zeros((17, 3))
        xyz[3] = np.nan
        result = serializable_pose(xyz)
        self.assertIsNone(result[3])
        self.assertEqual(result[2], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
