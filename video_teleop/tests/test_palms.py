from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from video_teleop.palm_tracking import PalmObservation, PalmTracker, match_palms


def hand(wrist_x, wrist_y):
    xy = np.tile([wrist_x, wrist_y], (21, 1)).astype(float)
    xy[[5, 9, 13, 17]] += [[-20, -40], [0, -50], [10, -45], [20, -35]]
    world = np.column_stack(((xy - xy[0]) / 600, np.zeros(21)))
    image_points = [SimpleNamespace(x=x / 640, y=y / 480) for x, y in xy]
    world_points = [SimpleNamespace(x=x, y=y, z=z) for x, y, z in world]
    return image_points, world_points


def result(*hands):
    return SimpleNamespace(hand_landmarks=[h[0] for h in hands], hand_world_landmarks=[h[1] for h in hands])


def body():
    points = np.full((17, 3), [100, 200, 0], dtype=float)
    points[7], points[9] = [350, 100, 1], [250, 100, 1]
    points[8], points[10] = [350, 500, 1], [250, 500, 1]
    return points


class PalmTests(unittest.TestCase):
    def test_hand_inference_is_skipped_without_current_visible_wrists(self):
        tracker = PalmTracker.__new__(PalmTracker)
        tracker.indices = np.arange(17)
        tracker.threshold = .5
        frame = np.zeros((480, 640, 3), np.uint8)
        box = (0, 0, 640, 480)
        tracker.detect = Mock(return_value=(result(hand(100, 250)), box))
        points = body()
        self.assertIn("left", tracker.update(frame, points, box))
        tracker.detect.assert_called_once()
        tracker.detect.reset_mock()
        points[9, 2] = 0
        points[10, 2] = 0
        self.assertEqual(tracker.update(frame, points, box), {})
        tracker.detect.assert_not_called()
        self.assertFalse(tracker.wrists_visible)
        # An inference already started before 2D completed is also discarded.
        self.assertEqual(tracker.update(frame, points, box,
            detection=(result(hand(100, 250)), box)), {})
        tracker.detect.assert_not_called()
        tracker.detect.return_value = result(), box
        self.assertEqual(tracker.update(frame, body(), box), {})
        self.assertTrue(tracker.wrists_visible)

    def test_reversed_detection_order_matches_anatomical_wrists(self):
        matched = match_palms(result(hand(500, 250), hand(100, 250)), (0, 0, 640, 480), body(), 640, 480, .5)
        self.assertEqual(set(matched), {"left", "right"})
        self.assertEqual(matched["left"].image_xy[0, 0], 100)
        self.assertEqual(matched["right"].image_xy[0, 0], 500)

    def test_other_person_and_ambiguous_crossed_wrists_are_rejected(self):
        points = body()
        points[8, 2] = 0
        matched = match_palms(result(hand(500, 250)), (0, 0, 640, 480), points, 640, 480, .5)
        self.assertEqual(matched, {})
        points = body()
        points[10] = points[9]
        matched = match_palms(result(hand(100, 250)), (0, 0, 640, 480), points, 640, 480, .5)
        self.assertEqual(matched, {})

    def test_missing_or_cropped_hand_is_not_reused(self):
        matched = match_palms(result(hand(100, 250)), (0, 0, 640, 480), body(), 640, 480, .5)
        self.assertIn("left", matched)
        self.assertEqual(match_palms(result(), (0, 0, 640, 480), body(), 640, 480, .5), {})
        self.assertEqual(match_palms(result(hand(5, 250)), (0, 0, 640, 480), body(), 640, 480, .5), {})
        points = body()
        points[9, 2] = 0
        self.assertEqual(match_palms(result(hand(100, 250)), (0, 0, 640, 480), points, 640, 480, .5), {})

    def test_crop_coordinates_and_attached_3d_center(self):
        source = hand(100, 250)
        crop = (40, 100, 360, 580)
        # Build normalized hand landmarks within the crop, retaining source pixels.
        image = [SimpleNamespace(x=(p.x * 640 - crop[0]) / 320, y=(p.y * 480 - crop[1]) / 480)
                 for p in source[0]]
        matched = match_palms(result((image, source[1])), crop, body(), 640, 640, .5)
        palm = matched["left"]
        np.testing.assert_allclose(palm.image_xy[0], [100, 250])
        wrist = np.array([1, 2, 3.0])
        payload = palm.to_json(wrist)
        np.testing.assert_allclose(payload["palm_xyz"], wrist + palm.offset_camera)
        self.assertIsNone(palm.to_json(np.full(3, np.nan))["palm_xyz"])


if __name__ == "__main__":
    unittest.main()
