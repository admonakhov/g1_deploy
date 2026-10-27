"""The complete image reaches ViTPose; selected boxes only gate association."""

from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np

from video_teleop.person_tracking import CroppedPersonInference, crop_box, local_points, select_person


class PersonTrackingTests(unittest.TestCase):
    def test_select_center_then_keep_id_and_clip_crop_at_frame_edge(self):
        boxes = {1: (0, 10, 250, 470), 2: (285, 140, 355, 340)}
        self.assertEqual(select_person(boxes, None, 640, 480), 2)
        self.assertEqual(select_person(boxes, 1, 640, 480), 1)
        self.assertIsNone(select_person({}, 1, 640, 480))
        self.assertEqual(crop_box((1, 2, 635, 479), 640, 480), (0, 0, 640, 480))

    def test_detect_runs_yolo_once_and_does_not_infer_pose_for_bystanders(self):
        detections = np.array([[100, 100, 200, 300, .8, 0], [250, 100, 400, 350, .9, 0]])
        tensor = Mock()
        tensor.cpu.return_value.numpy.return_value = detections
        model = NS(yolo=Mock(return_value=[NS(boxes=NS(data=tensor))]),
                   tracker=Mock(), yolo_size=640, device="cpu", yolo_classes=[0], _inference=Mock())
        model.tracker.update.return_value = np.column_stack((detections[:, :5], [1, 2]))
        frontend = CroppedPersonInference(model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = frontend.detect(frame)
        self.assertEqual(len(boxes), 2)
        model.yolo.assert_called_once()
        self.assertEqual(str(model.yolo.call_args.kwargs["device"]), "cpu")
        self.assertNotIsInstance(model.yolo.call_args.kwargs["device"], str)
        model._inference.assert_not_called()
        # SORT can emit predicted boxes for an empty detection list. Ignore them.
        tensor.cpu.return_value.numpy.return_value = np.empty((0, 6))
        self.assertEqual(frontend.detect(frame), {})

    def test_full_image_is_inferred_without_automatic_bbox_crop(self):
        # Padding a 640x480 image to 3/4 adds 186 rows above the full image.
        predicted=np.zeros((1,17,3))
        predicted[0,5]=[316.,210.,.9];predicted[0,6]=[320.,220.,.9];predicted[0,11]=[330.,205.,.9]
        predicted[0,9]=[336.,450.,.9] # An extended wrist outside the YOLO box, still inside the full image.
        predicted[0,0]=[50.,10.,.9] # Synthetic letterbox padding.
        model = NS(_inference=Mock(return_value=predicted))
        frame=np.zeros((480,640,3),dtype=np.uint8)
        frame[0,0]=[1,2,3];frame[479,639]=[4,5,6];frame[100:180,200:240]=[10,20,30]
        points=CroppedPersonInference(model).pose(frame,(200,100,240,180))
        inferred=model._inference.call_args.args[0]
        self.assertEqual(inferred.shape,(853,640,3))
        np.testing.assert_array_equal(inferred[186,0],[3,2,1])
        np.testing.assert_array_equal(inferred[186+479,639],[6,5,4])
        np.testing.assert_allclose(points[5],[130.,210.,.9])
        self.assertEqual(points[9,2],.9)
        self.assertEqual(points[0,2],0.)
        model._inference.assert_called_once()


if __name__ == "__main__":
    unittest.main()
