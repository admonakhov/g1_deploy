"""Full-frame inference with detector boxes used only for person association."""

import cv2
import numpy as np


def select_person(boxes, selected, width, height):
    if selected in boxes:
        return selected
    # Normalize by image dimensions so portrait/landscape use the same rule.
    def distance(person_id):
        x1, y1, x2, y2 = boxes[person_id]
        return ((x1 + x2) / (2 * width) - .5) ** 2 + ((y1 + y2) / (2 * height) - .5) ** 2
    return min(boxes, key=distance, default=None)


def crop_box(box, width, height, padding=10):
    x1, y1, x2, y2 = box
    return (max(0, x1 - padding), max(0, y1 - padding),
            min(width, x2 + padding), min(height, y2 + padding))


def local_points(points, box):
    local = np.array(points, copy=True)
    local[:, :2] -= [box[1], box[0]]
    return local


class CroppedPersonInference:
    """Full-frame Easy-ViTPose; bbox associates identity without cutting limbs."""

    def __init__(self, model):
        self.model = model
        self.detection_count = 0

    def detect(self, frame):
        import torch

        model = self.model
        # select_device('cpu') in Ultralytics resets PyTorch's thread budget on
        # lazy predictor setup. A resolved device preserves our configured limit.
        device = torch.device("cpu") if model.device == "cpu" else model.device
        results = model.yolo(frame, verbose=False, imgsz=model.yolo_size,
                             device=device if device != "cuda" else 0,
                             classes=model.yolo_classes)[0]
        detections = results.boxes.data.cpu().numpy()
        detections = detections[detections[:, 4] > .35, :5]
        self.detection_count = len(detections)
        tracked = model.tracker.update(detections)
        # SORT may return predicted boxes on empty detections; never run a pose
        # model for a person that YOLO did not observe in the current frame.
        if not len(detections):
            return {}
        height, width = frame.shape[:2]
        boxes = {}
        for row in tracked:
            x1, y1, x2, y2 = np.rint(row[:4]).astype(int)
            x1, x2 = np.clip([x1, x2], 0, width)
            y1, y2 = np.clip([y1, y2], 0, height)
            if x2 > x1 and y2 > y1:
                boxes[int(row[5])] = (int(x1), int(y1), int(x2), int(y2))
        return boxes

    def pose(self, frame, box):
        from easy_ViTPose.vit_utils.inference import pad_image

        x1, y1, x2, y2 = box
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        padded, (left_pad, top_pad) = pad_image(rgb, 3 / 4)
        points = self.model._inference(padded)[0].copy()
        points[:, :2] -= [top_pad, left_pad]
        height, width = frame.shape[:2]
        valid = (np.isfinite(points).all(axis=1) & (points[:, 0] >= 2)
                 & (points[:, 0] < height-2) & (points[:, 1] >= 2) & (points[:, 1] < width-2))
        # A full-image pose may describe another person. Require current body
        # support inside the selected detection, but keep their extended limbs
        # outside the bbox when they remain visible in the original image.
        body = np.arange(5,17) if len(points)>=17 else np.arange(len(points))
        associated = valid[body] & (points[body,2]>.35) & (points[body,0]>=y1) & (points[body,0]<y2) & (points[body,1]>=x1) & (points[body,1]<x2)
        if associated.sum()<min(3,len(body)):
            points[:,2]=0.
        else:
            points[~valid,2]=0.
        return points
