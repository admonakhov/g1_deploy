"""Current-frame hand detection, assigned to the selected person's wrists."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from video_teleop.pose_visibility import COCO_NAMES, coco_visibility

PALM_ANCHORS = [0, 5, 9, 17]
PALM_MCPS = [5, 9, 13, 17]


@dataclass
class PalmObservation:
    image_xy: np.ndarray
    world_xyz: np.ndarray

    @property
    def offset_camera(self):
        # Hand-local translation is irrelevant: attach the vector to body wrist.
        return self.world_xyz[PALM_MCPS].mean(axis=0) - self.world_xyz[0]

    @property
    def center_image(self):
        return self.image_xy[PALM_MCPS].mean(axis=0)

    def to_json(self, wrist_xyz=None):
        return {
            "observed": True,
            "landmarks_image_xy": self.image_xy.tolist(),
            "landmarks_hand_local_xyz_m": self.world_xyz.tolist(),
            "palm_center_image_xy": self.center_image.tolist(),
            "palm_offset_camera_xyz_m": self.offset_camera.tolist(),
            "palm_xyz": None if wrist_xyz is None or not np.isfinite(wrist_xyz).all()
                else (wrist_xyz + self.offset_camera).tolist(),
        }


def match_palms(result, crop_box, body_points, width, height, threshold):
    """Use 2D wrist proximity, never image-side ordering or stale handedness."""
    x1, y1, x2, y2 = crop_box
    valid = coco_visibility(body_points, width, height, threshold)
    candidates = []
    margin = max(2.0, min(width, height) * .01)
    for image_points, world_points in zip(result.hand_landmarks, result.hand_world_landmarks):
        if len(image_points) != 21 or len(world_points) != 21:
            continue
        xy = np.array([[p.x * (x2 - x1) + x1, p.y * (y2 - y1) + y1] for p in image_points])
        xyz = np.array([[p.x, p.y, p.z] for p in world_points])
        if not np.isfinite(xy).all() or not np.isfinite(xyz).all():
            continue
        anchors = xy[PALM_ANCHORS]
        if not ((anchors[:, 0] >= margin) & (anchors[:, 0] < width - margin)
                & (anchors[:, 1] >= margin) & (anchors[:, 1] < height - margin)).all():
            continue
        observation = PalmObservation(xy, xyz)
        length = np.linalg.norm(observation.offset_camera)
        if not .015 <= length <= .20:
            continue
        candidates.append(observation)
    wrists = []
    for side, shoulder, elbow, wrist in (("left", 5, 7, 9), ("right", 6, 8, 10)):
        if valid[[elbow, wrist]].all():
            body_xy = body_points[wrist, :2][::-1]
            forearm_length = np.linalg.norm(body_points[wrist, :2] - body_points[elbow, :2])
            wrists.append((side, body_xy, max(20.0, forearm_length * .5)))
    if not candidates or not wrists:
        return {}
    costs = np.array([[np.linalg.norm(obs.image_xy[0] - xy) / gate for obs in candidates]
                      for _, xy, gate in wrists])
    rows, cols = linear_sum_assignment(costs)
    matched = {}
    for row, col in zip(rows, cols):
        if costs[row, col] > 1:
            continue
        # Crossed wrists can be ambiguous; missing a frame is preferable to
        # controlling the wrong hand. Each detection is assigned at most once.
        alternatives = np.delete(costs[:, col], row)
        if len(alternatives) and alternatives.min() - costs[row, col] < .15:
            continue
        matched[wrists[row][0]] = candidates[col]
    return matched


class PalmTracker:
    def __init__(self, checkpoint, joint_names, body_threshold=.5, hand_threshold=.5):
        import mediapipe as mp

        self.mp = mp
        name_to_index = {name: index for index, name in joint_names.items()}
        self.indices = [name_to_index[name] for name in COCO_NAMES]
        self.threshold = body_threshold
        self.wrists_visible = False
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(checkpoint)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=hand_threshold,
            min_hand_presence_confidence=hand_threshold,
        )
        # Re-detect each frame: never reuse a hand after it disappears or ID changes.
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, frame, box):
        """Return current image detections without assigning body wrists yet."""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        pad = max(16, int((x2 - x1) * .15))
        x1, x2 = max(0, x1 - pad), min(width, x2 + pad)
        y1, y2 = max(0, y1 - pad), min(height, y2 + pad)
        rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.detector.detect(image)
        return result, (x1, y1, x2, y2)

    def update(self, frame, body_points, box, detection=None):
        self.wrists_visible = False
        if body_points is None or box is None:
            return {}
        body_points = body_points[self.indices]
        height, width = frame.shape[:2]
        valid = coco_visibility(body_points, width, height, self.threshold)
        self.wrists_visible = bool(valid[[7, 9]].all() or valid[[8, 10]].all())
        if not self.wrists_visible:
            return {}
        result, (x1, y1, x2, y2) = self.detect(frame, box) if detection is None else detection
        return match_palms(result, (x1, y1, x2, y2), body_points, width, height, self.threshold)

    def close(self):
        self.detector.close()


def draw_palms(image, palms):
    for side, observation in palms.items():
        start = tuple(np.rint(observation.image_xy[0]).astype(int))
        end = tuple(np.rint(observation.center_image).astype(int))
        cv2.line(image, start, end, (255, 220, 40), 3, cv2.LINE_AA)
        cv2.circle(image, end, 5, (255, 220, 40), -1, cv2.LINE_AA)
        cv2.putText(image, side + " palm", end, cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 220, 40), 1, cv2.LINE_AA)
    return image
