"""Image-based metric 3D estimates, matched to the selected ViTPose track."""

import cv2
import numpy as np

from video_teleop.pose_visibility import COCO_NAMES, H36M_ANCHORS, coco_visibility, h36m_visibility

# MediaPipe's 33 landmarks -> the existing COCO body slots.
MP_TO_COCO = np.array([0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])
BODY = np.arange(5, 17)


def image_points(landmarks, width, height):
    return np.array([
        [landmarks[i].y * height, landmarks[i].x * width,
         min(landmarks[i].visibility or 0., landmarks[i].presence or 0.)]
        for i in MP_TO_COCO
    ], dtype=float)


def select_pose(result, body_points, width, height, threshold):
    """Match current image landmarks, rejecting ambiguous people and limbs."""
    valid = coco_visibility(body_points, width, height, threshold)
    valid[:5] = False  # Face landmarks do not participate in body association.
    if valid.sum() < 3:
        return None
    extent = np.linalg.norm(np.ptp(body_points[valid, :2], axis=0))
    gate = max(30., extent * .20)
    candidates = []
    for index, landmarks in enumerate(result.pose_landmarks):
        if len(landmarks) != 33 or index >= len(result.pose_world_landmarks):
            continue
        if len(result.pose_world_landmarks[index]) != 33:
            continue
        points = image_points(landmarks, width, height)
        common = valid & coco_visibility(points, width, height, threshold)
        if common.sum() < 3:
            continue
        distances = np.linalg.norm(points[:, :2] - body_points[:, :2], axis=1)
        score = np.median(distances[common]) / gate
        if score <= 1.:
            observed = common & (distances <= gate * 1.5)
            if observed.sum() >= 3:
                candidates.append((score, index, observed))
    candidates.sort(key=lambda item: item[0])
    if not candidates or (len(candidates) > 1 and candidates[1][0] - candidates[0][0] < .15):
        return None
    _, index, observed = candidates[0]
    return result.pose_world_landmarks[index], observed


def h36m_pose(world_landmarks, visible_coco):
    """Preserve predicted depth; hide every joint without current image support.

    World landmarks use camera axes (right, down, forward), metres, hip origin.
    Recenter explicitly. Hidden hip coordinates are only an internal origin;
    they cannot enable pelvis, spine, leg or base targets.
    """
    world = np.array([[p.x, p.y, p.z] for p in world_landmarks], dtype=float)[MP_TO_COCO]
    observed_coco = visible_coco & np.isfinite(world).all(axis=1)
    origin = world[[11, 12]].mean(axis=0)
    if not np.isfinite(origin).all():
        return None, np.zeros(17, dtype=bool)
    world -= origin
    observed = h36m_visibility(observed_coco)
    # MediaPipe has no neck joint. Do not fabricate one from facial landmarks.
    observed[[9, 10]] = False
    xyz = np.full((17, 3), np.nan)
    for index in np.flatnonzero(observed):
        xyz[index] = world[list(H36M_ANCHORS[index])].mean(axis=0)
    if not observed.any():
        return None, observed
    return xyz, observed


class MediaPipePose:
    model_name = "MediaPipe Pose Heavy"
    temporal_mode = "per_frame_full_image"

    def __init__(self, checkpoint, joint_names, threshold=.5):
        import mediapipe as mp

        self.mp = mp
        self.checkpoint = checkpoint
        names_to_index = {name: index for index, name in joint_names.items()}
        self.indices = [names_to_index[name] for name in COCO_NAMES]
        self.threshold = threshold
        self.detector = None
        self.person_id = None
        self.observed = np.zeros(17, dtype=bool)
        self.failure_reason = "no_person"
        self.visible_body_2d = 0
        self.candidate_count = 0

    def reset(self):
        self.person_id = None
        self.observed[:] = False
        self.failure_reason = "no_person"
        self.visible_body_2d = 0
        self.candidate_count = 0

    def _open(self):
        options = self.mp.tasks.vision.PoseLandmarkerOptions(
            base_options=self.mp.tasks.BaseOptions(model_asset_path=str(self.checkpoint)),
            # Infer on the complete current image; association uses the selected
            # person's current 2D joints in the same full-frame coordinates.
            running_mode=self.mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=.5,
            min_pose_presence_confidence=.5,
            min_tracking_confidence=.5,
        )
        self.detector = self.mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame):
        """Image inference independent of ViTPose; association happens in update."""
        if self.detector is None:
            self._open()
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return self.detector.detect(image)

    def update(self, frame, points, person_id, result=None):
        height, width = frame.shape[:2]
        self.observed[:] = False
        if points is None or person_id is None:
            self.reset()
            return None
        if person_id != self.person_id:
            self.reset()
            self.person_id = person_id
        body_points = np.asarray(points)[self.indices]
        visible_count = int(coco_visibility(body_points, width, height, self.threshold)[BODY].sum())
        if visible_count < 3:
            self.reset()
            self.visible_body_2d = visible_count
            self.failure_reason = "insufficient_visible_2d"
            return None
        if result is None:
            result = self.detect(frame)
        match = select_pose(result, body_points, width, height, self.threshold)
        if match is None:
            # No stale estimates after a lost association.
            self.reset()
            self.visible_body_2d = visible_count
            self.candidate_count = len(result.pose_landmarks)
            self.failure_reason = "pose_association_rejected" if self.candidate_count else "no_mediapipe_pose"
            return None
        xyz, self.observed = h36m_pose(*match)
        self.visible_body_2d = visible_count
        self.candidate_count = len(result.pose_landmarks)
        self.failure_reason = "tracking" if xyz is not None else "no_visible_3d"
        return xyz

    def close(self):
        if self.detector is not None:
            self.detector.close()
            self.detector = None
