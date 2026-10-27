"""Current-frame visibility; inferred 3D joints never count as observations."""

import numpy as np

COCO_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
H36M_NAMES = (
    "pelvis", "right_hip", "right_knee", "right_ankle", "left_hip",
    "left_knee", "left_ankle", "spine", "thorax", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist", "right_shoulder",
    "right_elbow", "right_wrist",
)
# Virtual joints require their observed anatomical anchors, not a model score.
H36M_ANCHORS = (
    (11, 12), (12,), (14,), (16,), (11,), (13,), (15,),
    (5, 6, 11, 12), (5, 6), (0, 5, 6), (0, 1, 2),
    (5,), (7,), (9,), (6,), (8,), (10,),
)


def face_joint_mask(joint_names, count):
    """Identify facial/head landmarks without removing neck or hand joints."""
    def is_face(name):
        name = name.lower().replace(" ", "_")
        return (name in {"nose", "head", "head_top", "headtop"}
                or name.endswith(("_eye", "_ear")) or name.startswith(("face-", "face_")))
    return np.array([is_face(joint_names.get(index, "")) for index in range(count)], dtype=bool)


def without_face(points, joint_names):
    result = np.array(points, dtype=float, copy=True)
    result[face_joint_mask(joint_names, len(result))] = np.nan
    return result


def coco_visibility(points, width, height, threshold):
    points = np.asarray(points)
    if points.shape != (17, 3):
        raise ValueError("Expected COCO-17 (y, x, confidence) keypoints")
    margin = max(2.0, min(width, height) * 0.01)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] >= threshold)
    valid &= (points[:, 0] >= margin) & (points[:, 0] < height - margin)
    valid &= (points[:, 1] >= margin) & (points[:, 1] < width - margin)
    return valid


def h36m_visibility(coco_mask):
    mask = np.array([np.all(coco_mask[list(indices)]) for indices in H36M_ANCHORS], dtype=bool)
    mask[10] = False  # Head/face is not part of the exposed body skeleton.
    return mask


def serializable_pose(xyz):
    if xyz is None:
        return None
    return [point.tolist() if np.isfinite(point).all() else None for point in xyz]
