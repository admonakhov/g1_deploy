"""Describe current observations separately from held/fallback references."""

import numpy as np


def tracking_health(frontend, lifter, person_id, xyz, packet):
    active = packet.get("active_joint_mask", []) if packet else []
    names = packet.get("joint_names", []) if packet else []
    if person_id is None:
        reason = "no_person" if frontend.detection_count == 0 else "tracker_warmup"
    elif xyz is None:
        reason = getattr(lifter, "failure_reason", "no_visible_3d")
    elif packet and packet.get("status") == "invalid_pose":
        reason = "invalid_skeleton"
    elif packet and not any(active):
        reason = "no_retargetable_parts"
    else:
        reason = "tracking"
    return {
        "reason": reason,
        "detected_people": frontend.detection_count,
        "person_id": person_id,
        "visible_body_2d": getattr(lifter, "visible_body_2d", None),
        "mediapipe_candidates": getattr(lifter, "candidate_count", None),
        "observed_3d": int(np.count_nonzero(lifter.observed)) if lifter else 0,
        "active_robot_joints": sum(active),
        "arm_joints": {side: sum(bool(on) for name, on in zip(names, active)
            if name.startswith(side + "_") and any(part in name for part in ("shoulder", "elbow", "wrist")))
            for side in ("left", "right")},
    }
