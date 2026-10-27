"""Streaming camera-space body pose -> GMR inverse kinematics -> G1 qpos."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import xml.etree.ElementTree as ET
from tempfile import NamedTemporaryFile

import mink
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from video_teleop.third_party.gmr.general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
from video_teleop.third_party.gmr.general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT

ROOT = Path(__file__).resolve().parents[1]
LINKS = {
    "pelvis": "pelvis", "torso": "torso_link",
    **{f"{side}_{part}": f"{side}_{link}" for side in ("left", "right")
       for part, link in (("hip", "hip_pitch_link"), ("knee", "knee_link"),
                          ("foot", "ankle_roll_link"), ("shoulder", "shoulder_pitch_link"),
                          ("elbow", "elbow_link"), ("wrist", "wrist_yaw_link"),
                          ("palm", "palm"))},
}


def g1_palm_model():
    """Add two fixed end points; the upstream model and its 29 DoFs stay intact."""
    source = ROOT / "video_teleop/third_party/gmr/assets/unitree_g1/g1_mocap_29dof.xml"
    tree = ET.parse(source)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(source.parent / "meshes"))
    for side in ("left", "right"):
        wrist = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        palm = ET.SubElement(wrist, "body", name=f"{side}_palm", pos="0.10 0 0")
        ET.SubElement(palm, "geom", type="sphere", size="0.008", rgba="0.2 0.9 1 1",
                      contype="0", conaffinity="0", density="0")
    target = ROOT / ".cache/g1_with_palms.xml"
    target.parent.mkdir(exist_ok=True, parents=True)
    with NamedTemporaryFile(mode="w", dir=target.parent, suffix=".xml", delete=False) as output:
        tree.write(output, encoding="unicode")
        temporary = Path(output.name)
    temporary.replace(target)
    return target


def unit(vector):
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-5:
        raise ValueError("Degenerate skeleton: a required bone has zero length")
    return vector / length


def body_frame(left, up):
    """Anatomical frame: X forward, Y left, Z up (right-handed)."""
    up = unit(up)
    left = unit(left - np.dot(left, up) * up)
    return Rotation.from_matrix(np.column_stack((np.cross(left, up), left, up)))


def visible_targets(mask):
    """Each target requires its CURRENT observed anatomical chain."""
    active = set()
    torso = bool(mask[[0, 7, 8, 11, 14]].all())
    if torso:
        active.add("torso")
    if torso and mask[:7].all():
        active.add("pelvis")
    for side, hip, knee, ankle, shoulder, elbow, wrist in (
        ("left", 4, 5, 6, 11, 12, 13), ("right", 1, 2, 3, 14, 15, 16)
    ):
        # A leg is one indivisible observed chain. A missing ankle must also
        # disable the hip and knee, rather than retargeting a partial leg.
        if mask[[hip, knee, ankle]].all():
            active.update((f"{side}_hip", f"{side}_knee", f"{side}_foot"))
        if mask[[shoulder, elbow]].all():
            active.update((f"{side}_shoulder", f"{side}_elbow"))
            if mask[wrist]:
                active.add(f"{side}_wrist")
    return active


class FrozenDofs(mink.Limit):
    """Hard zero-displacement constraints for unobserved robot joints/base."""

    def __init__(self, model):
        self.identity = np.eye(model.nv)
        self.indices = []

    def compute_qp_inequalities(self, configuration, dt):
        if not len(self.indices):
            return mink.Constraint()
        rows = self.identity[self.indices]
        return mink.Constraint(G=np.vstack((rows, -rows)), h=np.zeros(len(rows) * 2))


class SkeletonAdapter:
    """Keep bone directions but use lengths/hip offsets from the robot model.

    Root translation is anchored; heading is relative to the first valid pose.
    Limb twist and foot orientation cannot be measured from 17 joint centres.
    """

    def __init__(self, model):
        self.model = model
        data = mj.MjData(model)
        mj.mj_forward(model, data)
        self.rest = {name: data.body(link).xpos.copy() for name, link in LINKS.items()}
        self.rest_rot = {name: Rotation.from_quat(data.body(link).xquat, scalar_first=True)
                         for name, link in LINKS.items()}
        self.heading = None

    def reset(self):
        self.heading = None

    def convert(self, xyz, observed=None, reference_qpos=None, palms=None):
        if np.shape(xyz) != (17, 3):
            raise ValueError("Expected 17 H36M XYZ joint slots")
        mask = np.isfinite(xyz).all(axis=1)
        if observed is not None:
            mask &= np.asarray(observed, dtype=bool)
        active = visible_targets(mask)
        points = np.asarray(xyz)[:, [2, 0, 1]] * [1, -1, -1]
        reference = mj.MjData(self.model)
        if reference_qpos is not None:
            reference.qpos[:] = reference_qpos
        mj.mj_forward(self.model, reference)
        held_root = Rotation.from_quat(reference.body("pelvis").xquat, scalar_first=True)
        held_torso = Rotation.from_quat(reference.body("torso_link").xquat, scalar_first=True)
        root_anchors = mask[[0, 1, 4, 7]].all()
        source_root = body_frame(points[4] - points[1], points[7] - points[0]) if root_anchors else None
        if self.heading is None:
            orientation = source_root
            if orientation is None and mask[[11, 14]].all():
                orientation = body_frame(points[11] - points[14], np.array([0, 0, 1]))
            if orientation is not None:
                forward = orientation.as_matrix()[:, 0]
                self.heading = Rotation.from_euler("z", -np.arctan2(forward[1], forward[0]))
        heading = self.heading if self.heading is not None else Rotation.identity()
        # NaN slots stay NaN and are never used to construct absent-part targets.
        points = points @ heading.as_matrix().T
        root_rotation = heading * source_root if "pelvis" in active else held_root
        torso_rotation = (body_frame(points[11] - points[14], points[8] - points[0])
                          if "torso" in active else held_torso)
        yaw = np.arctan2(root_rotation.as_matrix()[1, 0], root_rotation.as_matrix()[0, 0])
        foot_rotation = Rotation.from_euler("z", yaw)
        root_position = (np.array([0.0, 0.0, self.rest["pelvis"][2]]) if "pelvis" in active
                         else reference.body("pelvis").xpos.copy())
        # GMR needs a root reference for coordinate scaling even when its IK task
        # is disabled. It is the held robot base, not an inferred human pelvis.
        targets = {"pelvis": [root_position, root_rotation]}
        if "torso" in active:
            targets["torso"] = [root_position + root_rotation.apply(self.rest["torso"] - self.rest["pelvis"]), torso_rotation]
        for side, hip, knee, ankle, shoulder, elbow, wrist in (
            ("left", 4, 5, 6, 11, 12, 13), ("right", 1, 2, 3, 14, 15, 16)
        ):
            def rest_length(a, b):
                return np.linalg.norm(self.rest[f"{side}_{a}"] - self.rest[f"{side}_{b}"])

            if f"{side}_knee" in active:
                hip_pos = root_position + root_rotation.apply(self.rest[f"{side}_hip"] - self.rest["pelvis"])
                knee_pos = hip_pos + unit(points[knee] - points[hip]) * rest_length("knee", "hip")
                targets[f"{side}_hip"] = [hip_pos, root_rotation]
                targets[f"{side}_knee"] = [knee_pos, root_rotation]
                if f"{side}_foot" in active:
                    foot_pos = knee_pos + unit(points[ankle] - points[knee]) * rest_length("foot", "knee")
                    targets[f"{side}_foot"] = [foot_pos, foot_rotation * self.rest_rot[f"{side}_foot"]]
            if f"{side}_elbow" in active:
                shoulder_pos = (root_position + torso_rotation.apply(self.rest[f"{side}_shoulder"] - self.rest["pelvis"])
                                if "torso" in active else reference.body(LINKS[f"{side}_shoulder"]).xpos.copy())
                elbow_pos = shoulder_pos + unit(points[elbow] - points[shoulder]) * rest_length("elbow", "shoulder")
                targets[f"{side}_shoulder"] = [shoulder_pos, torso_rotation]
                targets[f"{side}_elbow"] = [elbow_pos, torso_rotation]
                if f"{side}_wrist" in active:
                    wrist_pos = elbow_pos + unit(points[wrist] - points[elbow]) * rest_length("wrist", "elbow")
                    targets[f"{side}_wrist"] = [wrist_pos, torso_rotation]
                    if palms and side in palms:
                        camera_direction = unit(np.asarray(palms[side]))
                        world_direction = heading.apply(camera_direction[[2, 0, 1]] * [1, -1, -1])
                        palm_pos = wrist_pos + world_direction * rest_length("palm", "wrist")
                        targets[f"{side}_palm"] = [palm_pos, torso_rotation]
        offset = 0.0
        if "pelvis" in active:
            neutral_height = min(self.rest[f"{side}_foot"][2] for side in ("left", "right"))
            offset = min(targets[f"{side}_foot"][0][2] for side in ("left", "right")) - neutral_height
        return {name: [position - [0, 0, offset], rotation.as_quat(scalar_first=True)]
                for name, (position, rotation) in targets.items()}


class StreamingRetargeter:
    def __init__(self, max_joint_speed=5.0, max_iterations=4):
        ROBOT_XML_DICT["unitree_g1"] = g1_palm_model()
        IK_CONFIG_DICT["h36m_stream"] = {"unitree_g1": ROOT / "video_teleop/configs/h36m_to_g1.json"}
        with redirect_stdout(io.StringIO()):
            self.gmr = GeneralMotionRetargeting("h36m_stream", "unitree_g1", solver="daqp", verbose=False)
        self.gmr.max_iter = max_iterations
        self.model = self.gmr.model
        self.adapter = SkeletonAdapter(self.model)
        self.max_joint_speed = max_joint_speed
        self.joint_ids = [self.model.actuator_trnid[i, 0] for i in range(self.model.nu)]
        self.qpos_addresses = np.array([self.model.jnt_qposadr[i] for i in self.joint_ids])
        self.joint_names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, i) for i in self.joint_ids]
        self.ranges = self.model.jnt_range[self.joint_ids].copy()
        self.posture = mink.PostureTask(self.model, cost=1e-3)
        self.posture.set_target(self.model.qpos0.copy())
        self.palm_pitch = mink.PostureTask(self.model, cost=np.zeros(self.model.nv))
        self.palm_pitch.set_target(self.model.qpos0.copy())
        self.gmr.tasks1.append(self.posture)
        self.frame_tasks = self.gmr.human_body_to_task1.copy()
        self.frozen = FrozenDofs(self.model)
        self.gmr.ik_limits.append(self.frozen)
        self.reset()

    def reset(self, keep_configuration=False):
        configuration = self.previous.copy() if keep_configuration else self.model.qpos0.copy()
        self.gmr.configuration.update(configuration)
        self.adapter.reset()
        self.previous = configuration
        self.person_id = None
        self.active_joint_mask = np.zeros(len(self.joint_names), dtype=bool)
        self.active_targets = set()

    def update(self, xyz, person_id, dt, observed=None, palms=None):
        if xyz is None:
            if person_id is not None and person_id != self.person_id:
                self.reset(keep_configuration=True)
            self.active_joint_mask[:] = False
            self.active_targets = set()
            return {**self.packet("no_pose"), "person_id": person_id}
        if self.person_id != person_id:
            # The robot pose is continuous even when the camera assigns a
            # new track ID. Reset human heading/targets, not the IK initial
            # joint configuration; all targets below come from the new frame.
            self.reset(keep_configuration=True)
            self.person_id = person_id
        try:
            mask = np.isfinite(xyz).all(axis=1)
            if observed is not None:
                mask &= np.asarray(observed, dtype=bool)
            self.active_targets = visible_targets(mask)
            valid_palms = {}
            for side, offset in (palms or {}).items():
                offset = np.asarray(offset)
                if (side in ("left", "right") and f"{side}_wrist" in self.active_targets
                        and offset.shape == (3,) and np.isfinite(offset).all()
                        and .015 <= np.linalg.norm(offset) <= .20):
                    self.active_targets.add(f"{side}_palm")
                    valid_palms[side] = offset
            targets = self.adapter.convert(xyz, mask, self.previous, palms=valid_palms)
            rejected = self._configure_palm_pitch(valid_palms)
            for side in rejected:
                self.active_targets.discard(f"{side}_palm")
                targets.pop(f"{side}_palm", None)
            self.active_joint_mask = np.array([self._joint_visible(name) for name in self.joint_names])
            if not self.active_joint_mask.any():
                return self.packet("no_pose")
        except ValueError:
            # An isolated bad camera frame must not restart the next IK solve
            # from zero. Preserve only internal state; export no observations.
            self.gmr.configuration.update(self.previous)
            self.active_joint_mask[:] = False
            self.active_targets = set()
            return self.packet("invalid_pose")
        self.gmr.human_body_to_task1 = {name: task for name, task in self.frame_tasks.items() if name in self.active_targets}
        self.gmr.tasks1 = list(self.gmr.human_body_to_task1.values()) + [self.posture, self.palm_pitch]
        inactive_joint_ids = np.asarray(self.joint_ids)[~self.active_joint_mask]
        self.frozen.indices = self.model.jnt_dofadr[inactive_joint_ids].tolist()
        freeze_base = "pelvis" not in self.active_targets
        if freeze_base:
            self.frozen.indices += list(range(6))
        before = self.gmr.configuration.q.copy()
        try:
            qpos = self.gmr.retarget(targets)
            if not np.isfinite(qpos).all():
                raise ValueError("IK produced nonfinite joint coordinates")
        except Exception as error:
            # Never publish a previous pose as a new successful IK result.
            self.gmr.configuration.update(before)
            self.reset()
            raise RuntimeError(f"GMR inverse kinematics failed: {error}") from error
        # GMR may perform several IK steps. Bound the TOTAL output change per frame.
        indices = self.qpos_addresses
        max_delta = self.max_joint_speed * min(max(float(dt), 0.001), 0.2)
        qpos[indices] = np.clip(qpos[indices], self.previous[indices] - max_delta, self.previous[indices] + max_delta)
        qpos[indices] = np.clip(qpos[indices], self.ranges[:, 0], self.ranges[:, 1])
        held_indices = indices[~self.active_joint_mask]
        qpos[held_indices] = self.previous[held_indices]
        if freeze_base:
            qpos[:7] = self.previous[:7]
        qpos[3:7] /= np.linalg.norm(qpos[3:7])
        self.gmr.configuration.update(qpos)
        self.previous = qpos.copy()
        return self.packet("tracking" if self.active_joint_mask.all() else "partial_tracking", qpos)

    def _configure_palm_pitch(self, palms):
        """Project the observed direction onto each G1 wrist's pitch plane.

        A joint-space objective prevents shoulder/yaw motion from absorbing the
        entire new palm constraint without actually bending the wrist.
        """
        cost = np.zeros(self.model.nv)
        target = self.previous.copy()
        heading = self.adapter.heading if self.adapter.heading is not None else Rotation.identity()
        rejected = []
        for side, offset in palms.items():
            world = heading.apply(unit(offset)[[2, 0, 1]] * [1, -1, -1])
            parent = self.gmr.configuration.data.body(f"{side}_wrist_roll_link")
            local = Rotation.from_quat(parent.xquat, scalar_first=True).inv().apply(world)
            if np.hypot(local[0], local[2]) < 1e-4:
                rejected.append(side)
                continue
            joint = self.model.joint(f"{side}_wrist_pitch_joint")
            target[joint.qposadr[0]] = np.clip(np.arctan2(-local[2], local[0]), *joint.range)
            cost[joint.dofadr[0]] = 20.0
        self.palm_pitch.set_cost(cost)
        self.palm_pitch.set_target(target)
        return rejected

    def _joint_visible(self, name):
        if "wrist_pitch_joint" in name:
            side = "left" if name.startswith("left_") else "right"
            return f"{side}_palm" in self.active_targets
        if name.startswith("waist_"):
            return "torso" in self.active_targets
        side = "left" if name.startswith("left_") else "right"
        if "hip_" in name:
            part = "knee"
        elif "knee_" in name or "ankle_" in name:
            part = "foot"
        elif "shoulder_" in name:
            part = "elbow"
        else:
            part = "wrist"
        return f"{side}_{part}" in self.active_targets

    def packet(self, status, qpos=None):
        active_mask = self.active_joint_mask if qpos is not None else np.zeros(len(self.joint_names), dtype=bool)
        return {
            "schema": "gmr-g1-v1", "robot": "unitree_g1", "status": status,
            "person_id": self.person_id, "joint_names": self.joint_names,
            "joint_units": "radians", "base_quaternion_order": "wxyz",
            "base_axes": ["forward", "left", "up"], "base_units": "metres",
            "active_joint_mask": active_mask.tolist(),
            "retargeted_body_parts": sorted(self.active_targets) if qpos is not None else [],
            "palms_tracked": {side: qpos is not None and f"{side}_palm" in self.active_targets
                              for side in ("left", "right")},
            "base_tracked": qpos is not None and "pelvis" in self.active_targets,
            "qpos_contains_held_joints": qpos is not None and not self.active_joint_mask.all(),
            "qpos": None if qpos is None else qpos.tolist(),
            "joint_positions": None if qpos is None else [float(qpos[index]) if active else None
                for index, active in zip(self.qpos_addresses, self.active_joint_mask)],
        }



# Compatibility import; transport startup does not require GMR or MuJoCo.
from video_teleop.teleop_stream import UDPPublisher
