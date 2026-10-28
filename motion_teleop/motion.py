"""Named G1 trajectories: metres, radians, wxyz; timed interpolation, no IK."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from video_teleop.teleop_stream import initial_reference

NAMES = tuple(initial_reference()["joint_names"])


def validate_packet(packet):
    """Reject malformed network input before it can replace the active reference."""
    if not isinstance(packet, dict) or packet.get("schema") != "gmr-g1-v1" or packet.get("robot") != "unitree_g1":
        raise ValueError("Expected a gmr-g1-v1 Unitree G1 packet")
    json.dumps(packet, allow_nan=False)  # Also check optional metadata before publishing.
    status = packet.get("status")
    if status in ("stopped", "no_pose", "invalid_pose"):
        return
    if status not in ("tracking", "partial_tracking", "fallback") or packet.get("joint_units") != "radians":
        raise ValueError("Invalid pose status or joint units")
    names = packet.get("joint_names", [])
    if len(names) != 29 or set(names) != set(NAMES):
        raise ValueError("Expected all 29 named G1 joints exactly once")
    values = packet.get("joint_positions")
    active = packet.get("active_joint_mask", [])
    if len(active) != 29 or any(type(v) is not bool for v in active):
        raise ValueError("Invalid active joint mask")
    if not isinstance(values, list) or len(values) != 29:
        raise ValueError("Invalid G1 joint positions")
    for value, observed in zip(values, active):
        if value is None and not observed and status != "fallback":
            continue
        if type(value) not in (int, float) or not np.isfinite(value):
            raise ValueError("Invalid G1 joint positions")
    if status != "fallback" and not any(active):
        raise ValueError("Tracking requires at least one active joint")
    if type(packet.get("frame")) is not int or packet["frame"] < 0 or not isinstance(packet.get("stream_id"), str) or not packet["stream_id"]:
        raise ValueError("Missing frame identity")
    if not np.isfinite(float(packet.get("reference_monotonic_s", float("nan")))):
        raise ValueError("Invalid reference timestamp")
    if packet.get("base_tracked"):
        base = np.asarray(packet.get("qpos"), dtype=float)
        if (base.shape != (36,) or not np.isfinite(base).all()
                or not .5 <= np.linalg.norm(base[3:7]) <= 1.5
                or packet.get("base_quaternion_order") != "wxyz"):
            raise ValueError("Invalid tracked base")
    if packet.get("reference_kind") == "motion":
        velocity = np.asarray(packet.get("base_linear_velocity_world"), dtype=float)
        observed = dict(zip(names, active))
        if (not packet.get("base_tracked") or not all(observed[n] for n in NAMES[:12])
                or velocity.shape != (3,) or not np.isfinite(velocity).all()):
            raise ValueError("Motion velocity requires a tracked base, complete legs, and a finite world velocity")
    if "wbc_reference" in packet:
        anchor = packet["wbc_reference"]
        if (not isinstance(anchor, dict) or packet.get("reference_kind") != "motion" or anchor.get("schema") != "wbc-torso-v1"
                or anchor.get("body_name") != "torso_link" or not all(active)):
            raise ValueError("WBC anchor requires a complete named motion reference")
        for key, size in (("position_world", 3), ("quaternion_wxyz", 4),
                          ("linear_velocity_world", 3), ("angular_velocity_world", 3)):
            value = np.asarray(anchor.get(key), dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError("Invalid WBC anchor " + key)
        if not .5 <= np.linalg.norm(anchor["quaternion_wxyz"]) <= 1.5:
            raise ValueError("Invalid WBC anchor quaternion")
    if status == "fallback":
        defaults = dict(zip(NAMES, initial_reference()["joint_positions"]))
        if (any(active) or packet.get("reference_pose") != "g1_standing" or packet.get("person_id") is not None
                or packet.get("base_tracked") or packet.get("reference_joint_mask") != [True] * 29
                or any(abs(value - defaults[name]) > 1e-8 for name, value in zip(names, values))):
            raise ValueError("Fallback must be the named initial WBC stance")


@dataclass
class MotionClip:
    qpos: np.ndarray
    fps: float
    names: tuple = NAMES
    align_origin: bool = True

    def __post_init__(self):
        self.qpos = np.asarray(self.qpos, dtype=float).copy()
        self.fps = float(self.fps)
        self.names = tuple(self.names)
        if len(self.names) != 29 or set(self.names) != set(NAMES):
            raise ValueError("Motion must contain all 29 unique named G1 joints")
        if self.qpos.ndim != 2 or self.qpos.shape[1] != 36 or not 2 <= len(self.qpos) <= 100000:
            raise ValueError("Expected 2..100000 qpos frames with shape [T,36]")
        if not np.isfinite(self.fps) or not 1 <= self.fps <= 240 or not np.isfinite(self.qpos).all():
            raise ValueError("Invalid motion FPS or nonfinite pose")
        norms = np.linalg.norm(self.qpos[:, 3:7], axis=1)
        if np.any((norms < .5) | (norms > 1.5)):
            raise ValueError("Invalid motion root quaternion")
        self.qpos[:, 3:7] /= norms[:, None]
        order = [self.names.index(name) for name in NAMES]
        self.qpos[:, 7:] = self.qpos[:, 7:][:, order]
        self.names = NAMES
        rotations = Rotation.from_quat(self.qpos[:, 3:7], scalar_first=True)
        forward = rotations[0].apply([1., 0., 0.])
        heading = Rotation.from_euler("z", -np.arctan2(forward[1], forward[0]))
        origin = self.qpos[0, :3].copy(); origin[2] = 0.
        if self.align_origin:
            self.qpos[:, :3] = heading.apply(self.qpos[:, :3] - origin)
            self.qpos[:, 3:7] = (heading * rotations).as_quat(scalar_first=True)
        self._wbc = None
        self._preview = None
        self.times = np.arange(len(self.qpos)) / self.fps
        self.rotations = Slerp(self.times, Rotation.from_quat(self.qpos[:, 3:7], scalar_first=True))

    @classmethod
    def load(cls, path, *, align_origin=True):
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(data["qpos"], float(data["fps"]), tuple(data["joint_names"].tolist()), align_origin=align_origin)

    @property
    def duration(self):
        return float(self.times[-1])

    def sample(self, seconds):
        t = float(np.clip(seconds, 0., self.duration))
        index = min(int(t * self.fps), len(self.qpos) - 2)
        alpha = np.clip((t - self.times[index]) * self.fps, 0., 1.)
        left, right = self.qpos[index:index + 2]
        qpos = (1 - alpha) * left + alpha * right
        qpos[3:7] = self.rotations(t).as_quat(scalar_first=True)
        velocity = (right[:3] - left[:3]) * self.fps
        if seconds >= self.duration or seconds < 0:
            velocity[:] = 0.
        return qpos, velocity

    def prepare_wbc(self):
        """Bake once before playback, outside the 50 Hz output loop."""
        if self._wbc is not None:
            return
        from .training_export import bake_clip
        data = bake_clip(self, include_endpoint=True)
        import xml.etree.ElementTree as ET
        from .training_export import ASSET
        names = data["body_names"].tolist()
        edges=[]
        def visit(body, parent=None):
            name=body.get("name")
            if parent in names and name in names:
                edges.append([names.index(parent), names.index(name)])
            for child in body.findall("body"):
                visit(child, name)
        for body in ET.parse(ASSET).getroot().findall("worldbody/body"):
            visit(body)
        self._preview = dict(fps=50., duration_s=self.duration, body_names=names, edges=edges,
                             positions=data["body_pos_w"].tolist(),
                             times=np.r_[np.arange(0., self.duration, .02), self.duration].tolist())
        anchor = data["body_names"].tolist().index("torso_link")
        times = np.r_[np.arange(0., self.duration, .02), self.duration]
        self._wbc = (times, data["body_pos_w"][:, anchor].astype(float),
                     Slerp(times, Rotation.from_quat(data["body_quat_w"][:, anchor], scalar_first=True)),
                     data["body_lin_vel_w"][:, anchor].astype(float),
                     data["body_ang_vel_w"][:, anchor].astype(float))

    def packet(self, seconds):
        self.prepare_wbc()
        qpos, velocity = self.sample(seconds)
        times, position, rotations, linear, angular = self._wbc
        t = float(np.clip(seconds, 0., self.duration))
        interp = lambda values: [float(np.interp(t, times, values[:, i])) for i in range(3)]
        anchor = dict(schema="wbc-torso-v1", body_name="torso_link",
                      position_world=interp(position), quaternion_wxyz=rotations(t).as_quat(scalar_first=True).tolist(),
                      linear_velocity_world=interp(linear), angular_velocity_world=interp(angular))
        if seconds >= self.duration or seconds < 0:
            anchor["linear_velocity_world"] = [0.] * 3
            anchor["angular_velocity_world"] = [0.] * 3
        return dict(initial_reference(), status="tracking", person_id="trace",
                    joint_positions=qpos[7:].tolist(), active_joint_mask=[True] * 29,
                    qpos=qpos.tolist(), base_tracked=True, reference_kind="motion",
                    base_linear_velocity_world=velocity.tolist(), wbc_reference=anchor, reference_source="trace",
                    retarget_status="generated_motion", tracking_state="tracking")


def boundary_pose(packet):
    """Snapshot the current G1 goal as TRACE pose35, in its documented SPS units."""
    defaults = initial_reference()
    q = dict(zip(defaults["joint_names"], defaults["joint_positions"]))
    for name, value, active in zip(packet["joint_names"], packet["joint_positions"], packet["active_joint_mask"]):
        if active and value is not None:
            q[name] = value
    base = packet.get("qpos") if packet.get("base_tracked") else None
    if base:
        position = np.asarray(base[:3])
    else:
        import yaml
        runtime = yaml.safe_load((Path(__file__).resolve().parents[1] / "deploy/g1/config/deploy.yaml").read_text())
        position = np.asarray([0., 0., float(runtime["wbc_base_height"]) - .044])
    quat = np.asarray(base[3:7] if base else [1., 0., 0., 0.])
    return np.r_[position[[1, 2, 0]] * 100,
                 Rotation.from_quat(quat, scalar_first=True).as_euler("xyz", degrees=True),
                 np.rad2deg([q[n] for n in NAMES])].astype(np.float32)
