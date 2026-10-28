"""TRACE -> wbc-mjlab converted motion NPZ, using exact G1 link kinematics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from .motion import MotionClip, NAMES

ROOT = Path(__file__).resolve().parents[1]
ASSET = Path(__file__).with_name("assets") / "g1_training_kinematics.xml"
DEFAULT_DATASET = ROOT / "motions"
SCHEMA = "wbc-mjlab-motion-v1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_npz(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def bake_motion(source, *, fps=50., mjcf=ASSET):
    """Match wbc-mjlab Baker's time grid, derivatives, and link-origin velocities.

    No dynamics rollout, clipping, height debias, heading reset, or joint smoothing
    is introduced here. This exports the final generated reference itself.
    """
    return bake_clip(MotionClip.load(source, align_origin=False), fps=fps, mjcf=mjcf)


def bake_clip(clip, *, fps=50., mjcf=ASSET, include_endpoint=False):
    """Shared FK and derivatives for dataset export and live WBC playback."""
    if not np.isfinite(fps) or not 1 <= fps <= 240:
        raise ValueError("Training FPS must be finite and between 1 and 240")
    model = mj.MjModel.from_xml_path(str(mjcf))
    bodies = list(range(1, model.nbody))  # Entity body order, without MuJoCo world.
    body_names = [model.body(i).name for i in bodies]
    joints = [i for i in range(model.njnt) if model.jnt_type[i] != mj.mjtJoint.mjJNT_FREE]
    joint_names = [model.joint(i).name for i in joints]
    free = [i for i in range(model.njnt) if model.jnt_type[i] == mj.mjtJoint.mjJNT_FREE]
    if len(free) != 1 or len(joints) != 29 or set(joint_names) != set(NAMES):
        raise ValueError("Training MJCF must be a free-root, 29-DoF G1")
    if any(model.jnt_type[i] != mj.mjtJoint.mjJNT_HINGE for i in joints):
        raise ValueError("G1 training joints must be scalar hinge joints")
    qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    base_q, base_v = int(model.jnt_qposadr[free[0]]), int(model.jnt_dofadr[free[0]])
    columns = [NAMES.index(n) for n in joint_names]
    dt = 1. / float(fps)
    times = np.arange(0., clip.duration, dt, dtype=np.float64)
    if include_endpoint:
        times = np.r_[times, clip.duration]
    if len(times) < 2:
        raise ValueError("Motion is too short to compute training velocities")
    samples = np.stack([clip.sample(t)[0] for t in times])
    previous = np.maximum(times - dt, 0.)
    following = np.minimum(times + dt, clip.duration)
    before = np.stack([clip.sample(t)[0] for t in previous])
    after = np.stack([clip.sample(t)[0] for t in following])
    span = (following - previous)[:, None]
    joint_pos = samples[:, 7:][:, columns]
    joint_vel = ((after[:, 7:] - before[:, 7:]) / span)[:, columns]
    base_lin_vel = (after[:, :3] - before[:, :3]) / span
    rotations = Rotation.from_quat(samples[:, 3:7], scalar_first=True)
    angular_world = (Rotation.from_quat(after[:, 3:7], scalar_first=True)
                     * Rotation.from_quat(before[:, 3:7], scalar_first=True).inv()).as_rotvec() / span
    angular_local = rotations.inv().apply(angular_world)
    count, nb = len(times), len(bodies)
    payload = dict(
        fps=np.asarray([fps], dtype=np.float32), robot=np.asarray("g1"),
        joint_names=np.asarray(joint_names), body_names=np.asarray(body_names),
        joint_pos=joint_pos.astype(np.float32), joint_vel=joint_vel.astype(np.float32),
        body_pos_w=np.empty((count, nb, 3), np.float32),
        body_quat_w=np.empty((count, nb, 4), np.float32),
        body_lin_vel_w=np.empty((count, nb, 3), np.float32),
        body_ang_vel_w=np.empty((count, nb, 3), np.float32),
    )
    data = mj.MjData(model)
    velocity = np.empty(6)
    for k in range(count):
        data.qpos[:] = model.qpos0
        data.qpos[base_q:base_q + 7] = samples[k, :7]
        data.qpos[qadr] = joint_pos[k]
        data.qvel[:] = 0.
        data.qvel[base_v:base_v + 3] = base_lin_vel[k]
        # MuJoCo free-joint rotational velocity is local, unlike NPZ *_w.
        data.qvel[base_v + 3:base_v + 6] = angular_local[k]
        data.qvel[vadr] = joint_vel[k]
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        mj.mj_comVel(model, data)
        payload["body_pos_w"][k] = data.xpos[bodies]
        payload["body_quat_w"][k] = data.xquat[bodies]
        for index, body in enumerate(bodies):
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_XBODY, body, velocity, 0)
            payload["body_ang_vel_w"][k, index] = velocity[:3]
            payload["body_lin_vel_w"][k, index] = velocity[3:]
    if any(not np.isfinite(v).all() for v in payload.values() if v.dtype.kind == "f"):
        raise ValueError("Nonfinite training kinematics")
    return payload


def export_training_motion(source, output=None, *, dataset_dir=DEFAULT_DATASET, clip_id=None,
                           fps=50., metadata=None, mjcf=ASSET):
    source = Path(source).resolve()
    output = Path(output) if output is not None else source.with_name("motion_wbc_mjlab.npz")
    if output.resolve() == source:
        raise ValueError("Training export must not overwrite the source TRACE NPZ")
    payload = bake_motion(source, fps=fps, mjcf=mjcf)
    provenance = dict(schema=SCHEMA, source_file=str(source), source_sha256=_sha(source),
                      kinematics_sha256=_sha(mjcf), fps=float(fps),
                      frames=len(payload["joint_pos"]), body_names=payload["body_names"].tolist(),
                      joint_names=payload["joint_names"].tolist(),
                      coordinates="metres, radians, wxyz, world link-origin velocities",
                      reference="TRACE final postprocessed motion, not a robot execution recording",
                      metadata=metadata or {})
    payload["export_metadata"] = np.asarray(json.dumps(provenance, ensure_ascii=False))
    _atomic_npz(output, payload)
    dataset_path = None
    if dataset_dir is not None:
        identifier = clip_id or source.parent.name
        if not identifier or Path(identifier).name != identifier or identifier in (".", ".."):
            raise ValueError("Clip ID must be a plain file name")
        dataset_path = Path(dataset_dir).resolve() / "npz" / f"{identifier}.npz"
        _atomic_npz(dataset_path, payload)
        info = dataset_path.parent.parent / "metadata" / f"{identifier}.json"
        info.parent.mkdir(parents=True, exist_ok=True)
        temporary = info.with_suffix(".tmp")
        temporary.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(info)
    return dict(file=str(output.resolve()), dataset_file=str(dataset_path) if dataset_path else None,
                fps=float(fps), frames=len(payload["joint_pos"]), bodies=len(payload["body_names"]),
                joints=len(payload["joint_names"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--source", type=Path, help="TRACE motion_final_qpos.npz")
    sources.add_argument("--jobs-dir", type=Path, help="Backfill all completed generation jobs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--clip-id")
    parser.add_argument("--fps", type=float, default=50.)
    parser.add_argument("--mjcf", type=Path, default=ASSET)
    args = parser.parse_args()
    if args.jobs_dir:
        for path in sorted(args.jobs_dir.glob("*/job.json")):
            job = json.loads(path.read_text())
            if job.get("state") != "ready":
                continue
            output = path.parent / "output"
            source = output / "motion_final_qpos.npz"
            if not source.is_file():
                continue
            config = json.loads((output / "request.json").read_text())
            report = export_training_motion(source, dataset_dir=args.dataset_dir, clip_id=job["id"], fps=args.fps,
                mjcf=args.mjcf, metadata={k: config[k] for k in ("text", "seed", "steps", "cfg", "contact_refine", "planner_provider", "planner_model") if k in config})
            (output / "training_export.json").write_text(json.dumps(report, indent=2))
            result_path = output / "result.json"
            result = json.loads(result_path.read_text())
            result["training_export"] = report
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            job["result"] = result
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2))
            temporary.replace(path)
            print(json.dumps(report))
        return
    print(json.dumps(export_training_motion(args.source, args.output, dataset_dir=args.dataset_dir,
                                           clip_id=args.clip_id, fps=args.fps, mjcf=args.mjcf), indent=2))


if __name__ == "__main__":
    main()
