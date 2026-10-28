"""Training NPZ: exact asset FK, world velocities, and wbc-mjlab's real reader."""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from motion_teleop.motion import NAMES
from motion_teleop.training_export import ASSET, bake_motion, export_training_motion
from motion_teleop.trace_worker import stage
from video_teleop.teleop_stream import initial_reference

TRAINING_REPO = Path(os.environ.get("WBC_MJLAB_ROOT", Path(__file__).resolve().parents[3] / "wbc-mjlab"))
TRAINING_XML = TRAINING_REPO / "src/wbc_mjlab/robots/g1/xmls/g1.xml"


class TrainingExportTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def source(self, *, moving_joints=False):
        time = np.arange(21) / 20.
        qpos = np.zeros((len(time), 36));qpos[:, 2] = .8
        qpos[:, 7:] = initial_reference()["joint_positions"]
        if moving_joints:
            qpos[:, 3] = 1.
            qpos[:, 7:] += time[:, None] * np.linspace(.01, .1, 29)
        else:
            # Both nonzero heading and a nonlocal rotation axis: export must
            # preserve world coordinates and convert free-joint omega to local.
            qpos[:, :3] += [2., -3., 0.]
            qpos[:, :3] += time[:, None] * [.2, -.1, .05]
            rotations = Rotation.from_rotvec(time[:, None] * [.4, .2, -.3]) * Rotation.from_euler("xyz", [.2, -.3, 1.7])
            qpos[:, 3:7] = rotations.as_quat(scalar_first=True)
        # TRACE export metadata, deliberately shuffled to catch index assumptions.
        qpos[:, 7:] = qpos[:, 7:][:, ::-1]
        source = self.root / "motion_final_qpos.npz"
        np.savez(source, qpos=qpos, fps=20., joint_names=np.array(NAMES[::-1]))
        return source

    def test_body_origin_world_velocities_and_preserved_heading(self):
        data = bake_motion(self.source())
        self.assertEqual(data["joint_pos"].shape, (50, 29))
        self.assertEqual(data["body_pos_w"].shape, (50, 30, 3))
        self.assertEqual(data["fps"].tolist(), [50.])
        pelvis = list(data["body_names"]).index("pelvis")
        time = np.arange(50) / 50.
        np.testing.assert_allclose(data["body_pos_w"][:, pelvis], np.array([2., -3., .8]) + time[:, None] * [.2, -.1, .05], atol=2e-7)
        np.testing.assert_allclose(data["body_ang_vel_w"], np.broadcast_to([.4, .2, -.3], (50, 30, 3)), atol=2e-6)
        offset = data["body_pos_w"] - data["body_pos_w"][:, pelvis:pelvis + 1]
        expected = [.2, -.1, .05] + np.cross([.4, .2, -.3], offset)
        np.testing.assert_allclose(data["body_lin_vel_w"], expected, atol=2e-6)
        self.assertTrue(all(v.dtype == np.float32 for k, v in data.items() if k not in ("robot", "joint_names", "body_names")))

    def test_joint_velocities_and_link_jacobians(self):
        exported = bake_motion(self.source(moving_joints=True))
        expected = np.linspace(.01, .1, 29)
        np.testing.assert_allclose(exported["joint_vel"], np.broadcast_to(expected, (50, 29)), atol=2e-7)
        model = mj.MjModel.from_xml_path(str(ASSET));data = mj.MjData(model)
        frame = 17
        data.qpos[:3] = [0., 0., .8];data.qpos[3:7] = [1., 0., 0., 0.]
        data.qpos[7:] = exported["joint_pos"][frame];mj.mj_forward(model, data)
        qvel = np.r_[np.zeros(6), expected]
        for body in range(1, model.nbody):
            linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mj.mj_jacBody(model, data, linear, angular, body)
            np.testing.assert_allclose(exported["body_lin_vel_w"][frame, body - 1], linear @ qvel, atol=2e-7)
            np.testing.assert_allclose(exported["body_ang_vel_w"][frame, body - 1], angular @ qvel, atol=2e-7)

    def test_actual_generation_training_stage_and_dataset_copy(self):
        output = self.root / "job" / "output";output.mkdir(parents=True)
        source = self.source()
        (output / source.name).write_bytes(source.read_bytes())
        (output / "request.json").write_text(json.dumps(dict(text="Wave", seed=42, steps=50, training_dataset_dir=str(self.root / "dataset"))))
        stage("training", output)
        report = json.loads((output / "training_export.json").read_text())
        self.assertEqual(report["fps"], 50.)
        self.assertEqual(Path(report["dataset_file"]), self.root / "dataset/npz/job.npz")
        self.assertEqual(Path(report["file"]).read_bytes(), Path(report["dataset_file"]).read_bytes())
        with np.load(report["file"], allow_pickle=False) as data:
            self.assertEqual(json.loads(str(data["export_metadata"]))["metadata"]["text"], "Wave")
            self.assertEqual(data["body_quat_w"].shape, (50, 30, 4))

    @unittest.skipUnless(TRAINING_XML.is_file(), "Local wbc-mjlab checkout not available")
    def test_compact_model_matches_actual_training_asset(self):
        a = mj.MjModel.from_xml_path(str(ASSET));b = mj.MjModel.from_xml_path(str(TRAINING_XML))
        self.assertEqual([a.body(i).name for i in range(a.nbody)], [b.body(i).name for i in range(b.nbody)])
        self.assertEqual([a.joint(i).name for i in range(a.njnt)], [b.joint(i).name for i in range(b.njnt)])
        rng = np.random.default_rng(123)
        for _ in range(5):
            qpos = a.qpos0.copy();qpos[7:] += rng.uniform(-.3, .3, 29)
            qpos[3:7] = Rotation.random(random_state=rng).as_quat(scalar_first=True)
            da, db = mj.MjData(a), mj.MjData(b);da.qpos[:] = db.qpos[:] = qpos
            mj.mj_forward(a, da);mj.mj_forward(b, db)
            np.testing.assert_allclose(da.xpos, db.xpos, atol=1e-12)
            np.testing.assert_allclose(da.xquat, db.xquat, atol=1e-12)

    @unittest.skipUnless(TRAINING_XML.is_file(), "Local wbc-mjlab checkout not available")
    def test_read_with_training_stack_and_actual_motionloader(self):
        import torch
        sys.path.insert(0, str(TRAINING_REPO / "src"))
        self.addCleanup(sys.path.remove, str(TRAINING_REPO / "src"))
        from wbc_mjlab.motion.library import MotionFrame
        from wbc_mjlab.motion.stack_bundle import load_motion_clip_npz, stack_clip_arrays_from_paths
        report = export_training_motion(self.source(), dataset_dir=self.root / "dataset", clip_id="wave")
        path = Path(report["dataset_file"])
        raw = load_motion_clip_npz(path)
        self.assertEqual(raw.joint_names, list(NAMES))
        stacked = stack_clip_arrays_from_paths([path])
        self.assertEqual(stacked.segment_length.tolist(), [50])
        # Execute the repository's actual loader class, without importing its
        # unrelated GPU RL environment dependencies. No reader logic is replaced.
        file = TRAINING_REPO / "src/wbc_mjlab/env/mdp/commands.py"
        node = next(n for n in ast.parse(file.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "MotionLoader")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
        namespace = dict(np=np, torch=torch, Path=Path, MotionFrame=MotionFrame)
        exec(compile(ast.fix_missing_locations(module), str(file), "exec"), namespace)
        constants = ast.parse((TRAINING_REPO / "src/wbc_mjlab/robots/g1/constants.py").read_text())
        tracked = ast.literal_eval(next(n.value for n in constants.body if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == "G1_MOTION_BODY_NAMES"))
        with np.load(path) as data:
            names = data["body_names"].tolist()
        indexes = torch.tensor([names.index(n) for n in tracked])
        anchor = tracked.index("torso_link")
        loader = namespace["MotionLoader"](str(path), indexes, anchor, .02, "cpu")
        batch = loader.frame(torch.tensor([0, 17, 49]))
        self.assertEqual(batch.joint_pos.shape, (3, 29))
        self.assertEqual(batch.body_pos_w.shape, (3, len(tracked), 3))
        self.assertTrue(torch.isfinite(batch.anchor_lin_acc_w).all())
        directory_loader = namespace["MotionLoader"](str(self.root / "dataset"), indexes, anchor, .02, "cpu")
        np.testing.assert_array_equal(directory_loader.joint_pos.numpy(), loader.joint_pos.numpy())


if __name__ == "__main__":unittest.main()
