"""The exact bridge packet -> native interpolator -> real WBC ONNX contract."""
from pathlib import Path
import unittest

import numpy as np

from deploy.tests.test_policies import Fixture
from motion_teleop.bridge import ReferenceBridge
from motion_teleop.motion import MotionClip
from video_teleop.teleop_stream import initial_reference

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless((ROOT / "deploy/g1/build/g1_policy_fixture").is_file(), "Build native fixtures")
class NativeMotionTests(unittest.TestCase):
    def test_motion_velocity_reaches_wbc_and_arms_suppresses_it(self):
        values = np.zeros((101, 36));values[:, 2] = .7842;values[:, 3] = 1.
        values[:, 7:] = initial_reference()["joint_positions"]
        values[:, 0] = np.arange(101) * .01
        clip = MotionClip(values, 20.)
        for mode in ("wbc", "arms"):
            with self.subTest(mode=mode):
                f = Fixture(mode)
                try:
                    bridge = ReferenceBridge();bridge.play(clip, "native", 0.)
                    for tick in range(101):
                        state = dict(time=tick * .02, q=f.constants["standing"], dq=[0.] * 29,
                                     root=[1., 0., 0., 0.], gyro=[0.] * 3, camera_stream=True,
                                     camera_packets=[bridge.packet(tick * .02)])
                        out = f.step(state)
                        self.assertEqual(out["camera_diagnostics"]["rejected_packets"], 0)
                        self.assertTrue(np.isfinite(out["command"]).all())
                    expected = .2 if mode == "wbc" else 0.
                    self.assertAlmostEqual(out["observations"][1], expected, places=5)
                    self.assertAlmostEqual(out["observations"][2], 0., places=5)
                    self.assertFalse(out["camera_waiting"])
                finally:
                    f.close()

    def test_all_39_reference_features_match_training_npz_without_filters(self):
        from scipy.spatial.transform import Rotation
        from motion_teleop.training_export import bake_clip
        values = np.zeros((81, 36));t = np.arange(81) / 20.
        values[:, 2] = .9 + .65 * np.sin(2. * t)  # outside the former height clamp
        values[:, 0] = .3 * t;values[:, 1] = -.2 * t
        values[:, 3:7] = Rotation.from_euler("xyz", np.c_[.3*np.sin(t), .2*np.cos(t), 4.*t]).as_quat(scalar_first=True)
        values[:, 7:] = initial_reference()["joint_positions"]
        values[:, 19] = .3*np.sin(3.*t)  # waist yaw
        values[:, 20] = .2*np.cos(2.*t)  # waist roll
        values[:, 21] = .15*np.sin(4.*t) # waist pitch
        values[:, 22] = 1.4*np.sin(6.*t) # faster than 5 rad/s
        clip = MotionClip(values, 20.)
        expected = bake_clip(clip)
        anchor = expected["body_names"].tolist().index("torso_link")
        f = Fixture("wbc")
        try:
            bridge=ReferenceBridge();bridge.play(clip,"exact",0.)
            for tick in range(3,180):
                # Include every packet; the native receiver interpolates with a 40 ms delay.
                start = 0 if tick==3 else tick
                packets=[bridge.packet(k*.02) for k in range(start,tick+1)]
                # Ingest all startup samples at the same receive time only on first step.
                if tick==3:
                    for k in range(3):
                        f.step(dict(time=k*.02,q=f.constants["standing"],dq=[0.]*29,root=[1.,0.,0.,0.],gyro=[0.]*3,
                                    camera_stream=True,camera_packets=[packets[k]]))
                    packets=packets[3:]
                out=f.step(dict(time=tick*.02,q=f.constants["standing"],dq=[0.]*29,root=[1.,0.,0.,0.],gyro=[0.]*3,
                                camera_stream=True,camera_packets=packets))
                self.assertEqual(out["camera_diagnostics"]["rejected_packets"],0)
                index=tick-2
                quat=Rotation.from_quat(expected["body_quat_w"][index,anchor],scalar_first=True)
                target=np.r_[expected["body_pos_w"][index,anchor,2],
                             quat.inv().apply(expected["body_lin_vel_w"][index,anchor]),
                             quat.inv().apply(expected["body_ang_vel_w"][index,anchor]),
                             quat.inv().apply([0.,0.,-1.]),expected["joint_pos"][index]]
                np.testing.assert_allclose(out["observations"][:39],target,atol=2e-6,err_msg=f"tick {tick}")
        finally:f.close()

    def test_wbc_output_is_unclipped_reference_residual_onnx(self):
        import yaml
        directory=ROOT/"deploy/g1/policies/wbc/params"
        raw_model=Fixture("wbc_reference")
        scale=np.array(yaml.safe_load((directory/"config.yaml").read_text())["action"]["scale"])
        f=Fixture("wbc")
        try:
            q=np.array(f.constants["standing"]);target=q.copy();target[15]=2.6;target[18]=2.
            previous=np.zeros(29)
            for tick in range(5):
                out=f.step(dict(time=tick*.02,q=q.tolist(),dq=[10.]*29,root=[1.,0.,0.,0.],gyro=[0.]*3,reference=target.tolist()))
                obs=np.asarray(out["observations"],dtype=np.float32)
                np.testing.assert_allclose(obs[10:39],target,atol=1e-7)
                np.testing.assert_allclose(obs[103:],previous,atol=1e-7)
                raw=np.asarray(raw_model.step({"obs":obs.tolist()})["raw"])
                np.testing.assert_allclose(out["command"],target+scale*raw,atol=2e-6)
                previous=raw
        finally:
            f.close()
            raw_model.close()


if __name__ == "__main__":unittest.main()
