"""Source deploy_policy behavior: linear PD, reset histories, WBC torque caps."""
import unittest

import numpy as np
import yaml
import mujoco as mj

from test_policies import BINARY, ROOT, Fixture
import test_policies as policies


@unittest.skipUnless(BINARY.is_file(), 'Build deployment fixtures')
class DeploymentLifecycleTests(unittest.TestCase):
    def fixture(self, mode):
        fixture = Fixture(mode)
        self.addCleanup(fixture.close)
        return fixture

    def test_fixstand_is_linear_from_measured_pose_and_keeps_pd_after_completion(self):
        f = self.fixture('fixstand')
        cfg = yaml.safe_load((ROOT/'deploy/g1/config/config.yaml').read_text())['FSM']['FixStand']
        start = np.linspace(-.05, .05, 29)
        state = policies.PolicyTests.state(start)
        state['root'] = [np.cos(np.deg2rad(81.)/2), 0., np.sin(np.deg2rad(81.)/2), 0.]
        f.step(state)
        target = np.array(cfg['qs'][1])
        for time in (0., .75, 1.5, 2.25, 3., 5.):
            state['time'] = time
            out = f.step(state)
            np.testing.assert_allclose(out['command'], start + min(time/3., 1.)*(target-start), atol=1e-9)
            np.testing.assert_array_equal(out['kp'], cfg['kp'])
            np.testing.assert_array_equal(out['kd'], cfg['kd'])
            self.assertEqual(out['backend'], 'pd')
        state.update(time=6., root=[1., 0., 0., 0.])
        self.assertEqual(f.step(state)['backend'], 'pd')

    def test_source_fixstand_gains_and_target(self):
        cfg = yaml.safe_load((ROOT/'deploy/g1/config/config.yaml').read_text())['FSM']['FixStand']
        np.testing.assert_array_equal(cfg['kp'], [100,100,100,150,40,40]*2 + [200]*3 + [40]*14)
        np.testing.assert_array_equal(cfg['kd'], [2,2,2,4,2,2]*2 + [5]*3 + [10]*14)
        np.testing.assert_array_equal(cfg['qs'][1], [-.1,0,0,.3,-.2,0]*2 + [0]*3
                                      + [0,.25,0,.97,.15,0,0,0,-.25,0,.97,-.15,0,0])

    def test_vq_policy_entry_resets_history_action_and_path(self):
        f = self.fixture('walk')
        state = policies.PolicyTests.state(f.constants['standing'])
        original = f.step(state)
        for tick in range(1,80):
            state.update(time=tick*.02, ly=.5, gyro=[.1, 0., 0.])
            f.step(state)
        state.update(time=2., reset=True, ly=0., gyro=[0., 0., 0.])
        reset = f.step(state)
        np.testing.assert_array_equal(reset['observations'], original['observations'])
        np.testing.assert_array_equal(reset['path_command'], original['path_command'])
        np.testing.assert_array_equal(reset['command'], original['command'])

    def test_wbc_pd_torque_caps_use_measured_q_and_dq(self):
        f = self.fixture('wbc')
        cfg = yaml.safe_load((ROOT/'deploy/g1/config/robot_defaults.yaml').read_text())['pd_torque_clipping']
        state = policies.PolicyTests.state(np.zeros(29))
        state.update(dq=[10.]*29)
        out = f.step(state)
        torque = np.array(out['kp'])*(np.array(out['command'])-state['q'])-np.array(out['kd'])*state['dq']
        self.assertTrue(np.isfinite(torque).all())
        self.assertTrue((abs(torque) <= cfg['fraction']*np.array(cfg['joint_max_torque']) + 1e-8).all())

    def test_source_pd_fixstand_does_not_provide_learned_balance(self):
        f = self.fixture('arms')
        model = mj.MjModel.from_xml_path(str(policies.SCENE))
        data = mj.MjData(model)
        joints = np.array([mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n) for n in f.constants['names']])
        qa,va = model.jnt_qposadr[joints],model.jnt_dofadr[joints]
        actuators = np.array([np.flatnonzero(model.actuator_trnid[:,0] == j)[0] for j in joints])
        q = np.array(f.constants['standing']);data.qpos[qa] = q;mj.mj_forward(model,data)
        feet = [i for i in range(model.ngeom) if 'ankle_roll' in
                (mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[i])) or '') and model.geom_contype[i]]
        low = min(data.geom_xpos[i,2]+data.geom_xmat[i].reshape(3,3)[2]@model.geom_aabb[i,:3]
                  -abs(data.geom_xmat[i].reshape(3,3)[2])@model.geom_aabb[i,3:] for i in feet)
        data.qpos[2] -= low-.002;mj.mj_forward(model,data)
        forward = q.copy();forward[[15,22]] = -.6;forward[[18,25]] = 1.2
        # Negative control: the source PD state replaces WBC and can lose
        # whole-body balance. Do not claim that a joint pose is a balance policy.
        peak = 0.
        for tick in range(500):
            state = dict(time=tick*.02,q=data.qpos[qa].tolist(),dq=data.qvel[va].tolist(),
                         root=data.qpos[3:7].tolist(),gyro=data.qvel[3:6].tolist())
            if 100 <= tick < 350:state['reference'] = forward.tolist()
            if tick == 350:state['switch_stand'] = True
            out = f.step(state)
            if tick >= 350:self.assertEqual(out['backend'],'pd')
            command,kp,kd = (np.array(out[key]) for key in ('command','kp','kd'))
            for _ in range(10):
                data.ctrl[:] = 0.;data.ctrl[actuators] = kp*(command-data.qpos[qa])-kd*data.qvel[va]
                mj.mj_step(model,data)
            quat = data.qpos[3:7]
            tilt = float(np.arccos(np.clip(1-2*(quat[1]**2+quat[2]**2),-1,1)));peak = max(peak,tilt)
            if tick < 350:self.assertLess(tilt,.2)
            if tick >= 350 and tilt>.5:break
        self.assertGreater(peak,.5,'Pinned source PD counterexample unexpectedly stabilized')
        print(f'\nSource PD FixStand counterexample: tilt {np.rad2deg(peak):.1f} deg at {tick*.02:.2f}s',flush=True)


if __name__ == '__main__':
    unittest.main()
