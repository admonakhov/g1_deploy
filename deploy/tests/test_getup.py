"""Exact ROS2 get_up command construction and bounded take completion, no DDS."""
import unittest
import numpy as np
import yaml
import mujoco as mj
from scipy.spatial.transform import Rotation
from test_policies import BINARY, ROOT, Fixture
import test_policies as policies

@unittest.skipUnless(BINARY.is_file(), 'Build deployment fixtures')
class GetUpTests(unittest.TestCase):
    def test_getup_matches_source_observations_offsets_gains_and_unclipped_targets(self):
        directory=ROOT/'deploy/g1/policies/getup'
        cfg=yaml.safe_load((directory/'source_config.yaml').read_text())
        g1=cfg['g1'];ids=g1['control_joints'];defaults=np.array(g1['default_dof_pos'],dtype=np.float32)
        old_limits=__import__('json').loads((ROOT/'deploy/g1/config/joint_limits.json').read_text())
        largest_old_clip=0.
        for angle,branch in ((-1.45,'front_get_up'),(1.45,'back_get_up')):
            f=Fixture('getup');reference=Fixture('getup_reference');self.addCleanup(f.close);self.addCleanup(reference.close)
            take=g1['punch']['trajs'][g1['punch'][branch]]
            rows=np.loadtxt(directory/'aux'/f"{take['file']}.csv",delimiter=',',dtype=np.float32)
            rows=rows[round(take['start']*50):round(take['end']*50)+1]
            np.testing.assert_allclose(f.constants['kp'],g1['kp'],atol=0.)
            np.testing.assert_allclose(f.constants['kd'],g1['kd'],atol=0.)
            np.testing.assert_allclose(f.constants['standing'],defaults,atol=3e-8)
            history=[];last=np.zeros(23,dtype=np.float32)
            for tick,row in enumerate(rows):
                q=defaults+(.1*np.sin(np.arange(29)+tick*.03)).astype(np.float32)
                dq=(.3*np.cos(np.arange(29)+tick*.03)).astype(np.float32)
                rotation=Rotation.from_euler('xyz',[.02*np.sin(tick*.02),angle+.02*np.sin(tick*.04),.3+tick*.001])
                xyzw=rotation.as_quat();root=xyzw[[3,0,1,2]];gyro=np.array([.03,.02,-.01],dtype=np.float32)
                gravity=rotation.inv().apply([0.,0.,-1.]).astype(np.float32)
                frame=np.r_[gyro*np.float32(.2),gravity,q[ids]-defaults[ids],dq[ids]*np.float32(.05),last].astype(np.float32)
                history.append(np.clip(frame,-100.,100.));history=history[-10:]
                observations=np.array([history[max(0,i-(10-len(history)))] for i in range(10)]).ravel()
                ref=Rotation.from_quat(row[[68,69,70,67]])
                if tick==0:
                    matrix=(rotation*ref.inv()).as_matrix();yaw=np.arctan2(matrix[1,0],matrix[0,0]);offset=Rotation.from_euler('z',yaw)
                matrix=(offset*ref*rotation.inv()).as_matrix()
                command=np.r_[row[:64],np.zeros(3),matrix[:,:2].ravel()].astype(np.float32)
                golden=reference.step(dict(p_obs=observations.tolist(),cmd=command.tolist()))
                last=np.array(golden['raw'],dtype=np.float32)
                target=defaults.copy();target[ids]+=last;target[[19,20,21,26,27,28]]=0.
                out=f.step(dict(time=tick*.02,q=q.tolist(),dq=dq.tolist(),root=root.tolist(),gyro=gyro.tolist()))
                np.testing.assert_allclose(out['observations'],observations,atol=3e-5,rtol=2e-5)
                np.testing.assert_allclose(out['getup_command'],command,atol=2e-6)
                np.testing.assert_allclose(out['command'],target,atol=5e-5,rtol=3e-5)
                bounds=np.array([old_limits[n] for n in f.constants['names']])
                largest_old_clip=max(largest_old_clip,float(np.max(abs(target-np.clip(target,bounds[:,0],bounds[:,1])))))
        self.assertGreater(largest_old_clip,.1,'Reference tape must exercise the removed GMR command clamp')
        print(f'\nGetUp audit: prior GMR clamp changed targets by up to {largest_old_clip:.3f} rad',flush=True)

    def test_getup_then_walk_stabilizes_both_fallen_directions_in_mujoco(self):
        stand=yaml.safe_load((ROOT/'deploy/g1/config/config.yaml').read_text())['FSM']['FixStand']
        for take,index,frames in (('t061',212,134),('t067',288,171)):
            with self.subTest(take=take):
                f=Fixture('getup');model=mj.MjModel.from_xml_path(str(policies.SCENE));data=mj.MjData(model)
                joints=np.array([mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n) for n in f.constants['names']]);qa=model.jnt_qposadr[joints];va=model.jnt_dofadr[joints]
                actuators=np.array([np.flatnonzero(model.actuator_trnid[:,0]==j)[0] for j in joints])
                q=np.array(stand['qs'][1]);data.qpos[qa]=q
                rows=np.loadtxt(ROOT/f'deploy/g1/policies/getup/aux/{take}.csv',delimiter=',')
                data.qpos[3:7]=rows[index,67:71];data.qpos[3:7]/=np.linalg.norm(data.qpos[3:7]);mj.mj_forward(model,data)
                geoms=[i for i in range(model.ngeom) if model.geom_bodyid[i]!=0 and model.geom_contype[i]]
                low=min(data.geom_xpos[i,2]+data.geom_xmat[i].reshape(3,3)[2]@model.geom_aabb[i,:3]
                        -abs(data.geom_xmat[i].reshape(3,3)[2])@model.geom_aabb[i,3:] for i in geoms)
                data.qpos[2]-=low-.002;mj.mj_forward(model,data)
                for _ in range(500):
                    data.ctrl[:]=0.;data.ctrl[actuators]=np.array(stand['kp'])*(q-data.qpos[qa])-np.array(stand['kd'])*data.qvel[va];mj.mj_step(model,data)
                self.assertLess(data.qpos[2],.2)
                try:
                    for tick in range(frames+150):
                        if tick==frames:
                            self.assertTrue(out['finished'])
                            f.close();f=Fixture('walk') # same reset on the FSM's GetUp completion event
                        state=dict(time=tick*.02,q=data.qpos[qa].tolist(),dq=data.qvel[va].tolist(),root=data.qpos[3:7].tolist(),gyro=data.qvel[3:6].tolist())
                        out=f.step(state);command,kp,kd=(np.array(out[k]) for k in ('command','kp','kd'))
                        for _ in range(10):
                            data.ctrl[:]=0.;data.ctrl[actuators]=kp*(command-data.qpos[qa])-kd*data.qvel[va];mj.mj_step(model,data)
                        self.assertTrue(np.isfinite(data.qpos).all())
                    quat=data.qpos[3:7];tilt=float(np.arccos(np.clip(1-2*(quat[1]**2+quat[2]**2),-1,1)))
                    self.assertGreater(data.qpos[2],.7);self.assertLess(tilt,.2)
                    print(f'\n{take} → Walk: height {data.qpos[2]:.2f} m, tilt {np.rad2deg(tilt):.1f} deg',flush=True)
                finally:f.close()

    def test_both_gravity_branches_use_original_latents_and_finish(self):
        source=ROOT/'deploy/g1/policies/getup'
        cfg=yaml.safe_load((source/'source_config.yaml').read_text())['g1']['punch']
        for angle,branch in ((-np.pi/2,'front_get_up'),(np.pi/2,'back_get_up')):
            with self.subTest(branch=branch):
                f=Fixture('getup');self.addCleanup(f.close)
                take=cfg['trajs'][cfg[branch]];rows=np.loadtxt(source/'aux'/f"{take['file']}.csv",delimiter=',')
                begin=int(np.floor(take['start']*50+.5));end=int(np.floor(take['end']*50+.5))+1;rows=rows[begin:end]
                state=dict(q=f.constants['standing'],dq=[0.]*29,root=[float(np.cos(angle/2)),0.,float(np.sin(angle/2)),0.],gyro=[0.]*3)
                for tick,row in enumerate(rows):
                    state['time']=tick*.02;out=f.step(state)
                    self.assertEqual(out['front'],branch=='front_get_up');self.assertEqual(out['frames'],len(rows));self.assertEqual(out['frame'],tick+1)
                    self.assertEqual(out['finished'],tick==len(rows)-1)
                    self.assertEqual(len(out['observations']),750);self.assertEqual(len(out['getup_command']),73)
                    np.testing.assert_allclose(out['getup_command'][:64],row[:64],atol=1e-7)
                    np.testing.assert_array_equal(out['getup_command'][64:67],[0.,0.,0.])
                    self.assertTrue(np.isfinite(out['command']).all())
                    for wrist in (19,20,21,26,27,28):self.assertEqual(out['command'][wrist],0.)
                self.assertIn(len(rows),(134,171))

if __name__=='__main__':unittest.main()
