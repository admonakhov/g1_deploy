"""Real imported ONNX bundles, closed-loop MuJoCo; no DDS or robot commands."""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

import mujoco as mj
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
BINARY=next((ROOT/p for p in ('deploy/g1/build/g1_policy_fixture','.cache/g1-deploy/g1_policy_fixture') if (ROOT/p).is_file()),ROOT/'deploy/g1/build/g1_policy_fixture')
SCENE=ROOT/'deploy/g1/sim/scene.xml'

class Fixture:
    def __init__(self,mode):
        self.process=subprocess.Popen([str(BINARY),mode],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.constants=json.loads(self.process.stdout.readline())
    def step(self,state):
        self.process.stdin.write(json.dumps(state)+'\n');self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise AssertionError('Native fixture stopped: '+self.process.stderr.read())
        return json.loads(line)
    def close(self):
        self.process.stdin.close()
        try:self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        code,error=self.process.returncode,self.process.stderr.read();self.process.stdout.close();self.process.stderr.close()
        if code:raise AssertionError(error)

@unittest.skipUnless(BINARY.is_file(),'Build the deployment fixtures')
class PolicyTests(unittest.TestCase):
    def fixture(self,mode):
        f=Fixture(mode);self.addCleanup(f.close);return f
    @staticmethod
    def state(q,time=0.):return dict(time=time,q=list(q),dq=[0.]*29,root=[1.,0.,0.,0.],gyro=[0.,0.,0.])
    def test_only_requested_exact_source_bundles(self):
        directory=ROOT/'deploy/g1/policies';manifest=json.loads((directory/'SOURCE.json').read_text())
        self.assertEqual({p.name for p in directory.iterdir() if p.is_dir()},{'traj_loco','wbc','getup'})
        for relative,digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((directory/relative).read_bytes()).hexdigest(),digest,relative)
        self.assertEqual(len(list(directory.rglob('*.onnx'))),5) # walking + two auxiliaries + WBC + GetUp
    def test_vq_observation_order_history_and_path_contract(self):
        f=self.fixture('walk');q=np.array(f.constants['standing']);state=self.state(q);state.update(gyro=[.1,-.2,.3],dq=[.2]*29)
        out=f.step(state);obs=np.array(out['observations']).reshape(10,75)
        expected=np.r_[[.02,-.04,.06],[0.,0.,-1.],np.zeros(23),np.full(23,.01),np.zeros(23)]
        np.testing.assert_allclose(obs,np.tile(expected,(10,1)),atol=1e-7)
        self.assertTrue(out['adjusting'])
        cmd=np.array(out['path_command']);self.assertEqual(cmd.shape,(163,));np.testing.assert_array_equal(cmd[64:73],cmd[154:])
        cfg=yaml.safe_load((ROOT/'deploy/g1/policies/traj_loco/config.yaml').read_text())
        np.testing.assert_allclose(cmd[:64],cfg['g1']['vq']['anchor'],atol=1e-7)
        for i in (19,20,21,26,27,28):self.assertEqual(out['command'][i],0.)
        state.update(time=.02,gyro=[0.,0.,0.]);second=np.array(f.step(state)['observations']).reshape(10,75)
        np.testing.assert_array_equal(second[:-1],obs[1:]);np.testing.assert_allclose(second[-1,:3],0.)
    def test_idle_ignores_sticks_without_changing_policy(self):
        f=self.fixture('walk');state=self.state(f.constants['standing']);state.update(idle=True,lx=1.,ly=1.,rx=1.)
        out=f.step(state);cmd=np.array(out['path_command'])[64:154].reshape(10,9)
        np.testing.assert_allclose(cmd[:,:3],0.,atol=1e-8)
    def test_wbc_exact_observation_and_reference_residual_contract(self):
        f=self.fixture('wbc');q=np.array(f.constants['standing']);state=self.state(q);state.update(gyro=[.1,.2,.3],dq=[.02]*29)
        out=f.step(state);obs=np.array(out['observations']);self.assertEqual(obs.shape,(132,))
        np.testing.assert_allclose(obs[10:39],q,atol=1e-7);np.testing.assert_allclose(obs[39:42],[.1,.2,.3],atol=1e-7)
        np.testing.assert_array_equal(obs[42:45],[0.,0.,-1.]);np.testing.assert_allclose(obs[45:74],0.,atol=1e-8)
        np.testing.assert_allclose(obs[74:103],.02,atol=1e-7);np.testing.assert_array_equal(obs[103:],0.)
        scale=np.array(yaml.safe_load((ROOT/'deploy/g1/policies/wbc/params/config.yaml').read_text())['action']['scale'])
        raw=(np.array(out['command'])-q)/scale
        state['time']=.02;second=f.step(state)
        np.testing.assert_allclose(second['observations'][103:],raw,atol=2e-6)
    def test_arms_and_missing_legs_cannot_become_camera_goals(self):
        for mode in ('arms','wbc'):
            with self.subTest(mode=mode):
                f=self.fixture(mode);q=np.array(f.constants['standing']);state=self.state(q);f.step(state)
                target=q.copy();target[:15]+=.4;target[[15,22]]=-.5
                state.update(time=3.,reference=target.tolist())
                if mode=='wbc':state['active']=[False]*15+[True]*14
                out=f.step(state);np.testing.assert_allclose(out['reference'][:15],q[:15],atol=1e-8)
    def test_waiting_ignores_unusable_camera_goals_and_resumes_smoothly(self):
        for mode in ('wbc','arms'):
            with self.subTest(mode=mode):
                f=self.fixture(mode);q=np.array(f.constants['standing']);previous=q.copy()
                pose=q.copy();pose[[15,22]]=-.6;pose[[18,25]]=1.2
                def packet(tick,session='first'):
                    return dict(schema='gmr-g1-v1',robot='unitree_g1',status='tracking',person_id=1,
                                stream_id=session,joint_units='radians',joint_names=f.constants['names'],
                                joint_positions=pose.tolist(),active_joint_mask=[True]*29,
                                base_tracked=False,latency_s=0.,frame=tick,reference_monotonic_s=tick*.02)
                for tick in range(700):
                    state=self.state(q,tick*.02);state['camera_stream']=True
                    if 200<=tick<300 or 400<=tick<450:state['camera_packets']=[packet(tick)]
                    if tick==450:state['camera_packets']=[dict(packet(tick),status='stopped')]
                    if tick>=550:state['camera_packets']=[packet(tick-550,'restarted')]
                    out=f.step(state);reference=np.array(out['reference'])
                    self.assertEqual(out['backend'],'TeleopArms' if mode=='arms' else 'Teleop')
                    self.assertTrue(np.isfinite(out['command']).all())
                    self.assertTrue((np.array(out['kp'])>0.).all())
                    self.assertLessEqual(np.max(np.abs(reference-previous)),.100001)
                    previous=reference
                    if tick<201 or 350<=tick<401 or 450<=tick<551:self.assertTrue(out['camera_waiting'],tick)
                    if 201<=tick<350 or 401<=tick<450 or tick>=551:self.assertFalse(out['camera_waiting'],tick)
                    if tick in (199,399,549):np.testing.assert_allclose(reference,q,atol=1e-4)
                    if tick in (299,449,699):np.testing.assert_allclose(reference[[15,22]],pose[[15,22]],atol=1e-4)
                    self.assertEqual(out['camera_diagnostics']['rejected_packets'],0)
                self.assertEqual(out['camera_diagnostics']['last_valid_frame'],149)
    def test_invalid_reference_cannot_override_waiting_stance(self):
        f=self.fixture('wbc');q=np.array(f.constants['standing'])
        state=self.state(q);state.update(camera_stream=True);f.step(state)
        state.update(time=30.,camera_packets=[dict(schema='wrong')])
        out=f.step(state)
        self.assertTrue(out['camera_waiting']);np.testing.assert_array_equal(out['reference'],q)
        self.assertEqual(out['camera_diagnostics']['rejected_packets'],1)
    def rollout(self,mode,walking=False,arms=False,loss=False,from_walk=False,failover=False,camera_wait=False):
        f=self.fixture(mode);m=mj.MjModel.from_xml_path(str(SCENE));d=mj.MjData(m)
        joints=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n) for n in f.constants['names']]);qa=m.jnt_qposadr[joints];va=m.jnt_dofadr[joints]
        actuators=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in joints]);q=np.array(f.constants['standing'])
        initial=yaml.safe_load((ROOT/'deploy/g1/policies/traj_loco/config.yaml').read_text())['g1']['default_dof_pos'] if from_walk else q
        d.qpos[qa]=initial;mj.mj_forward(m,d)
        feet=[i for i in range(m.ngeom) if 'ankle_roll' in (mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[i])) or '') and m.geom_contype[i]]
        low=min(d.geom_xpos[i,2]+d.geom_xmat[i].reshape(3,3)[2]@m.geom_aabb[i,:3]-abs(d.geom_xmat[i].reshape(3,3)[2])@m.geom_aabb[i,3:] for i in feet)
        d.qpos[2]-=low-.002;mj.mj_forward(m,d);kp,kd=np.array(f.constants['kp']),np.array(f.constants['kd']);positions=[];tilts=[]
        for tick in range(700):
            state=dict(time=tick*.02,q=d.qpos[qa].tolist(),dq=d.qvel[va].tolist(),root=d.qpos[3:7].tolist(),gyro=d.qvel[3:6].tolist(),ly=.2 if walking and 150<=tick<450 else 0.)
            if arms and tick>=150:
                target=q.copy();target[[15,22]]=-.6;target[[18,25]]=1.2
                if mode=='arms':target[:15]+=.3
                state['reference']=target.tolist()
                if loss and 300<=tick<400:state['active']=[False]*29
            if failover and tick>=350:
                if tick==350:state['teleop_error']=True
                state['ly']=1. if tick<450 else .2 if 475<=tick<600 else 0.
            if camera_wait:
                state['camera_stream']=True
                if 200<=tick<300 or 400<=tick<450 or tick>=550:
                    target=q.copy();target[[15,22]]=-.6;target[[18,25]]=1.2
                    state['camera_packets']=[dict(schema='gmr-g1-v1',robot='unitree_g1',status='tracking',person_id=1,
                        stream_id='restarted' if tick>=550 else 'first',joint_units='radians',joint_names=f.constants['names'],
                        joint_positions=target.tolist(),active_joint_mask=[True]*29,base_tracked=False,latency_s=0.,
                        frame=tick-550 if tick>=550 else tick,reference_monotonic_s=(tick-550 if tick>=550 else tick)*.02)]
                if tick==450:state['camera_packets']=[dict(schema='gmr-g1-v1',robot='unitree_g1',status='stopped',stream_id='first')]
            out=f.step(state);command=np.array(out['command']);self.assertTrue(np.isfinite(command).all())
            if camera_wait:
                self.assertEqual(out['backend'],'TeleopArms' if mode=='arms' else 'Teleop')
                self.assertTrue((np.array(out['kp'])>0.).all())
                if tick in (199,399,549):self.assertTrue(out['camera_waiting'])
                if tick in (299,449,699):self.assertFalse(out['camera_waiting'])
            if failover and tick>=350:
                self.assertEqual(out['backend'],'Walk');self.assertTrue((np.array(out['kp'])>0.).all())
                if tick<450:np.testing.assert_array_equal(out['target_velocity'],[0.,0.,0.]);self.assertTrue(out['recovery_waiting'])
                if tick==474:self.assertFalse(out['recovery_waiting'])
            kp,kd=np.array(out['kp']),np.array(out['kd'])
            for _ in range(10):
                d.ctrl[:]=0.;d.ctrl[actuators]=kp*(command-d.qpos[qa])-kd*d.qvel[va];mj.mj_step(m,d)
            quat=d.qpos[3:7];tilt=np.arccos(np.clip(1-2*(quat[1]**2+quat[2]**2),-1,1));tilts.append(tilt);positions.append(float(d.qpos[0]))
            self.assertLess(tilt,.45,f'{mode} tilt at {tick*.02:.2f}s');self.assertGreater(d.qpos[2],.6)
        if failover:
            self.assertLess(abs(positions[449]-positions[349]),.25,'Held stick caused movement on recovery')
            self.assertGreater(positions[599]-positions[474],.5,'Walking did not resume after centering sticks')
        elif walking:self.assertGreater(positions[449]-positions[149],1.)
        else:self.assertLess(abs(positions[-1]-positions[100]),.25)
        self.assertLess(abs(positions[-1]-positions[-101]),.15,'Did not settle after stopping')
        print(f'\n{mode}: peak tilt {np.rad2deg(max(tilts)):.1f} deg, displacement {positions[-1]:.2f} m',flush=True)
    def test_vq_idle_balance(self):self.rollout('walk')
    def test_vq_walk_and_stop(self):self.rollout('walk',walking=True)
    def test_wbc_forward_arms_balance(self):self.rollout('wbc',arms=True)
    def test_wbc_smooth_entry_from_walking_stance(self):self.rollout('wbc',arms=True,from_walk=True)
    def test_wbc_arms_body_is_independent_and_tracking_recovers(self):self.rollout('arms',arms=True,loss=True)
    def test_wholebody_keeps_balance_while_waiting_and_reconnecting(self):self.rollout('wbc',arms=True,camera_wait=True)
    def test_arms_keeps_balance_while_waiting_and_reconnecting(self):self.rollout('arms',arms=True,camera_wait=True)
    def test_arms_failure_enters_balanced_walk_and_rearms_sticks(self):self.rollout('arms',arms=True,failover=True)
    def test_wholebody_failure_enters_balanced_walk_and_rearms_sticks(self):self.rollout('wbc',arms=True,failover=True)
    def test_teleop_start_failure_uses_walk_without_passive_command(self):
        f=self.fixture('arms');state=self.state(f.constants['standing']);state.update(teleop_error=True,ly=1.)
        out=f.step(state);self.assertEqual(out['backend'],'Walk');self.assertTrue((np.array(out['kp'])>0.).all())
        np.testing.assert_array_equal(out['target_velocity'],[0.,0.,0.]);self.assertTrue(out['recovery_waiting'])

if __name__=='__main__':unittest.main()
