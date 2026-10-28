"""Unitree FSM validation, without creating DDS channels."""
from pathlib import Path
import json
import subprocess
import tempfile
import unittest
import yaml
ROOT=Path(__file__).resolve().parents[2]
BINARY=next((ROOT/p for p in ('deploy/g1/build/g1_ctrl','.cache/g1-deploy/g1_ctrl') if (ROOT/p).is_file()),ROOT/'deploy/g1/build/g1_ctrl')
CONFIG=ROOT/'deploy/g1/config/config.yaml'
DEPLOY=ROOT/'deploy/g1/config/deploy.yaml'
STATES={'Passive','FixStand','Idle','Walk','Run','GetUp','Teleop','TeleopArms'}

@unittest.skipUnless(BINARY.is_file(),'Build g1_ctrl')
class ConfigTests(unittest.TestCase):
    def check(self,*args,cwd=None):return subprocess.run([str(BINARY),'--check-config',*args],cwd=cwd or ROOT,capture_output=True,text=True,timeout=15)
    def custom(self,config=None,deploy=None):
        with tempfile.TemporaryDirectory() as directory:
            c=config if config is not None else yaml.safe_load(CONFIG.read_text())
            for name in ('Idle','Walk','Run','GetUp','Teleop','TeleopArms'):
                c['FSM'][name]['policy_dir']=str(ROOT/'deploy/g1/policies'/('traj_loco' if name in ('Idle','Walk','Run') else 'getup' if name=='GetUp' else 'wbc'))
            p=Path(directory)/'config.yaml';p.write_text(yaml.safe_dump(c,sort_keys=False));d=Path(directory)/'deploy.yaml'
            runtime=deploy if deploy is not None else yaml.safe_load(DEPLOY.read_text())
            runtime['robot_defaults_yaml']=str(ROOT/'deploy/g1/config/robot_defaults.yaml');d.write_text(yaml.safe_dump(runtime))
            return self.check('--config',str(p),'--deploy',str(d))
    def test_three_bundles_and_eight_states(self):
        with tempfile.TemporaryDirectory() as directory:r=self.check(cwd=directory)
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual({line.split(' id=')[0] for line in r.stdout.splitlines() if ' id=' in line},STATES)
        self.assertIn('no DDS initialized',r.stdout)
    def test_removed_policy_types_cannot_be_enabled(self):
        for kind in ('Teleop','Locomotion','SmplTeleop','Mimic','RLBase'):
            c=yaml.safe_load(CONFIG.read_text());c['FSM']['_']['OldPolicy']={'id':999,'type':kind};c['FSM']['OldPolicy']={'transitions':{}}
            r=self.custom(c);self.assertEqual(r.returncode,1);self.assertIn('unsupported policy',r.stderr)
    def test_invalid_frequency_and_endpoint_fail_before_dds(self):
        for field,value in (('policy_hz',30),('step_dt',.03),('udp_port',0),('udp_host','bad-host'),('stream_timeout_s',0.),('wbc_base_height',float('nan'))):
            with self.subTest(field=field):
                d=yaml.safe_load(DEPLOY.read_text());d[field]=value;r=self.custom(deploy=d)
                self.assertEqual(r.returncode,1,r.stdout+r.stderr);self.assertNotIn('Waiting for G1',r.stdout)
    def test_state_cannot_override_training_frequency(self):
        c=yaml.safe_load(CONFIG.read_text());c['FSM']['Walk']['policy_hz']=25;r=self.custom(c)
        self.assertEqual(r.returncode,1);self.assertIn('trained policy frequency',r.stderr)
    def test_invalid_camera_scope_is_rejected(self):
        c=yaml.safe_load(CONFIG.read_text());c['FSM']['TeleopArms']['retarget_scope']='legs';r=self.custom(c)
        self.assertEqual(r.returncode,1);self.assertIn('retarget_scope',r.stderr)
    def test_fixstand_waypoints_and_auto_transition_validated(self):
        for field,value in (('ts',[0.,0.]),('qs',[[],[1.]]),('auto_transition','Absent')):
            with self.subTest(field=field):
                c=yaml.safe_load(CONFIG.read_text());c['FSM']['FixStand'][field]=value;r=self.custom(c)
                self.assertEqual(r.returncode,1);self.assertIn('FixStand',r.stderr)
    def test_orientation_checks_match_ros2_policy_configuration(self):
        c=yaml.safe_load(CONFIG.read_text())
        self.assertNotIn('enable_bad_orientation_passive',c['FSM']['FixStand'])
        for name in ('Idle','Walk','Run','GetUp','Teleop','TeleopArms'):
            self.assertNotIn('entry_check',c['FSM'][name]);c['FSM'][name]['enable_bad_orientation_passive']=False
        r=self.custom(c);self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        c['FSM']['Walk']['enable_bad_orientation_passive']='invalid';r=self.custom(c);self.assertEqual(r.returncode,1)
    def test_invalid_runtime_initial_state_orientation_and_pd_limits(self):
        for field,value in (('initial_state','Missing'),('bad_orientation_limit_rad',0.),('max_sensor_age_s',0.),('pd_torque_clipping',{'enabled':True,'fraction':2.,'joint_max_torque':[1.]*29})):
            d=yaml.safe_load(DEPLOY.read_text());d[field]=value;r=self.custom(deploy=d);self.assertEqual(r.returncode,1)
    def test_obsolete_custom_entry_gates_fail_explicitly(self):
        c=yaml.safe_load(CONFIG.read_text());c['FSM']['Walk']['check_posture']=False
        r=self.custom(c);self.assertEqual(r.returncode,1);self.assertIn('enable_bad_orientation_passive',r.stderr)
    def test_removed_wbc_reference_processing_fails_explicitly(self):
        for field in ('transition_duration_s','reference_smoothing_s','reference_max_speed','max_joint_speed'):
            with self.subTest(field=field):
                d=yaml.safe_load(DEPLOY.read_text());d[field]=.1
                r=self.custom(deploy=d);self.assertEqual(r.returncode,1);self.assertIn('Removed WBC',r.stderr)
                c=yaml.safe_load(CONFIG.read_text());c['FSM']['Teleop'][field]=.1
                r=self.custom(config=c);self.assertEqual(r.returncode,1);self.assertIn('Removed WBC',r.stderr)

    def test_camera_failure_routes_to_walk(self):
        for state in ('Teleop','TeleopArms'):
            c=yaml.safe_load(CONFIG.read_text());self.assertEqual(c['FSM'][state]['failure_state'],'Walk')
            c['FSM'][state]['failure_state']='Passive';r=self.custom(c)
            self.assertEqual(r.returncode,1);self.assertIn('failure_state',r.stderr)
    def test_invalid_velocity_limits_and_getup_return_fail_before_dds(self):
        for field,value in (('forward',0.),('backward',-.5),('lateral',float('nan')),('planar',-1.)):
            c=yaml.safe_load(CONFIG.read_text());c['FSM']['Run']['velocity_limits'][field]=value;r=self.custom(c)
            self.assertEqual(r.returncode,1);self.assertIn('velocity_limits',r.stderr)
        c=yaml.safe_load(CONFIG.read_text());c['FSM']['GetUp']['return_state']='Absent';r=self.custom(c)
        self.assertEqual(r.returncode,1);self.assertIn('return_state',r.stderr)
    def test_disabled_transition_and_duplicate_id_rejected(self):
        c=yaml.safe_load(CONFIG.read_text());c['FSM']['Walk']['transitions']['Absent']='RB + X.on_pressed';r=self.custom(c)
        self.assertEqual(r.returncode,1);self.assertIn('disabled state',r.stderr)
        c=yaml.safe_load(CONFIG.read_text());c['FSM']['_']['Teleop']['id']=c['FSM']['_']['Walk']['id'];r=self.custom(c)
        self.assertEqual(r.returncode,1);self.assertIn('FSM ID',r.stderr)
    def test_real_dsl_routes_camera_walk_idle_and_emergency(self):
        binary=ROOT/'.cache/g1-deploy/g1_fsm_keys_fixture'
        self.assertTrue(binary.is_file())
        def route(source,*keys):
            r=subprocess.run([str(binary),source,*keys],capture_output=True,text=True,timeout=5);self.assertEqual(r.returncode,0,r.stderr);return json.loads(r.stdout)
        targets={'Walk':('RB','X'),'Run':('RB','Y'),'Idle':('LB','up'),'Teleop':('RT','A'),'TeleopArms':('RT','X')}
        for source in STATES-{'Passive'}:
            for target,keys in targets.items():self.assertEqual(route(source,*keys)['matches'],[] if source==target or (source=='GetUp' and target!='Walk') else [target])
            self.assertEqual(route(source,'LT','RB','B')['target'],'Passive')
            self.assertEqual(route(source,'RB','LB','Y')['matches'],[])
        self.assertEqual(route('Passive','RB','Y')['matches'],[])
        self.assertEqual(route('FixStand','LT','up')['target'],'GetUp')
        self.assertEqual(route('Passive','LT','up')['target'],'FixStand')
        for source in STATES-{'Passive','FixStand'}:
            self.assertEqual(route(source,'LT','up')['target'],'FixStand')
            self.assertEqual(route(source,'RB','RT','X')['matches'],[])
            self.assertEqual(route(source,'RB','RT','Y')['matches'],[])
    def test_supported_binary_locations(self):
        for path in ('deploy/g1/build/g1_ctrl','.cache/g1-deploy/g1_ctrl','.cache/g1-unitree/g1_ctrl'):
            binary=ROOT/path
            if binary.is_file():
                r=subprocess.run([str(binary),'--check-config'],cwd=binary.parent,capture_output=True,text=True,timeout=15)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertIn('TeleopArms id=',r.stdout)

if __name__=='__main__':unittest.main()
