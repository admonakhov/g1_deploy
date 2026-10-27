"""Real ONNX planner commands; speed limits are not physical speed guarantees."""
import unittest
import numpy as np
import yaml
from test_policies import Fixture, BINARY, ROOT

@unittest.skipUnless(BINARY.is_file(), 'Build deployment fixtures')
class LocomotionModesTests(unittest.TestCase):
    @staticmethod
    def run_limits():
        return yaml.safe_load((ROOT/"deploy/g1/config/config.yaml").read_text())["FSM"]["Run"]["velocity_limits"]
    def exercise(self,mode,lx,ly,ticks=550):
        f=Fixture(mode);self.addCleanup(f.close)
        state=dict(q=f.constants['standing'],dq=[0.]*29,root=[1.,0.,0.,0.],gyro=[0.]*3,lx=lx,ly=ly)
        out=None
        for tick in range(ticks):
            state['time']=tick*.02;out=f.step(state)
            target=np.array(out['target_velocity']);planned=np.array(out['planned_velocity'])
            forward=1.5 if mode=='walk' else self.run_limits()['forward']
            for velocity in (target,planned):
                self.assertLessEqual(velocity[0],forward+1e-5);self.assertGreaterEqual(velocity[0],-.5-1e-5)
                self.assertLessEqual(abs(velocity[1]),1.+1e-5)
                if mode=='run':self.assertLessEqual(np.linalg.norm(velocity[:2]),self.run_limits()['planar']+1e-5)
        return out
    def test_walk_forward_backward_and_lateral_limits(self):
        for lx,ly,expected in ((0.,1.,[1.5,0.]),(0.,-1.,[-.5,0.]),(1.,0.,[0.,-1.]),(-1.,0.,[0.,1.]),(-1.,1.,[1.5,1.])):
            with self.subTest(lx=lx,ly=ly):
                out=self.exercise('walk',lx,ly,160)
                np.testing.assert_allclose(out['target_velocity'][:2],expected,atol=1e-5)
                np.testing.assert_allclose(out['planned_velocity'][:2],expected,atol=1e-5)
    def test_run_reaches_configured_speed_and_uses_appropriate_path_clip(self):
        out=self.exercise('run',0.,1.)
        np.testing.assert_allclose(out['planned_velocity'][:2],[self.run_limits()['forward'],0.],atol=1e-5)
        self.assertEqual(out['path_speed_limit'],max(5.,self.run_limits()['planar']))
        if self.run_limits()['forward']>5.:
            self.assertGreater(abs(out['path_command'][64+9*9]),5.*.92+.1)
    def test_run_diagonal_obeys_configured_planar_limit(self):
        out=self.exercise('run',-1.,1.)
        self.assertAlmostEqual(np.linalg.norm(out['planned_velocity'][:2]),self.run_limits()['planar'],places=5)
    def test_run_backward_and_sideways_keep_walk_limits(self):
        for lx,ly,expected in ((0.,-1.,[-.5,0.]),(1.,0.,[0.,-1.])):
            out=self.exercise('run',lx,ly,160)
            np.testing.assert_allclose(out['planned_velocity'][:2],expected,atol=1e-5)

if __name__=='__main__':unittest.main()
