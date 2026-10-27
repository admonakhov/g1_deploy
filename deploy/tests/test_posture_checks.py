"""Any-orientation joint FixStand and per-policy posture opt-out, without DDS."""
import json
import subprocess
import unittest

import numpy as np
import yaml

from test_policies import BINARY, ROOT, Fixture


@unittest.skipUnless(BINARY.is_file(), 'Build deployment fixtures')
class PostureTests(unittest.TestCase):
    @staticmethod
    def state(angle=81.):
        a = np.deg2rad(angle) / 2.
        return dict(time=0., q=[0.]*29, dq=[0.]*29,
                    root=[float(np.cos(a)), 0., float(np.sin(a)), 0.], gyro=[0.]*3)

    def run_policy(self, mode, unchecked, state):
        args = [str(BINARY), mode] + (['--no-posture-check'] if unchecked else ['--posture-check'])
        return subprocess.run(args, input=json.dumps(state)+'\n', capture_output=True,
                              text=True, cwd=ROOT, timeout=10)

    def test_fixstand_from_fallen_pose_remains_active_and_moves_smoothly(self):
        f = Fixture('fixstand')
        self.addCleanup(f.close)
        state = self.state()
        before = np.array(state['q'])
        for tick in range(201):
            state['time'] = tick*.02
            out = f.step(state)
            command = np.array(out['command'])
            self.assertEqual(out['backend'], 'pd')
            self.assertLessEqual(float(np.max(abs(command-before))), .06000001)
            if tick == 0:
                np.testing.assert_array_equal(command, before)
            # Joint servo following on a supported robot; no get-up assumption.
            state['q'] = command.tolist()
            before = command
        cfg = yaml.safe_load((ROOT/'deploy/g1/config/config.yaml').read_text())['FSM']['FixStand']
        np.testing.assert_allclose(command, cfg['qs'][1], atol=1e-8)
        self.assertTrue(out['finished'])
        # No auto WBC takeover when the body is made upright.
        state.update(root=[1.,0.,0.,0.],time=5.)
        self.assertEqual(f.step(state)['backend'],'pd')

    def test_lying_policy_inference_requires_explicit_opt_out(self):
        for mode in ('walk', 'wbc', 'arms'):
            with self.subTest(mode=mode):
                checked = self.run_policy(mode, False, self.state())
                self.assertEqual(checked.returncode, 1)
                self.assertIn('G1 tilt 81.0', checked.stderr)
                unchecked = self.run_policy(mode, True, self.state())
                self.assertEqual(unchecked.returncode, 0, unchecked.stderr)
                out = json.loads(unchecked.stdout.splitlines()[-1])
                self.assertTrue(np.isfinite(out['command']).all())

    def test_invalid_measurements_still_rejected_with_posture_check_disabled(self):
        invalid = [dict(root=[0.]*4), dict(dq=[36.]*29)]
        for mode in ('fixstand', 'walk', 'wbc', 'arms'):
            for values in invalid:
                with self.subTest(mode=mode, values=values):
                    state = self.state()
                    state.update(values)
                    result = self.run_policy(mode, True, state)
                    self.assertEqual(result.returncode, 1)
                    self.assertIn('Invalid G1', result.stderr)


if __name__ == '__main__':
    unittest.main()
