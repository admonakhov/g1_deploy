"""A fresh controller configuration must use only local support, not submodules."""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]

@unittest.skipUnless(shutil.which('cmake'),'CMake required')
class StandaloneTests(unittest.TestCase):
    def test_fresh_controller_config_without_removed_submodules(self):
        for name in ('unitree_rl_lab','GR00T-WholeBodyControl'):
            self.assertFalse((ROOT/name).exists())
            self.assertNotIn(f'path = {name}',(ROOT/'.gitmodules').read_text())
        with tempfile.TemporaryDirectory() as directory:
            result=subprocess.run(['cmake','-S',str(ROOT/'deploy/g1'),'-B',directory,
                '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DBUILD_TESTING=OFF'],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            commands=json.loads((Path(directory)/'compile_commands.json').read_text())
            controller=[row for row in commands if row['file'].endswith('/src/main.cpp')]
            self.assertEqual(len(controller),1)
            for row in commands:
                self.assertNotIn('/unitree_rl_lab/',row['command'])
                self.assertNotIn('/GR00T-WholeBodyControl/',row['command'])
            self.assertIn('/third_party/unitree_deploy/include',controller[0]['command'])

if __name__=='__main__':unittest.main()
