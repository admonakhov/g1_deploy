"""Real 50 Hz camera publisher -> native UDP interpolation, without DDS."""
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest
import sys
import signal
import yaml
from video_teleop.teleop_stream import ContinuousPosePublisher
from video_teleop.robot_retarget import UDPPublisher

ROOT=Path(__file__).resolve().parents[2]
BINARY=ROOT/'.cache/g1-deploy/g1_stream_preview'

@unittest.skipUnless(BINARY.is_file(),'Build g1_stream_preview')
class CameraPipelineTests(unittest.TestCase):
    def test_missing_and_empty_video_keep_native_reference_ready_without_loading_models(self):
        for empty in (False,True):
            with self.subTest(empty=empty),tempfile.TemporaryDirectory() as directory:
                path=Path(directory);source=path/'missing.mp4'
                if empty:source.write_bytes(b'not a playable video')
                with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as reserve:
                    reserve.bind(('127.0.0.1',0));port=reserve.getsockname()[1]
                output=path/'reference.jsonl'
                native=subprocess.Popen([str(BINARY),'--port',str(port),'--seconds','2.3','--output',str(output)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                camera=subprocess.Popen([sys.executable,'-m','video_teleop.pose_stream','--source',str(source),
                    '--retarget','--udp',f'127.0.0.1:{port}','--no-display','--model',str(path/'not-loaded.pth')],
                    cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                try:
                    stdout,error=native.communicate(timeout=6)
                    self.assertEqual(native.returncode,0,stdout+error)
                    self.assertIsNone(camera.poll(),'Input failure terminated the teleop process')
                    rows=[json.loads(line) for line in output.read_text().splitlines()]
                    ready=[r for r in rows if r['diagnostics']['last_status']=='fallback']
                    self.assertGreater(len(ready),50)
                    self.assertLess(ready[-1]['diagnostics']['last_valid_pose_age_s'],.1)
                    self.assertEqual(ready[-1]['diagnostics']['rejected_packets'],0)
                    expected=json.loads((ROOT/'deploy/g1/config/initial_pose.json').read_text())
                    names=yaml.safe_load((ROOT/'deploy/g1/policies/wbc/params/config.yaml').read_text())['joint_names']
                    self.assertEqual(ready[-1]['q'],[expected[n] for n in names])
                finally:
                    camera.send_signal(signal.SIGINT)
                    try:camera.communicate(timeout=5)
                    except subprocess.TimeoutExpired:camera.kill();camera.communicate(timeout=3)
                    if native.poll() is None:native.kill();native.communicate(timeout=3)

    def test_low_fps_reference_is_interpolated_and_loss_sends_wbc_stance(self):
        angles=json.loads((ROOT/'deploy/g1/config/initial_pose.json').read_text());names=list(angles);q=list(angles.values())
        initial=dict(schema='gmr-g1-v1',robot='unitree_g1',status='fallback',person_id=None,joint_names=names,joint_positions=q,
                     joint_units='radians',active_joint_mask=[False]*29,reference_joint_mask=[True]*29,
                     reference_pose='g1_standing',base_tracked=False,base_quaternion_order='wxyz',qpos=None)
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as reserve:
            reserve.bind(('127.0.0.1',0));port=reserve.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'reference.jsonl'
            process=subprocess.Popen([str(BINARY),'--port',str(port),'--seconds','2.3','--output',str(output)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            publisher=ContinuousPosePublisher(UDPPublisher(f'127.0.0.1:{port}'),initial,'pipeline',pose_timeout=.75,loss_grace=.1)
            try:
                publisher.start();time.sleep(.1)
                for i in range(5):
                    pose=dict(initial);pose.update(status='tracking',person_id=1,active_joint_mask=[True]*29,
                        joint_positions=q.copy(),capture_monotonic_s=time.monotonic(),latency_s=0.,processing_completed_monotonic_s=time.monotonic())
                    pose['joint_positions'][15]=-.5 if i%2 else .2
                    publisher.publish(pose);time.sleep(.18)
                publisher.publish(dict(initial,status='no_pose',processing_completed_monotonic_s=time.monotonic()))
                stdout,error=process.communicate(timeout=5)
                self.assertEqual(process.returncode,0,stdout+error)
                rows=[json.loads(line) for line in output.read_text().splitlines()]
                self.assertGreater(len(rows),60)
                moving=[r for r in rows if r['diagnostics']['last_status']=='tracking']
                self.assertTrue(any(.05<r['alpha']<.95 for r in moving))
                self.assertTrue(any(-.45<r['q'][15]<.15 for r in moving))
                self.assertEqual(rows[-1]['diagnostics']['last_status'],'fallback')
                self.assertGreater(rows[-1]['diagnostics']['reference_holds'],0)
                self.assertEqual(rows[-1]['diagnostics']['rejected_packets'],0)
            finally:
                publisher.close()
                if process.poll() is None:process.kill();process.communicate(timeout=3)

if __name__=='__main__':unittest.main()
