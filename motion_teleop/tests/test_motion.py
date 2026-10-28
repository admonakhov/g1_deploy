import asyncio
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
from aiohttp.test_utils import TestClient, TestServer
from scipy.spatial.transform import Rotation

from motion_teleop.bridge import ReferenceBridge
from motion_teleop.motion import MotionClip, NAMES, boundary_pose, validate_packet
from motion_teleop.server import BRIDGE, JOBS, create_app, parser
from video_teleop.teleop_stream import HTTPPublisher, initial_reference


def clip():
    values = np.zeros((41, 36))
    values[:, 2] = .7842
    values[:, 3] = 1.
    values[:, 7:] = initial_reference()["joint_positions"]
    values[:, 0] = np.arange(41) * .01
    values[:, 22] = np.linspace(.35, -.5, 41)
    return MotionClip(values, 20.)


class MotionTests(unittest.TestCase):
    def test_playback_pause_holds_pose_and_zero_velocity_then_resumes(self):
        bridge = ReferenceBridge()
        bridge.play(clip(), "test", now=10.)
        before = bridge.packet(10.5)
        epoch = bridge.epoch
        bridge.pause(now=10.5)
        paused = bridge.packet(20.)
        later = bridge.packet(21.)
        np.testing.assert_allclose(paused["qpos"], before["qpos"])
        np.testing.assert_allclose(later["qpos"], paused["qpos"])
        self.assertEqual(paused["base_linear_velocity_world"], [0., 0., 0.])
        for name in ("linear_velocity_world", "angular_velocity_world"):
            self.assertEqual(paused["wbc_reference"][name], [0., 0., 0.])
        self.assertFalse(bridge.playing)
        self.assertEqual(bridge.epoch, epoch + 1)
        self.assertEqual(bridge.status()["position_s"], .5)
        bridge.pause(now=22.)
        self.assertEqual(bridge.epoch, epoch + 1)
        bridge.resume(now=30.)
        np.testing.assert_allclose(bridge.packet(30.)["qpos"], paused["qpos"])
        self.assertGreater(bridge.packet(30.1)["qpos"][0], paused["qpos"][0])
        bridge.select("video")
        self.assertIsNone(bridge.paused_at)
        with self.assertRaises(ValueError):bridge.resume()

    def test_named_order_units_quaternion_interpolation_and_velocity(self):
        motion = clip()
        qpos, velocity = motion.sample(.025)
        self.assertAlmostEqual(qpos[0], .005)
        self.assertAlmostEqual(qpos[22], .35 + (-.5 - .35) / 80)
        np.testing.assert_allclose(velocity, [.2, 0., 0.])
        np.testing.assert_array_equal(motion.sample(20.)[1], [0., 0., 0.])
        base = boundary_pose(motion.packet(.5))
        np.testing.assert_allclose(base[:3], [0., 78.42, 10.], atol=1e-4)
        np.testing.assert_allclose(base[6:], np.rad2deg(motion.sample(.5)[0][7:]), atol=1e-4)
        shuffled = motion.qpos.copy()
        shuffled[:, 7:] = shuffled[:, 7:][:, ::-1]
        np.testing.assert_allclose(MotionClip(shuffled, 20., NAMES[::-1]).qpos, motion.qpos)

    def test_heading_normalization_and_slerp_short_path(self):
        values = clip().qpos
        values[:, 3:7] = Rotation.from_euler("z", np.linspace(170., 190., len(values)), degrees=True).as_quat(scalar_first=True)
        motion = MotionClip(values, 20.)
        self.assertAlmostEqual(Rotation.from_quat(motion.sample(1.)[0][3:7], scalar_first=True).as_euler("xyz", degrees=True)[2], 10.)

    def test_reject_invalid_arrays_and_network_packets(self):
        for field in ("nan", "quaternion"):
            values = clip().qpos.copy()
            if field == "nan":values[0, 7] = np.nan
            if field == "quaternion":values[:, 3:7] = 0.
            with self.assertRaises(ValueError):MotionClip(values, 20.)
        packet = dict(initial_reference(), stream_id="video", frame=0, reference_monotonic_s=1.)
        validate_packet(packet)
        packet["joint_names"] = [NAMES[0]] * 29
        with self.assertRaises(ValueError):validate_packet(packet)

    def test_partial_video_null_joints_are_kept_unobserved(self):
        packet = dict(initial_reference(), status="partial_tracking", person_id=1, stream_id="video", frame=0, reference_monotonic_s=1.)
        packet["active_joint_mask"][15] = True
        packet["joint_positions"] = [None] * 29
        packet["joint_positions"][15] = -.4
        validate_packet(packet)
        pose = boundary_pose(packet)
        self.assertTrue(np.isfinite(pose).all())
        self.assertAlmostEqual(pose[21], np.rad2deg(-.4), places=4)
        bridge = ReferenceBridge();bridge.receive(packet, 1.)
        self.assertIsNone(bridge.packet(1.02)["joint_positions"][0])
        self.assertFalse(bridge.last["active_joint_mask"][0])

    def test_bridge_preserves_video_cadence_and_selects_one_source(self):
        bridge = ReferenceBridge()
        packet = dict(initial_reference(), status="tracking", person_id=1, active_joint_mask=[True] * 29,
                      stream_id="camera", frame=10, reference_monotonic_s=1000., latency_s=.1)
        bridge.receive(packet, 1.)
        first, held = bridge.packet(1.), bridge.packet(1.02)
        self.assertEqual(first["frame"], 10)
        self.assertFalse(first["reference_hold"])
        self.assertTrue(held["reference_hold"])
        self.assertEqual(held["reference_monotonic_s"], 1000.)
        self.assertEqual(bridge.packet(2.)["status"], "fallback")
        bridge.play(clip(), "test", 3.)
        a, b = bridge.packet(3.), bridge.packet(3.02)
        self.assertNotEqual(a["frame"], b["frame"])
        self.assertEqual(b["reference_kind"], "motion")
        self.assertEqual(b["reference_source"], "trace")
        self.assertAlmostEqual(b["qpos"][0], .004)
        self.assertTrue(bridge.playing)
        bridge.packet(10.)
        self.assertFalse(bridge.playing)
        bridge.select("idle")
        self.assertEqual(bridge.packet(10.02)["status"], "fallback")


class Transport:
    def __init__(self):self.packets = [];self.closed = False
    def publish(self, packet):self.packets.append((time.monotonic(), packet))
    def close(self):self.closed = True


class WebTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        args = parser().parse_args(["--jobs", self.temporary.name, "--models", self.temporary.name,
                                    "--trace-python", sys.executable])
        self.output = Transport()
        self.app = create_app(args, self.output)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_http_websocket_camera_and_50hz_output(self):
        self.assertEqual((await self.client.get("/")).status, 200)
        ws = await self.client.ws_connect("/ws/poses")
        self.assertEqual((await ws.receive_json())["type"], "status")
        packet = dict(initial_reference(), stream_id="video", frame=1, reference_monotonic_s=time.monotonic())
        response = await self.client.post("/api/pose", json=packet)
        self.assertEqual(response.status, 200)
        await asyncio.sleep(.5)
        self.assertGreaterEqual(len(self.output.packets), 20)
        received = await ws.receive_json()
        self.assertEqual(received["type"], "pose")
        self.assertIn("schema", received["packet"])
        await ws.close()
        duration = self.output.packets[-1][0] - self.output.packets[0][0]
        self.assertGreater((len(self.output.packets) - 1) / duration, 40.)

    async def test_http_publisher_and_bad_input_do_not_interrupt_output(self):
        address = str(self.client.make_url("/"))
        publisher = HTTPPublisher(address)
        packet = dict(initial_reference(), stream_id="video", frame=1, reference_monotonic_s=time.monotonic())
        await asyncio.to_thread(publisher.publish, packet)
        publisher.close()
        self.assertEqual(self.app[BRIDGE].camera["stream_id"], "video")
        self.assertEqual((await self.client.post("/api/pose", json={"wrong": True})).status, 400)
        self.assertEqual((await self.client.post("/api/source", json={"source": "trace"})).status, 400)
        self.assertEqual((await self.client.post("/api/source", json={"source": "idle"})).status, 200)
        count = len(self.output.packets)
        await asyncio.sleep(.1)
        self.assertGreater(len(self.output.packets), count)
        self.assertEqual((await self.client.post("/api/source", json={"source": "video"}, headers={"Origin": "http://elsewhere"})).status, 403)

    async def test_openai_configuration_never_exposes_or_persists_credentials(self):
        key='test-only-private-planner-key'
        response=await self.client.post('/api/planner',json=dict(provider='openai',model='gpt-4.1-mini',api_key=key))
        self.assertEqual(response.status,200)
        public=await response.json()
        self.assertTrue(public['planner']['api_key_configured'])
        self.assertNotIn(key,json.dumps(public))
        self.assertEqual(public['planner']['provider'],'openai')
        generated=await self.client.post('/api/generate',json=dict(text='Wave the right hand',autoplay=False))
        self.assertEqual(generated.status,202)
        job=await generated.json()
        self.assertEqual(job['planner_provider'],'openai')
        from motion_teleop.training_export import DEFAULT_DATASET
        self.assertEqual(DEFAULT_DATASET.name, 'motions')
        self.assertEqual(self.app[JOBS].training_dataset_dir, DEFAULT_DATASET)
        request=json.loads((self.app[JOBS].directory/job['id']/'input.json').read_text())
        self.assertEqual(Path(request['training_dataset_dir']), DEFAULT_DATASET)
        await self.app[JOBS].cancel()
        self.assertFalse(self.app[JOBS]._job_keys)
        for file in self.app[JOBS].directory.rglob('*'):
            if file.is_file():self.assertNotIn(key.encode(),file.read_bytes(),str(file))
        status=await(await self.client.get('/api/status')).json()
        self.assertNotIn(key,json.dumps(status))
        cleared=await self.client.post('/api/planner',json=dict(clear_api_key=True))
        self.assertFalse((await cleared.json())['planner']['api_key_configured'])
        count=len(self.output.packets)
        await asyncio.sleep(.1)
        self.assertGreater(len(self.output.packets),count)

    async def test_codex_subscription_mode_needs_no_api_key_and_probe_is_public(self):
        from unittest.mock import patch
        from motion_teleop import codex_planner
        fake=dict(available=True,authenticated=True,auth_mode='chatgpt',checked=True,compatible=True,binary='/test/codex',version='codex-cli test',message='Signed in')
        with patch.object(codex_planner,'find_codex',return_value='/test/codex'):
            settings=await self.client.post('/api/planner',json=dict(provider='codex',codex_model='',clear_api_key=True))
            self.assertEqual(settings.status,200)
            data=await settings.json()
            self.assertEqual(data['planner']['provider'],'codex')
            self.assertFalse(data['planner']['api_key_configured'])
            with patch.object(codex_planner,'probe_codex',return_value=fake):
                response=await self.client.post('/api/codex/status',json={})
                self.assertTrue((await response.json())['planner']['codex']['authenticated'])
            generated=await self.client.post('/api/generate',json=dict(text='Wave the right hand',autoplay=False))
            self.assertEqual(generated.status,202)
            self.assertEqual((await generated.json())['planner_provider'],'codex')
            await self.app[JOBS].cancel()
        self.assertEqual(self.app[BRIDGE].source,'video')
        self.assertEqual((await self.client.post('/api/codex/status',json={},headers={'Origin':'http://elsewhere'})).status,403)

    async def test_missing_key_and_bad_settings_leave_stream_and_source_running(self):
        await self.client.post('/api/planner',json=dict(provider='openai',clear_api_key=True))
        response=await self.client.post('/api/generate',json=dict(text='Wave'))
        self.assertEqual(response.status,400)
        self.assertIn('key is missing',(await response.json())['error'])
        self.assertFalse(self.app[JOBS].jobs)
        bad=await self.client.post('/api/planner',json=dict(provider='unexpected',api_key='fake'))
        self.assertEqual(bad.status,400)
        self.assertEqual(self.app[JOBS].planner.provider,'openai')
        denied=await self.client.post('/api/planner',json=dict(api_key='fake'),headers={'Origin':'http://elsewhere'})
        self.assertEqual(denied.status,403)
        count=len(self.output.packets)
        await asyncio.sleep(.1)
        self.assertGreater(len(self.output.packets),count)
        self.assertEqual(self.app[BRIDGE].source,'video')

    async def test_generation_failure_leaves_reference_stream_running(self):
        result = await self.client.post("/api/generate", json=dict(text="Raise both hands", steps=1))
        self.assertEqual(result.status, 202)
        await asyncio.wait_for(self.app[JOBS].task, timeout=20.)
        self.assertEqual(next(iter(self.app[JOBS].jobs.values()))["state"], "failed")
        count = len(self.output.packets)
        await asyncio.sleep(.1)
        self.assertGreater(len(self.output.packets), count)
        self.assertEqual(self.app[BRIDGE].source, "video")

    async def test_cancel_before_worker_starts_is_persisted(self):
        jobs = self.app[JOBS]
        job = jobs.create(dict(text="Wave the right hand"))
        await jobs.cancel()
        self.assertEqual(job["state"], "cancelled")
        stored = json.loads((jobs.directory / job["id"] / "job.json").read_text())
        self.assertEqual(stored["state"], "cancelled")

    async def test_nonfinite_metadata_cannot_break_output(self):
        packet = dict(initial_reference(), stream_id="video", frame=1, reference_monotonic_s=time.monotonic(), extra=float("nan"))
        response = await self.client.post("/api/pose", json=packet)
        self.assertEqual(response.status, 400)
        count = len(self.output.packets)
        await asyncio.sleep(.1)
        self.assertGreater(len(self.output.packets), count)

    async def test_preview_is_full_3d_reference_and_does_not_change_robot_source(self):
        jobs=self.app[JOBS]
        jobs.jobs["preview"]=dict(id="preview",state="ready",created=time.time(),text="Wave")
        folder=jobs.directory/"preview/output";folder.mkdir(parents=True)
        motion=clip()
        np.savez(folder/"motion_final_qpos.npz",qpos=motion.qpos,fps=20.,joint_names=motion.names)
        before=self.app[BRIDGE].epoch
        response=await self.client.get("/api/jobs/preview/preview")
        self.assertEqual(response.status,200)
        data=await response.json()
        self.assertEqual(np.shape(data["positions"])[1:],(30,3))
        self.assertEqual(len(data["edges"]),29)
        self.assertEqual(data["times"][-1],motion.duration)
        self.assertEqual(data["fps"],50.)
        self.assertEqual(self.app[BRIDGE].epoch,before)
        self.assertEqual(self.app[BRIDGE].source,"video")
        self.assertEqual((await self.client.get("/preview.js")).status,200)
        for asset in ("/style.css", "/studio.js", "/fonts/Manrope.ttf", "/voice.js", "/voice-worklet.js"):
            response = await self.client.get(asset)
            self.assertEqual(response.status, 200, asset)
            self.assertGreater(len(await response.read()), 100)
        self.assertEqual((await self.client.get("/api/jobs/missing/preview")).status,404)

    async def test_playback_api_pause_resume_and_bad_requests(self):
        self.assertEqual((await self.client.post("/api/playback", json={"action":"pause"})).status, 400)
        self.app[BRIDGE].play(clip(), "fixture")
        response = await self.client.post("/api/playback", json={"action":"pause"})
        self.assertTrue((await response.json())["bridge"]["paused"])
        response = await self.client.post("/api/playback", json={"action":"resume"})
        self.assertFalse((await response.json())["bridge"]["paused"])
        self.assertEqual((await self.client.post("/api/playback", json={"action":"other"})).status, 400)
        self.assertEqual((await self.client.post("/api/playback", json=[], headers={"Origin":"http://other.test"})).status, 403)

    async def test_download_training_npz_for_completed_job(self):
        import io
        from motion_teleop.training_export import export_training_motion
        jobs = self.app[JOBS]
        jobs.jobs["wave"] = dict(id="wave", state="ready", created=time.time(), text="Wave")
        folder = jobs.directory / "wave/output";folder.mkdir(parents=True)
        motion = clip()
        source = folder / "motion_final_qpos.npz"
        np.savez(source, qpos=motion.qpos, fps=20., joint_names=motion.names)
        export_training_motion(source, dataset_dir=None)
        response = await self.client.get("/api/jobs/wave/training.npz")
        self.assertEqual(response.status, 200)
        self.assertIn('attachment;', response.headers["Content-Disposition"])
        with np.load(io.BytesIO(await response.read()), allow_pickle=False) as data:
            self.assertEqual(data["fps"].tolist(), [50.])
            self.assertEqual(data["body_pos_w"].shape[1:], (30, 3))
        self.assertEqual((await self.client.get("/api/jobs/missing/training.npz")).status, 404)


if __name__ == "__main__":unittest.main()
