"""Continuous camera references without network or robot connections."""

from copy import deepcopy
import threading
import unittest

from video_teleop.robot_retarget import StreamingRetargeter
from video_teleop.teleop_stream import ContinuousPosePublisher, PoseReference, initial_reference


class MemoryTransport:
    def __init__(self):
        self.packets = []
        self.enough = threading.Event()
        self.closed = False

    def publish(self, packet):
        self.packets.append(deepcopy(packet))
        if len(self.packets) >= 6:
            self.enough.set()

    def close(self):
        self.closed = True


class TeleopStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = StreamingRetargeter()
        cls.initial = initial_reference(cls.robot)

    def setUp(self):
        self.reference = PoseReference(self.initial, "test", pose_timeout=.75)

    def observed(self, frame=0, captured=10.):
        packet = deepcopy(self.initial)
        packet.update(status="partial_tracking", person_id=7, frame=frame,
                      capture_monotonic_s=captured, active_joint_mask=[True] * 29)
        packet.pop("reference_joint_mask")
        packet.pop("reference_pose")
        # Missing left leg remains absent, even with a full reference qpos.
        for i, name in enumerate(packet["joint_names"]):
            if name.startswith("left_") and any(p in name for p in ("hip_", "knee_", "ankle_")):
                packet["active_joint_mask"][i] = False
                packet["joint_positions"][i] = None
        return packet

    def test_startup_and_no_pose_send_named_initial_pose(self):
        startup = self.reference.sample(10., 100.)
        self.assertEqual(startup["status"], "fallback")
        self.assertEqual(startup["fallback_reason"], "startup")
        self.assertIsNone(startup["person_id"])
        self.assertFalse(any(startup["active_joint_mask"]))
        self.assertTrue(all(startup["reference_joint_mask"]))
        self.assertEqual(len(startup["joint_positions"]), 29)
        for address, angle in zip(self.robot.qpos_addresses, startup["joint_positions"]):
            self.assertEqual(startup["qpos"][address], angle)
        for status in ("no_pose", "invalid_pose"):
            packet = self.robot.packet(status)
            packet.update(frame=3, capture_monotonic_s=10.)
            self.reference.update(packet)
            fallback = self.reference.sample(10.1, 100.1)
            self.assertEqual(fallback["fallback_reason"], status)
            self.assertEqual(fallback["joint_positions"], startup["joint_positions"])
            self.assertFalse(any(fallback["active_joint_mask"]))

    def test_repeats_preserve_source_cadence_and_partial_masks(self):
        source = self.observed()
        untouched = deepcopy(source)
        self.reference.update(source)
        first = self.reference.sample(10.1, 100.)
        for i in range(1, 10):
            repeat = self.reference.sample(10.1 + i * .02, 100. + i * .02)
            self.assertTrue(repeat["reference_hold"])
            self.assertEqual(repeat["frame"], first["frame"])
            self.assertEqual(repeat["reference_monotonic_s"], first["reference_monotonic_s"])
            self.assertEqual(repeat["capture_monotonic_s"], 10.)
            self.assertEqual(repeat["joint_positions"], source["joint_positions"])
            self.assertEqual(repeat["active_joint_mask"], source["active_joint_mask"])
            self.assertNotIn("reference_joint_mask", repeat)
        self.assertEqual(source, untouched)
        self.reference.update(self.observed(frame=1, captured=10.3))
        second = self.reference.sample(10.4, 100.4)
        self.assertFalse(second["reference_hold"])
        self.assertGreater(second["frame"], first["frame"])
        self.assertAlmostEqual(second["reference_monotonic_s"] - first["reference_monotonic_s"], .3)

    def test_stall_returns_initial_pose_and_reacquires(self):
        self.reference.update(self.observed())
        tracked = self.reference.sample(10.1, 100.)
        lost = self.reference.sample(10.76, 100.76)
        self.assertEqual(lost["status"], "fallback")
        self.assertEqual(lost["fallback_reason"], "stale_pose")
        self.assertGreater(lost["retarget_latency_s"], .75)
        self.assertFalse(lost["reference_hold"])
        next_fallback = self.reference.sample(10.78, 100.78)
        self.assertGreater(next_fallback["frame"], lost["frame"])
        self.reference.update(self.observed(frame=1, captured=10.77))
        acquired = self.reference.sample(10.8, 100.8)
        self.assertEqual(acquired["status"], tracked["status"])
        self.assertGreater(acquired["frame"], next_fallback["frame"])
        self.assertFalse(acquired["reference_hold"])

    def test_no_active_joint_is_also_fallback(self):
        source = self.observed()
        source["active_joint_mask"] = [False] * 29
        self.reference.update(source)
        packet = self.reference.sample(10.1, 100.)
        self.assertEqual(packet["status"], "fallback")
        self.assertEqual(packet["fallback_reason"], "no_active_joints")

    def test_single_missing_frame_holds_reference_and_continuous_loss_expires(self):
        self.reference.update(self.observed())
        first = self.reference.sample(10., 100.)
        for time in (10.1, 10.2, 10.3):
            self.reference.update(dict(status="no_pose", frame=1, capture_monotonic_s=time,
                tracking_reason="no_mediapipe_pose"))
            held = self.reference.sample(time, 100. + time)
            self.assertEqual(held["tracking_state"], "holding")
            self.assertTrue(held["reference_hold"])
            self.assertEqual(held["frame"], first["frame"])
            self.assertEqual(held["joint_positions"], first["joint_positions"])
            self.assertEqual(held["retarget_status"], "no_pose")
            self.assertEqual(held["retarget_reason"], "no_mediapipe_pose")
        # More invalid inputs cannot postpone the fallback forever.
        self.reference.update(dict(status="invalid_pose", frame=4, capture_monotonic_s=10.5))
        lost = self.reference.sample(10.5, 100.5)
        self.assertEqual(lost["status"], "fallback")
        self.assertFalse(any(lost["active_joint_mask"]))

    def test_slow_inference_does_not_expire_a_newly_completed_pose(self):
        pose = self.observed(captured=10.)
        pose.update(latency_s=.6, processing_completed_monotonic_s=10.6)
        self.reference.update(pose)
        fresh = self.reference.sample(10.8, 100.8)
        self.assertEqual(fresh["tracking_state"], "tracking")
        self.assertAlmostEqual(fresh["latency_s"], .8)
        self.assertAlmostEqual(fresh["result_age_s"], .2)
        self.assertEqual(self.reference.sample(11.36, 101.36)["status"], "fallback")

    def test_reacquisition_during_grace_does_not_visit_initial_pose(self):
        self.reference.update(self.observed())
        packets = [self.reference.sample(10., 100.)]
        self.reference.update(dict(status="no_pose", frame=1, capture_monotonic_s=10.2))
        packets.append(self.reference.sample(10.3, 100.3))
        self.reference.update(self.observed(frame=2, captured=10.4))
        packets.append(self.reference.sample(10.4, 100.4))
        self.assertTrue(all(p["status"] == "partial_tracking" for p in packets))
        self.assertEqual(packets[-1]["tracking_state"], "tracking")
        self.assertEqual(packets[-1]["fallback_entries"], 0)
        self.assertEqual(packets[-1]["holding_entries"], 1)

    def test_background_publishes_without_camera_updates_and_stops_last(self):
        transport = MemoryTransport()
        publisher = ContinuousPosePublisher(transport, self.initial, "test", hz=50.)
        try:
            publisher.start()
            self.assertTrue(transport.enough.wait(1.), "No background reference packets")
            count, last = publisher.snapshot()
            self.assertGreaterEqual(count, 6)
            self.assertEqual(last["status"], "fallback")
        finally:
            publisher.close()
        self.assertTrue(transport.closed)
        self.assertFalse(publisher.thread.is_alive())
        self.assertEqual(transport.packets[-1]["status"], "stopped")
        times = [p["reference_monotonic_s"] for p in transport.packets[:-1]]
        self.assertGreater(times[-1] - times[0], .075)
        self.assertLess(times[-1] - times[0], .5)

    def test_thread_transport_failure_is_reported(self):
        transport = MemoryTransport()
        failed = threading.Event()
        def fail(packet):
            failed.set()
            raise OSError("test transport failure")
        transport.publish = fail
        publisher = ContinuousPosePublisher(transport, self.initial, "test")
        publisher.start()
        self.assertTrue(failed.wait(1.))
        publisher.thread.join(timeout=1.)
        with self.assertRaisesRegex(RuntimeError, "test transport failure"):
            publisher.snapshot()
        with self.assertRaisesRegex(OSError, "test transport failure"):
            publisher.close()
        self.assertTrue(transport.closed)


if __name__ == "__main__":
    unittest.main()
