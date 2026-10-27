"""Continuous reference packets, independent of camera capture and inference.

Held observations retain their frame/time so controller interpolation still
uses the camera cadence. Initial references are explicit commands, not inferred
body parts. Loss of this process or transport still expires the robot watchdog.
"""

from copy import deepcopy
import json
import math
from pathlib import Path
import threading
import time
import socket


def initial_reference(retargeter=None):
    """The WBC standing reference, mapped by name to the actual MJCF qpos."""
    path = Path(__file__).resolve().parents[1] / "deploy/g1/config/initial_pose.json"
    angles = json.loads(path.read_text())
    if retargeter is None:
        return dict(schema="gmr-g1-v1",robot="unitree_g1",status="fallback",person_id=None,
                    joint_names=list(angles),joint_positions=list(angles.values()),joint_units="radians",
                    base_quaternion_order="wxyz",base_axes=["forward","left","up"],base_units="metres",
                    qpos=None,reference_pose="g1_standing",base_tracked=False,
                    active_joint_mask=[False]*29,reference_joint_mask=[True]*29,
                    retargeted_body_parts=[],palms_tracked={"left":False,"right":False},qpos_contains_held_joints=False)
    if set(angles) != set(retargeter.joint_names):
        raise ValueError("Initial pose must describe all 29 G1 joints by name")
    qpos = retargeter.model.qpos0.copy()
    for name, address in zip(retargeter.joint_names, retargeter.qpos_addresses):
        qpos[address] = angles[name]
    packet = retargeter.packet("fallback")
    packet.update(person_id=None, reference_pose="g1_standing", base_tracked=False,
                  qpos=qpos.tolist(), joint_positions=[angles[n] for n in retargeter.joint_names],
                  active_joint_mask=[False] * 29, reference_joint_mask=[True] * 29,
                  retargeted_body_parts=[], palms_tracked={"left": False, "right": False},
                  qpos_contains_held_joints=False)
    return packet


class PoseReference:
    """Single-owner packet composer; the publisher serializes calls with a lock."""

    def __init__(self, initial, stream_id, pose_timeout=.75, loss_grace=.35):
        if not math.isfinite(pose_timeout) or pose_timeout <= 0:
            raise ValueError("Pose timeout must be finite and positive")
        if not math.isfinite(loss_grace) or not 0 <= loss_grace < pose_timeout:
            raise ValueError("Loss grace must be nonnegative and shorter than pose timeout")
        self.initial = deepcopy(initial)
        self.stream_id, self.pose_timeout = stream_id, pose_timeout
        self.loss_grace = loss_grace
        self.latest = self.current = None
        self.last_good, self.lost_at = None, None
        self.good_revision = -1
        self.mode = None
        self.fallback_entries = self.holding_entries = 0
        self.revision, self.current_revision, self.next_frame = 0, -1, 0

    def update(self, packet):
        self.latest = deepcopy(packet)
        self.revision += 1
        if packet.get("status") in ("tracking", "partial_tracking") and any(packet.get("active_joint_mask", [])):
            self.last_good = deepcopy(packet)
            self.good_revision = self.revision
            self.lost_at = None
        elif self.lost_at is None:
            self.lost_at = self.completed(packet)

    @staticmethod
    def completed(packet):
        return packet.get("processing_completed_monotonic_s",
                          packet.get("capture_monotonic_s", 0.) + packet.get("latency_s", 0.))

    def sample(self, now, unix):
        latest = self.latest or {}
        status = latest.get("status", "startup")
        source = latest
        revision = self.revision
        holding = (status not in ("tracking", "partial_tracking") or not any(latest.get("active_joint_mask", [])))
        holding = (holding and self.last_good is not None and self.lost_at is not None
                   and 0 <= now - self.lost_at < self.loss_grace)
        if holding:
            source, revision = self.last_good, self.good_revision
        captured = source.get("capture_monotonic_s", now)
        age = now - captured
        reason = status
        usable = source.get("status") in ("tracking", "partial_tracking")
        if usable and not any(source.get("active_joint_mask", [])):
            usable, reason = False, "no_active_joints"
        result_age = now - self.completed(source)
        if usable and (not math.isfinite(age) or age < 0 or not math.isfinite(result_age)
                       or result_age < 0 or result_age > self.pose_timeout):
            usable, reason = False, "stale_pose"
        repeated = usable and self.current_revision == revision and self.current is not None
        if not repeated:
            self.current = deepcopy(source if usable else self.initial)
            self.current.update(frame=self.next_frame, reference_monotonic_s=now)
            self.next_frame += 1
            self.current_revision = revision if usable else -1
        packet = deepcopy(self.current)
        mode = "holding" if usable and holding else "tracking" if usable else "fallback"
        if mode != self.mode:
            self.fallback_entries += mode == "fallback"
            self.holding_entries += mode == "holding"
            self.mode = mode
        packet.update(stream_id=self.stream_id, timestamp_unix_s=unix, reference_hold=repeated,
                      latency_s=age if usable else 0., retarget_status=status,
                      tracking_state=mode, fallback_entries=self.fallback_entries, holding_entries=self.holding_entries,
                      input_frame=latest.get("frame"), result_age_s=result_age if usable else None,
                      retarget_frame=source.get("frame"),
                      retarget_reason=latest.get("tracking_reason", status),
                      retarget_capture_monotonic_s=source.get("capture_monotonic_s"),
                      retarget_latency_s=age if self.latest is not None and math.isfinite(age) else None)
        if not usable:
            packet.update(fallback_reason=reason, capture_monotonic_s=now, capture_unix_s=unix)
        return packet


class ContinuousPosePublisher:
    def __init__(self, transport, initial, stream_id, hz=50., pose_timeout=.75, loss_grace=.35):
        if not math.isfinite(hz) or not 1 <= hz <= 1000:
            raise ValueError("UDP frequency must be between 1 and 1000 Hz")
        self.transport = transport
        self.reference = PoseReference(initial, stream_id, pose_timeout, loss_grace)
        self.period = 1 / hz
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None
        self.sent = 0
        self.last = None
        self.thread = threading.Thread(target=self._run, name="camera-teleop-udp", daemon=True)

    def start(self):
        self.thread.start()

    def publish(self, packet):
        with self.lock:
            self._check_error()
            self.reference.update(packet)

    def set_initial(self, packet):
        with self.lock:
            self._check_error()
            self.reference.initial = deepcopy(packet)

    def use_initial(self, reason):
        """Input failure switches immediately, without holding an old human pose."""
        with self.lock:
            self._check_error()
            self.reference.update(dict(status=reason,processing_completed_monotonic_s=time.monotonic()))
            self.reference.last_good = None

    def snapshot(self):
        with self.lock:
            self._check_error()
            return self.sent, deepcopy(self.last)

    def _check_error(self):
        if self.error is not None:
            raise RuntimeError(f"Camera UDP publisher failed: {self.error}") from self.error

    def _run(self):
        deadline = time.monotonic()
        try:
            while not self.stop.is_set():
                with self.lock:
                    packet = self.reference.sample(time.monotonic(), time.time())
                self.transport.publish(packet)
                with self.lock:
                    self.sent += 1
                    self.last = packet
                deadline += self.period
                now = time.monotonic()
                if deadline < now:
                    deadline = now + self.period
                self.stop.wait(max(0., deadline - now))
        except Exception as error:
            with self.lock:
                self.error = error
            self.stop.set()

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join()
        try:
            # Send only after joining, so no in-flight reference can follow it.
            packet = deepcopy(self.reference.initial)
            packet.update(status="stopped", stream_id=self.reference.stream_id,
                          timestamp_unix_s=time.time(), qpos=None, joint_positions=None,
                          reference_joint_mask=[False] * 29)
            self.transport.publish(packet)
        finally:
            self.transport.close()
        self._check_error()


class UDPPublisher:
    def __init__(self, address):
        host, port = address.rsplit(":", 1)
        self.address = (host, int(port))
        if not host or not 1 <= self.address[1] <= 65535:
            raise ValueError("UDP destination must be HOST:PORT, port 1..65535")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)

    def publish(self, packet):
        payload = json.dumps(packet, allow_nan=False, separators=(",", ":")).encode("utf-8")
        try:
            self.socket.sendto(payload, self.address)
        except BlockingIOError:
            pass

    def close(self):
        self.socket.close()
