"""One selected source, one 50 Hz output; generation never blocks playback."""
from copy import deepcopy
import math
import time
import uuid

from video_teleop.teleop_stream import initial_reference
from .motion import validate_packet


class ReferenceBridge:
    def __init__(self, hz=50., video_timeout=1.):
        if not math.isfinite(hz) or not 1 <= hz <= 200:
            raise ValueError("Output frequency must be 1..200 Hz")
        if not math.isfinite(video_timeout) or video_timeout <= 0:
            raise ValueError("Video timeout must be positive")
        self.hz, self.video_timeout = hz, video_timeout
        self.session = uuid.uuid4().hex
        self.source = "video"
        self.epoch = self.frame = self.sent = 0
        self.camera = None
        self.camera_received = -math.inf
        self.clip = None
        self.clip_id = None
        self.started = 0.
        self.playing = False
        self.paused_at = None
        self.last = initial_reference()
        self.last_identity = None
        self.output_hz = 0.
        self.output_error = None

    def receive(self, packet, now=None):
        validate_packet(packet)
        self.camera = deepcopy(packet)
        self.camera_received = time.monotonic() if now is None else now

    def select(self, source, now=None):
        if source not in ("video", "trace", "idle"):
            raise ValueError("Source must be video, trace or idle")
        if source == "trace" and self.clip is None:
            raise ValueError("Generate or select a motion first")
        self.source = source
        self.epoch += 1
        self.last_identity = None
        if source != "trace":
            self.playing = False
            self.paused_at = None

    def play(self, clip, clip_id, now=None):
        clip.prepare_wbc()
        self.clip, self.clip_id = clip, clip_id
        self.select("trace")
        self.started = time.monotonic() if now is None else now
        self.paused_at = None
        self.playing = True

    def pause(self, now=None):
        if self.source != "trace" or self.clip is None:
            raise ValueError("Пауза доступна при воспроизведении TRACE.")
        if self.paused_at is None:
            self.paused_at = min(self.clip.duration, max(0., (time.monotonic() if now is None else now) - self.started))
            self.epoch += 1  # An in-flight generation must not auto-start after Stop.
            self.last_identity = None
        self.playing = False

    def resume(self, now=None):
        if self.source != "trace" or self.clip is None or self.paused_at is None:
            raise ValueError("Сначала поставьте движение TRACE на паузу.")
        self.started = (time.monotonic() if now is None else now) - self.paused_at
        self.paused_at = None
        self.playing = True

    def packet(self, now=None, unix=None):
        now = time.monotonic() if now is None else now
        unix = time.time() if unix is None else unix
        packet = None
        if self.source == "trace" and self.clip is not None:
            elapsed = self.paused_at if self.paused_at is not None else now - self.started
            packet = self.clip.packet(elapsed)
            if self.paused_at is not None:
                packet["base_linear_velocity_world"] = [0.] * 3
                for name in ("linear_velocity_world", "angular_velocity_world"):
                    packet["wbc_reference"][name] = [0.] * 3
            # Hold the final generated pose until another source is selected.
            # The default generator constrains both endpoints to the snapshot.
            self.playing = self.paused_at is None and elapsed < self.clip.duration
        elif self.source == "video" and self.camera is not None:
            if now - self.camera_received <= self.video_timeout:
                if self.camera.get("status") in ("tracking", "partial_tracking", "fallback"):
                    packet = deepcopy(self.camera)
                    packet["stream_id"] = f"{self.session}:{self.epoch}:video:{packet['stream_id']}"
                    packet["latency_s"] = float(packet.get("latency_s", 0.)) + max(0., now - self.camera_received)
                    if not math.isfinite(packet["latency_s"]) or packet["latency_s"] >= self.video_timeout:
                        packet = None
        if packet is None:
            packet = initial_reference()
            packet.update(fallback_reason="waiting_for_video" if self.source == "video" else "idle")
        if "stream_id" not in packet:
            kind = "trace" if self.source == "trace" and self.clip is not None else "idle"
            packet.update(stream_id=f"{self.session}:{self.epoch}:{kind}", frame=self.frame,
                          reference_monotonic_s=now, capture_monotonic_s=now, latency_s=0.)
        identity = (packet["stream_id"], packet["frame"], packet["reference_monotonic_s"])
        packet["reference_hold"] = identity == self.last_identity
        packet.update(timestamp_unix_s=unix, selected_source=self.source)
        self.last_identity = identity
        self.frame += 1
        self.last = packet
        return packet

    def status(self):
        return dict(source=self.source, playing=self.playing, paused=self.paused_at is not None, motion_id=self.clip_id,
                    duration_s=self.clip.duration if self.clip else None,
                    position_s=(self.paused_at if self.paused_at is not None else min(max(0., time.monotonic() - self.started), self.clip.duration)) if self.clip else 0.,
                    target_hz=self.hz, output_hz=round(self.output_hz, 1), sent=self.sent,
                    output_error=self.output_error, video_age_s=max(0., time.monotonic() - self.camera_received) if self.camera else None,
                    reference_status=self.last.get("status"))
