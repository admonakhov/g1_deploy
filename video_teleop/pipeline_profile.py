"""Optional wall-clock stage timings; parallel stages are reported separately."""

from collections import defaultdict
from time import perf_counter


class PipelineProfile:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.samples = defaultdict(list)
        self.started = perf_counter()
        self.frames = 0

    def call(self, name, function, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        started = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            self.samples[name].append(perf_counter() - started)

    def frame_done(self, force=False):
        if not self.enabled:
            return
        self.frames += not force
        now = perf_counter()
        elapsed = now - self.started
        if not self.frames or (elapsed < 2. and not force):
            return
        stages = " | ".join(f"{name}={1000 * sum(values) / len(values):.1f} ms"
                            for name, values in sorted(self.samples.items()) if values)
        # The caller has joined the hand worker before reporting/clearing samples.
        print(f"Profile: {self.frames / elapsed:.2f} FPS | {stages}", flush=True)
        self.samples.clear()
        self.frames = 0
        self.started = now
