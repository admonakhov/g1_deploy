"""Keep just the newest live frame; read video files sequentially."""

import threading
import time

import cv2


class StreamCapture:
    def __init__(self, capture, latest=False):
        self.capture = capture
        self.latest = latest
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.pending = None
        self.finished = False
        self.index = 0
        self.worker = None
        if latest:
            self.worker = threading.Thread(target=self._collect, name="latest-video-frame", daemon=True)
            self.worker.start()

    def _collect(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    break
                item = (True, frame, self.index, time.monotonic(), time.time())
                self.index += 1
                with self.condition:
                    self.pending = item
                    self.condition.notify()
        finally:
            self.capture.release()
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def read(self):
        if not self.latest:
            ok, frame = self.capture.read()
            item = (ok, frame, self.index, time.monotonic(), time.time())
            self.index += 1
            return item
        with self.condition:
            while self.pending is None and not self.finished and not self.stop.is_set():
                self.condition.wait(timeout=0.25)
            if self.pending is None:
                return False, None, self.index, time.monotonic(), time.time()
            item, self.pending = self.pending, None
            return item

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        if self.worker:
            self.worker.join(timeout=2)
        else:
            self.capture.release()


def open_source(source):
    if isinstance(source, str) and source.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000,
        ])
    return cv2.VideoCapture(source)


class RecoveringCapture:
    """Retry unavailable inputs; keep EOF in fallback rather than replaying a file."""
    def __init__(self, source, latest=False, retry_interval=1., opener=None):
        self.source=source
        self.is_live=isinstance(source,int) or "://" in str(source)
        self.latest=latest and self.is_live
        self.opener=opener or open_source
        self.retry_interval=retry_interval
        self.reader=None
        self.pending=None
        self.next_attempt=0.
        self.fps=25.
        self.failure_reason="source_unavailable"
        self.eof=False
        self.frames=0

    def read(self):
        if self.pending is not None:
            item,self.pending=self.pending,None
            return item
        if self.reader is None and not self.eof and time.monotonic()>=self.next_attempt:
            self.next_attempt=time.monotonic()+self.retry_interval
            capture=None
            try:
                capture=self.opener(self.source)
                if not capture.isOpened():
                    capture.release()
                    return False,None,self.frames,time.monotonic(),time.time()
                capture.set(cv2.CAP_PROP_BUFFERSIZE,1)
                fps=capture.get(cv2.CAP_PROP_FPS)
                if fps>0 and fps<float("inf"):self.fps=fps
                self.reader=StreamCapture(capture,latest=self.latest)
            except (OSError,RuntimeError,cv2.error):
                if capture is not None:capture.release()
        if self.reader is not None:
            try:
                item=self.reader.read()
                if item[0] and item[1] is not None and item[1].size:
                    self.frames+=1
                    return item
            except (OSError,RuntimeError,cv2.error):
                pass
            self.reader.close();self.reader=None
            self.eof=not self.is_live and self.frames>0
            self.failure_reason="source_ended" if self.eof else "source_read_failed"
            self.next_attempt=time.monotonic()+self.retry_interval
        return False,None,self.frames,time.monotonic(),time.time()

    def ready(self):
        item=self.read()
        if item[0]:self.pending=item
        return item[0]

    def close(self):
        if self.reader is not None:self.reader.close();self.reader=None
