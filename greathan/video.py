from __future__ import annotations
from collections import deque
import os
import math
import threading
import time

import cv2


class VideoSource:
    def __init__(self, camera, stop):
        self.camera, self.stop = camera, stop
        source = os.environ.get(camera["source_env"],"") if camera.get("source_env") else camera["source"]
        if not source:
            raise ValueError("Camera source environment variable is empty")
        source = int(source) if source.isdecimal() else source
        self.mode = camera.get("mode","offline")
        network = isinstance(source,str) and "://" in source
        if self.mode == "live" and not network:
            raise ValueError("Live mode currently requires a network stream; local files use offline/realtime mode")
        if network:
            self.capture = cv2.VideoCapture(source,cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,5000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,5000])
        else:
            self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError("Cannot open configured video source")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps_estimated=False
        if not self.fps or not 0 < self.fps <= 1000:
            if self.mode=="live":
                self.fps=30.0
                self.fps_estimated=True
            else:
                self.capture.release()
                raise ValueError("Source FPS is unavailable; configure a file with valid timestamps")
        self.total = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_id = 0
        self.dropped = 0
        self.latest = deque(maxlen=1)
        self.condition = threading.Condition()
        self.error = None
        self.done = False
        self.thread = None
        self.started = time.monotonic()
        self.last_pts=-1.0
        self.time_basis="capture_monotonic" if self.mode=="live" else "source_pts"
        if self.mode == "live":
            self.thread = threading.Thread(target=self._read_live,name="latest-frame-capture",daemon=True)
            self.thread.start()

    def _read_live(self):
        try:
            while not self.stop.is_set():
                ok,frame = self.capture.read()
                if not ok:
                    raise RuntimeError("Live video source disconnected")
                self.frame_id += 1
                with self.condition:
                    if self.latest:
                        self.dropped += 1
                    self.latest.append((self.frame_id,time.monotonic()-self.started,frame))
                    self.condition.notify_all()
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.capture.release()
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def read(self):
        if self.mode == "live":
            with self.condition:
                self.condition.wait_for(lambda:self.latest or self.done or self.stop.is_set(),timeout=10)
                if self.latest:
                    return self.latest.popleft()
                if self.stop.is_set():
                    return None
                raise RuntimeError(self.error or "No live frame for 10 seconds")
        while not self.stop.is_set():
            ok,frame = self.capture.read()
            if not ok:
                if self.total > 0 and self.frame_id+2 < self.total:
                    raise RuntimeError("Video ended before expected frame count (decode failure)")
                return None
            self.frame_id += 1
            pts=self.capture.get(cv2.CAP_PROP_POS_MSEC)/1000
            if math.isfinite(pts) and pts>self.last_pts:
                source_s=pts
            else:
                source_s=(self.frame_id-1)/self.fps
                self.time_basis="fps_fallback"
            self.last_pts=source_s
            if self.mode == "realtime":
                now=time.monotonic()
                # A slow backend must not accumulate an ever older file replay.
                if self.started+source_s < now-.25:
                    self.dropped+=1
                    continue
                self.stop.wait(max(0,self.started+source_s-now))
            return self.frame_id,source_s,frame
        return None

    def close(self):
        if self.thread:
            self.stop.set()
            with self.condition:
                self.condition.notify_all()
            self.thread.join(timeout=7)
            return not self.thread.is_alive()
        self.capture.release()
        return True
