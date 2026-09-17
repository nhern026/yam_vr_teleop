"""Threaded wrist camera capture for the teleop pipeline.

Each camera runs its own grab thread so the control loop never blocks on
a frame read. The loop calls ``latest()`` to get whatever frame is newest;
if the camera runs at 30 Hz and the loop at 100 Hz, most ticks see the
same frame (same timestamp), which the recorder uses to deduplicate.

    cameras:
      wrist_right:
        device: "/dev/video0"
        width: 640
        height: 480
        fps: 30
      wrist_left:
        device: "/dev/video2"
        width: 640
        height: 480
        fps: 30
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraFrame:
    image: np.ndarray
    timestamp: float
    seq: int


class WristCamera:
    """One USB camera on its own grab thread."""

    def __init__(
        self,
        name: str,
        device: str | int,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.name = name
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps

        self._lock = threading.Lock()
        self._frame: CameraFrame | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._error: str | None = None

    def start(self) -> None:
        dev = self._device
        if isinstance(dev, str) and dev.startswith("/dev/"):
            dev = int(dev.replace("/dev/video", ""))
        self._cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"camera {self.name}: cannot open {self._device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Read one frame to confirm the camera works.
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._cap.release()
            raise RuntimeError(f"camera {self.name}: first read failed on {self._device}")
        with self._lock:
            self._seq = 1
            self._frame = CameraFrame(image=frame, timestamp=time.monotonic(), seq=1)
        self._thread = threading.Thread(
            target=self._grab_loop, name=f"cam-{self.name}", daemon=True
        )
        self._thread.start()

    def _grab_loop(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                with self._lock:
                    self._error = f"camera {self.name}: read failed"
                break
            now = time.monotonic()
            with self._lock:
                self._seq += 1
                self._frame = CameraFrame(image=frame, timestamp=now, seq=self._seq)

    def latest(self) -> CameraFrame | None:
        with self._lock:
            return self._frame

    @property
    def resolution(self) -> tuple[int, int]:
        f = self.latest()
        if f is not None:
            return f.image.shape[1], f.image.shape[0]
        return self._width, self._height

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def alive(self) -> bool:
        return (
            self._thread is not None
            and self._thread.is_alive()
            and self._error is None
        )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()


def open_cameras(config: dict) -> dict[str, WristCamera]:
    """Open all cameras listed under ``config["cameras"]``."""
    cameras_cfg = config.get("cameras")
    if not cameras_cfg or not isinstance(cameras_cfg, dict):
        return {}
    cameras: dict[str, WristCamera] = {}
    for name, cam_cfg in cameras_cfg.items():
        if not isinstance(cam_cfg, dict) or "device" not in cam_cfg:
            raise ValueError(f"cameras.{name} must have a 'device' key")
        cam = WristCamera(
            name=str(name),
            device=cam_cfg["device"],
            width=int(cam_cfg.get("width", 640)),
            height=int(cam_cfg.get("height", 480)),
            fps=int(cam_cfg.get("fps", 30)),
        )
        cam.start()
        cameras[name] = cam
    return cameras
