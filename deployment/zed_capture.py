"""Threaded, timestamped capture for the synchronized ZED cameras.

Each SDK camera object is owned by exactly one thread.  The bundle broker keeps
every successfully retrieved frame: matching frames become a bundle and frames
whose mates never arrive become explicit orphan events.  Nothing is silently
shifted onto the following camera exposure.  The wrist cameras use CameraOne;
the overhead ZED X uses Camera and records its left RGB view.

``pyzed`` is imported lazily so the rest of the teleop and its unit tests still
run on machines without a ZED SDK installation.
"""

from __future__ import annotations

import argparse
import collections
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FramePacket:
  side: str
  serial: int
  sequence: int
  zed_timestamp_ns: int
  host_epoch_ns: int
  host_monotonic_ns: int
  image_monotonic_ns: int
  image_rgb: np.ndarray
  grab_start_ns: int = 0
  grab_end_ns: int = 0
  retrieve_end_ns: int = 0


@dataclass(frozen=True)
class CaptureEvent:
  index: int
  kind: str
  left: FramePacket | None
  right: FramePacket | None
  reason: str = ""
  overhead: FramePacket | None = None

  @property
  def frames(self) -> dict[str, FramePacket]:
    return {side: frame for side in ("left", "right", "overhead")
            if (frame := getattr(self, side)) is not None}

  @property
  def image_monotonic_ns(self) -> int:
    """Causal timestamp: no command selected for a pair predates either view."""
    frames = list(self.frames.values())
    if not frames:
      raise RuntimeError("capture event contains no frame")
    return max(frame.image_monotonic_ns for frame in frames)

  @property
  def pair_skew_ns(self) -> int | None:
    frames = list(self.frames.values())
    if len(frames) < 2:
      return None
    timestamps = [frame.zed_timestamp_ns for frame in frames]
    return max(timestamps) - min(timestamps)


class FramePairBroker:
  """One-to-one timestamp bundling with explicit orphan retention.

  A one-frame look-ahead prevents a late-starting stream from pairing its
  first frame with the adjacent exposure on the other camera.  Events are
  emitted outside the lock because a recorder may perform queue operations in
  its callback.
  """

  def __init__(
    self,
    *,
    tolerance_ns: int,
    max_pending_per_side: int,
    callback: Callable[[CaptureEvent], None],
    sides: tuple[str, ...] = ("left", "right"),
  ):
    if tolerance_ns <= 0 or max_pending_per_side < 2:
      raise ValueError("pairing tolerance must be positive and pending depth at least two")
    if len(sides) < 2 or len(set(sides)) != len(sides):
      raise ValueError("at least two unique camera sides are required")
    if not set(sides) <= {"left", "right", "overhead"}:
      raise ValueError("camera sides must be left, right, and/or overhead")
    self._tolerance_ns = int(tolerance_ns)
    self._max_pending = int(max_pending_per_side)
    self._callback = callback
    self.sides = tuple(sides)
    self._queues = {side: collections.deque() for side in self.sides}
    # Kept for callers/tests that inspect the original pair queues.
    self._left = self._queues.get("left", collections.deque())
    self._right = self._queues.get("right", collections.deque())
    self._lock = threading.Lock()
    self._next_index = 0
    self._last_timestamp = {side: -1 for side in self.sides}
    self.pairs = 0
    self.orphans = {side: 0 for side in self.sides}

  def set_callback(self, callback: Callable[[CaptureEvent], None], *, reset_counts: bool = False) -> None:
    """Atomically switch the event sink after camera-only warm-up."""
    with self._lock:
      self._callback = callback
      if reset_counts:
        for pending in self._queues.values():
          pending.clear()
        self._next_index = 0
        self.pairs = 0
        self.orphans = {side: 0 for side in self.sides}

  def add(self, frame: FramePacket) -> None:
    if frame.side not in self._queues:
      raise ValueError(f"unknown camera side {frame.side!r}")
    with self._lock:
      queue = self._queues[frame.side]
      if frame.zed_timestamp_ns <= self._last_timestamp[frame.side]:
        event = self._orphan(frame, f"{frame.side}_timestamp_not_monotonic")
        events = [event]
      else:
        self._last_timestamp[frame.side] = frame.zed_timestamp_ns
        queue.append(frame)
        events = self._drain(force=False)
        for side, pending in self._queues.items():
          while len(pending) > self._max_pending:
            events.append(self._orphan(pending.popleft(), f"{side}_pairing_queue_overflow"))
      callback = self._callback
    for event in events:
      callback(event)

  def close(self) -> None:
    with self._lock:
      events = self._drain(force=True)
      for side, pending in self._queues.items():
        while pending:
          events.append(self._orphan(pending.popleft(), f"{side}_missing_camera_at_shutdown"))
    for event in events:
      self._callback(event)

  def _event(
    self,
    kind: str,
    frames: dict[str, FramePacket],
    reason: str = "",
  ) -> CaptureEvent:
    event = CaptureEvent(self._next_index, kind, frames.get("left"), frames.get("right"),
                         reason, frames.get("overhead"))
    self._next_index += 1
    return event

  def _orphan(self, frame: FramePacket, reason: str) -> CaptureEvent:
    self.orphans[frame.side] += 1
    return self._event(f"{frame.side}_orphan", {frame.side: frame}, reason)

  def _drain(self, *, force: bool) -> list[CaptureEvent]:
    events: list[CaptureEvent] = []
    while all(self._queues[side] for side in self.sides):
      heads = {side: self._queues[side][0] for side in self.sides}
      timestamps = [frame.zed_timestamp_ns for frame in heads.values()]
      difference = max(timestamps) - min(timestamps)

      # Use one-frame look-ahead on every stream as a disambiguator before
      # accepting a boundary-sized timestamp difference, especially at startup.
      replacement = None
      replacement_difference = difference
      for side, pending in self._queues.items():
        if len(pending) < 2:
          continue
        candidate = dict(heads)
        candidate[side] = pending[1]
        candidate_times = [frame.zed_timestamp_ns for frame in candidate.values()]
        candidate_difference = max(candidate_times) - min(candidate_times)
        if candidate_difference < replacement_difference and candidate_difference <= self._tolerance_ns:
          replacement, replacement_difference = side, candidate_difference
      if replacement is not None:
        events.append(self._orphan(self._queues[replacement].popleft(),
                                   f"{replacement}_has_no_timestamp_mate"))
        continue

      if difference <= self._tolerance_ns and (force or any(len(q) > 1 for q in self._queues.values())):
        frames = {side: pending.popleft() for side, pending in self._queues.items()}
        self.pairs += 1
        events.append(self._event("pair", frames))
        continue

      if difference > self._tolerance_ns:
        oldest = min(self.sides, key=lambda side: heads[side].zed_timestamp_ns)
        events.append(self._orphan(self._queues[oldest].popleft(),
                                   f"{oldest}_has_no_timestamp_mate"))
        continue
      break
    return events

  @property
  def left_orphans(self) -> int:
    return self.orphans.get("left", 0)

  @property
  def right_orphans(self) -> int:
    return self.orphans.get("right", 0)


class _ZedCameraWorker:
  def __init__(
    self,
    *,
    side: str,
    serial: int,
    camera_type: str,
    view: str,
    fps: int,
    warmup_frames: int,
    start_barrier: threading.Barrier,
    broker: FramePairBroker,
    stop_event: threading.Event,
    fatal: Callable[[str], None],
  ):
    self.side = side
    self.serial = int(serial)
    self.camera_type = camera_type
    self.view = view
    self.fps = int(fps)
    self.warmup_frames = int(warmup_frames)
    self._barrier = start_barrier
    self._broker = broker
    self._stop = stop_event
    self._fatal = fatal
    self.ready = threading.Event()
    self.thread = threading.Thread(target=self._run, name=f"zed-{side}", daemon=True)
    self.frames = 0
    self.grab_errors = 0
    self.first_timestamp = self.last_timestamp = 0
    self.long_intervals = 0

  def start(self) -> None:
    self.thread.start()

  def _run(self) -> None:
    camera = None
    try:
      import pyzed.sl as sl

      if self.camera_type == "mono":
        camera = sl.CameraOne()
        init = sl.InitParametersOne()
        sdk_view = None
      elif self.camera_type == "stereo":
        camera = sl.Camera()
        init = sl.InitParameters()
        init.depth_mode = sl.DEPTH_MODE.NONE
        try:
          sdk_view = getattr(sl.VIEW, self.view.upper())
        except AttributeError as exc:
          raise ValueError(f"unsupported ZED view {self.view!r}") from exc
      else:
        raise ValueError(f"unsupported camera type {self.camera_type!r}")
      init.camera_resolution = sl.RESOLUTION.SVGA
      init.camera_fps = self.fps
      init.set_from_serial_number(self.serial)
      status = camera.open(init)
      if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"open failed: {status}")

      info = camera.get_camera_information()
      actual = info.camera_configuration
      if (info.serial_number != self.serial or actual.resolution.width != 960
          or actual.resolution.height != 600 or abs(actual.fps - self.fps) > .1):
        raise RuntimeError("opened camera serial/resolution/FPS differs from configuration")
      image = sl.Mat()
      warmed = 0
      warmup_deadline = time.monotonic() + 30.0
      while warmed < self.warmup_frames and not self._stop.is_set():
        if time.monotonic() > warmup_deadline:
          raise RuntimeError("camera warmup timed out")
        if camera.grab() == sl.ERROR_CODE.SUCCESS:
          status = camera.retrieve_image(image) if sdk_view is None else camera.retrieve_image(image, sdk_view)
          if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"warmup image retrieval failed: {status}")
          warmed += 1
      if self._stop.is_set():
        return
      self.ready.set()
      self._barrier.wait(timeout=30.0)

      sequence = 0
      consecutive_errors = 0
      clock_offset = time.monotonic_ns() - time.time_ns()
      while not self._stop.is_set():
        grab_start_ns = time.monotonic_ns()
        status = camera.grab()
        grab_end_ns = time.monotonic_ns()
        if status != sl.ERROR_CODE.SUCCESS:
          self.grab_errors += 1
          consecutive_errors += 1
          if consecutive_errors >= 30:
            raise RuntimeError(f"30 consecutive grab failures; last status {status}")
          continue
        consecutive_errors = 0
        retrieve_status = (camera.retrieve_image(image) if sdk_view is None
                           else camera.retrieve_image(image, sdk_view))
        if retrieve_status != sl.ERROR_CODE.SUCCESS:
          raise RuntimeError("image retrieval failed")
        retrieve_end_ns = time.monotonic_ns()
        zed_timestamp_ns = int(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
        if not self.first_timestamp:
          self.first_timestamp = zed_timestamp_ns
        if self.last_timestamp and zed_timestamp_ns - self.last_timestamp > 1.5e9 / self.fps:
          self.long_intervals += 1
        self.last_timestamp = zed_timestamp_ns
        sdk_current_ns = int(camera.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds())
        host_epoch_ns = time.time_ns()
        host_monotonic_ns = time.monotonic_ns()
        # SDK 5.2 reports IMAGE in host epoch time.  Mapping it to the monotonic
        # clock avoids NTP/wall-clock use inside the control loop; both raw
        # timestamps are retained so a clock step can be diagnosed.
        offset = host_monotonic_ns - host_epoch_ns
        if abs(offset - clock_offset) > 2_000_000:
          raise RuntimeError("wall/monotonic clock offset changed by more than 2 ms")
        # Validate the clock domain independently of image delivery latency.
        if abs(host_epoch_ns - sdk_current_ns) > 50_000_000:
          raise RuntimeError("SDK CURRENT clock is not aligned with host epoch")
        if not -33_333_333 <= sdk_current_ns - zed_timestamp_ns <= 5_000_000_000:
          raise RuntimeError("IMAGE timestamp outside the five-second history window")
        image_monotonic_ns = zed_timestamp_ns + clock_offset
        raw = np.asarray(image.get_data())
        if raw.ndim != 3 or raw.shape[:2] != (600, 960) or raw.shape[2] not in (3, 4):
          raise RuntimeError(f"unexpected ZED image shape {raw.shape}")
        # PyZED returns BGR/BGRA.  Everything downstream is explicitly RGB.
        image_rgb = np.ascontiguousarray(raw[..., 2::-1])
        self._broker.add(
          FramePacket(
            side=self.side,
            serial=self.serial,
            sequence=sequence,
            zed_timestamp_ns=zed_timestamp_ns,
            host_epoch_ns=host_epoch_ns,
            host_monotonic_ns=host_monotonic_ns,
            image_monotonic_ns=image_monotonic_ns,
            image_rgb=image_rgb,
            grab_start_ns=grab_start_ns, grab_end_ns=grab_end_ns, retrieve_end_ns=retrieve_end_ns,
          )
        )
        sequence += 1
        self.frames = sequence
    except threading.BrokenBarrierError:
      if not self._stop.is_set():
        self._fatal(f"{self.side} camera did not reach the capture barrier")
    except Exception as exc:
      self._fatal(f"{self.side} camera {self.serial}: {exc}")
      try:
        self._barrier.abort()
      except Exception:
        pass
    finally:
      self.ready.set()
      if camera is not None:
        camera.close()


class ZedPairCapture:
  """Lifecycle wrapper around two wrist cameras and an optional overhead ZED X."""

  def __init__(
    self,
    *,
    left_serial: int,
    right_serial: int,
    fps: int,
    warmup_frames: int,
    pair_tolerance_ms: float,
    pending_frames: int,
    callback: Callable[[CaptureEvent], None],
    overhead_serial: int | None = None,
    overhead_view: str = "left",
    error_callback: Callable[[str], None] | None = None,
  ):
    serials = [int(left_serial), int(right_serial)]
    if overhead_serial is not None:
      serials.append(int(overhead_serial))
    if len(set(serials)) != len(serials) or fps != 30 or not 0 < pair_tolerance_ms < 1000 / fps / 2:
      raise ValueError("require distinct cameras, 30 FPS, and tolerance below half a frame")
    sides = ("left", "right") + (("overhead",) if overhead_serial is not None else ())
    self._stop = threading.Event()
    self._error_callback = error_callback or (lambda message: None)
    self._errors: list[str] = []
    self._broker = FramePairBroker(
      tolerance_ns=int(float(pair_tolerance_ms) * 1_000_000),
      max_pending_per_side=int(pending_frames),
      callback=callback,
      sides=sides,
    )
    barrier = threading.Barrier(len(sides))
    common = dict(
      fps=fps,
      warmup_frames=warmup_frames,
      start_barrier=barrier,
      broker=self._broker,
      stop_event=self._stop,
      fatal=self._fatal,
    )
    self._workers = [
      _ZedCameraWorker(side="left", serial=left_serial, camera_type="mono", view="left", **common),
      _ZedCameraWorker(side="right", serial=right_serial, camera_type="mono", view="left", **common),
    ]
    if overhead_serial is not None:
      self._workers.append(_ZedCameraWorker(side="overhead", serial=overhead_serial,
                                            camera_type="stereo", view=overhead_view, **common))

  def _fatal(self, message: str) -> None:
    self._errors.append(message)
    self._error_callback(message)
    self._stop.set()

  def start(self, timeout_s: float = 45.0) -> None:
    for worker in self._workers:
      worker.start()
    deadline = time.monotonic() + timeout_s
    for worker in self._workers:
      remaining = deadline - time.monotonic()
      if remaining <= 0.0 or not worker.ready.wait(remaining):
        self.close()
        raise RuntimeError("timed out opening/warming the ZED cameras")
    if self._errors:
      self.close()
      raise RuntimeError("; ".join(self._errors))

  def begin_recording(
    self,
    callback: Callable[[CaptureEvent], None],
    error_callback: Callable[[str], None],
  ) -> None:
    """Start emitting events after camera warm-up and arm initialization."""
    if self._errors or self._stop.is_set():
      raise RuntimeError("camera capture stopped before recording began")
    self._error_callback = error_callback
    self._broker.set_callback(callback, reset_counts=True)

  def end_recording(self) -> dict[str, object]:
    """Atomically return to warm/discard mode without stopping the cameras."""
    self._broker.set_callback(lambda event: None)
    self._error_callback = lambda message: None
    return self.status()

  def close(self) -> None:
    self._stop.set()
    for worker in self._workers:
      if worker.thread.is_alive():
        worker.thread.join(timeout=5.0)
    if any(worker.thread.is_alive() for worker in self._workers):
      self._fatal("camera thread did not stop; capture incomplete")
    else:
      self._broker.close()

  def status(self) -> dict[str, object]:
    return {
      "frames": {worker.side: worker.frames for worker in self._workers},
      "measured_fps": {worker.side: (worker.frames - 1) * 1e9 / (worker.last_timestamp - worker.first_timestamp)
                       if worker.last_timestamp > worker.first_timestamp else 0.0 for worker in self._workers},
      "long_intervals": {worker.side: worker.long_intervals for worker in self._workers},
      "grab_errors": {worker.side: worker.grab_errors for worker in self._workers},
      "pairs": self._broker.pairs,
      "left_orphans": self._broker.left_orphans,
      "right_orphans": self._broker.right_orphans,
      "orphans": dict(self._broker.orphans),
      "errors": list(self._errors),
    }


def main() -> None:
  parser = argparse.ArgumentParser(description="Soak-test and bundle the configured ZED cameras")
  parser.add_argument("--left-serial", type=int, default=301058360)
  parser.add_argument("--right-serial", type=int, default=306353224)
  parser.add_argument("--overhead-serial", type=int, default=41925345)
  parser.add_argument("--overhead-view", choices=("left", "right"), default="left")
  parser.add_argument("--frames", type=int, default=300)
  parser.add_argument("--warmup-frames", type=int, default=60)
  args = parser.parse_args()
  done = threading.Event()
  pair_skews = []
  pair_count = 0

  def receive(event: CaptureEvent) -> None:
    nonlocal pair_count
    if event.kind == "pair":
      pair_count += 1
      pair_skews.append(event.pair_skew_ns / 1e6)
    if pair_count >= args.frames:
      done.set()

  capture = ZedPairCapture(
    left_serial=args.left_serial,
    right_serial=args.right_serial,
    overhead_serial=args.overhead_serial,
    overhead_view=args.overhead_view,
    fps=30,
    warmup_frames=args.warmup_frames,
    pair_tolerance_ms=12.0,
    pending_frames=4,
    callback=receive,
  )
  capture.start()
  deadline = time.monotonic() + args.frames / 30 * 3 + 10
  try:
    while not done.wait(0.5):
      print(capture.status(), flush=True)
      if time.monotonic() > deadline:
        raise RuntimeError("not enough valid camera pairs before deadline; inspect timestamp pairing")
      if capture.status()["errors"]:
        raise RuntimeError(str(capture.status()["errors"]))
  finally:
    capture.close()
  print(capture.status())
  if pair_skews:
    print(f"pair skew ms: median={float(np.median(pair_skews)):.3f} max={max(pair_skews):.3f}")


if __name__ == "__main__":
  main()
