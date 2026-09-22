"""Bounded asynchronous raw recording. No robot or SDK imports.

The timestamp of a control sample is the earliest arm read start: a causal
match therefore never selects a state read before either camera timestamp.
Encoder cache age is unknown and is explicitly not an exposure-time guarantee.
"""
from __future__ import annotations

import csv
import json
import os
import queue
import re
import shutil
import threading
import time
import unicodedata
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from deployment.zed_capture import CaptureEvent


def task_slug(task: str, max_length: int = 64) -> str:
    """Stable, shell-friendly folder prefix while metadata keeps the exact task."""
    ascii_task = unicodedata.normalize("NFKD", task).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_task.lower()).strip("_")
    return (slug[:max_length].rstrip("_") or "task")


@dataclass(frozen=True)
class ControlSample:
    tick: int
    timestamp_ns: int
    state: tuple[float, ...]
    action: tuple[float, ...]
    mode: str
    read_intervals: tuple[tuple[int, int], ...]
    command_intervals: tuple[tuple[int, int], ...]
    quest_received_ns: tuple[int, ...]
    valid: bool
    reason: str = ""

    def usable(self):
        return (self.valid and len(self.state) == len(self.action) == 14
                and np.isfinite(self.state + self.action).all())


class EpisodeRecorder:
    def __init__(self, root, *, task, metadata=None, max_skew_ms=20.,
                 queue_size=8, min_free_gb=2., jpeg_quality=95, max_invalid_per_window=5,
                 camera_sides=("left", "right")):
        import cv2  # Fail before opening cameras or moving arms.
        if not task.strip() or max_skew_ms <= 0 or queue_size < 1:
            raise ValueError("task, positive skew and queue size required")
        self.camera_sides = tuple(camera_sides)
        if (len(self.camera_sides) < 2 or len(set(self.camera_sides)) != len(self.camera_sides)
                or not set(self.camera_sides) <= {"left", "right", "overhead"}):
            raise ValueError("camera_sides must contain two or three unique supported views")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.min_free = int(min_free_gb * 1e9)
        if shutil.disk_usage(self.root).free < self.min_free:
            raise RuntimeError("insufficient recording disk space")
        name = task_slug(task) + "_" + time.strftime("%Y-%m-%d_at_%I-%M-%S%p").lower()
        self.path = self.root / (name + ".partial")
        if self.path.exists():
            name += "_" + str(time.time_ns() % 1_000_000)
            self.path = self.root / (name + ".partial")
        self.path.mkdir()
        for side in self.camera_sides:
            (self.path / "images" / side).mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata or {}, task=task, fps=30, schema_version=2,
                             color_order="RGB", state_order="left arm(6), left gripper, right arm(6), right gripper",
                             arm_units="radians", gripper_units="driver units; verify on hardware",
                             encoder_cache_age="unknown", camera_sides=list(self.camera_sides),
                             status="recording", errors=[])
        self.max_skew_ns = int(max_skew_ms * 1e6)
        self.quality = jpeg_quality
        self.max_invalid_per_window = max_invalid_per_window
        self._recent_invalid = deque(maxlen=30)
        self._seen_valid = False
        self._counts = {"captures": 0, "valid": 0, "orphans": 0, "unmatched": 0}
        self._max_queue = 0
        self._history = deque(maxlen=1000)
        self._condition = threading.Condition()
        self._frames = queue.Queue(queue_size)
        self._controls = queue.Queue(2000)
        self._stop = threading.Event()
        self._controls_finished = threading.Event()
        self._errors_lock = threading.Lock()
        self._last_tick = -1
        self._last_time = -1
        self._started_monotonic = time.monotonic()
        self._write_metadata()
        self._thread = threading.Thread(target=self._run, name="episode-writer", daemon=True)
        self._thread.start()

    def _write_metadata(self):
        tmp = self.path / "metadata.json.tmp"
        tmp.write_text(json.dumps(self.metadata, indent=2) + "\n")
        tmp.replace(self.path / "metadata.json")

    def fail(self, reason):
        with self._errors_lock:
            if reason not in self.metadata["errors"]:
                self.metadata["errors"].append(str(reason))
                self.metadata["status"] = "invalid"
                print(f"\nRECORDING INVALID: {reason}", flush=True)
                try:
                    self._write_metadata()
                except Exception:
                    pass
                self._stop.set()

    def add_control(self, sample):
        if self._stop.is_set():
            return
        with self._condition:
            if sample.timestamp_ns <= self._last_time or sample.tick <= self._last_tick:
                self.fail("control timestamps or tick IDs moved backwards")
                return
            self._last_time, self._last_tick = sample.timestamp_ns, sample.tick
            self._history.append(sample)
            self._condition.notify_all()
        try:
            self._controls.put_nowait(sample)
        except queue.Full:
            self.fail("raw control queue overflow; raw stream incomplete")

    def add_capture(self, event):
        if self._stop.is_set():
            return
        try:
            self._frames.put_nowait(event)
            self._max_queue = max(self._max_queue, self._frames.qsize())
        except queue.Full:
            self.fail("image writer queue overflow; raw capture incomplete")

    def _match(self, event):
        deadline = time.monotonic() + .15
        with self._condition:
            while True:
                for sample in self._history:
                    delta = sample.timestamp_ns - event.image_monotonic_ns
                    if delta > self.max_skew_ns:
                        return None
                    if delta >= 0 and sample.usable():
                        return sample
                if self._stop.is_set() or self._controls_finished.is_set() or time.monotonic() >= deadline:
                    return None
                self._condition.wait(.01)

    def _run(self):
        import cv2
        fields = ["capture_index", "kind"] + [f"{side}_image" for side in self.camera_sides] + ["valid", "reason",
                  "image_timestamp_ns", "control_tick", "control_timestamp_ns", "skew_ns", "pair_skew_ns"]
        fields += [f"{kind}_{i}" for kind in ("state", "action") for i in range(14)]
        try:
            with (self.path / "data.csv").open("w", newline="") as data, \
                 (self.path / "debug_timing.csv").open("w", newline="") as timing, \
                 (self.path / "controls.jsonl").open("w") as controls:
                writer = csv.DictWriter(data, fieldnames=fields)
                writer.writeheader()
                debug = csv.writer(timing)
                debug.writerow(["capture_index", "side", "sequence", "serial", "zed_timestamp_ns",
                                "host_epoch_ns", "host_monotonic_ns", "image_monotonic_ns",
                                "grab_start_ns", "grab_end_ns", "retrieve_end_ns"])
                while not self._stop.is_set() or not self._frames.empty() or not self._controls.empty():
                    # Bound each drain so camera encoding cannot starve.
                    for _ in range(200):
                        try:
                            sample = self._controls.get_nowait()
                        except queue.Empty:
                            break
                        controls.write(json.dumps(asdict(sample), allow_nan=False) + "\n")
                    try:
                        event = self._frames.get(timeout=.02)
                    except queue.Empty:
                        for file in (data, timing, controls):
                            file.flush()
                        continue
                    if event.index % 30 == 0 and shutil.disk_usage(self.path).free < self.min_free:
                        raise RuntimeError("recording disk below free-space threshold")
                    complete = event.kind == "pair" and all(side in event.frames for side in self.camera_sides)
                    sample = self._match(event) if complete else None
                    row = dict(capture_index=event.index, kind=event.kind,
                               valid=int(sample is not None), reason=event.reason or ("" if sample else "no_valid_causal_control"),
                               image_timestamp_ns=event.image_monotonic_ns, pair_skew_ns=event.pair_skew_ns)
                    for side in self.camera_sides:
                        frame = event.frames.get(side)
                        if frame is None:
                            continue
                        relative = Path("images") / side / f"{frame.sequence:09d}_{event.index:09d}.jpg"
                        image_path = self.path / relative
                        if not cv2.imwrite(str(image_path), frame.image_rgb[..., ::-1],
                                           [cv2.IMWRITE_JPEG_QUALITY, self.quality]):
                            raise RuntimeError(f"image write failed: {image_path}")
                        row[side + "_image"] = str(relative)
                        debug.writerow([event.index, side, frame.sequence, frame.serial, frame.zed_timestamp_ns,
                                        frame.host_epoch_ns, frame.host_monotonic_ns, frame.image_monotonic_ns,
                                        frame.grab_start_ns, frame.grab_end_ns, frame.retrieve_end_ns])
                    if sample:
                        row.update(control_tick=sample.tick, control_timestamp_ns=sample.timestamp_ns,
                                   skew_ns=sample.timestamp_ns-event.image_monotonic_ns)
                        for kind in ("state", "action"):
                            row.update({f"{kind}_{i}": v for i, v in enumerate(getattr(sample, kind))})
                    self._counts["captures"] += 1
                    self._counts["valid"] += int(sample is not None)
                    self._counts["orphans"] += int(event.kind != "pair")
                    self._counts["unmatched"] += int(event.kind == "pair" and sample is None)
                    writer.writerow(row)
                    self._seen_valid = self._seen_valid or sample is not None
                    with self._condition:
                        active = bool(self._history and self._history[-1].mode in ("idle", "engaged"))
                    if self._seen_valid and active:
                        self._recent_invalid.append(sample is None)
                        if sum(self._recent_invalid) >= self.max_invalid_per_window:
                            self.fail("repeated invalid captures during active teleoperation")
                    if event.index % 30 == 0:
                        for file in (data, timing, controls):
                            file.flush()
                for file in (data, timing, controls):
                    file.flush()
                    os.fsync(file.fileno())
        except Exception as exc:
            self.fail(f"writer failed: {exc}")
            self._stop.set()

    def finish_controls(self):
        """Do not wait for future ticks after the robot loop has stopped."""
        self._controls_finished.set()
        with self._condition:
            self._condition.notify_all()

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            self.fail("writer failed to stop")
            return self.path
        self.metadata["counts"] = self._counts
        self.metadata["max_writer_queue"] = self._max_queue
        self.metadata["duration_s"] = round(time.monotonic() - self._started_monotonic, 3)
        self.metadata["status"] = "invalid" if self.metadata["errors"] else "complete"
        self._write_metadata()
        if not self.metadata["errors"]:
            destination = self.path.with_suffix("")
            self.path.rename(destination)
            self.path = destination
        return self.path
