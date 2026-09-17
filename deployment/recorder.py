"""Record teleop demonstration data to HDF5.

Each demonstration is one HDF5 file containing synchronized, per-tick data
from every arm and controller at the teleop loop rate (~100 Hz).

Datasets per arm (keyed by hand: "right" or "left"):

  /right/joint_position          (N, 6)   measured joint angles [rad]
  /right/joint_velocity          (N, 6)   measured joint velocities [rad/s]
  /right/joint_target            (N, 6)   commanded joint targets [rad]
  /right/gripper_position        (N,)     measured gripper opening [0-1]
  /right/gripper_command         (N,)     commanded gripper target [0-1]
  /right/ee_position             (N, 3)   end-effector position [m]
  /right/ee_quaternion           (N, 4)   end-effector orientation [xyzw]
  /right/controller_position     (N, 3)   Quest controller position [m]
  /right/controller_quaternion   (N, 4)   Quest controller orientation [xyzw]
  /right/controller_trigger      (N,)     trigger value [0-1]
  /right/controller_grip         (N,)     grip value [0-1]
  /right/controller_clutch       (N,)     clutch engaged [bool as 0/1]

Scalar / session-level:

  /timestamps                    (N,)     seconds since the first tick
  /mode                          (N,)     session mode per tick

Writing is split from recording on purpose. ``tick`` appends to plain lists and
is cheap enough for the control loop; ``detach`` swaps those lists out in O(1)
and hands back a `DemoSnapshot`; `write_snapshot` turns one into an HDF5 file
and takes tens to hundreds of milliseconds, so it must NOT run on the control
thread -- a stalled loop stops commanding the arms and can trip the
"arm is not following its target" check on the next tick.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class _ArmBuffer:
    joint_position: list[np.ndarray] = field(default_factory=list)
    joint_velocity: list[np.ndarray] = field(default_factory=list)
    joint_target: list[np.ndarray] = field(default_factory=list)
    gripper_position: list[float] = field(default_factory=list)
    gripper_command: list[float] = field(default_factory=list)
    ee_position: list[np.ndarray] = field(default_factory=list)
    ee_quaternion: list[np.ndarray] = field(default_factory=list)
    controller_position: list[np.ndarray] = field(default_factory=list)
    controller_quaternion: list[np.ndarray] = field(default_factory=list)
    controller_trigger: list[float] = field(default_factory=list)
    controller_grip: list[float] = field(default_factory=list)
    controller_clutch: list[int] = field(default_factory=list)


@dataclass
class DemoSnapshot:
    """One finished recording, detached from the recorder and safe to write
    from any thread: it owns its buffers and shares nothing with live state."""

    hands: list[str]
    hz: float
    arms: dict[str, _ArmBuffer]
    timestamps: list[float]
    modes: list[str]
    start_wall: str
    camera_frames: dict[str, list[bytes]] = field(default_factory=dict)
    camera_resolutions: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def num_ticks(self) -> int:
        return len(self.timestamps)

    @property
    def duration_s(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]


class DemoRecorder:
    def __init__(self, hands: list[str], hz: float = 100.0, camera_names: list[str] | None = None):
        self._hands = list(hands)
        self._hz = hz
        self._arms: dict[str, _ArmBuffer] = {h: _ArmBuffer() for h in hands}
        self._timestamps: list[float] = []
        self._modes: list[str] = []
        self._start_wall: str | None = None
        self._recording = False
        self._camera_names = list(camera_names or [])
        self._camera_frames: dict[str, list[bytes]] = {n: [] for n in self._camera_names}
        self._camera_resolutions: dict[str, tuple[int, int]] = {}
        self._last_camera_seq: dict[str, int] = {n: -1 for n in self._camera_names}

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def tick_count(self) -> int:
        return len(self._timestamps)

    def start(self) -> None:
        self.reset()
        self._recording = True
        self._start_wall = time.strftime("%Y-%m-%dT%H:%M:%S")

    def stop(self) -> None:
        self._recording = False

    def tick_cameras(self, cameras: dict) -> None:
        """Grab the latest frame from each camera and JPEG-encode it.

        Called once per control tick. Skips cameras whose seq hasn't advanced
        (the camera runs slower than the control loop). The recorder stores
        one frame per tick at most; if the camera is slower we repeat the
        previous frame to keep rows aligned with timestamps.
        """
        if not self._recording:
            return
        import cv2
        for name in self._camera_names:
            cam = cameras.get(name)
            if cam is None:
                self._camera_frames[name].append(b"")
                continue
            frame = cam.latest()
            if frame is None:
                self._camera_frames[name].append(b"")
                continue
            if name not in self._camera_resolutions:
                self._camera_resolutions[name] = cam.resolution
            if frame.seq == self._last_camera_seq.get(name, -1):
                prev = self._camera_frames[name]
                self._camera_frames[name].append(prev[-1] if prev else b"")
                continue
            self._last_camera_seq[name] = frame.seq
            ok, buf = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self._camera_frames[name].append(buf.tobytes() if ok else b"")

    def tick(
        self,
        states: list,
        targets: list[np.ndarray],
        samples: dict,
        channels: list,
        mode: str,
        now: float,
    ) -> None:
        if not self._recording:
            return
        self._timestamps.append(now)
        self._modes.append(mode)

        for channel, state, target in zip(channels, states, targets):
            buf = self._arms.get(channel.hand)
            if buf is None:
                continue

            buf.joint_position.append(state.joint_position.copy())
            buf.joint_velocity.append(state.joint_velocity.copy())
            buf.joint_target.append(np.asarray(target, dtype=np.float64).copy())
            buf.gripper_position.append(
                float(state.gripper_position) if state.gripper_position is not None else 0.0
            )
            buf.gripper_command.append(float(channel._gripper_command))

            ee_pos, ee_quat = channel._ik.ee_pose(state)
            buf.ee_position.append(np.asarray(ee_pos).copy())
            buf.ee_quaternion.append(np.asarray(ee_quat).copy())

            sample = samples.get(channel.hand)
            if sample is not None:
                buf.controller_position.append(sample.position.copy())
                buf.controller_quaternion.append(sample.quat.copy())
                buf.controller_trigger.append(float(sample.trigger))
                buf.controller_grip.append(float(sample.grip))
                buf.controller_clutch.append(int(sample.clutch))
            else:
                # The arm still moved this tick, so the row must exist; a lost
                # controller reads as a neutral pose with nothing pressed.
                buf.controller_position.append(np.zeros(3))
                buf.controller_quaternion.append(np.array([0.0, 0.0, 0.0, 1.0]))
                buf.controller_trigger.append(0.0)
                buf.controller_grip.append(0.0)
                buf.controller_clutch.append(0)

    def detach(self) -> DemoSnapshot | None:
        """Stop recording and hand the buffers off. O(1); safe in the loop.

        Returns None when nothing was captured. Call `write_snapshot` on the
        result from a thread that is not driving the arms.
        """
        self._recording = False
        if not self._timestamps:
            self.reset()
            return None
        snapshot = DemoSnapshot(
            hands=list(self._hands),
            hz=self._hz,
            arms=self._arms,
            timestamps=self._timestamps,
            modes=self._modes,
            start_wall=self._start_wall or "",
            camera_frames=self._camera_frames,
            camera_resolutions=dict(self._camera_resolutions),
        )
        # Fresh containers: the snapshot owns the old ones from here.
        self._arms = {h: _ArmBuffer() for h in self._hands}
        self._timestamps = []
        self._modes = []
        self._start_wall = None
        self._camera_frames = {n: [] for n in self._camera_names}
        self._last_camera_seq = {n: -1 for n in self._camera_names}
        return snapshot

    def save(self, demo_dir: str | Path) -> Path:
        """Detach and write in one step. Convenience for offline use; the
        teleop session detaches and writes separately so the loop never waits."""
        snapshot = self.detach()
        if snapshot is None:
            raise ValueError("No data recorded")
        return write_snapshot(snapshot, demo_dir)

    def reset(self) -> None:
        self._arms = {h: _ArmBuffer() for h in self._hands}
        self._timestamps = []
        self._modes = []
        self._start_wall = None
        self._recording = False
        self._camera_frames = {n: [] for n in self._camera_names}
        self._last_camera_seq = {n: -1 for n in self._camera_names}


def write_snapshot(snapshot: DemoSnapshot, demo_dir: str | Path) -> Path:
    """Write one snapshot to a new HDF5 file. Tens to hundreds of ms."""
    import h5py

    demo_dir = Path(demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)

    n = snapshot.num_ticks
    if n == 0:
        raise ValueError("No data recorded")

    base = f"demo_{time.strftime('%Y%m%d_%H%M%S')}"
    path = demo_dir / f"{base}.hdf5"
    suffix = 1
    while path.exists():
        path = demo_dir / f"{base}_{suffix}.hdf5"
        suffix += 1

    with h5py.File(path, "w") as f:
        f.attrs["start_time"] = snapshot.start_wall
        f.attrs["hz"] = snapshot.hz
        f.attrs["arms"] = snapshot.hands
        f.attrs["num_ticks"] = n
        f.attrs["duration_s"] = round(snapshot.duration_s, 3)

        t0 = snapshot.timestamps[0]
        f.create_dataset(
            "timestamps", data=np.array([t - t0 for t in snapshot.timestamps], dtype=np.float64)
        )
        f.create_dataset("mode", data=[m.encode("ascii") for m in snapshot.modes])

        for hand in snapshot.hands:
            buf = snapshot.arms[hand]
            g = f.create_group(hand)
            g.create_dataset("joint_position", data=np.array(buf.joint_position, dtype=np.float32))
            g.create_dataset("joint_velocity", data=np.array(buf.joint_velocity, dtype=np.float32))
            g.create_dataset("joint_target", data=np.array(buf.joint_target, dtype=np.float32))
            g.create_dataset("gripper_position", data=np.array(buf.gripper_position, dtype=np.float32))
            g.create_dataset("gripper_command", data=np.array(buf.gripper_command, dtype=np.float32))
            g.create_dataset("ee_position", data=np.array(buf.ee_position, dtype=np.float32))
            g.create_dataset("ee_quaternion", data=np.array(buf.ee_quaternion, dtype=np.float32))
            g.create_dataset("controller_position", data=np.array(buf.controller_position, dtype=np.float32))
            g.create_dataset("controller_quaternion", data=np.array(buf.controller_quaternion, dtype=np.float32))
            g.create_dataset("controller_trigger", data=np.array(buf.controller_trigger, dtype=np.float32))
            g.create_dataset("controller_grip", data=np.array(buf.controller_grip, dtype=np.float32))
            g.create_dataset("controller_clutch", data=np.array(buf.controller_clutch, dtype=np.int8))

        for cam_name, frames in snapshot.camera_frames.items():
            if not frames:
                continue
            vlen_dt = h5py.special_dtype(vlen=np.uint8)
            g = f.create_group(f"cameras/{cam_name}")
            ds = g.create_dataset("frames", shape=(len(frames),), dtype=vlen_dt)
            for i, jpg in enumerate(frames):
                if jpg:
                    ds[i] = np.frombuffer(jpg, dtype=np.uint8)
                else:
                    ds[i] = np.zeros(0, dtype=np.uint8)
            res = snapshot.camera_resolutions.get(cam_name, (0, 0))
            g.attrs["width"] = res[0]
            g.attrs["height"] = res[1]
            g.attrs["codec"] = "jpeg"

    return path
