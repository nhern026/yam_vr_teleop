"""Meta Quest teleoperation of the YAM arm over USB.

The headset runs the `oculus_reader` teleop app (`com.rail.oculus.teleop`),
which logs both controllers' 4x4 poses and button states at ~70 Hz. This
module reads them straight off `adb logcat` over the USB cable -- no browser,
no network, no TLS, no firewall -- and turns them into safe joint targets.

Control path, per tick, at ``teleop.control_hz``:

1. Read the measured arm state and trip on a runaway (measured joint speed
   against `safety.max_joint_velocity_rad_s`).
2. Smooth the raw controller pose with a One Euro filter -- the standard VR
   input smoother: heavy on a resting hand's jitter, near-transparent on a
   moving hand -- and declutch outright if the pose teleports (a tracking
   glitch must never be chased).
3. While the clutch (index trigger or grip) is held, the filtered motion
   since the moment it was pressed is mapped into the world frame and added
   to the end-effector pose measured at that same moment. Releasing the
   trigger freezes the arm and re-anchors on the next press.
4. A wide slew cap (`teleop.max_tray_speed_m_s`, `teleop.max_tray_omega_rad_s`)
   bounds the commanded pose as a final guard; smoothness comes from the
   filter, not from this.
5. `TeleopIK` -- i2rt's own YAM model, its `grasp_site`, one damped `mink` QP,
   exactly the stack `i2rt.robots.kinematics.Kinematics` uses -- turns that
   pose into a joint target inside the safety envelope (command offset, the
   operator's joint box intersected with the arm's mechanical limits).

Startup holds the current position. B/Y pauses both arms; A/X resumes after
a pause. Faults latch until restart. No automatic homing or parking is used.
Exiting closes the motor drivers: support the arms before Ctrl-C.

`--dump` prints the controller stream and touches no hardware at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mink
import mujoco
import numpy as np
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

from balancing_act.assets import YAM_ARM_HOME_JOINT_POS, YAM_ARM_HOME_VERIFIED
from deployment.camera import OneEuroFilter
from deployment.config import load_deployment_config, validate_config
from deployment.dashboard import Dashboard
from deployment.recorder import DemoRecorder, DemoSnapshot, write_snapshot
from deployment.robot import ArmState, make_arm

# The headset app, its log tag, and where to get it. The app is `oculus_reader`'s
# (rail-berkeley); the binary is the Quest-3 maintained fork's build, which is
# distributed only through git-lfs, hence the media URL and the hash pin.
QUEST_PACKAGE = "com.rail.oculus.teleop"
QUEST_ACTIVITY = f"{QUEST_PACKAGE}/{QUEST_PACKAGE}.MainActivity"
QUEST_LOG_TAG = "wE9ryARX"
QUEST_APK_URL = (
  "https://media.githubusercontent.com/media/jborbik/oculus_reader/main/"
  "oculus_reader/APK/teleop-debug.apk"
)
QUEST_APK_SHA256 = "6ddd90d8bced3a9533ae36099c950238fdb3ffd5feb5a6daa0afd63ac484cdb0"

# Modes. `homing` walks to the training reset pose at start-up, `returning`
# walks to the folded home pose on the way out; both are smoothstep moves the
# operator cannot interrupt with the clutch.
HOMING = "homing"
IDLE = "idle"
ENGAGED = "engaged"
RETURNING = "returning"
PARKED = "parked"
FAULT = "fault"

# The app reports controller poses in the headset's tracking frame: right
# handed, +x to the operator's right, +y up, +z toward the operator. The
# deployment world is MuJoCo's z-up frame with +x pointing away from the arm's
# base. This is the rotation between them for an operator standing behind the
# arm and facing the same way it does; `teleop.yaw_deg` rotates it about world
# +z for any other standing position.
_HEADSET_TO_WORLD = np.array(
  [
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
  ],
  dtype=np.float64,
)

# Per-hand button names in the app's log format. Bare markers appear only while
# pressed; the analogue axes are always reported.
_BUTTONS = {
  "right": {"home": "A", "stop": "B", "trigger": "rightTrig", "grip": "rightGrip"},
  "left": {"home": "X", "stop": "Y", "trigger": "leftTrig", "grip": "leftGrip"},
}


def frame_rotation(yaw_deg: float) -> np.ndarray:
  """Headset-to-world rotation for an operator standing at ``yaw_deg``."""
  yaw = np.deg2rad(float(yaw_deg))
  about_z = np.array(
    [
      [np.cos(yaw), -np.sin(yaw), 0.0],
      [np.sin(yaw), np.cos(yaw), 0.0],
      [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
  )
  return about_z @ _HEADSET_TO_WORLD


# The headset's own up axis. The tracking frame is y-up and its yaw is set
# wherever the headset happened to be looking when tracking was established,
# which is exactly the part a fixed matrix cannot know.
_HEADSET_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def frame_from_push(push: np.ndarray) -> np.ndarray:
  """Headset-to-world rotation measured from one push toward the arm.

  ``push`` is the controller displacement, in the headset's tracking frame,
  of a motion the operator made straight away from themselves toward the arm.
  Its horizontal part becomes world ``+x`` (away from the arm's base) and the
  headset's up axis becomes world ``+z``, which pins the one degree of freedom
  -- the operator's heading in the tracking frame -- that `frame_rotation`
  can only assume. With a push along ``-z`` this reproduces `frame_rotation(0)`
  exactly.
  """
  forward = np.asarray(push, dtype=np.float64) - _HEADSET_UP * float(
    np.dot(push, _HEADSET_UP)
  )
  length = float(np.linalg.norm(forward))
  if length < 1.0e-6:
    raise ValueError("the calibration push has no horizontal component")
  forward = forward / length
  right = np.cross(_HEADSET_UP, forward)
  return np.stack([forward, right, _HEADSET_UP])


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
  mat = np.zeros(9, dtype=np.float64)
  mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
  return mat.reshape(3, 3)


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
  quat = np.zeros(4, dtype=np.float64)
  mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=np.float64).reshape(9))
  return quat


def smoothstep(fraction: float) -> float:
  f = min(max(fraction, 0.0), 1.0)
  return f * f * (3.0 - 2.0 * f)


@dataclass
class ControllerSample:
  """One controller as the headset last reported it."""

  hand: str
  position: np.ndarray
  quat: np.ndarray
  trigger: float
  grip: float
  clutch: bool
  home: bool
  stop: bool
  joystick: tuple[float, float]
  received_s: float
  buttons: dict[str, Any] = field(default_factory=dict)

  def describe(self) -> str:
    p = self.position
    q = self.quat
    return (
      f"{self.hand:>5}: p=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})m "
      f"q=({q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f}) "
      f"trig={self.trigger:.2f} grip={self.grip:.2f} "
      f"js=({self.joystick[0]:+.2f},{self.joystick[1]:+.2f}) "
      f"clutch={int(self.clutch)} home={int(self.home)} stop={int(self.stop)}"
    )


def parse_buttons(text: str) -> dict[str, Any]:
  """The app's button field: bare pressed markers plus `name x [y]` axes."""
  buttons: dict[str, Any] = {}
  for item in text.split(","):
    item = item.strip()
    if not item:
      continue
    parts = item.split()
    if len(parts) == 1:
      buttons[parts[0]] = True
      continue
    try:
      values = tuple(float(value) for value in parts[1:])
    except ValueError:
      continue
    if not all(np.isfinite(value) for value in values):
      continue
    buttons[parts[0]] = values[0] if len(values) == 1 else values
  return buttons


def parse_log_payload(payload: str, now: float) -> dict[str, ControllerSample]:
  """One log line's payload into per-hand samples.

  Format: ``l:<16 floats>|r:<16 floats>&<buttons>``. The 16 floats are a
  row-major 4x4 pose, so the rotation is the upper-left 3x3 and the position
  the last column.
  """
  transforms_text, _, buttons_text = payload.partition("&")
  buttons = parse_buttons(buttons_text)
  samples: dict[str, ControllerSample] = {}
  for chunk in transforms_text.split("|"):
    key, _, values_text = chunk.strip().partition(":")
    hand = {"l": "left", "r": "right"}.get(key.strip())
    if hand is None:
      continue
    try:
      values = np.asarray([float(value) for value in values_text.split()])
    except ValueError:
      continue
    if values.size != 16 or not np.all(np.isfinite(values)):
      continue
    matrix = values.reshape(4, 4)
    rotation = matrix[:3, :3]
    # A dropped or half-flushed line can still parse as 16 numbers; a pose
    # whose rotation block is not a rotation is not usable as one.
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-3):
      continue
    names = _BUTTONS[hand]
    try:
      trigger = float(buttons.get(names["trigger"], 0.0))
      grip = float(buttons.get(names["grip"], 0.0))
    except (TypeError, ValueError):
      continue
    if not (np.isfinite(trigger) and np.isfinite(grip) and 0 <= trigger <= 1 and 0 <= grip <= 1):
      continue
    joystick = buttons.get("rightJS" if hand == "right" else "leftJS", (0.0, 0.0))
    if not isinstance(joystick, tuple) or len(joystick) != 2:
      continue
    samples[hand] = ControllerSample(
      hand=hand,
      position=matrix[:3, 3].copy(),
      quat=mat_to_quat(rotation),
      trigger=trigger,
      grip=grip,
      # Either trigger engages the clutch, so the operator can use whichever
      # is comfortable. The analogue value is what the SDK always reports.
      clutch=trigger > 0.5 or grip > 0.5,
      home=bool(buttons.get(names["home"], False)),
      stop=bool(buttons.get(names["stop"], False)),
      joystick=(float(joystick[0]), float(joystick[1])),
      received_s=now,
      buttons=buttons,
    )
  return samples


class QuestReader:
  """Controller poses off `adb logcat`, with the headset app's lifecycle."""

  def __init__(self, *, serial: str | None = None, apk_path: Path | None = None):
    self._adb = shutil.which("adb")
    if self._adb is None:
      raise RuntimeError("adb is not on PATH; install Android platform-tools")
    self._serial = serial
    self._apk_path = apk_path
    self._lock = threading.Lock()
    self._samples: dict[str, ControllerSample] = {}
    self._lines = 0
    self._process: subprocess.Popen[str] | None = None
    self._thread: threading.Thread | None = None
    self._stop = threading.Event()
    self._clock_offset = 0.0
    self._clock_uncertainty = 0.0

  # ----------------------------------------------------------------- adb glue

  def _run(self, *args: str, timeout: float = 30.0) -> str:
    command = [self._adb]
    if self._serial:
      command += ["-s", self._serial]
    result = subprocess.run(
      [*command, *args], capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode != 0:
      raise RuntimeError(f"adb {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout

  def device_serial(self) -> str:
    """The one attached device, with the usual failure modes named."""
    devices = {}
    for line in self._run("devices").splitlines()[1:]:
      fields = line.split(maxsplit=1)
      if len(fields) == 2:
        devices[fields[0]] = fields[1].strip()
    if self._serial is None:
      if len(devices) != 1:
        raise RuntimeError("Connect exactly one Quest, or pass --serial from adb devices")
      self._serial = next(iter(devices))
    state = devices.get(self._serial)
    if state != "device":
      raise RuntimeError(f"Quest {self._serial}: {state or 'not connected'}; check USB debugging authorization")
    return self._serial

  def ensure_app(self) -> str:
    """Install the headset app if it is missing, then launch it."""
    installed = QUEST_PACKAGE in self._run("shell", "pm", "list", "packages", QUEST_PACKAGE)
    if not installed:
      apk = self._fetch_apk()
      self._run("install", "-r", "-t", str(apk), timeout=300.0)
      if QUEST_PACKAGE not in self._run("shell", "pm", "list", "packages", QUEST_PACKAGE):
        raise RuntimeError(f"Installing {QUEST_PACKAGE} did not take")
    self._run(
      "shell",
      "am",
      "start",
      "-n",
      QUEST_ACTIVITY,
      "-a",
      "android.intent.action.MAIN",
      "-c",
      "android.intent.category.LAUNCHER",
    )
    return "installed and launched" if not installed else "launched"

  def _fetch_apk(self) -> Path:
    if self._apk_path is None:
      raise RuntimeError(f"{QUEST_PACKAGE} is not installed and no apk_path is configured")
    apk = self._apk_path
    if apk.exists() and hashlib.sha256(apk.read_bytes()).hexdigest() == QUEST_APK_SHA256:
      return apk
    apk.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(QUEST_APK_URL, timeout=120) as response:
      payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != QUEST_APK_SHA256:
      raise RuntimeError(f"Downloaded apk hashes {digest}, expected {QUEST_APK_SHA256}")
    apk.write_bytes(payload)
    return apk

  # -------------------------------------------------------------------- stream

  def start(self) -> None:
    # Map Android log timestamps to this process's monotonic clock. Old buffered
    # lines must retain their age instead of becoming "fresh" when read.
    estimates = []
    for _ in range(3):
      before = time.monotonic()
      device_time = float(self._run("shell", "date", "+%s.%N").strip())
      after = time.monotonic()
      if not np.isfinite(device_time):
        raise RuntimeError("Invalid Quest clock")
      estimates.append((after-before, (before+after)/2-device_time))
    duration, self._clock_offset = min(estimates)
    self._clock_uncertainty = duration / 2
    if duration > 0.1:
      raise RuntimeError("Quest USB clock check too slow; reconnect USB before teleop")
    command = [self._adb]
    if self._serial:
      command += ["-s", self._serial]
    # `-T 1` starts one line back rather than replaying the whole buffer.
    self._process = subprocess.Popen(
      [*command, "logcat", "-v", "epoch", "-T", "1", "-s", f"{QUEST_LOG_TAG}:V", "*:S"],
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
      text=True,
      bufsize=1,
    )
    self._thread = threading.Thread(target=self._read, name="quest-logcat", daemon=True)
    self._thread.start()

  def _read(self) -> None:
    assert self._process is not None and self._process.stdout is not None
    for line in self._process.stdout:
      if self._stop.is_set():
        break
      samples = self._parse_line(line, time.monotonic())
      if not samples:
        continue
      with self._lock:
        self._samples = samples
        self._lines += 1

  def _parse_line(self, line: str, now: float) -> dict[str, ControllerSample]:
    match = re.match(r"^\s*(\d+\.\d+)\s+\d+\s+\d+\s+[A-Z]\s+" + QUEST_LOG_TAG + r"\s*:\s*(.*)$", line)
    if match is None:
      return {}
    stamp = float(match.group(1)) + self._clock_offset - self._clock_uncertainty
    if not 0 <= now - stamp <= 0.15:
      return {}
    return parse_log_payload(match.group(2).strip(), stamp)

  def samples(self) -> dict[str, ControllerSample]:
    with self._lock:
      return dict(self._samples)

  def sample(self, hand: str) -> ControllerSample | None:
    with self._lock:
      return self._samples.get(hand)

  @property
  def lines(self) -> int:
    with self._lock:
      return self._lines

  def alive(self) -> bool:
    return (self._process is not None and self._process.poll() is None
            and self._thread is not None and self._thread.is_alive())

  def close(self) -> None:
    self._stop.set()
    if self._process is not None:
      self._process.terminate()
      try:
        self._process.wait(timeout=3.0)
      except subprocess.TimeoutExpired:
        self._process.kill()
        self._process.wait(timeout=3.0)
    if self._thread is not None:
      self._thread.join(timeout=2.0)


class OneEuroQuatFilter:
  """One Euro on an orientation: the low-pass step becomes a slerp fraction,
  with the adaptive cutoff driven by the smoothed angular speed."""

  def __init__(self, *, min_cutoff_hz: float, beta: float, d_cutoff_hz: float = 1.0):
    if min_cutoff_hz <= 0.0 or beta < 0.0 or d_cutoff_hz <= 0.0:
      raise ValueError("filter cutoffs must be positive and beta non-negative")
    self._min_cutoff = float(min_cutoff_hz)
    self._beta = float(beta)
    self._d_cutoff = float(d_cutoff_hz)
    self._q: np.ndarray | None = None
    self._speed = 0.0

  def reset(self) -> None:
    self._q = None
    self._speed = 0.0

  def __call__(self, quat: np.ndarray, dt: float) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    if self._q is None:
      self._q = quat.copy()
      return quat.copy()
    velocity = np.zeros(3, dtype=np.float64)
    mujoco.mju_subQuat(velocity, quat, self._q)
    angle = float(np.linalg.norm(velocity))
    a_d = OneEuroFilter._alpha(self._d_cutoff, dt)
    self._speed = self._speed + a_d * (angle / dt - self._speed)
    cutoff = self._min_cutoff + self._beta * self._speed
    a = OneEuroFilter._alpha(cutoff, dt)
    if angle > 1.0e-12:
      smoothed = self._q.copy()
      mujoco.mju_quatIntegrate(smoothed, velocity / angle, a * angle)
      self._q = smoothed
    return self._q.copy()


class TeleopIK:
  """IK on i2rt's own YAM model: the arm's standard kinematics and limits.

  This is the stack i2rt's teleop uses (`i2rt.robots.kinematics.Kinematics`):
  the shipped YAM MJCF, the `grasp_site` frame, a `mink.FrameTask` and one
  damped QP -- solved once per tick here, since a streaming target does not
  need iteration to convergence. Joint bounds are the *mechanical* ranges from
  that model intersected with the operator's `safety` box; the policy
  deployment's trained-envelope intersection is deliberately absent, because
  it exists to keep a trained policy inside its data and does nothing for a
  human operator except pin joints mid-move.
  """

  # i2rt's `Kinematics.ik` default damping.
  _DAMPING = 1.0e-4
  _SITE = "grasp_site"

  def __init__(self, safety: dict, robot_config: dict | None = None):
    robot_config = robot_config or {}
    arm = ArmType[str(robot_config.get("arm_type", "yam")).upper()]
    gripper = GripperType[str(robot_config.get("gripper_type", "no_gripper")).upper()]
    xml_path = combine_arm_and_gripper_xml(arm, gripper)
    model = mujoco.MjModel.from_xml_path(xml_path)
    joint_ids = np.asarray(
      [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 7)],
      dtype=np.int32,
    )
    if np.any(joint_ids < 0):
      raise RuntimeError("i2rt's YAM model is missing an arm joint")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self._SITE) < 0:
      raise RuntimeError(f"i2rt's YAM model is missing the {self._SITE!r} site")
    self._qpos_ids = model.jnt_qposadr[joint_ids]

    self._max_offset_rad = float(safety["max_command_offset_rad"])
    self._max_velocity_rad_s = float(safety["max_joint_velocity_rad_s"])
    operator_min = np.asarray(safety["joint_position_min_rad"], dtype=np.float64)
    operator_max = np.asarray(safety["joint_position_max_rad"], dtype=np.float64)
    if operator_min.shape != (6,) or operator_max.shape != (6,):
      raise ValueError("safety joint position bounds must each contain six values")
    limited = model.jnt_limited[joint_ids].astype(bool)
    mech_min = model.jnt_range[joint_ids, 0].astype(np.float64)
    mech_max = model.jnt_range[joint_ids, 1].astype(np.float64)
    self.mechanical_limits = (mech_min.copy(), mech_max.copy())
    self._position_min = np.where(limited, np.maximum(operator_min, mech_min), operator_min)
    self._position_max = np.where(limited, np.minimum(operator_max, mech_max), operator_max)

    if np.any(self._position_min >= self._position_max):
      raise ValueError("operator joint box does not intersect mechanical limits")
    self._command_speed = float(safety.get("max_command_velocity_rad_s", 0.3))
    self._configuration = mink.Configuration(model)
    self._task = mink.FrameTask(
      frame_name=self._SITE,
      frame_type="site",
      position_cost=1.0,
      orientation_cost=1.0,
    )

  @property
  def home_joint_position(self) -> np.ndarray:
    return np.asarray(YAM_ARM_HOME_JOINT_POS, dtype=np.float64).copy()

  @property
  def joint_position_limits(self) -> tuple[np.ndarray, np.ndarray]:
    return self._position_min.copy(), self._position_max.copy()

  def _update(self, state: ArmState) -> None:
    qpos = np.zeros(self._configuration.model.nq, dtype=np.float64)
    qpos[self._qpos_ids] = state.joint_position
    self._configuration.update(qpos)

  def ee_pose(self, state: ArmState) -> tuple[np.ndarray, np.ndarray]:
    """Grasp-site position and quaternion (w, x, y, z) in the world."""
    self._update(state)
    frame = self._configuration.get_transform_frame_to_world(self._SITE, "site")
    return frame.wxyz_xyz[4:].copy(), frame.wxyz_xyz[:4].copy()

  def check_velocity(self, state: ArmState) -> None:
    speed = float(np.max(np.abs(np.asarray(state.joint_velocity, dtype=np.float64))))
    if not np.isfinite(speed) or speed > self._max_velocity_rad_s:
      raise RuntimeError(
        f"Joint speed {speed:.2f} rad/s exceeds the "
        f"{self._max_velocity_rad_s:.2f} rad/s safety limit"
      )

  def clamp(self, state: ArmState, target: np.ndarray) -> np.ndarray:
    current = np.asarray(state.joint_position, dtype=np.float64)
    offset = np.clip(
      np.asarray(target, dtype=np.float64) - current, -self._max_offset_rad, self._max_offset_rad
    )
    return np.clip(current + offset, self._position_min, self._position_max)

  def joint_target_from_pose(
    self,
    state: ArmState,
    target_position: np.ndarray,
    target_quat: np.ndarray,
    period_s: float,
  ) -> np.ndarray:
    self._update(state)
    self._task.set_target(
      mink.SE3(
        wxyz_xyz=np.concatenate(
          [
            np.asarray(target_quat, dtype=np.float64),
            np.asarray(target_position, dtype=np.float64),
          ]
        )
      )
    )
    velocity = mink.solve_ik(
      self._configuration,
      [self._task],
      dt=period_s,
      solver="quadprog",
      damping=self._DAMPING,
    )
    new_qpos = self._configuration.integrate(velocity, period_s)
    return self.clamp(state, new_qpos[self._qpos_ids])


class TrayTarget:
  """Clutched controller motion mapped to a slew-limited absolute tray pose.

  ``engage`` anchors the mapping on the controller pose and the measured tray
  pose at the instant the clutch was pressed, so the arm never jumps when the
  operator takes hold: at that instant the requested pose is the measured one.
  ``step`` then walks the commanded pose toward the requested one at no more
  than ``max_speed_m_s`` and ``max_omega_rad_s``, which is what stops a
  flicked wrist from asking for a motion the joint-speed trip would catch.
  """

  def __init__(
    self,
    *,
    frame: np.ndarray,
    position_scale: float,
    orientation: bool,
    max_speed_m_s: float,
    max_omega_rad_s: float,
    period_s: float,
  ):
    self._frame = np.asarray(frame, dtype=np.float64)
    self._position_scale = float(position_scale)
    self._orientation = bool(orientation)
    self._max_step_m = float(max_speed_m_s) * float(period_s)
    self._max_step_rad = float(max_omega_rad_s) * float(period_s)
    self._controller_pos = np.zeros(3, dtype=np.float64)
    self._controller_mat = np.eye(3, dtype=np.float64)
    self._tray_pos = np.zeros(3, dtype=np.float64)
    self._tray_mat = np.eye(3, dtype=np.float64)
    self._command_pos = np.zeros(3, dtype=np.float64)
    self._command_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    self._engaged = False
    self._backlog_m = 0.0
    self._hand_world = np.zeros(3, dtype=np.float64)

  @property
  def frame(self) -> np.ndarray:
    return self._frame

  @property
  def engaged(self) -> bool:
    return self._engaged

  @property
  def command(self) -> tuple[np.ndarray, np.ndarray]:
    return self._command_pos.copy(), self._command_quat.copy()

  @property
  def hand_world(self) -> np.ndarray:
    """The hand's travel since the clutch press, in world axes and unscaled.

    Printed beside the tray's own travel: the two together say whether lost
    travel is the scale, the slew limiter, or the joint envelope.
    """
    return self._hand_world.copy()

  @property
  def tray_travel(self) -> np.ndarray:
    """The commanded tray pose's travel since the clutch press."""
    return self._command_pos - self._tray_pos

  @property
  def backlog_m(self) -> float:
    """How far the commanded pose is behind the hand, after slew limiting.

    A backlog that stays up while the operator moves means
    ``max_speed_m_s`` is binding, and every millimetre of it is travel the
    tray owes the hand -- which is what under-responsive position control
    feels like.
    """
    return self._backlog_m

  def engage(
    self,
    controller_pos: np.ndarray,
    controller_quat: np.ndarray,
    tray_pos: np.ndarray,
    tray_quat: np.ndarray,
  ) -> None:
    self._controller_pos = np.asarray(controller_pos, dtype=np.float64).copy()
    self._controller_mat = quat_to_mat(controller_quat)
    self._tray_pos = np.asarray(tray_pos, dtype=np.float64).copy()
    self._tray_mat = quat_to_mat(tray_quat)
    self._command_pos = self._tray_pos.copy()
    self._command_quat = np.asarray(tray_quat, dtype=np.float64).copy()
    self._engaged = True

  def release(self) -> None:
    self._engaged = False

  def desired(
    self, controller_pos: np.ndarray, controller_quat: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    """Absolute tray pose the controller is asking for, before slew limiting."""
    if not self._engaged:
      raise RuntimeError("the clutch must be engaged before a target exists")
    delta = np.asarray(controller_pos, dtype=np.float64) - self._controller_pos
    self._hand_world = self._frame @ delta
    position = self._tray_pos + self._position_scale * self._hand_world
    if not self._orientation:
      return position, mat_to_quat(self._tray_mat)
    # The controller's rotation since the clutch press, carried from the
    # headset's frame into the world frame (a rotation changes basis by
    # conjugation), then applied to the tray orientation held at that press.
    controller_delta = quat_to_mat(controller_quat) @ self._controller_mat.T
    world_delta = self._frame @ controller_delta @ self._frame.T
    return position, mat_to_quat(world_delta @ self._tray_mat)

  def step(
    self, controller_pos: np.ndarray, controller_quat: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    position, quat = self.desired(controller_pos, controller_quat)
    step = position - self._command_pos
    distance = float(np.linalg.norm(step))
    if distance > self._max_step_m:
      self._command_pos = self._command_pos + step * (self._max_step_m / distance)
    else:
      self._command_pos = position
    self._backlog_m = float(np.linalg.norm(position - self._command_pos))

    # `mju_subQuat` is the rotation vector taking the commanded orientation to
    # the requested one in unit time; `mju_quatIntegrate` applies a capped
    # slice of it, which is a slerp step without the quaternion algebra.
    velocity = np.zeros(3, dtype=np.float64)
    mujoco.mju_subQuat(velocity, quat, self._command_quat)
    angle = float(np.linalg.norm(velocity))
    if angle > self._max_step_rad:
      turned = self._command_quat.copy()
      mujoco.mju_quatIntegrate(turned, velocity / angle, self._max_step_rad)
      self._command_quat = turned
    else:
      self._command_quat = quat
    return self._command_pos.copy(), self._command_quat.copy()


class ArmChannel:
  """One hand driving one arm: its IK, clutch mapping, filters and hardware.

  Deliberately holds no mode of its own. The session owns the state machine and
  tells the channel what to do each tick, which is what lets one loop, one clock
  and one fault domain drive two arms: a trip on either channel parks both, and
  both arms' targets come from the same tick, so a recorded frame is consistent.
  """

  def __init__(
    self,
    config: dict,
    *,
    period_s: float,
    backend: str | None = None,
    frame: np.ndarray | None = None,
    label: str | None = None,
  ):
    teleop = config["teleop"]
    robot_config = dict(config["robot"])
    if backend is not None:
      robot_config["backend"] = backend
    self.backend = str(robot_config["backend"])

    self.hand = str(teleop["hand"]).lower()
    if self.hand not in {"left", "right"}:
      raise ValueError("teleop.hand must be 'left' or 'right'")
    self.label = label or self.hand
    self._period_s = float(period_s)

    position_scale = float(teleop["position_scale"])
    max_speed = float(teleop["max_tray_speed_m_s"])
    max_omega = float(teleop["max_tray_omega_rad_s"])
    if min(position_scale, max_speed, max_omega) <= 0.0:
      raise ValueError("teleop scales and slew limits must be positive")
    self._target = TrayTarget(
      frame=frame_rotation(teleop["yaw_deg"]) if frame is None else frame,
      position_scale=position_scale,
      orientation=bool(teleop["orientation"]),
      max_speed_m_s=max_speed,
      max_omega_rad_s=max_omega,
      period_s=self._period_s,
    )
    # The smoothness path: One Euro on the raw controller pose, plus a jump
    # guard that declutches on a tracking glitch instead of chasing it.
    min_cutoff = float(teleop["filter_min_cutoff_hz"])
    beta = float(teleop["filter_beta"])
    self._filter_pos = OneEuroFilter(min_cutoff_hz=min_cutoff, beta=beta)
    self._filter_quat = OneEuroQuatFilter(min_cutoff_hz=min_cutoff, beta=beta)
    self._max_hand_jump_m = float(teleop["max_hand_jump_m"])
    self._last_raw_pos: np.ndarray | None = None
    # Joint-space smoothing on the IK output. Near the base axis a fraction of
    # a millimetre of end-effector motion maps to milliradians of j1/j5, so
    # even a well-filtered hand chatters the motors; a short exponential stage
    # on the commanded joints removes that for ~one servo lag of extra delay.
    tau = float(teleop["joint_smoothing_tau_s"])
    self._joint_alpha = 1.0 if tau <= 0.0 else self._period_s / (self._period_s + tau)
    self._smoothed_target: np.ndarray | None = None

    self._ik = TeleopIK(config["safety"], robot_config)
    # Both poses are per-arm. `yam_home.json` is only the fallback, because two
    # arms that face each other -- or sit at different heights, or are mirrored
    # -- do not share a ready pose, and homing the left arm to the right arm's
    # pose is exactly the move that puts them through each other.
    reset = config["deploy"].get("reset_joint_position_rad")
    self.reset_pose = (
      self._ik.home_joint_position
      if reset is None
      else np.asarray(reset, dtype=np.float64)
    )
    self.park_pose = np.asarray(config["deploy"]["home_joint_position_rad"], dtype=np.float64)
    # Gripper: the analogue trigger drives it, which is why the grip button
    # takes over the clutch when it is on. Without a gripper the channel never
    # calls command_gripper and the trigger keeps its clutch duty.
    self.gripper_enabled = bool(teleop.get("gripper", False))
    self._gripper_command = 0.0

    self._arm = make_arm(robot_config)
    try:
      if self.gripper_enabled and not getattr(self._arm, "_gripper_in_vector", False):
        raise RuntimeError(f"{self.label}: backend has no gripper")
      state = self._arm.read_state()
      self._ik.check_velocity(state)
      lower, upper = self._ik.joint_position_limits
      # Allow up to 0.05 rad (~3 deg) past limits — the arm rests on its
      # mechanical stops when powered off, and the operator cannot move it
      # without gravity comp (which requires this constructor to finish).
      startup_tol = 0.05
      if np.any(state.joint_position < lower - startup_tol) or np.any(state.joint_position > upper + startup_tol):
        outside = (state.joint_position < lower - startup_tol) | (state.joint_position > upper + startup_tol)
        joints = ", ".join(f"j{i+1}={state.joint_position[i]:+.4f}" for i in np.flatnonzero(outside))
        raise RuntimeError(f"Initial arm position is outside the configured joint box: {joints}")
    except BaseException:
      self._arm.close()
      raise
    if state.gripper_position is not None:
      self._gripper_command = float(state.gripper_position)
    self._last_command = state.joint_position.copy()
    self._require_release = True
    self._last_sample_s = None

    self.hold = np.clip(state.joint_position, lower, upper)
    self._move_from = state.joint_position.copy()
    self._move_target = self.reset_pose.copy()
    self.joint_position = state.joint_position.copy()
    self.tray_position, _ = self._ik.ee_pose(state)
    self._backlog_m = 0.0
    self._pinned_joints: list[int] = []
    self._hand_travel = np.zeros(3, dtype=np.float64)
    self._tray_travel = np.zeros(3, dtype=np.float64)
    self._engaged = False

  # ----------------------------------------------------------------- hardware

  def read_state(self) -> ArmState:
    return self._arm.read_state()

  def command(self, target: np.ndarray) -> None:
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (6,) or not np.all(np.isfinite(target)):
      raise ValueError("invalid joint target")
    step = self._ik._command_speed * self._period_s
    bounded = np.clip(target, self._last_command - step, self._last_command + step)
    self._arm.command_arm_positions(bounded)
    self._last_command = bounded.copy()
    self.hold = bounded.copy()

  def enter_gravity_comp(self) -> None:
    """Drop PD targets but keep gravity comp running, if the backend supports it."""
    if hasattr(self._arm, 'enter_gravity_comp'):
      self._arm.enter_gravity_comp()

  def close(self) -> None:
    self._arm.close()

  def check_velocity(self, state: ArmState) -> None:
    """Raises RuntimeError on a runaway; the session turns that into a fault."""
    self._ik.check_velocity(state)

  def observe(self, state: ArmState) -> None:
    """Refresh the diagnostics that hold while the arm is not being driven."""
    self.joint_position = state.joint_position.copy()
    if not self._engaged:
      self.tray_position, _ = self._ik.ee_pose(state)

  # -------------------------------------------------------------------- moves

  def begin_move(self, target: np.ndarray) -> None:
    self._move_from = self.joint_position.copy()
    self._move_target = np.asarray(target, dtype=np.float64).copy()
    self.release_input()

  def move_step(
    self, state: ArmState, elapsed: float, move_s: float, settle_s: float
  ) -> tuple[np.ndarray, bool]:
    eased = smoothstep(elapsed / max(move_s, 1.0e-3))
    target = self._move_from + eased * (self._move_target - self._move_from)
    if elapsed < move_s + settle_s:
      return self._ik.clamp(state, target), False
    return self._ik.clamp(state, self._move_target), True

  # ------------------------------------------------------------------- clutch

  def release_input(self) -> None:
    """Drop the clutch and forget the filter state."""
    self._target.release()
    self._filter_pos.reset()
    self._filter_quat.reset()
    self._last_raw_pos = None
    self._smoothed_target = None
    self._backlog_m = 0.0
    self._pinned_joints = []
    self._engaged = False
    self._require_release = True
    self._last_sample_s = None

  def drive(
    self, state: ArmState, sample: ControllerSample | None, now: float, watchdog_s: float
  ) -> tuple[np.ndarray, bool, str | None]:
    """One tick while the session is live. Returns (target, engaged, note).

    ``engaged`` is False whenever this arm is holding rather than tracking: no
    sample, a stale one, the clutch released, or a tracking glitch.
    """
    stale = sample is None or (now - sample.received_s) > watchdog_s
    if stale:
      self.release_input()
      return self.hold.copy(), False, "controller stream lost; release clutch to resume"
    pressed = self._clutch_of(sample)
    if not pressed:
      if self.gripper_enabled and sample.trigger > 0.5:
        self._gripper_command = float(np.clip(1.0, self._gripper_command - self._period_s, self._gripper_command + self._period_s))
        self._arm.command_gripper(self._gripper_command)
      self.release_input()
      self._require_release = False
      return self.hold.copy(), False, None
    if self._require_release:
      return self.hold.copy(), False, "release clutch before engaging"
    # Process a controller sample once, using its actual interval for filtering.
    if self._last_sample_s == sample.received_s:
      return self.hold.copy(), self._engaged, None
    sample_dt = self._period_s if self._last_sample_s is None else max(1e-5, sample.received_s - self._last_sample_s)
    self._last_sample_s = sample.received_s

    # A tracking glitch (occlusion, reacquisition) teleports the reported
    # pose. Chasing it would slam the arm, so the mapping is dropped and
    # re-anchored: the arm holds still through the jump and the hand simply
    # owns its new position from here, since engaging never moves the arm.
    if (
      self._last_raw_pos is not None
      and float(np.linalg.norm(sample.position - self._last_raw_pos)) > self._max_hand_jump_m
    ):
      self.release_input()
      return self.hold.copy(), False, "controller pose jumped (tracking glitch); re-anchored"
    self._last_raw_pos = sample.position.copy()

    # One Euro runs at the loop rate on the newest raw pose: jitter from a
    # resting hand is filtered hard, a moving hand is followed nearly 1:1,
    # and the 70 Hz log stream's steps are smoothed over instead of relayed.
    position_f = self._filter_pos(sample.position, sample_dt)
    quat_f = self._filter_quat(sample.quat, sample_dt)

    note = None
    if not self._target.engaged:
      ee_pos, ee_quat = self._ik.ee_pose(state)
      self._target.engage(position_f, quat_f, ee_pos, ee_quat)
      note = "driving the arm"
    self._engaged = True

    position, quat = self._target.step(position_f, quat_f)
    solved = self._ik.joint_target_from_pose(state, position, quat, self._period_s)
    if self._smoothed_target is None:
      self._smoothed_target = state.joint_position.copy()
    self._smoothed_target = self._smoothed_target + self._joint_alpha * (
      solved - self._smoothed_target
    )
    target = self._ik.clamp(state, self._smoothed_target)

    if self.gripper_enabled:
      desired_gripper = self._gripper_of(sample)
      self._gripper_command = float(np.clip(desired_gripper, self._gripper_command - sample_dt, self._gripper_command + sample_dt))
      self._arm.command_gripper(self._gripper_command)

    self.tray_position = position
    self._backlog_m = self._target.backlog_m
    self._hand_travel = self._target.hand_world
    self._tray_travel = self._target.tray_travel
    # Which joints the commanded target is pinned against. Once one is, the
    # arm cannot travel further that way no matter how far the hand goes,
    # and this is the only place that is visible.
    lower, upper = self._ik.joint_position_limits
    pinned = (target <= lower + 1.0e-3) | (target >= upper - 1.0e-3)
    self._pinned_joints = [index + 1 for index in np.flatnonzero(pinned)]
    return target, True, note

  def _clutch_of(self, sample: ControllerSample) -> bool:
    """Grip alone clutches once the trigger is spoken for by the gripper."""
    if self.gripper_enabled:
      return sample.grip > 0.5
    return sample.clutch

  def _gripper_of(self, sample: ControllerSample) -> float:
    """Proportional: the trigger's analog value sets the grip target.

    Full squeeze → closed (0), released → open (1). The ramp in drive()
    limits how fast the gripper moves; the trigger just says where.
    """
    return 1.0 - sample.trigger

  # ------------------------------------------------------------------- status

  def status(self) -> dict[str, Any]:
    return {
      "label": self.label,
      "hand": self.hand,
      "backend": self.backend,
      "joints": [round(float(v), 4) for v in self.joint_position],
      "tray": [round(float(v), 4) for v in self.tray_position],
      "lag_mm": round(self._backlog_m * 1000.0, 1),
      "pinned": list(self._pinned_joints),
      "engaged": self._engaged,
      "gripper": round(float(self._gripper_command), 3) if self.gripper_enabled else None,
      "hand_travel": [round(float(v), 4) for v in self._hand_travel],
      "tray_travel": [round(float(v), 4) for v in self._tray_travel],
    }


class TeleopSession:
  """Fixed-rate control loop driving one or two arms from Quest controllers.

  One thread, one clock, one mode. With two channels both arms home together,
  park together and share a fault domain: a runaway on either arm, a stop press
  from either hand, or a Ctrl-C parks both. That shared stop is the whole reason
  two arms belong in one process rather than in two.
  """

  def __init__(
    self,
    config: dict,
    reader: QuestReader,
    *,
    backend: str | None = None,
    frame: np.ndarray | None = None,
    extra: list[tuple[dict, np.ndarray | None]] | None = None,
    record_dir: Path | None = None,
    dashboard_port: int | None = None,
  ):
    validate_config(config)
    teleop = config["teleop"]
    self._period_s = 1.0 / float(teleop["control_hz"])
    self._watchdog_s = float(teleop["watchdog_s"])
    self._move_s = float(teleop["move_s"])
    self._settle_s = float(teleop["settle_s"])
    if min(self._watchdog_s, self._move_s) <= 0.0:
      raise ValueError("teleop watchdog and move duration must be positive")
    self._reader = reader

    specs: list[tuple[dict, np.ndarray | None]] = [(config, frame)]
    specs += list(extra or [])
    for cfg, _ in specs:
      validate_config(cfg)
    hands = [str(cfg["teleop"]["hand"]).lower() for cfg, _ in specs]
    buses = [cfg["robot"].get("channel") for cfg, _ in specs if (backend or cfg["robot"]["backend"]) == "i2rt"]
    if len(set(hands)) != len(hands) or len(set(buses)) != len(buses):
      raise ValueError("Each arm must have its own controller and CAN interface")
    for cfg, frm in specs:
      if (backend or cfg["robot"]["backend"]) == "i2rt" and frm is None:
        raise RuntimeError("Calibrate each controller frame before opening real hardware")
    self._channels = []
    try:
      for cfg, frm in specs:
        self._channels.append(ArmChannel(cfg, period_s=self._period_s, backend=backend, frame=frm))
    except BaseException:
      for channel in self._channels:
        try:
          channel.close()
        except Exception as exc:
          print(f"Startup cleanup failed: {exc}", file=sys.stderr)
      raise
    print_hz = float(teleop["print_hz"])
    self._print_period_s = 0.0 if print_hz <= 0.0 else 1.0 / print_hz
    self._last_print_s = 0.0

    self._record_dir = record_dir
    self._recorder = DemoRecorder(hands=hands, hz=1.0 / self._period_s) if record_dir else None
    self._writers: list[threading.Thread] = []

    self._dashboard: Dashboard | None = None
    if dashboard_port is not None:
      self._dashboard = Dashboard(port=dashboard_port, record_dir=record_dir)
      self._dashboard.start()
    self._last_dashboard_push_s = 0.0

    self._lock = threading.Lock()
    self._button_edges = {hand: {"home": True, "stop": False, "js_click": False} for hand in hands}
    self._mode = IDLE
    self._fault: str | None = None
    self._note = "holding current position; release clutch before engaging"
    self._move_started_s = time.monotonic()
    self._measured_hz = 0.0
    self._shutdown = threading.Event()
    self._parked = threading.Event()
    self._closed = False
    self._thread = threading.Thread(target=self._run_loop, name="quest-teleop", daemon=True)

  # ---------------------------------------------------------------- lifecycle

  @property
  def backend(self) -> str:
    return self._channels[0].backend

  @property
  def channels(self) -> list[ArmChannel]:
    return list(self._channels)

  def start(self) -> None:
    self._thread.start()

  def alive(self) -> bool:
    return self._thread.is_alive()

  def enter_gravity_comp(self) -> None:
    """Stop the control loop and enter gravity comp. Arms float but don't fall.

    Call this before close() to give the operator time to support the arms.
    The motor driver stays alive — gravity comp torques keep flowing.
    """
    self._shutdown.set()
    if self._thread.is_alive():
      self._thread.join(timeout=5.0)
      if self._thread.is_alive():
        raise RuntimeError("Control thread did not stop. Use physical emergency stop.")
    for channel in self._channels:
      channel.enter_gravity_comp()

  def close(self) -> None:
    """Disable motors and release all resources. Arms will fall."""
    if self._closed:
      return
    if not self._shutdown.is_set():
      self.enter_gravity_comp()
    errors = []
    for channel in self._channels:
      try:
        channel.close()
      except Exception as exc:
        errors.append(str(exc))
    self._finish_recording()
    if self._dashboard is not None:
      self._dashboard.close()
    if errors:
      raise RuntimeError("Driver shutdown failed; use emergency stop: " + "; ".join(errors))
    self._closed = True

  def _run_loop(self) -> None:
    try:
      self._loop()
    except BaseException as exc:
      with self._lock:
        self._fault = f"Control loop failed: {exc}"
        self._mode = FAULT
      print(self._fault + "; support arms and use physical stop if needed", file=sys.stderr)
      self._shutdown.set()

  def status(self) -> dict[str, Any]:
    with self._lock:
      return {
        "mode": self._mode,
        "fault": self._fault,
        "note": self._note,
        "hz": round(self._measured_hz, 1),
        "arms": [channel.status() for channel in self._channels],
      }

  # --------------------------------------------------------------------- loop

  def _loop(self) -> None:
    next_tick = time.perf_counter()
    last = next_tick
    while not self._shutdown.is_set():
      # Every arm is read, then every arm is commanded: both channels are
      # computed from one tick, so a recorded frame is a consistent snapshot.
      states = [channel.read_state() for channel in self._channels]
      for channel, state in zip(self._channels, states):
        if np.max(np.abs(channel._last_command - state.joint_position)) > channel._ik._max_offset_rad + 1e-6:
          raise RuntimeError(f"{channel.label}: arm is not following its target")
        if (state.joint_position.shape != (6,) or state.joint_velocity.shape != (6,)
            or not np.all(np.isfinite(state.joint_position)) or not np.all(np.isfinite(state.joint_velocity))):
          raise RuntimeError("Invalid motor feedback")
      now = time.monotonic()
      samples = {channel.hand: self._reader.sample(channel.hand) for channel in self._channels}
      now = time.monotonic()
      fresh_samples = [sample for sample in samples.values()
                       if sample is not None and 0 <= now - sample.received_s <= self._watchdog_s]
      stop_pressed = any(sample.stop for sample in fresh_samples)
      for sample in fresh_samples:
        if not stop_pressed or sample.stop:
          self._apply_buttons(sample)
      targets = self._step(states, samples, now)
      for channel, target in zip(self._channels, targets):
        channel.command(target)

      if self._recorder is not None:
        with self._lock:
          mode = self._mode
        self._recorder.tick(states, targets, samples, self._channels, mode, now)

      tick = time.perf_counter()
      period = max(tick - last, 1.0e-6)
      last = tick
      with self._lock:
        for channel, state in zip(self._channels, states):
          channel.observe(state)
        # Exponential average over ~0.5 s: the number the operator reads.
        alpha = min(1.0, self._period_s / 0.5)
        self._measured_hz = (1.0 - alpha) * self._measured_hz + alpha / period
      self._maybe_print(samples, now)
      self._maybe_push_dashboard(samples, targets, now)

      next_tick += self._period_s
      sleep_s = next_tick - time.perf_counter()
      if sleep_s > 0.0:
        time.sleep(sleep_s)
      else:
        # Overran: give up the missed ticks rather than sprinting to catch up.
        next_tick = time.perf_counter()

  def _apply_buttons(self, sample: ControllerSample) -> None:
    """B/Y pauses. A/X resumes a deliberate pause, never a fault.
    Joystick click toggles recording (when --record is active)."""
    if time.monotonic() - sample.received_s > self._watchdog_s:
      return
    js_key = "RJ" if sample.hand == "right" else "LJ"
    js_pressed = bool(sample.buttons.get(js_key, False))
    with self._lock:
      edges = self._button_edges[sample.hand]
      stop = sample.stop and not edges["stop"]
      resume = sample.home and not edges["home"]
      js_click = js_pressed and not edges["js_click"]
      edges.update(stop=sample.stop, home=sample.home, js_click=js_pressed)
      if stop:
        self._mode = FAULT if self._fault else PARKED
        self._note = "paused; A/X resumes, then release and press clutch"
        for channel in self._channels:
          channel.release_input()
          channel.hold = channel._last_command.copy()
        if self._recorder is not None and self._recorder.recording:
          self._save_recording()
      elif resume and self._mode == PARKED and self._fault is None:
        self._begin_move(HOMING, [channel.reset_pose for channel in self._channels])
        self._note = "homing to start position..."
      if js_click and self._recorder is not None:
        if self._recorder.recording:
          self._save_recording()
        elif self._mode in {PARKED, FAULT}:
          # Parking already saved whatever was running; starting again here
          # would only capture a stationary arm.
          self._note = "resume with A/X before recording"
        else:
          self._recorder.start()
          self._note = "RECORDING started (joystick click to stop)"
          print("\n*** RECORDING STARTED ***", flush=True)

  def _step(
    self,
    states: list[ArmState],
    samples: dict[str, ControllerSample | None],
    now: float,
  ) -> list[np.ndarray]:
    # The runaway trip is checked on every arm before anything is commanded,
    # and a trip on one parks them all.
    for channel, state in zip(self._channels, states):
      try:
        channel.check_velocity(state)
      except RuntimeError as exc:
        raise RuntimeError(f"{channel.label}: {exc}; no parking move will be attempted") from exc

    with self._lock:
      missing = any(samples.get(c.hand) is None or not 0 <= now - samples[c.hand].received_s <= self._watchdog_s
                    for c in self._channels)
      if missing and self._mode in {IDLE, ENGAGED}:
        self._mode = PARKED
        self._note = "input lost; A/X resumes after fresh input returns"
        for channel in self._channels:
          channel.release_input()
          channel.hold = channel._last_command.copy()
      mode = self._mode
      if mode in {HOMING, RETURNING}:
        elapsed = now - self._move_started_s
        targets = []
        finished = True
        for channel, state in zip(self._channels, states):
          target, done = channel.move_step(state, elapsed, self._move_s, self._settle_s)
          targets.append(target)
          finished = finished and done
        if finished:
          if mode == HOMING:
            self._mode = IDLE
            self._note = "hold the clutch to drive the arm"
          else:
            self._mode = FAULT if self._fault else PARKED
            self._note = self._fault or "parked; press A to re-arm"
            self._parked.set()
        for channel, target in zip(self._channels, targets):
          channel.hold = target.copy()
        return targets

      if mode in {PARKED, FAULT}:
        return [channel.hold.copy() for channel in self._channels]

      targets = []
      any_engaged = False
      for channel, state in zip(self._channels, states):
        target, engaged, note = channel.drive(
          state, samples.get(channel.hand), now, self._watchdog_s
        )
        if note is not None:
          self._note = f"{channel.label}: {note}"
        channel.hold = target.copy()
        targets.append(target)
        any_engaged = any_engaged or engaged
      # ENGAGED while either hand drives; the other arm simply holds.
      self._mode = ENGAGED if any_engaged else IDLE
      return targets

  # --------------------------------------------------------- recording helpers

  def _save_recording(self) -> None:
    """Detach the buffer and write it off the control thread.

    Writing a demo takes tens to hundreds of milliseconds and grows with its
    length; doing it here would stall the loop for whole control periods and
    can trip the follow check on the next tick. `detach` is O(1), so only the
    handoff happens on this thread. Caller holds self._lock.
    """
    if self._recorder is None:
      return
    snapshot = self._recorder.detach()
    if snapshot is None:
      self._note = "recording discarded (nothing captured)"
      print("\n*** RECORDING DISCARDED (empty) ***", flush=True)
      return
    self._writers = [thread for thread in self._writers if thread.is_alive()]
    writer = threading.Thread(
      target=self._write_snapshot, args=(snapshot,), name="demo-writer", daemon=False
    )
    self._writers.append(writer)
    writer.start()
    self._note = f"saving {snapshot.num_ticks} ticks"

  def _write_snapshot(self, snapshot: DemoSnapshot) -> None:
    """Runs on a writer thread, never on the control loop."""
    try:
      path = write_snapshot(snapshot, self._record_dir)
      print(
        f"\n*** RECORDING SAVED: {path.name} "
        f"({snapshot.num_ticks} ticks, {snapshot.duration_s:.1f}s) ***",
        flush=True,
      )
    except Exception as exc:
      print(f"\n*** RECORDING SAVE FAILED: {exc} ***", flush=True)

  def _finish_recording(self) -> None:
    """Flush an in-progress recording and wait for pending writes. Called
    once the control thread has stopped, so detaching needs no lock."""
    if self._recorder is None:
      return
    snapshot = self._recorder.detach()
    if snapshot is not None:
      self._write_snapshot(snapshot)
    for writer in self._writers:
      writer.join(timeout=60.0)
    self._writers.clear()

  # ------------------------------------------------------------ moves/console

  def _begin_move(self, mode: str, targets: list[np.ndarray]) -> None:
    """Caller holds the lock."""
    for channel, target in zip(self._channels, targets):
      channel.begin_move(target)
    self._move_started_s = time.monotonic()
    self._parked.clear()
    self._mode = mode

  def _begin_return(self) -> None:
    """Caller holds the lock."""
    self._begin_move(RETURNING, [channel.park_pose for channel in self._channels])

  def _maybe_print(self, samples: dict[str, ControllerSample | None], now: float) -> None:
    """The operator's live view: controller state in, arm state out."""
    if self._print_period_s <= 0.0 or now - self._last_print_s < self._print_period_s:
      return
    self._last_print_s = now
    status = self.status()
    line = f"{status['mode']:<9} {status['hz']:5.1f}Hz"
    for arm in status["arms"]:
      sample = samples.get(arm["hand"])
      tag = arm["hand"][0].upper()
      if sample is None:
        line += f"  | {tag} no pose yet"
        continue
      age_ms = (now - sample.received_s) * 1000.0
      joints = ",".join(f"{value:+.2f}" for value in arm["joints"])
      line += f"  | {tag} clutch={int(sample.clutch)}"
      if arm["gripper"] is not None:
        line += f" grip={arm['gripper']:.2f}"
      if arm["engaged"]:
        # Hand travel beside tray travel: the whole diagnosis of "position
        # feels weak" is the ratio and direction of these two vectors.
        hand = arm["hand_travel"]
        moved = arm["tray_travel"]
        line += (
          f" hand=({hand[0]:+.2f},{hand[1]:+.2f},{hand[2]:+.2f})"
          f" tray=({moved[0]:+.2f},{moved[1]:+.2f},{moved[2]:+.2f})"
        )
      else:
        tray = arm["tray"]
        line += f" tray=({tray[0]:+.2f},{tray[1]:+.2f},{tray[2]:+.2f})"
      line += f" q=[{joints}] age={age_ms:3.0f}ms"
      if arm["lag_mm"] >= 1.0:
        line += f" lag={arm['lag_mm']:3.0f}mm"
      if arm["pinned"]:
        line += " pinned=j" + ",j".join(str(index) for index in arm["pinned"])
    if self._recorder is not None:
      if self._recorder.recording:
        ticks = self._recorder.tick_count
        secs = ticks * self._period_s
        line += f"  [REC {secs:.0f}s]"
      else:
        line += "  [js=rec]"
    if status["fault"]:
      line += f"  FAULT: {status['fault']}"
    _emit(line)

  def _maybe_push_dashboard(
    self,
    samples: dict[str, ControllerSample | None],
    targets: list[np.ndarray],
    now: float,
  ) -> None:
    if self._dashboard is None:
      return
    # ~10 Hz to the browser — no need for 100 Hz visual updates.
    if now - self._last_dashboard_push_s < 0.1:
      return
    self._last_dashboard_push_s = now
    with self._lock:
      status = {
        "mode": self._mode,
        "fault": self._fault,
        "note": self._note,
        "hz": round(self._measured_hz, 1),
        "dashboard_clients": self._dashboard.client_count,
        "arms": [],
        "controller_samples": {},
        "recording": {
          "active": self._recorder.recording if self._recorder else False,
          "ticks": self._recorder.tick_count if self._recorder else 0,
          "duration_s": round(self._recorder.tick_count * self._period_s, 1) if self._recorder else 0,
        },
      }
      for i, channel in enumerate(self._channels):
        arm_status = channel.status()
        arm_status["targets"] = targets[i].tolist() if i < len(targets) else []
        status["arms"].append(arm_status)
        sample = samples.get(channel.hand)
        if sample is not None:
          status["controller_samples"][channel.hand] = {
            "position": sample.position.tolist(),
            "quat": sample.quat.tolist(),
            "trigger": float(sample.trigger),
            "grip": float(sample.grip),
            "clutch": bool(sample.clutch),
          }
    self._dashboard.push(status)


def _emit(line: str) -> None:
  """One live line on a terminal, one log line otherwise."""
  if sys.stdout.isatty():
    print(f"\r\x1b[2K{line}", end="", flush=True)
  else:
    print(line, flush=True)


def dump(reader: QuestReader, hz: float) -> None:
  """Print the controller stream and touch no hardware."""
  period = 1.0 / max(hz, 1.0e-3)
  print("controller stream (Ctrl-C to stop):\n", flush=True)
  while reader.alive():
    samples = reader.samples()
    if not samples:
      _emit("waiting for poses... (wake the headset; the teleop app must be in focus)")
    else:
      _emit("   ".join(samples[hand].describe() for hand in sorted(samples)))
    time.sleep(period)
  raise RuntimeError("the adb logcat stream ended; is the Quest still attached?")


def save_frame(path: Path, frame: np.ndarray, push: np.ndarray) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    json.dumps(
      {
        "frame": [[float(value) for value in row] for row in frame],
        "push": [float(value) for value in push],
        "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
      },
      indent=2,
    )
    + "\n"
  )


def load_frame(path: Path) -> np.ndarray | None:
  if not path.exists():
    return None
  frame = np.asarray(json.loads(path.read_text())["frame"], dtype=np.float64)
  if (frame.shape != (3, 3) or not np.all(np.isfinite(frame))
      or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-6)
      or abs(float(np.linalg.det(frame)) - 1.0) > 1.0e-6):
    raise ValueError(f"{path} does not hold a rotation; delete it and recalibrate")
  return frame


def calibrate(reader: QuestReader, hand: str, path: Path) -> np.ndarray:
  """Measure the operator's heading from one push toward the arm.

  The headset's tracking frame has whatever yaw it was given when tracking
  started, so where "away from me" points in that frame is not knowable in
  advance -- and getting it wrong is what makes hand motion drive the tray
  along the wrong axes. One measured push fixes it.
  """
  print(
    "Frame calibration.\n"
    "  1. Stand where you will teleoperate from, facing the arm.\n"
    f"  2. Hold the {hand} trigger and push the controller ~30 cm straight at the arm.\n"
    "  3. Release the trigger.\n",
    flush=True,
  )
  # A controller the headset cannot see reports a frozen pose metres from the
  # origin, and reacquiring it teleports back. Either way the recorded push is
  # the teleport rather than the hand, and a frame built from that silently
  # sends every later run's motion along the wrong axes. Both bounds below
  # exist to reject that: no single sample may jump, and no whole push may be
  # longer than an arm.
  MAX_SAMPLE_JUMP_M = 0.30
  MAX_PUSH_M = 1.50

  start: np.ndarray | None = None
  last: np.ndarray | None = None
  glitched = False
  while reader.alive():
    sample = reader.sample(hand)
    if sample is None or time.monotonic() - sample.received_s > 0.15:
      start = last = None
      _emit("waiting for fresh poses... (wake the headset)")
      time.sleep(0.05)
      continue
    if sample.clutch:
      if start is None:
        start = sample.position.copy()
        glitched = False
      elif last is not None:
        if float(np.linalg.norm(sample.position - last)) > MAX_SAMPLE_JUMP_M:
          glitched = True
      last = sample.position.copy()
      travel = last - start
      _emit(f"recording: {np.linalg.norm(travel) * 100.0:5.1f} cm pushed")
    elif start is not None and last is not None:
      push = last - start
      length = float(np.linalg.norm(push))
      if glitched or length > MAX_PUSH_M:
        print(
          f"\nThat push measured {length * 100.0:.0f} cm, which is a tracking glitch, "
          "not a hand: the headset lost sight of the controller. Keep the controller "
          "in front of the headset and try again.",
          flush=True,
        )
        start = last = None
        glitched = False
        continue
      if length < 0.10:
        print("\nThat push was under 10 cm; try again, longer.", flush=True)
        start = last = None
        continue
      frame = frame_from_push(push)
      save_frame(path, frame, push)
      # The yaw this push implies, relative to the built-in assumption, is the
      # number worth reading: it says how far off a fixed frame would have been.
      offset = np.rad2deg(np.arctan2(-push[0], -push[2]))
      print(
        f"\npush = ({push[0]:+.3f},{push[1]:+.3f},{push[2]:+.3f}) m in the headset frame\n"
        f"operator heading is {offset:+.0f} deg from the built-in assumption\n"
        f"saved to {path}\n"
        "Rerun without --calibrate to drive the arm.",
        flush=True,
      )
      return frame
    time.sleep(0.02)
  raise RuntimeError("the adb logcat stream ended; is the Quest still attached?")


def main() -> None:
  parser = argparse.ArgumentParser(prog="python -m deployment.quest_teleop")
  parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
  parser.add_argument(
    "--backend",
    choices=("i2rt", "sim", "mock"),
    default=None,
    help=(
      "Override robot.backend. 'sim' drives i2rt's own MuJoCo SimRobot -- the same "
      "Robot protocol as the hardware, gripper included -- instead of CAN; 'mock' is "
      "the bare first-order servo."
    ),
  )
  parser.add_argument(
    "--second-config",
    type=Path,
    default=None,
    help=(
      "Drive a second arm from the other hand in the SAME loop: one clock, one "
      "stop button, one fault domain. This is what data collection needs; two "
      "separate processes share none of it. e.g. --second-config "
      "deployment/config_left.yaml"
    ),
  )
  parser.add_argument("--hand", choices=("left", "right"), default=None)
  parser.add_argument(
    "--gripper",
    action="store_true",
    help=(
      "Drive the gripper from the analogue trigger (squeeze to close). The grip "
      "button then becomes the clutch on its own. Needs robot.gripper_type set "
      "to what is mounted."
    ),
  )
  parser.add_argument("--serial", default=None, help="adb serial, if several devices are attached.")
  parser.add_argument(
    "--position-scale",
    type=float,
    default=None,
    help="Metres of tray motion per metre of controller motion.",
  )
  parser.add_argument(
    "--yaw-deg",
    type=float,
    default=None,
    help="Rotate the operator frame about world +z; 0 means standing behind the arm.",
  )
  parser.add_argument(
    "--max-speed",
    type=float,
    default=None,
    help="Tray slew cap in m/s; raise it if the console line shows a standing lag.",
  )
  parser.add_argument("--max-omega", type=float, default=None, help="Tray slew cap in rad/s.")
  parser.add_argument(
    "--lock-orientation",
    action="store_true",
    help="Drive tray position only and hold the orientation captured at each clutch press.",
  )
  parser.add_argument("--print-hz", type=float, default=None, help="Console line rate; 0 silences.")
  parser.add_argument(
    "--dump",
    action="store_true",
    help="Print the controller stream only: no CAN, no IK, no motion.",
  )
  parser.add_argument(
    "--calibrate",
    action="store_true",
    help=(
      "Measure the operator frame from one push toward the arm and save it. "
      "Run this whenever you change where you stand; it is what makes hand "
      "motion drive the tray along the axes you expect."
    ),
  )
  parser.add_argument(
    "--record",
    type=Path,
    default=None,
    metavar="DIR",
    help=(
      "Record demonstration data to HDF5 files in DIR. Recording starts/stops "
      "with the right joystick click. Each segment saves automatically."
    ),
  )
  parser.add_argument(
    "--dashboard",
    type=int,
    nargs="?",
    const=8080,
    default=None,
    metavar="PORT",
    help=(
      "Start a live web dashboard on PORT (default 8080). Open "
      "http://localhost:PORT in a browser to see joint angles, EE position, "
      "controller state, recording status and saved demos."
    ),
  )
  args = parser.parse_args()

  config = load_deployment_config(args.config)
  teleop = config["teleop"]
  for name, key in (
    ("hand", "hand"),
    ("position_scale", "position_scale"),
    ("yaw_deg", "yaw_deg"),
    ("max_speed", "max_tray_speed_m_s"),
    ("max_omega", "max_tray_omega_rad_s"),
    ("print_hz", "print_hz"),
  ):
    value = getattr(args, name)
    if value is not None:
      teleop[key] = value
  if args.lock_orientation:
    teleop["orientation"] = False
  if args.gripper:
    teleop["gripper"] = True

  # The second arm is a whole config of its own -- its own hand, CAN channel,
  # joint box, home poses and operator frame -- because nothing about the two
  # arms is required to match. Only the loop-wide numbers (rate, move times,
  # watchdog) are taken from the primary config, since one loop has one clock.
  second: dict | None = None
  if args.second_config is not None:
    second = load_deployment_config(args.second_config)
    if args.backend is not None:
      second["robot"]["backend"] = args.backend
    if args.gripper:
      second["teleop"]["gripper"] = True
    # Feel overrides apply to both arms; --hand and --yaw-deg deliberately do
    # not, since the hands must differ and each arm's base has its own heading.
    for name, key in (
      ("position_scale", "position_scale"),
      ("max_speed", "max_tray_speed_m_s"),
      ("max_omega", "max_tray_omega_rad_s"),
    ):
      value = getattr(args, name)
      if value is not None:
        second["teleop"][key] = value
    if args.lock_orientation:
      second["teleop"]["orientation"] = False
    if str(second["teleop"]["hand"]).lower() == str(teleop["hand"]).lower():
      raise SystemExit(
        f"Both configs use the {teleop['hand']} hand. The second arm needs the other "
        "controller: set teleop.hand in one of them."
      )

  reader = QuestReader(serial=args.serial, apk_path=Path(teleop["apk_path"]))
  try:
    serial = reader.device_serial()
    print(f"Quest: {serial} ({reader.ensure_app()})")
    reader.start()
    # The app needs a moment to come up and start logging poses.
    deadline = time.monotonic() + 10.0
    while not reader.samples() and time.monotonic() < deadline:
      time.sleep(0.2)
    if not reader.samples():
      print(
        "No controller poses yet. Put the headset on (it stops tracking when idle) "
        "and make sure the teleop app is the app in focus.",
        flush=True,
      )

    frame_path = Path(teleop["frame_path"])
    if args.calibrate:
      calibrate(reader, str(teleop["hand"]), frame_path)
      return
    if args.dump:
      dump(reader, float(teleop["print_hz"]) or 10.0)
      return

    # An explicit --yaw-deg is the operator overriding the measurement.
    frame = None if args.yaw_deg is not None else load_frame(frame_path)
    if frame is None:
      print(
        f"operator frame: assumed (yaw {teleop['yaw_deg']} deg). "
        "If hand motion drives the tray along the wrong axes, run --calibrate."
      )
    else:
      print(f"operator frame: measured, from {frame_path}")

    extra: list[tuple[dict, np.ndarray | None]] = []
    if second is not None:
      second_frame = load_frame(Path(second["teleop"]["frame_path"]))
      if second_frame is None:
        print(
          f"operator frame ({second['teleop']['hand']}): assumed "
          f"(yaw {second['teleop']['yaw_deg']} deg); run --calibrate with that config."
        )
      else:
        print(f"operator frame ({second['teleop']['hand']}): measured")
      extra.append((second, second_frame))

    configs = [config] + ([second] if second is not None else [])
    for cfg in configs:
      sample = reader.sample(cfg["teleop"]["hand"])
      if sample is None or time.monotonic() - sample.received_s > float(teleop["watchdog_s"]):
        raise RuntimeError("Fresh controller input is required before opening an arm")
    try:
      session = TeleopSession(config, reader, backend=args.backend, frame=frame, extra=extra, record_dir=args.record, dashboard_port=args.dashboard)
    except Exception as exc:
      backend = args.backend or config["robot"]["backend"]
      if backend == "i2rt":
        print(f"Could not open the arm: {exc}\n")
        print("CAN must be up first:")
        print("  sudo ip link set can0 down")
        print("  sudo ip link set can0 type can bitrate 1000000")
        print("  sudo ip link set can0 txqueuelen 1000")
        print("  sudo ip link set can0 up")
        print("\nOr run without hardware: --backend sim")
        raise SystemExit(1) from exc
      raise

    arms = ", ".join(f"{channel.hand}->{channel.label}" for channel in session.channels)
    print(
      f"backend: {session.backend}   arms: {arms}   "
      f"scale: {teleop['position_scale']}   yaw: {teleop['yaw_deg']} deg"
    )
    if teleop.get("gripper"):
      print("grip button clutches, trigger drives the gripper (squeeze to close).")
    else:
      print("trigger (or grip) drives the arm.")
    print("B/Y pauses both arms; A/X resumes. Release and press clutch to engage.")
    if args.record:
      print(f"Recording to {args.record}/. Click joystick to start/stop recording.")
    if args.dashboard:
      print(f"Dashboard: http://localhost:{args.dashboard}")
    print("B/Y parks the arms. After parking, Ctrl-C to begin shutdown.\n")
    is_real = any(ch.backend == "i2rt" for ch in session.channels)
    session.start()
    try:
      while session.alive():
        if not reader.alive():
          print("\nadb stream stopped.", flush=True)
          break
        time.sleep(0.2)
    except KeyboardInterrupt:
      pass
    finally:
      if is_real:
        session.enter_gravity_comp()
        print("\nArms in gravity comp — they will float but not fall.")
        print("Support the arms, then press Enter to disable motors.", flush=True)
        try:
          input()
        except (KeyboardInterrupt, EOFError):
          pass
        print("Disabling motors...", flush=True)
      session.close()
    if session.status()["fault"]:
      raise RuntimeError(session.status()["fault"])
  finally:
    reader.close()


if __name__ == "__main__":
  main()
