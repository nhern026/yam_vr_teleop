"""Arm backends for the deployment stack. Hardware construction enables motors
and runs gripper calibration; it is not a passive connection.

Reconstructed from `quest_teleop`'s use of it:

* ``make_arm(robot_config)`` returns an arm for ``robot_config["backend"]``
  (``"i2rt"`` or ``"mock"``);
* ``arm.read_state()`` returns an `ArmState` with six-element
  ``joint_position`` and ``joint_velocity``;
* ``arm.command_arm_positions(q)`` takes six arm joint targets;
* ``arm.close()`` releases the hardware.

Only the six arm joints cross this interface. A physical gripper, if the arm
has one, is held where it was found at start-up; `command_gripper` exists for
when the teleop grows a gripper channel.
"""

from __future__ import annotations

import math
import hashlib
import os
from pathlib import Path
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

ARM_DOF = 6


@dataclass
class ArmState:
  joint_position: np.ndarray
  joint_velocity: np.ndarray
  gripper_position: float | None = None
  timestamp_s: float = 0.0


class _VelocityEstimate:
  """Finite-difference joint velocity with a light low-pass.

  Used only when the driver does not report velocity itself. The teleop's
  runaway trip compares this against `safety.max_joint_velocity_rad_s`, so it
  is smoothed just enough that encoder noise cannot trip it.
  """

  def __init__(self, cutoff_hz: float = 20.0):
    self._tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    self._q: np.ndarray | None = None
    self._t = 0.0
    self._v = np.zeros(ARM_DOF, dtype=np.float64)

  def update(self, q: np.ndarray, t: float) -> np.ndarray:
    if self._q is not None and t > self._t:
      dt = t - self._t
      raw = (q - self._q) / dt
      alpha = dt / (dt + self._tau)
      self._v = self._v + alpha * (raw - self._v)
    self._q = q.copy()
    self._t = t
    return self._v.copy()


def _import_get_yam_robot():
  # The factory moved between i2rt releases.
  try:
    from i2rt.robots.get_robot import get_yam_robot
  except ImportError:
    from i2rt.robots.motor_chain_robot import get_yam_robot
  return get_yam_robot


def _claim_bus(config: dict):
  import fcntl
  channel = str(config.get("channel", "can0"))
  if config.get("mapping_verified") is not True:
    raise RuntimeError("Identify this adapter's physical arm and controller before setting mapping_verified")
  expected = str(config.get("adapter_serial", ""))
  device = Path("/sys/class/net") / channel / "device"
  actual = None
  for parent in [device.resolve(), *device.resolve().parents]:
    serial = parent / "serial"
    if serial.is_file():
      actual = serial.read_text().strip()
      break
  if not expected or actual != expected:
    raise RuntimeError(f"{channel}: USB serial {actual!r} does not match configured {expected!r}")
  # Process-wide bus ownership. Never start two driver threads on one CAN bus.
  lock_name = hashlib.sha256(channel.encode()).hexdigest()[:16]
  lock_dir = Path(f"/tmp/yam-teleop-{os.getuid()}")
  lock_dir.mkdir(mode=0o700, exist_ok=True)
  bus_lock = (lock_dir / lock_name).open("a")
  try:
    fcntl.flock(bus_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BaseException:
    bus_lock.close()
    raise RuntimeError(f"{channel} is already in use by another teleop process")
  return bus_lock


class I2rtArm:
  """A YAM on CAN through i2rt."""

  def __init__(self, config: dict[str, Any]):
    from i2rt.robots.utils import ArmType, GripperType

    get_yam_robot = _import_get_yam_robot()
    channel = str(config.get("channel", "can0"))
    arm_type = ArmType[str(config.get("arm_type", "yam")).upper()]
    gripper_type = GripperType[str(config.get("gripper_type", "no_gripper")).upper()]
    self._bus_lock = _claim_bus(config)
    self._robot = None
    try:
      # This pinned driver API enables motors and calibrates the gripper.
      # Do not retry TypeError: a failed constructor may already have hardware open.
      self._robot = get_yam_robot(channel=channel, arm_type=arm_type,
                                  gripper_type=gripper_type, zero_gravity_mode=True)
      self._lock = threading.Lock()
      self._velocity = _VelocityEstimate()
      q = self._raw_joint_position()
      if q.size not in (ARM_DOF, ARM_DOF + 1) or not np.all(np.isfinite(q)):
        raise RuntimeError("i2rt returned invalid initial joint feedback")
      self._gripper_in_vector = q.size == ARM_DOF + 1
      self._gripper_target = float(q[ARM_DOF]) if self._gripper_in_vector else None
    except BaseException:
      try:
        if self._robot is not None:
          self._robot.close()
      finally:
        self._bus_lock.close()
      raise

  def _raw_joint_position(self) -> np.ndarray:
    return np.asarray(self._robot.get_joint_pos(), dtype=np.float64).reshape(-1).copy()

  def _reported_velocity(self) -> np.ndarray | None:
    velocity = np.asarray(self._robot.get_observations()["joint_vel"], dtype=np.float64)
    if velocity.shape != (ARM_DOF,) or not np.all(np.isfinite(velocity)):
      raise RuntimeError("Invalid joint velocity feedback")
    return velocity.copy()

  def read_state(self) -> ArmState:
    with self._lock:
      now = time.monotonic()
      chain = self._robot.motor_chain
      if not chain.running or not self._robot._server_thread.is_alive():
        raise RuntimeError("Robot driver stopped; motor feedback is unavailable")
      stamp = chain.last_feedback_monotonic
      if not math.isfinite(stamp) or not 0 <= now - stamp <= 0.15:
        raise RuntimeError("Motor feedback is stale")
      q = self._raw_joint_position()
      if q.shape != ((7,) if self._gripper_in_vector else (6,)) or not np.all(np.isfinite(q)):
        raise RuntimeError("Invalid joint position feedback")
      position = q[:ARM_DOF].copy()
      velocity = self._reported_velocity()
      gripper = float(q[ARM_DOF]) if self._gripper_in_vector else None
      return ArmState(position, velocity, gripper, now)

  def command_arm_positions(self, target: np.ndarray) -> None:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if target.size != ARM_DOF or not np.all(np.isfinite(target)):
      raise ValueError(f"arm target must be {ARM_DOF} finite values")
    with self._lock:
      if self._gripper_in_vector:
        target = np.append(target, self._gripper_target)
      self._robot.command_joint_pos(target)

  def command_gripper(self, position: float) -> None:
    """Set the held gripper target, in whatever units i2rt reports it."""
    if not self._gripper_in_vector:
      raise RuntimeError("this i2rt arm does not expose the gripper in its joint vector")
    with self._lock:
      if not math.isfinite(position) or not 0 <= position <= 1:
        raise ValueError("gripper opening must be finite and between 0 and 1")
      self._gripper_target = float(position)

  def enter_gravity_comp(self) -> None:
    """Drop PD targets but keep gravity comp running. Arms float but don't fall."""
    if self._robot is not None:
      self._robot.enter_gravity_comp_idle()

  def close(self) -> None:
    if self._robot is not None:
      self._robot.close()
      self._robot = None
    self._bus_lock.close()


class SimArm:
  """i2rt's own MuJoCo `SimRobot`: the same `Robot` protocol as the hardware.

  Preferred over `MockArm` for anything beyond a smoke test, because it is the
  interface the real arm presents -- same joint vector, same gripper index,
  same mechanical limits (with i2rt's own +-0.15 rad buffer) -- so teleop code
  validated against it transfers to CAN unchanged.

  Two behaviours of `SimRobot` shape this adapter:

  * `command_joint_pos` teleports `qpos` and zeroes `qvel`, so the robot's own
    ``joint_vel`` is always zero once we start commanding. The runaway trip in
    `TeleopIK.check_velocity` reads that field, so trusting it would silently
    disable the trip. Velocity is finite-differenced here instead.
  * It always starts at all zeros, which for the YAM is *on* the j2/j3 limits.
    ``sim_initial_joint_position_rad`` seeds a non-degenerate pose instead.
  """

  def __init__(self, config: dict[str, Any]):
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType

    arm_type = ArmType[str(config.get("arm_type", "yam")).upper()]
    gripper_type = GripperType[str(config.get("gripper_type", "no_gripper")).upper()]
    self._robot = get_yam_robot(arm_type=arm_type, gripper_type=gripper_type, sim=True)

    self._lock = threading.Lock()
    self._velocity = _VelocityEstimate()
    dofs = int(self._robot.num_dofs())
    if dofs not in (ARM_DOF, ARM_DOF + 1):
      raise RuntimeError(f"sim robot reports {dofs} dofs; expected {ARM_DOF} or {ARM_DOF + 1}")
    self._gripper_in_vector = dofs == ARM_DOF + 1
    self._gripper_target = 0.0 if self._gripper_in_vector else None

    start = np.asarray(
      config.get("sim_initial_joint_position_rad", [0.0, 0.9, 1.2, 0.0, 0.6, 0.0]),
      dtype=np.float64,
    )
    if start.shape != (ARM_DOF,):
      raise ValueError(f"robot.sim_initial_joint_position_rad must contain {ARM_DOF} values")
    self.command_arm_positions(start)

  def read_state(self) -> ArmState:
    with self._lock:
      now = time.monotonic()
      q = np.asarray(self._robot.get_joint_pos(), dtype=np.float64).reshape(-1)
      position = q[:ARM_DOF].copy()
      # Deliberately not `get_observations()["joint_vel"]`; see the class docstring.
      velocity = self._velocity.update(position, now)
      gripper = float(q[ARM_DOF]) if self._gripper_in_vector else None
      return ArmState(position, velocity, gripper, now)

  def command_arm_positions(self, target: np.ndarray) -> None:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if target.size != ARM_DOF or not np.all(np.isfinite(target)):
      raise ValueError(f"arm target must be {ARM_DOF} finite values")
    with self._lock:
      if self._gripper_in_vector:
        target = np.append(target, self._gripper_target)
      self._robot.command_joint_pos(target)

  def command_gripper(self, position: float) -> None:
    """Set the gripper target. i2rt's sim gripper is normalised: 0 shut, 1 open."""
    if not self._gripper_in_vector:
      raise RuntimeError("this sim arm was built with no_gripper")
    if not math.isfinite(position) or not 0 <= position <= 1:
      raise ValueError("gripper opening must be finite and between 0 and 1")
    with self._lock:
      self._gripper_target = float(position)

  def close(self) -> None:
    close = getattr(self._robot, "close", None)
    if callable(close):
      close()


class MockArm:
  """A first-order servo: each joint closes on its target with time constant ``tau``.

  Integrated on the wall clock at every call, so it behaves the same whatever
  rate the control loop actually achieves.
  """

  def __init__(self, config: dict[str, Any]):
    self._tau = float(config.get("mock_servo_tau_s", 0.05))
    if self._tau <= 0.0:
      raise ValueError("robot.mock_servo_tau_s must be positive")
    start = np.asarray(
      config.get("mock_initial_joint_position_rad", [0.0] * ARM_DOF), dtype=np.float64
    )
    if start.shape != (ARM_DOF,):
      raise ValueError(f"robot.mock_initial_joint_position_rad must contain {ARM_DOF} values")
    self._lock = threading.Lock()
    self._q = start.copy()
    self._target = start.copy()
    self._v = np.zeros(ARM_DOF, dtype=np.float64)
    self._t = time.monotonic()

  def _advance(self) -> float:
    now = time.monotonic()
    dt = now - self._t
    if dt > 0.0:
      step = (1.0 - math.exp(-dt / self._tau)) * (self._target - self._q)
      self._v = step / dt
      self._q = self._q + step
      self._t = now
    return now

  def read_state(self) -> ArmState:
    with self._lock:
      now = self._advance()
      return ArmState(self._q.copy(), self._v.copy(), None, now)

  def command_arm_positions(self, target: np.ndarray) -> None:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if target.size != ARM_DOF or not np.all(np.isfinite(target)):
      raise ValueError(f"arm target must be {ARM_DOF} finite values")
    with self._lock:
      self._advance()
      self._target = target.copy()

  def close(self) -> None:
    pass


def make_arm(robot_config: dict[str, Any]):
  backend = str(robot_config["backend"]).lower()
  if backend == "i2rt":
    return I2rtArm(robot_config)
  if backend == "sim":
    return SimArm(robot_config)
  if backend == "mock":
    return MockArm(robot_config)
  raise ValueError(f"unknown robot.backend {backend!r}; expected 'i2rt', 'sim' or 'mock'")
