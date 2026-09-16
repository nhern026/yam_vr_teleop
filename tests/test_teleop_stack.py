"""Full-stack teleop tests against the real mujoco/mink/i2rt install.

Unlike `test_backform.py`, which stubs the heavy imports so it can run anywhere,
this exercises the actual stack: i2rt's YAM model, a real `mink` IK solve, and
i2rt's own `SimRobot` standing in for the arm. It needs the real dependencies
but no hardware, which is exactly the envelope a laptop can cover.

Run: python tests/test_teleop_stack.py
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import deployment.quest_teleop as qt  # noqa: E402
from deployment.config import load_deployment_config  # noqa: E402
from deployment.robot import ArmState, make_arm  # noqa: E402


def _cfg(name="config.yaml", *, gripper=False):
  cfg = load_deployment_config(ROOT / "deployment" / name)
  cfg["robot"]["backend"] = "sim"
  cfg["teleop"]["print_hz"] = 0.0
  cfg["teleop"]["move_s"] = 1.0
  cfg["teleop"]["settle_s"] = 0.2
  if gripper:
    cfg["robot"]["gripper_type"] = "linear_4310"
    cfg["teleop"]["gripper"] = True
  return cfg


class FakeReader:
  """Both controllers tracing circles, with scriptable buttons."""

  def __init__(self):
    self.t0 = time.monotonic()
    self.on = {"right": False, "left": False}
    self.trig = {"right": 0.0, "left": 0.0}
    self.stop = {"right": False, "left": False}
    self.home = {"right": False, "left": False}
    self.frozen = False

  def alive(self):
    return True

  def close(self):
    pass

  def samples(self):
    return {h: self.sample(h) for h in ("right", "left")}

  def sample(self, hand):
    t = 0.0 if self.frozen else time.monotonic() - self.t0
    sign = 1.0 if hand == "right" else -1.0
    p = np.array([0.10 * np.cos(t) * sign, 0.10 * np.sin(t), -0.30])
    grip = 1.0 if self.on[hand] else 0.0
    return qt.ControllerSample(
      hand=hand,
      position=p,
      quat=np.array([1.0, 0.0, 0.0, 0.0]),
      trigger=self.trig[hand],
      grip=grip,
      clutch=self.trig[hand] > 0.5 or grip > 0.5,
      home=self.home[hand],
      stop=self.stop[hand],
      joystick=(0.0, 0.0),
      received_s=time.monotonic(),
    )


# --------------------------------------------------------------------- the IK

ik = qt.TeleopIK(_cfg()["safety"])
lower, upper = ik.joint_position_limits

# The YAM's j2/j3 mechanical lower limits are exactly 0.0. Anything that homes
# to all zeros therefore starts pinned against two limits, which is why the
# reset pose is a recorded file and not a default.
assert abs(lower[1]) < 1e-9 and abs(lower[2]) < 1e-9, "expected j2/j3 lower limits at zero"
home = np.asarray(ik.home_joint_position, dtype=np.float64)
margin = float(np.minimum(home - lower, upper - home).min())
assert margin > 0.1, f"reset pose is only {margin:.3f} rad from a joint limit"

# IK tracks a commanded end-effector move to sub-millimetre accuracy.
start = np.array([0.0, 0.9, 1.2, 0.0, 0.6, 0.0])
pos0, quat0 = ik.ee_pose(ArmState(start, np.zeros(6)))
goal = pos0 + np.array([0.03, -0.02, 0.01])
q = start.copy()
for _ in range(50):
  q = ik.joint_target_from_pose(ArmState(q, np.zeros(6)), goal, quat0, 0.01)
reached, _ = ik.ee_pose(ArmState(q, np.zeros(6)))
err_mm = float(np.linalg.norm(reached - goal)) * 1000.0
assert err_mm < 1.0, f"IK error {err_mm:.2f} mm"
print(f"ik ok (tracked a 37 mm move to {err_mm:.3f} mm, reset pose {margin:.2f} rad clear)")

# The runaway trip fires on measured speed, whatever the backend reports.
try:
  ik.check_velocity(ArmState(start, np.full(6, 99.0)))
  raise SystemExit("runaway trip did not fire")
except RuntimeError as exc:
  assert "safety limit" in str(exc)
print("runaway trip ok")


# ------------------------------------------------------------- the sim arm

arm = make_arm({"backend": "sim", "gripper_type": "linear_4310"})
seeded = arm.read_state()
assert seeded.joint_position.shape == (6,)
# Seeded away from the degenerate zeros pose.
assert float(np.abs(seeded.joint_position).sum()) > 0.5, "sim arm seeded at zeros"
target = np.array([0.1, 1.2, 1.0, 0.2, 0.5, -0.1])
arm.command_arm_positions(target)
assert np.allclose(arm.read_state().joint_position, target, atol=1e-3)
# SimRobot zeroes its own qvel on command, so velocity must be derived here or
# the runaway trip silently dies. Moving the arm must show a non-zero speed.
speeds = []
for k in range(12):
  arm.command_arm_positions(target + np.array([0.0, 0.02 * k, 0.0, 0.0, 0.0, 0.0]))
  time.sleep(0.01)
  speeds.append(float(np.abs(arm.read_state().joint_velocity).max()))
assert max(speeds) > 0.05, f"sim arm reports no velocity ({max(speeds):.4f} rad/s)"
arm.command_gripper(0.25)
arm.command_arm_positions(target)
assert abs(float(arm.read_state().gripper_position) - 0.25) < 1e-6
arm.close()
print(f"sim arm ok (velocity derived, peak {max(speeds):.2f} rad/s; gripper held)")


# ------------------------------------------------------- one loop, two arms

reader = FakeReader()
session = qt.TeleopSession(
  _cfg(gripper=True), reader, extra=[(_cfg("config_left.yaml", gripper=True), None)]
)
assert [c.hand for c in session.channels] == ["right", "left"]
session.start()
try:
  time.sleep(0.1)  # let the released-clutch samples reach both channels
  assert session.status()["mode"] == qt.IDLE, session.status()

  # Both hands clutch with different trigger pulls; grippers stay independent.
  reader.on["right"] = reader.on["left"] = True
  reader.trig["right"], reader.trig["left"] = 0.9, 0.2
  time.sleep(1.5)
  status = session.status()
  assert status["mode"] == qt.ENGAGED, status["mode"]
  assert status["hz"] > 50.0, f"loop fell to {status['hz']} Hz with two arms"
  assert all(a["engaged"] for a in status["arms"])
  grips = [a["gripper"] for a in status["arms"]]
  assert abs(grips[0] - 0.1) < 1e-6 and abs(grips[1] - 0.8) < 1e-6, grips
  travel = [float(np.linalg.norm(a["tray_travel"])) for a in status["arms"]]
  assert all(t > 0.005 for t in travel), travel
  print(f"dual-arm ok ({status['hz']:.0f} Hz, grippers {grips}, travel {np.round(travel, 3)})")

  # One hand releasing must not disturb the other arm.
  reader.on["right"] = False
  time.sleep(0.5)
  status = session.status()
  assert not status["arms"][0]["engaged"] and status["arms"][1]["engaged"], status["arms"]
  assert status["mode"] == qt.ENGAGED
  held = np.asarray(status["arms"][0]["joints"])
  time.sleep(0.4)
  assert np.allclose(held, session.status()["arms"][0]["joints"], atol=1e-3), "released arm drifted"
  print("independent release ok (released arm holds, other keeps driving)")

  # A stale stream declutches rather than chasing the last pose.
  reader.on["left"] = False
  time.sleep(0.3)
  assert session.status()["mode"] == qt.IDLE

  # Shared pause holds both arms without a parking movement.
  held = [c.read_state().joint_position.copy() for c in session.channels]
  reader.stop["left"] = True
  time.sleep(0.2)
  reader.stop["left"] = False
  assert session.status()["mode"] == qt.PARKED, session.status()
  for expected, channel in zip(held, session.channels):
    assert np.allclose(channel.read_state().joint_position, expected, atol=1e-3)
  reader.home["left"] = True
  time.sleep(.1)
  reader.home["left"] = False
  assert session.status()["mode"] == qt.HOMING, session.status()
  time.sleep(1.5)  # move_s=1.0 + settle_s=0.2
  assert session.status()["mode"] == qt.IDLE, session.status()
  for channel in session.channels:
    assert np.allclose(channel.read_state().joint_position, channel.reset_pose, atol=1e-2)
  print("shared pause/resume ok (homes to reset pose)")
finally:
  session.close()

print("ALL PASS")
