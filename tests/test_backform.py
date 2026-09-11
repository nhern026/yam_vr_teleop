import sys, time, json, tempfile, shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests" / "stubs"), str(ROOT)]

from deployment.camera import OneEuroFilter
from deployment.config import load_deployment_config
from deployment.robot import make_arm, ArmState
import deployment.quest_teleop as qt   # proves every import in the original resolves

# One Euro: still signal passes, noise shrinks, fast motion tracked, reset works
f = OneEuroFilter(min_cutoff_hz=1.5, beta=8.0)
rng = np.random.default_rng(0)
dt = 0.01
out = [f(np.array([0.2, 0.1, 0.0]) + rng.normal(0, 0.002, 3), dt) for _ in range(300)]
assert np.std(np.array(out)[100:], axis=0).max() < 0.0012, "resting jitter not reduced"
f.reset(); f(np.zeros(3), dt)
xs = [f(np.array([0.5 * k * dt, 0, 0]), dt)[0] for k in range(1, 101)]
lag = 0.5 - xs[-1]
assert lag < 0.02, f"moving lag too large: {lag}"
assert OneEuroFilter._alpha(1.0, 0.01) > 0
print(f"filter ok (moving lag at 0.5 m/s: {lag*1000:.1f} mm)")

# Config: loads, resolves paths next to the file, names missing keys
cfg = load_deployment_config(str(ROOT / "deployment") + "/config.yaml")
assert Path(cfg["teleop"]["frame_path"]).parent == ROOT / "deployment" / "calibration"
left = load_deployment_config(str(ROOT / "deployment") + "/config_left.yaml")
assert left["teleop"]["hand"] == "left" and left["robot"]["channel"] == "can1"
tmp = Path(tempfile.mkdtemp()) / "bad.yaml"
tmp.write_text("teleop: {hand: right}\nrobot: {backend: mock}\n")
try: load_deployment_config(tmp); raise SystemExit("should have failed")
except ValueError as e: assert "teleop.control_hz" in str(e) and "safety" in str(e)
print("config ok")

# Mock arm: converges on target, reports velocity
arm = make_arm({"backend": "mock", "mock_servo_tau_s": 0.05})
target = np.array([0.3, -0.2, 0.1, 0.0, 0.4, -0.1])
arm.command_arm_positions(target)
time.sleep(0.05); mid = arm.read_state()
assert np.abs(mid.joint_velocity).max() > 0
time.sleep(0.4); s = arm.read_state()
assert isinstance(s, ArmState) and np.allclose(s.joint_position, target, atol=1e-3)
print("mock arm ok")

# i2rt backend against a fake driver: 6-vector and gripper-appended 7-vector
from i2rt.robots import get_robot as fake
for gtype, n in (("no_gripper", 6), ("linear_4310", 7)):
  with patch("deployment.robot._claim_bus", return_value=Mock()):
    a = make_arm({"backend": "i2rt", "channel": "can1", "gripper_type": gtype})
  a._robot.motor_chain = Mock(running=True, last_feedback_monotonic=time.monotonic())
  a._robot._server_thread = Mock()
  a._robot._server_thread.is_alive.return_value = True
  st = a.read_state(); assert st.joint_position.shape == (6,)
  a.command_arm_positions(np.zeros(6))
  sent = fake.LAST["robot"].sent[-1]
  assert sent.size == n and fake.LAST["channel"] == "can1"
  if n == 7: assert np.isclose(sent[6], 0.6), "gripper not held"
  a.close(); assert fake.LAST["robot"].closed
print("i2rt adapter ok")

# Original script's pure parts, now runnable
assert np.allclose(qt.frame_from_push(np.array([0, 0, -0.3])), qt.frame_rotation(0.0))
pose = " ".join(str(v) for v in np.eye(4).reshape(-1))
line = f"l:{pose}|r:{pose}&A,rightTrig 0.9,rightGrip 0.0,rightJS 0.1 -0.2"
smp = qt.parse_log_payload(line, 0.0)
r = smp["right"]
assert r.clutch and r.home and not r.stop and r.joystick == (0.1, -0.2) and not smp["left"].clutch
print("quest_teleop helpers ok")
print("ALL PASS")
