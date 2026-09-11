"""Record a YAM's ready or park pose by hand-guiding the real arm.

Opening enables motors and calibrates the gripper, which moves. After startup,
no arm position targets are sent; gravity compensation assists hand guiding. Positions print live; pressing
Enter captures wherever it is standing.

    # ready pose -> balancing_act/yam_home.json, marked verified
    python -m deployment.record_pose --channel can0 --gripper linear_4310 --save-home

    # per-arm ready pose, or a park pose -> printed for you to paste
    python -m deployment.record_pose --channel can1 --gripper linear_4310

This is the step that clears the pre-flight's "reset pose is unverified"
failure, and it is deliberately the only way to clear it: the flag means a
human watched this arm reach this pose, which is not something a config edit
can honestly assert.

Hold the arm before you start -- gravity compensation carries the arm's own
weight, not a payload, and not a gripper that is heavier than the model thinks.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

from balancing_act.assets import HOME_POSE_FILE

ARM_DOF = 6


def main() -> None:
  parser = argparse.ArgumentParser(prog="python -m deployment.record_pose")
  parser.add_argument("--channel", default="can0", help="CAN channel the arm is on.")
  parser.add_argument("--arm", default="yam", help="Arm variant, as named in i2rt's ArmType.")
  parser.add_argument(
    "--gripper",
    default="linear_4310",
    help="Gripper variant, as named in i2rt's GripperType. Must match what is mounted.",
  )
  parser.add_argument(
    "--save-home",
    action="store_true",
    help=(
      f"Write the captured pose to {HOME_POSE_FILE.name} and set verified_on_hardware, "
      "which is what lets the i2rt backend start."
    ),
  )
  args = parser.parse_args()

  from i2rt.robots.get_robot import get_yam_robot
  from i2rt.robots.utils import ArmType, GripperType

  arm_type = ArmType[args.arm.upper()]
  gripper_type = GripperType[args.gripper.upper()]

  print(f"Opening {args.arm} on {args.channel} with gripper {args.gripper}.")
  print("Opening enables motors and moves the gripper for calibration. Support the arm; keep fingers clear.\n")
  # zero_gravity_mode is get_yam_robot's default and we never command a
  # position, so the arm stays free the whole time this script runs.
  robot = get_yam_robot(args.channel, arm_type=arm_type, gripper_type=gripper_type)

  captured: list[np.ndarray] = []
  done = threading.Event()

  def wait_for_enter() -> None:
    try:
      line = sys.stdin.readline()
      if not line:
        done.set()
        return
    except Exception:  # noqa: BLE001 - stdin closed is just "stop"
      pass
    captured.append(np.empty(0))  # main thread reads feedback before closing
    done.set()

  threading.Thread(target=wait_for_enter, daemon=True).start()
  print("Guide the arm to the pose you want, then press Enter. Ctrl-C aborts.\n")
  try:
    while not done.is_set():
      q = np.asarray(robot.get_joint_pos(), dtype=np.float64).reshape(-1)
      joints = " ".join(f"{value:+7.3f}" for value in q[:ARM_DOF])
      gripper = f"  gripper={q[ARM_DOF]:+.3f}" if q.size > ARM_DOF else ""
      print(f"\r\x1b[2K  q = [{joints}]{gripper}", end="", flush=True)
      time.sleep(0.05)
    if captured:
      captured[0] = np.asarray(robot.get_joint_pos(), dtype=np.float64).reshape(-1).copy()
  except KeyboardInterrupt:
    print("\naborted; nothing written.")
    return
  finally:
    close = getattr(robot, "close", None)
    if callable(close):
      close()

  if not captured:
    raise RuntimeError("stdin closed; no pose recorded")
  pose = captured[0][:ARM_DOF]
  if pose.shape != (6,) or not np.all(np.isfinite(pose)):
    raise RuntimeError("Invalid pose feedback; nothing written")
  values = [round(float(v), 4) for v in pose]
  print(f"\n\ncaptured: {values}\n")

  if not args.save_home:
    print("Paste it into this arm's config as:\n")
    print("deploy:")
    print(f"  reset_joint_position_rad: {values}")
    print("\nor rerun with --save-home to write it to the shared home file.")
    return

  payload = {}
  if HOME_POSE_FILE.exists():
    payload = json.loads(HOME_POSE_FILE.read_text())
  payload["joint_position_rad"] = values
  payload["verified_on_hardware"] = True
  payload["recorded"] = time.strftime("%Y-%m-%d %H:%M:%S")
  payload["recorded_on_channel"] = args.channel
  HOME_POSE_FILE.write_text(json.dumps(payload, indent=2) + "\n")
  print(f"written to {HOME_POSE_FILE}, marked verified_on_hardware.")
  print("\nNOTE: this file is shared by both arms. If the second arm is mounted")
  print("differently, record its pose too and put it in that arm's config as")
  print("deploy.reset_joint_position_rad instead of overwriting this file.")


if __name__ == "__main__":
  main()
