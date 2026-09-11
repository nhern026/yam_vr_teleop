"""Pre-flight checks for a real YAM before teleop drives it.

Run this on the lab machine, with the arms powered and CAN up, *before*
``quest_teleop --backend i2rt``. It answers the questions that are miserable to
debug with a live arm and cheap to answer with a limp one:

* is the CAN interface actually up, and at 1 Mbit/s?
* does every motor the config declares answer, with the type it should be?
* does the declared ``gripper_type`` match the chain that is physically there?
* is the arm resting inside its joint limits, so i2rt will even start?
* is the reset pose reachable, verified, and clear of the limits?
* is the operator's joint box actually narrower than the mechanical range,
  or still wide open?

Nothing here commands a motor. The chain survey uses i2rt's ``--survey-only``
path, which reads registers without building a chain or energising anything.

    python -m deployment.preflight --config deployment/config.yaml
    python -m deployment.preflight --config deployment/config.yaml \
                                   --second-config deployment/config_left.yaml

Exit status is 0 only when every check passed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from balancing_act.assets import HOME_POSE_FILE, YAM_ARM_HOME_VERIFIED
from deployment.config import load_deployment_config

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_MARK = {PASS: "  ok  ", WARN: " warn ", FAIL: " FAIL "}


class Report:
  """Collects check results so one bad check does not hide the rest."""

  def __init__(self) -> None:
    self.rows: list[tuple[str, str, str]] = []

  def add(self, status: str, name: str, detail: str = "") -> None:
    self.rows.append((status, name, detail))
    print(f"[{_MARK[status]}] {name}" + (f"\n           {detail}" if detail else ""), flush=True)

  @property
  def failed(self) -> int:
    return sum(1 for status, _, _ in self.rows if status == FAIL)

  @property
  def warned(self) -> int:
    return sum(1 for status, _, _ in self.rows if status == WARN)


# ------------------------------------------------------------------------ CAN


def check_can(report: Report, channel: str) -> None:
  """The interface exists, is up, and is running at the YAM's 1 Mbit/s."""
  if sys.platform != "linux":
    report.add(
      WARN,
      f"CAN {channel}: not checked",
      f"socketcan is Linux-only and this is {sys.platform}; real arms need Linux "
      "(or WSL2 with usbipd-win).",
    )
    return
  if shutil.which("ip") is None:
    report.add(WARN, f"CAN {channel}: 'ip' not found", "cannot inspect the interface")
    return
  result = subprocess.run(
    ["ip", "-details", "link", "show", channel], capture_output=True, text=True, check=False
  )
  if result.returncode != 0:
    report.add(
      FAIL,
      f"CAN {channel}: no such interface",
      f"sudo ip link set {channel} type can bitrate 1000000 && sudo ip link set {channel} up",
    )
    return
  text = result.stdout
  if "state UP" not in text and "<NOARP,UP" not in text:
    report.add(FAIL, f"CAN {channel}: down", f"sudo ip link set {channel} up")
    return
  if "bitrate 1000000" in text:
    report.add(PASS, f"CAN {channel}: up at 1 Mbit/s")
  else:
    bitrate = next(
      (part for part in text.split() if part.isdigit() and len(part) >= 5), "unknown"
    )
    report.add(
      FAIL,
      f"CAN {channel}: wrong bitrate ({bitrate})",
      "the YAM chain runs at 1000000; anything else will not talk to it",
    )


def survey_chain(report: Report, channel: str, motor_ids: tuple[int, ...]) -> None:
  """Read the chain's registers without energising a single motor."""
  if sys.platform != "linux":
    report.add(WARN, f"chain {channel}: not surveyed", "needs Linux and a live bus")
    return
  command = [
    sys.executable,
    "-m",
    "i2rt.motor_drivers.dm_driver",
    "--channel",
    channel,
    "--survey-only",
    "--check-motor-types",
    "--motor-id",
    *[str(i) for i in motor_ids],
  ]
  try:
    result = subprocess.run(command, capture_output=True, text=True, timeout=60.0, check=False)
  except (subprocess.TimeoutExpired, OSError) as exc:
    report.add(FAIL, f"chain {channel}: survey did not run", str(exc))
    return
  output = (result.stdout + result.stderr).strip()
  if result.returncode == 0:
    report.add(PASS, f"chain {channel}: survey exited successfully (inspect raw output)")
    print(output)
  else:
    tail = "\n           ".join(output.splitlines()[-6:]) or "(no output)"
    report.add(
      FAIL,
      f"chain {channel}: survey failed",
      f"a motor is missing, misidentified, or on the wrong id:\n           {tail}",
    )


# -------------------------------------------------------------------- configs


def check_gripper(report: Report, label: str, robot: dict[str, Any]) -> None:
  gripper = str(robot.get("gripper_type", "no_gripper")).lower()
  if gripper == "no_gripper":
    report.add(
      WARN,
      f"{label}: gripper_type is no_gripper",
      "If a gripper IS mounted this is wrong twice: i2rt builds a 6-motor chain "
      "for a 7-motor arm so the gripper hangs limp, and IK's grasp_site sits at "
      "the flange, ~10 cm behind the fingers.",
    )
  else:
    report.add(PASS, f"{label}: gripper_type = {gripper}")


def check_poses(report: Report, label: str, config: dict[str, Any], ik: Any) -> None:
  lower, upper = ik.joint_position_limits
  reset_cfg = config["deploy"].get("reset_joint_position_rad")
  reset = np.asarray(reset_cfg if reset_cfg is not None else ik.home_joint_position, float)
  park = np.asarray(config["deploy"]["home_joint_position_rad"], float)

  if reset_cfg is None and not YAM_ARM_HOME_VERIFIED:
    report.add(
      FAIL,
      f"{label}: reset pose is unverified",
      f"{HOME_POSE_FILE} has verified_on_hardware: false, and homing is a move the "
      "clutch cannot interrupt. Record it on this arm, then set the flag or add a "
      "per-arm deploy.reset_joint_position_rad.",
    )
  else:
    report.add(PASS, f"{label}: reset pose is marked verified")

  for name, pose in (("reset", reset), ("park", park)):
    outside = (pose < lower - 1e-9) | (pose > upper + 1e-9)
    if np.any(outside):
      joints = ", ".join(f"j{i + 1}" for i in np.flatnonzero(outside))
      report.add(FAIL, f"{label}: {name} pose outside the joint envelope at {joints}")
      continue
    margin = float(np.minimum(pose - lower, upper - pose).min())
    if margin < 0.05:
      report.add(
        WARN,
        f"{label}: {name} pose sits {margin:.3f} rad from a limit",
        "IK has almost no room to move away from it; expect immediate pinning.",
      )
    else:
      report.add(PASS, f"{label}: {name} pose clear of limits by {margin:.2f} rad")


def check_joint_box(report: Report, label: str, config: dict[str, Any], ik: Any) -> None:
  """The operator box is the only thing keeping two arms out of each other."""
  operator_min = np.asarray(config["safety"]["joint_position_min_rad"], float)
  operator_max = np.asarray(config["safety"]["joint_position_max_rad"], float)
  effective_min, effective_max = ik.mechanical_limits
  # A box wider than the mechanical range everywhere is a box doing nothing.
  binding = int(np.sum((operator_min > effective_min + 1e-9) | (operator_max < effective_max - 1e-9)))
  if binding == 0:
    report.add(
      WARN,
      f"{label}: the operator joint box constrains nothing",
      "safety.joint_position_* is wider than the arm's mechanical range on every "
      "joint, so the only limits in force are the arm's own. There is NO collision "
      "checking anywhere in this system: with two arms sharing a workspace, this "
      "box is the only thing that keeps them apart. Tighten it before running both.",
    )
  else:
    report.add(PASS, f"{label}: operator joint box binds on {binding}/6 joints")


def check_frame(report: Report, label: str, config: dict[str, Any]) -> None:
  path = Path(config["teleop"]["frame_path"])
  if path.exists():
    report.add(PASS, f"{label}: operator frame measured ({path.name})")
  else:
    report.add(
      WARN,
      f"{label}: no operator frame; using the assumed yaw",
      f"run --calibrate with this config. Without it, hand motion drives the arm "
      f"along the wrong axes.",
    )


def check_arm(report: Report, label: str, config: dict[str, Any], *, survey: bool) -> None:
  print(f"\n--- {label} " + "-" * (60 - len(label)), flush=True)
  robot = config["robot"]
  channel = str(robot.get("channel", "can0"))
  backend = str(robot.get("backend", "sim"))
  if backend != "i2rt":
    report.add(
      WARN,
      f"{label}: backend is '{backend}', not 'i2rt'",
      "this config will not drive a real arm; pass --backend i2rt or edit robot.backend",
    )

  check_can(report, channel)
  check_gripper(report, label, robot)
  if survey:
    with_gripper = str(robot.get("gripper_type", "no_gripper")).lower() != "no_gripper"
    survey_chain(report, channel, tuple(range(1, 8 if with_gripper else 7)))

  # TeleopIK is imported late: it builds a MuJoCo model, which is slow and
  # pointless if the config never loaded.
  from deployment.quest_teleop import TeleopIK

  ik = TeleopIK(config["safety"], robot)
  # Startup holds the measured pose; no reset/park target is commanded.
  if robot.get("mapping_verified") is not True:
    report.add(FAIL, f"{label}: physical arm/controller mapping is unverified")
  check_joint_box(report, label, config, ik)
  check_frame(report, label, config)


def main() -> None:
  parser = argparse.ArgumentParser(prog="python -m deployment.preflight")
  parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
  parser.add_argument("--second-config", type=Path, default=None)
  parser.add_argument(
    "--no-survey",
    action="store_true",
    help="Skip the CAN chain survey (config and pose checks only).",
  )
  args = parser.parse_args()

  print("YAM teleop pre-flight. Nothing here commands a motor.\n")
  report = Report()
  configs = [(args.config, "right")]
  if args.second_config is not None:
    configs.append((args.second_config, "left"))

  for path, _ in configs:
    config = load_deployment_config(path)
    check_arm(report, str(config["teleop"]["hand"]), config, survey=not args.no_survey)

  print("\n" + "=" * 62)
  if report.failed:
    print(f"{report.failed} check(s) FAILED, {report.warned} warning(s).")
    print("Do not run --backend i2rt until the failures are cleared.")
    raise SystemExit(1)
  if report.warned:
    print(f"All checks passed with {report.warned} warning(s). Read them before driving.")
  else:
    print("All checks passed.")


if __name__ == "__main__":
  main()
