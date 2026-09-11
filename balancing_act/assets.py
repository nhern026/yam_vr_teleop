"""Robot assets shared with the deployment stack.

`YAM_ARM_HOME_JOINT_POS` is the reset pose teleop walks to at start-up and on
the A button: the arm raised over the table, ready to be driven. The original
value was tuned for another setup and is not recoverable, so it is read from
``yam_home.json`` beside this file. Record it on your own arm (see README).

Without that file the reset pose falls back to all zeros, the i2rt zero pose.
That keeps start-up safe (the arm homes to where it rests) but leaves the IK
starting from a folded configuration, so set the file before real use.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HOME_POSE_FILE = Path(__file__).with_name("yam_home.json")


def _load_home_pose() -> tuple[tuple[float, ...], bool]:
  """The reset pose, and whether anyone has confirmed it on a real arm.

  The flag is what the i2rt backend gates on. Homing is an uninterruptible
  smoothstep move, so a pose that has only ever been checked in simulation is
  a pose the arm will drive into a table without giving the operator a way to
  stop it. Shipping a plausible default and shipping a *trusted* one are
  different things, and only the file can say which this is.
  """
  if not HOME_POSE_FILE.exists():
    print(
      f"[balancing_act] {HOME_POSE_FILE.name} not found; the reset pose is all zeros. "
      "Record a raised pose for your table before driving the real arm.",
      file=sys.stderr,
    )
    return (0.0,) * 6, False
  loaded = json.loads(HOME_POSE_FILE.read_text())
  values = loaded["joint_position_rad"]
  if len(values) != 6:
    raise ValueError(f"{HOME_POSE_FILE} must hold six joint positions")
  return tuple(float(value) for value in values), bool(loaded.get("verified_on_hardware", False))


YAM_ARM_HOME_JOINT_POS, YAM_ARM_HOME_VERIFIED = _load_home_pose()
