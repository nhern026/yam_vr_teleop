"""Deployment config loading.

Every key listed in `_REQUIRED` is read somewhere in `quest_teleop`; a missing
one fails here with its full name instead of as a KeyError mid-startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
import math

_REQUIRED: dict[str, tuple[str, ...]] = {
  "teleop": (
    "hand",
    "control_hz",
    "position_scale",
    "yaw_deg",
    "orientation",
    "max_tray_speed_m_s",
    "max_tray_omega_rad_s",
    "filter_min_cutoff_hz",
    "filter_beta",
    "max_hand_jump_m",
    "joint_smoothing_tau_s",
    "watchdog_s",
    "move_s",
    "settle_s",
    "print_hz",
    "apk_path",
    "frame_path",
  ),
  "robot": ("backend",),
  "safety": (
    "max_command_offset_rad",
    "max_joint_velocity_rad_s",
    "joint_position_min_rad",
    "joint_position_max_rad",
  ),
  "deploy": ("home_joint_position_rad",),
}

# Paths in the config are taken relative to the config file, not the shell's
# working directory, so the calibration file lands in the same place however
# the script is launched.
_PATH_KEYS = (("teleop", "apk_path"), ("teleop", "frame_path"))


def load_deployment_config(path: str | Path) -> dict[str, Any]:
  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(f"config not found: {path}")
  config = yaml.safe_load(path.read_text()) or {}
  if not isinstance(config, dict):
    raise ValueError(f"{path} must hold a mapping at the top level")

  missing = []
  for section, keys in _REQUIRED.items():
    block = config.get(section)
    if not isinstance(block, dict):
      missing.append(section)
      continue
    missing.extend(f"{section}.{key}" for key in keys if key not in block)
  if missing:
    raise ValueError(f"{path} is missing: {', '.join(missing)}")

  for name in ("joint_position_min_rad", "joint_position_max_rad"):
    if len(config["safety"][name]) != 6:
      raise ValueError(f"safety.{name} must contain six values")
  lower = config["safety"]["joint_position_min_rad"]
  upper = config["safety"]["joint_position_max_rad"]
  if any(lo >= hi for lo, hi in zip(lower, upper)):
    raise ValueError("every safety joint minimum must be below its maximum")

  validate_config(config)
  base = path.resolve().parent
  for section, key in _PATH_KEYS:
    value = Path(config[section][key]).expanduser()
    config[section][key] = str(value if value.is_absolute() else base / value)
  return config


def validate_config(config: dict) -> None:
  """Reject invalid numeric settings before opening any hardware."""
  for section in ("teleop", "safety"):
    for key, value in config[section].items():
      if isinstance(value, (float, int)) and not isinstance(value, bool) and not math.isfinite(value):
        raise ValueError(f"{section}.{key} must be finite")
  teleop = config["teleop"]
  for key in ("control_hz", "position_scale", "max_tray_speed_m_s", "max_tray_omega_rad_s",
              "filter_min_cutoff_hz", "max_hand_jump_m", "watchdog_s", "move_s"):
    value = teleop[key]
    if isinstance(value, bool) or not isinstance(value, (float, int)) or value <= 0:
      raise ValueError(f"teleop.{key} must be a positive number")
  if not 1 <= teleop["control_hz"] <= 250:
    raise ValueError("control_hz must be between 1 and 250")
  for key in ("filter_beta", "joint_smoothing_tau_s", "settle_s", "print_hz"):
    if not isinstance(teleop[key], (int, float)) or teleop[key] < 0:
      raise ValueError(f"teleop.{key} must be nonnegative")
  if teleop["hand"] not in ("left", "right"):
    raise ValueError("teleop.hand must be left or right")
  if not isinstance(teleop["orientation"], bool):
    raise ValueError("teleop.orientation must be true or false")
  if config["robot"]["backend"] not in ("i2rt", "sim", "mock"):
    raise ValueError("unknown robot.backend")
  for key in ("max_command_offset_rad", "max_joint_velocity_rad_s", "max_command_velocity_rad_s"):
    value = config["safety"].get(key, 0.3)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
      raise ValueError(f"safety.{key} must be finite and positive")
  for section, keys in (("safety", ("joint_position_min_rad", "joint_position_max_rad")),
                         ("deploy", ("home_joint_position_rad", "reset_joint_position_rad"))):
    for key in keys:
      if key not in config[section]:
        continue
      vector = config[section][key]
      if not isinstance(vector, list) or len(vector) != 6 or any(
          isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
        raise ValueError(f"{section}.{key} must be six finite numbers")
  if any(lo >= hi for lo, hi in zip(config["safety"]["joint_position_min_rad"], config["safety"]["joint_position_max_rad"])):
    raise ValueError("joint minimum must be below maximum")
