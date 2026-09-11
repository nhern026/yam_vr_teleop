# yam_vr_teleop

Meta Quest VR teleoperation for YAM 6-DOF arms. One or two arms in a single process, driven by Quest 3 controllers over USB.

## Prerequisites (fresh Linux machine)

```bash
# Python 3.11
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3.11-dev

# uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or restart your shell

# Android Debug Bridge (for Quest communication)
sudo apt install -y android-tools-adb

# CAN utilities (for real hardware)
sudo apt install -y can-utils iproute2
```

### Quest setup

1. Enable **Developer Mode** on your Quest 3 (Settings → System → Developer)
2. Connect Quest to the PC via USB-C
3. Put on the headset and accept the **Allow USB debugging** prompt
4. Verify: `adb devices` should show your device as `device` (not `unauthorized`)

## Quick start

```bash
# Clone and setup
git clone git@github.com:nhern026/yam_vr_teleop.git
cd yam_vr_teleop
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install "i2rt @ git+https://github.com/i2rt-robotics/i2rt"  # or: uv pip install -e /path/to/local/i2rt
```

### CAN setup (real hardware)

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 up
# Repeat for can1 if using two arms
```

### Running teleop

```bash
cd yam_vr_teleop

# Right arm only
.venv/bin/python -m deployment.quest_teleop --config deployment/config.yaml --gripper

# Left arm only
.venv/bin/python -m deployment.quest_teleop --config deployment/config_left.yaml --gripper

# Both arms 
.venv/bin/python -m deployment.quest_teleop --config deployment/config.yaml --second-config deployment/config_left.yaml --gripper

# Sim mode (no hardware)
.venv/bin/python -m deployment.quest_teleop --backend sim --gripper
.venv/bin/python -m deployment.quest_teleop --backend sim --second-config deployment/config_left.yaml --gripper
```

### Other commands

```bash
# Debug: print controller stream (no hardware)
.venv/bin/python -m deployment.quest_teleop --dump

# Calibrate operator frame (only needed if you change where you stand)
# right arm
.venv/bin/python -m deployment.quest_teleop --config deployment/config.yaml --calibrate
# left arm
.venv/bin/python -m deployment.quest_teleop --config deployment/config_left.yaml --calibrate

# Pre-flight checks (Checks CAN(s), motors, joint limits. No motors activate)
.venv/bin/python -m deployment.preflight --config deployment/config.yaml --second-config deployment/config_left.yaml
```

### Controls

| Action | Button |
|--------|--------|
| Clutch (engage arm tracking) | Grip button (side button) |
| Close gripper | Trigger while clutched |
| Open gripper | Trigger while NOT clutched |
| Pause both arms | B (right) / Y (left) |
| Resume from pause | A (right) / X (left) |

**Exiting:** Press B/Y **on the Quest controller** to park the arms first, wait for them to settle, then Ctrl-C **in the terminal**. If you Ctrl-C without parking first, gravity comp turns off instantly and the arms drop — support them.

## Architecture
```
Quest 3 (USB)
  │
  │  adb logcat (~70 Hz pose stream)
  ▼
┌─────────────┐     ┌──────────────┐     ┌──────────┐     ┌───────────┐
│ QuestReader │────▶│  ArmChannel  │────▶│ TeleopIK │────▶│  I2rtArm  │
│ (adb parse) │     │  (per arm)   │     │ (mink QP)│     │  (CAN)    │
└─────────────┘     └──────────────┘     └──────────┘     └───────────┘
                           │
                    TeleopSession
                    (state machine,
                     shared clock,
                     fault domain)
```

### Control path (per tick, 100 Hz)

1. **Read** — `I2rtArm.read_state()` gets joint positions/velocities from the motor chain. Trip immediately if any joint exceeds `max_joint_velocity_rad_s`.
2. **Filter** — One Euro filter smooths the raw Quest pose. Jitter on a resting hand is filtered hard; fast motion passes through nearly 1:1. Pose teleports (tracking glitches) are rejected outright.
3. **Map** — While the clutch is held, filtered hand motion since the press is rotated into the world frame (using the calibrated operator heading) and scaled by `position_scale`. The tray pose is slew-capped at `max_tray_speed_m_s` / `max_tray_omega_rad_s`.
4. **IK** — `TeleopIK` solves one damped QP (via `mink`) to turn the target end-effector pose into joint angles, clamped to the intersection of mechanical limits and the operator's joint box.
5. **Command** — Joint targets are rate-limited (`max_command_velocity_rad_s`) and offset-clamped (`max_command_offset_rad`) before being sent to the motor chain.

Releasing the clutch freezes the arm. Re-pressing re-anchors without jumping — the arm picks up from where it is, not where the controller is.

### Terminology

| Term | Meaning |
|------|---------|
| **Tray** | The end-effector frame (grasp site) at the tip of the arm — the point IK controls. |
| **Clutch** | Grip button hold that engages arm tracking. Release to freeze the arm. Re-press to re-anchor without jumping. |
| **Operator frame** | A rotation matrix mapping Quest tracking coordinates to the robot's world frame. Measured once per standing position via `--calibrate`. |
| **Joint box** | Per-joint position limits (`safety.joint_position_min/max_rad`) intersected with mechanical limits. The only collision avoidance in the system. |
| **Slew cap** | Maximum tray velocity (`max_tray_speed_m_s`). Prevents fast hand flicks from commanding unsafe motions. |
| **One Euro filter** | Adaptive low-pass filter (Casiez et al., CHI 2012). Filters hard on a resting hand, passes fast motion nearly 1:1. |
| **Gravity comp** | i2rt's zero-gravity mode — motors compensate for the arm's weight so it holds position without being driven. The arm is "limp" to the touch but doesn't fall. |
| **Runaway trip** | Safety check: if any joint's measured velocity exceeds `max_joint_velocity_rad_s`, the session faults immediately. |
| **Fault** | A latched error state. Both arms stop. Requires restart — no button can clear it. |
| **Park** | A deliberate pause (B/Y). Arms hold position. A/X resumes. |


### Key classes

| Class | File | Role |
|-------|------|------|
| `QuestReader` | `quest_teleop.py` | Reads Quest controller poses via `adb logcat`. No network, no browser. |
| `TeleopIK` | `quest_teleop.py` | Builds a MuJoCo model from i2rt's YAM MJCF, solves IK with `mink`, enforces joint limits. |
| `TrayTarget` | `quest_teleop.py` | Clutch-anchored mapping from controller space to world-frame tray pose, with slew limiting. |
| `ArmChannel` | `quest_teleop.py` | One hand driving one arm: filters, clutch, gripper, IK — no mode of its own. |
| `TeleopSession` | `quest_teleop.py` | State machine (idle/engaged/homing/parked/fault) over one or two `ArmChannel`s. Single loop, single clock. |
| `I2rtArm` | `robot.py` | Real hardware backend. Claims the CAN bus (fcntl lock + USB serial verification), wraps `get_yam_robot(zero_gravity_mode=True)`. |
| `SimArm` | `robot.py` | i2rt's MuJoCo `SimRobot` — same `Robot` protocol as hardware. Preferred for development. |

### Files

```
deployment/
  quest_teleop.py    Main teleop script (QuestReader, TeleopIK, TrayTarget, ArmChannel, TeleopSession)
  robot.py           Arm backends: I2rtArm (CAN), SimArm (MuJoCo), MockArm (first-order servo)
  config.py          YAML config loader with validation
  config.yaml        Right arm config (hand, channel, backend, safety limits, tuning)
  config_left.yaml   Left arm config
  preflight.py       Pre-flight checks for real hardware
  camera.py          One Euro filter implementation
  calibration/       Saved operator frame calibration files
balancing_act/
  assets.py          Reset pose loader (yam_home.json)
```

## Configuration

Each arm has its own YAML config. Key sections:

**`teleop`** — Control feel: `position_scale` (hand-to-tray ratio), `max_tray_speed_m_s` (slew cap), `filter_min_cutoff_hz` / `filter_beta` (One Euro filter), `orientation` (wrist tracking on/off).

**`robot`** — Hardware: `backend` (i2rt/sim/mock), `channel` (can0/can1), `arm_type`, `gripper_type`, `adapter_serial`, `mapping_verified`.

**`safety`** — Limits: `max_command_velocity_rad_s`, `max_command_offset_rad`, `max_joint_velocity_rad_s`, `joint_position_min/max_rad` (operator box).

**`deploy`** — Poses: `home_joint_position_rad` (park pose), `reset_joint_position_rad` (startup pose).

### Tuning tips

- **Arm feels sluggish:** Raise `position_scale` (1.0 = 1:1), `max_tray_speed_m_s`, `max_command_velocity_rad_s`.
- **Arm jitters at rest:** Lower `filter_min_cutoff_hz`.
- **Arm lags behind fast movements:** Raise `filter_beta`, `max_tray_speed_m_s`.
- **Console shows `lag=` values:** The slew cap is binding. Raise `max_tray_speed_m_s`.
- **Console shows `pinned=j2,j3`:** IK is against a joint limit. Tighten the operator box or adjust your approach angle.

## Lab setup notes

The configs ship with the correct CAN adapter serials and channel mappings for our lab's two YAM arms (`mapping_verified: true`). The operator frame calibration files are also included — they're valid as long as you stand in the same spot as the original calibration. If you move to a different position, re-run `--calibrate` for each arm.

**CAN mapping:** Right arm is on `can1`, left arm is on `can0`. This is already set in the configs.

**If you're setting up on a different machine or different arms**, you'll need to:

1. **Verify CAN adapter mapping.** Check which USB serial is on which CAN interface (`/sys/class/net/canX/device/.../serial`), update `adapter_serial` and `channel` in both configs, and set `mapping_verified: true`.
2. **Run preflight.** It checks CAN, motor chain, gripper type, joint limits, and operator frame — without energizing anything.
3. **Calibrate the operator frame.** Run `--calibrate` for each arm from where you'll stand.
4. **Tighten the joint box.** The default `±3.0 rad` constrains nothing. There is **no collision checking** — the operator box is the only thing keeping two arms apart.

## Requirements

- Linux (socketcan for CAN hardware)
- Python 3.11+
- Meta Quest 3 with developer mode enabled, connected via USB
- `adb` on PATH
- GS_USB CAN adapters (one per arm)
- [i2rt](https://github.com/i2rt-robotics/i2rt) installed (provides YAM models, motor drivers, MuJoCo sim)

## Known gaps

- **No recording.** Nothing writes demonstration data yet.
- **No cameras.** `camera.py` is just the One Euro filter.
- **No collision checking.** The operator joint box is the only thing keeping arms apart.
