# yam_vr_teleop

Meta Quest VR teleoperation for YAM 6-DOF arms. For architecture, data formats, configuration, and tuning — see [docs/architecture.md](docs/architecture.md).

## Running

All commands assume you're in the repo root with the venv active (`source .venv/bin/activate`). See [Setup](#setup) below if this is a fresh clone.

### Calibration

Run once per standing position. Saves a frame file that teleop reads automatically.

```bash
# Right arm
.venv/bin/python -m deployment.quest_teleop --config deployment/config.yaml --calibrate
```

```bash
# Left arm
.venv/bin/python -m deployment.quest_teleop --config deployment/config_left.yaml --calibrate
```

### Teleop

```bash
# Right arm only
.venv/bin/python -m deployment.quest_teleop --config deployment/config.yaml --gripper
```

```bash
# Left arm only
.venv/bin/python -m deployment.quest_teleop --config deployment/config_left.yaml --gripper
```

```bash
# Both arms
.venv/bin/python -m deployment.quest_teleop --config deployment/config.yaml --second-config deployment/config_left.yaml --gripper
```

```bash
# Sim mode (no hardware needed)
.venv/bin/python -m deployment.quest_teleop --backend sim --gripper
```

```bash
# Sim, both arms
.venv/bin/python -m deployment.quest_teleop --backend sim --second-config deployment/config_left.yaml --gripper
```

### Data recording + dashboard

Add `--record <dir>` to record demos and `--dashboard` to launch a live browser UI. These work with any teleop command above.

```bash
# Right arm with recording + dashboard
.venv/bin/python -m deployment.quest_teleop --config deployment/config.yaml --gripper --record demos/ --dashboard
```

```bash
# Both arms with recording + dashboard
.venv/bin/python -m deployment.quest_teleop --config deployment/config.yaml --second-config deployment/config_left.yaml --gripper --record demos/ --dashboard
```

```bash
# Sim with recording + dashboard
.venv/bin/python -m deployment.quest_teleop --backend sim --gripper --record demos/ --dashboard
```

```bash
# Dashboard on a custom port
.venv/bin/python -m deployment.quest_teleop --backend sim --gripper --record demos/ --dashboard 9090
```

Open `http://localhost:8080` (or your custom port) in a browser to see live joint angles, recording status, controller state, and saved demos.

**Recording:** joystick click to start, joystick click to stop. Parking (B/Y) auto-saves. Each segment saves as `demos/demo_YYYYMMDD_HHMMSS.hdf5`.

```bash
# Inspect a saved demo
.venv/bin/python -m deployment.inspect_demo demos/demo_20260916_143022.hdf5
```

### Other commands

```bash
# Debug: print raw controller stream (no hardware)
.venv/bin/python -m deployment.quest_teleop --dump
```

```bash
# Pre-flight checks (CAN, motors, joint limits — nothing energizes)
.venv/bin/python -m deployment.preflight --config deployment/config.yaml --second-config deployment/config_left.yaml
```

### Controls

| Action | Button |
|--------|--------|
| Clutch (engage arm tracking) | Grip button (side button) |
| Gripper (proportional) | Trigger while clutched (squeeze = close) |
| Open gripper | Trigger while NOT clutched |
| Pause both arms | B (right) / Y (left) |
| Home to start position | A (right) / X (left) |
| Start/stop recording | Joystick click (either hand, requires `--record`) |

The gripper is proportional: squeezing the trigger halfway closes the gripper halfway, which matters for delicate tasks like handling vials.

**Homing:** pressing A/X from a paused state moves both arms to their start position (`deploy.reset_joint_position_rad` in the config, or the shared `yam_home.json` fallback). The move takes `teleop.move_s` seconds with smooth easing. Use this between demos so each one starts from the same pose.

**Exiting:** Press B/Y to park the arms, then Ctrl-C in the terminal. The arms enter gravity comp (they float but don't fall). Support the arms, then press Enter to disable motors.

### All CLI flags

| Flag | Effect |
|------|--------|
| `--config <path>` | Primary arm config YAML |
| `--second-config <path>` | Second arm for bimanual |
| `--backend sim` | MuJoCo sim (no hardware) |
| `--gripper` | Enable gripper control |
| `--record <dir>` | Enable HDF5 recording to directory |
| `--dashboard [PORT]` | Start live dashboard (default 8080) |
| `--calibrate` | Calibrate operator frame and exit |
| `--dump` | Print raw controller stream and exit |

---

## Setup

### Prerequisites (fresh Linux machine)

```bash
# Python 3.11
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

```bash
# uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

```bash
# Android Debug Bridge (for Quest communication)
sudo apt install -y android-tools-adb
```

```bash
# CAN utilities (for real hardware)
sudo apt install -y can-utils iproute2
```

### Quest setup

1. Enable **Developer Mode** on your Quest 3 (Settings → System → Developer)
2. Connect Quest to the PC via USB-C
3. Put on the headset and accept the **Allow USB debugging** prompt
4. Verify: `adb devices` should show your device as `device` (not `unauthorized`)

### Install

```bash
git clone git@github.com:nhern026/yam_vr_teleop.git
cd yam_vr_teleop
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install "i2rt @ git+https://github.com/i2rt-robotics/i2rt"
```

Or with a local i2rt checkout:

```bash
uv pip install -e /path/to/local/i2rt
```

### CAN setup (real hardware)

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 up
```

Repeat for `can1` if using two arms. Right arm is on `can1`, left arm is on `can0` (already set in configs).

### Recording a start pose

Record a pose so that A/X always homes the arm to the same position between demos.

```bash
# Right arm — guide the arm by hand, press Enter to capture
.venv/bin/python -m deployment.record_pose --channel can1 --gripper linear_4310
```

```bash
# Left arm
.venv/bin/python -m deployment.record_pose --channel can0 --gripper linear_4310
```

Paste the printed values into the arm's config under `deploy.reset_joint_position_rad`. Without this key the arm homes to the shared `yam_home.json` fallback.

### New machine / different arms

1. **Verify CAN adapter mapping.** Check which USB serial is on which CAN interface (`/sys/class/net/canX/device/.../serial`), update `adapter_serial` and `channel` in both configs, set `mapping_verified: true`.
2. **Run preflight.** Checks CAN, motor chain, gripper type, joint limits — without energizing anything.
3. **Calibrate the operator frame.** Run `--calibrate` for each arm from where you'll stand.
4. **Record a start pose.** Run `record_pose` for each arm and paste the result into the config (see above).
5. **Tighten the joint box.** The default `±3.0 rad` constrains nothing. There is **no collision checking** — the operator box is the only thing keeping two arms apart.

## Requirements

- Linux (socketcan for CAN hardware)
- Python 3.11+
- Meta Quest 3 with developer mode enabled, connected via USB
- `adb` on PATH
- GS_USB CAN adapters (one per arm)
- [i2rt](https://github.com/i2rt-robotics/i2rt) installed (provides YAM models, motor drivers, MuJoCo sim)
