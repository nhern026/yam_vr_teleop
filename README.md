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

`--record <dir> --gripper` arms joystick-delimited episode recording from the
two ZED X wrist cameras and the overhead ZED X ([deployment/zed_capture.py](deployment/zed_capture.py)),
timestamp-matched against the control loop's state/action ticks
([deployment/recording.py](deployment/recording.py)). It requires both arms
(`--second-config`) and `--gripper`; camera serials and writer settings come
from [deployment/recording.yaml](deployment/recording.yaml) (override with
`--record-config`). `--dashboard` launches a live browser UI and works with
any teleop command above, recording or not.

```bash
# Both arms, recording armed + dashboard
.venv/bin/python -m deployment.quest_teleop --config deployment/config.yaml --second-config deployment/config_left.yaml --gripper --record episodes/ --dashboard
```

```bash
# Sim, recording armed + dashboard (ZED cameras still required for --record)
.venv/bin/python -m deployment.quest_teleop --backend sim --second-config deployment/config_left.yaml --gripper --record episodes/ --dashboard
```

```bash
# Dashboard on a custom port, no recording
.venv/bin/python -m deployment.quest_teleop --backend sim --gripper --dashboard 9090
```

Open `http://localhost:8080` (or your custom port) in a browser to see live joint angles, recording status, controller state, and saved episodes.

**Recording:** `--record` arms the session but does not start capturing.
Either joystick click begins an episode and prompts for a task name in the
terminal (or reuses `--task`, given once for every episode or repeated to
pre-supply names in order); the same click ends it. Ending an episode
finalizes it on a background thread while teleop stays live and recording
goes back to armed — any number of episodes per run. Each episode is a
directory under `episodes/`, named from the task and start time, holding raw
images, `data.csv`, and `metadata.json`.

```bash
# Inspect a raw episode's integrity and timing before exporting it
.venv/bin/python -m deployment.inspect_episode episodes/place_vial_2026-09-21_at_02-14-05pm
```

```bash
# Export one episode into per-segment HDF5 trajectories
.venv/bin/python -m deployment.export_dataset episodes/place_vial_2026-09-21_at_02-14-05pm --output out/segments
```

```bash
# ... and also convert those segments into a LeRobot v2 dataset for XPolicyLab/pi0.5
.venv/bin/python -m deployment.export_dataset episodes/place_vial_2026-09-21_at_02-14-05pm --output out/segments --lerobot-root out/yam_dataset --repo-id local/yam
```

See [OPERATING.md](OPERATING.md) for the full teleop→record→export→train
workflow, and [deployment/openpi_config.py](deployment/openpi_config.py) /
[deployment/yam_policy.py](deployment/yam_policy.py) for the π0.5/OpenPI side.

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
| Start/stop an episode | Joystick click, either hand (requires `--record`) |

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
| `--record <dir>` | Arm joystick-delimited episode recording into directory (needs `--second-config`, `--gripper`) |
| `--task <text>` | Task instruction for an episode; repeat to pre-supply names in order, or omit to be prompted |
| `--record-config <path>` | Camera/writer settings for `--record` (default `deployment/recording.yaml`) |
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

### Jetson / conda setup

The venv/`uv` route above targets a generic Linux dev machine. On a Jetson
(aarch64), some of this stack has no manylinux wheel and is easier to pull
from conda-forge: see [environment.yml](environment.yml) and
`bash scripts/bootstrap.sh`, or [OPERATING.md](OPERATING.md) for the full
bring-up-through-training workflow on that path. [HANDOVER.md](HANDOVER.md)
has the hardware bring-up order and what is/isn't verified on real arms.

## Requirements

- Linux (socketcan for CAN hardware)
- Python 3.11+
- Meta Quest 3 with developer mode enabled, connected via USB
- `adb` on PATH
- GS_USB CAN adapters (one per arm)
- [i2rt](https://github.com/i2rt-robotics/i2rt) installed (provides YAM models, motor drivers, MuJoCo sim)
- ZED SDK + `pyzed` only if using `--record` (see [requirements.txt](requirements.txt))
