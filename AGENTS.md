# AGENTS.md

Agent handoff document for the `yam_vr_teleop` project. Read this first.

## What this is

VR teleoperation for YAM 6-DOF robot arms using Meta Quest 3 controllers. One or two arms in a single process, driven by Quest controllers over USB (`adb logcat`). The operator holds grip to clutch, moves their hand, and the arm follows via IK. Trigger controls the gripper proportionally (squeeze = close). Data collection records joints + cameras to HDF5 at 100 Hz, exportable to CSV or LeRobot v2.1 for pi0.5 fine-tuning.

## Environment

```bash
# Python venv (NOT uv-managed — uses requirements.txt)
cd /home/nico/yam_teleop
source .venv/bin/activate

# Run teleop in sim (no hardware needed)
.venv/bin/python -m deployment.quest_teleop --backend sim --gripper

# Run with recording + live dashboard
.venv/bin/python -m deployment.quest_teleop --backend sim --gripper --record demos/ --dashboard

# Both arms with cameras + recording + dashboard
.venv/bin/python -m deployment.quest_teleop --config deployment/config.yaml --second-config deployment/config_left.yaml --gripper --cameras --record demos/ --dashboard

# Inspect a recorded demo
.venv/bin/python -m deployment.inspect_demo demos/<file>.hdf5

# Export demos to CSV (full 100 Hz, 13D state with velocities)
.venv/bin/python -m deployment.export_demos demos/ --csv out/csv

# Export demos to LeRobot v2.1 (7D state/action, resampled to 30 Hz)
.venv/bin/python -m deployment.export_demos demos/ --lerobot out/dataset --fps 30
```

There is no test suite in this repo. Test by running `--backend sim` and checking behavior. The venv is at `/home/nico/yam_teleop/.venv/bin/python` — the system has `python3` but no `python` alias.

## Dependencies

- `requirements.txt` lists: numpy, mujoco, mink, quadprog, pyyaml, h5py, pyarrow, opencv-python-headless
- `i2rt` is installed separately (`uv pip install "i2rt @ git+https://github.com/i2rt-robotics/i2rt"` or editable from `/home/nico/i2rt`). i2rt provides the Robot protocol, motor drivers, MuJoCo sim, URDF/MJCF models, gravity comp. Even sim mode needs i2rt because IK builds from its MJCF.
- No aiohttp, no websockets, no Flask — the dashboard uses only Python stdlib.

## Architecture at a glance

```
Quest 3 (USB)
  │  adb logcat (~70 Hz)
  ▼
QuestReader ──▶ ArmChannel ──▶ TeleopIK ──▶ I2rtArm (CAN) or SimArm (MuJoCo)
                    │
             TeleopSession (state machine: idle/engaged/homing/parked/fault)
                    │
              ┌─────┼──────┬──────────┐
              ▼     ▼      ▼          ▼
          Recorder  Dashboard  Console  WristCameras
          (HDF5)    (SSE@10Hz) (stdout)  (threaded USB)
```

Control loop runs at 100 Hz on a single thread. Each tick: read joints → filter Quest pose → map hand motion to tray frame → solve IK (mink QP) → rate-limit and command joints → grab latest camera frames → record everything.

**Timing**: Nothing blocks waiting for data at source rates. Quest (~70 Hz) and cameras (~30 Hz) run on their own threads; the loop calls `latest()` and gets whatever's newest. Joint reads are synchronous CAN calls (~μs). If a tick overruns, missed ticks are skipped rather than sprinted.

## Key files

All code lives in `deployment/`:

| File | What it does |
|------|-------------|
| `quest_teleop.py` | Everything: QuestReader, TeleopIK, TrayTarget, ArmChannel, TeleopSession, CLI main. ~1850 lines. |
| `robot.py` | Arm backends: `I2rtArm` (real CAN hardware), `SimArm` (MuJoCo), `MockArm` (first-order servo). `enter_gravity_comp()` for safe shutdown. |
| `recorder.py` | `DemoRecorder` appends per-tick data in-memory. `tick_cameras()` JPEG-encodes frames. `detach()` swaps buffers in O(1). `write_snapshot()` serializes to HDF5 on a background thread. |
| `camera_capture.py` | `WristCamera` — threaded USB camera capture. One grab thread per camera, `latest()` returns newest frame without blocking. `open_cameras(config)` factory. |
| `export_demos.py` | Export HDF5 demos to CSV (13D state at 100 Hz) or LeRobot v2.1 (7D state/action, resampled with `--fps`). Writes mp4 videos for LeRobot when camera data is present. |
| `dashboard.py` | `Dashboard` class runs an HTTP server on a background thread. SSE at `/events`, HTML at `/`, demo list at `/demos`. |
| `dashboard.html` | Single-page dark-theme dashboard. `EventSource('/events')` auto-reconnects. |
| `config.py` | YAML loader with validation. |
| `config.yaml` | Right arm config + camera config. |
| `config_left.yaml` | Left arm config. |
| `inspect_demo.py` | CLI to print contents of a recorded HDF5 demo. |
| `record_pose.py` | Hand-guide arm to a pose and capture it for `deploy.reset_joint_position_rad`. |
| `preflight.py` | Hardware pre-flight checks (CAN, motors, limits). |
| `camera.py` | One Euro filter (NOT a camera — just the filter math). |

## Config structure (config.yaml)

- **`teleop`**: `hand`, `control_hz`, `position_scale`, `max_tray_speed_m_s`, filter params, `watchdog_s`, `move_s`, `settle_s`
- **`robot`**: `backend` (i2rt/sim/mock), `channel` (can0/can1), `arm_type`, `gripper_type`, `adapter_serial`
- **`safety`**: `max_command_velocity_rad_s`, `max_command_offset_rad`, `max_joint_velocity_rad_s`, `joint_position_min/max_rad`
- **`deploy`**: `home_joint_position_rad` (park pose), `reset_joint_position_rad` (homing pose)
- **`cameras`**: Named cameras with `device` (/dev/videoN), `width`, `height`, `fps`. Only in primary config.

## Data recording

HDF5 files at 100 Hz. Per-arm groups (`/right/`, `/left/`) each contain: `joint_position` (N,6), `joint_velocity` (N,6), `joint_target` (N,6), `gripper_position` (N,), `gripper_command` (N,), `ee_position` (N,3), `ee_quaternion` (N,4), `controller_position` (N,3), `controller_quaternion` (N,4), `controller_trigger` (N,), `controller_grip` (N,), `controller_clutch` (N,). Plus `/timestamps` (N,) and `/mode` (N,). File attrs: `start_time`, `hz`, `arms`, `num_ticks`, `duration_s`.

When `--cameras` is enabled, `/cameras/<name>/frames` stores JPEG-encoded frames (vlen uint8) with `width`, `height`, `codec` attrs.

Toggle recording with joystick click on either controller. Parking (B/Y) auto-saves.

## Export formats

**CSV**: Full 100 Hz, 13D state per arm (6 joint positions + 6 joint velocities + gripper) + 7D action (6 joint targets + gripper command). Velocities included for analysis.

**LeRobot v2.1**: XPolicyLab/pi0.5 canonical format. 7D state (joint positions + gripper) + 7D action (joint targets + gripper command). State and action share physical dimensions (enables relative-action mode). `--fps 30` resamples to camera rate. Camera data exports as mp4 videos under `videos/chunk-000/<cam_name>/`.

## Controls

| Action | Button |
|--------|--------|
| Clutch (engage arm tracking) | Grip button (side) |
| Gripper (proportional) | Trigger while clutched (squeeze = close) |
| Open gripper | Trigger while NOT clutched |
| Pause both arms | B (right) / Y (left) |
| Home to start position | A (right) / X (left) |
| Start/stop recording | Joystick click (either hand) |

**Shutdown**: B/Y parks arms → Ctrl-C → arms enter gravity comp (float but don't fall) → operator supports arms → Enter to disable motors.

## CLI flags (quest_teleop.py)

| Flag | Effect |
|------|--------|
| `--config <path>` | Primary arm config YAML |
| `--second-config <path>` | Second arm for bimanual |
| `--backend sim` | MuJoCo sim (no hardware) |
| `--gripper` | Enable proportional gripper control |
| `--cameras` | Enable wrist cameras (needs `cameras` section in config) |
| `--record <dir>` | Enable HDF5 recording to directory |
| `--dashboard [PORT]` | Start live dashboard (default 8080) |
| `--calibrate` | Calibrate operator frame and exit |
| `--dump` | Print raw controller stream and exit |

## Known issues and next steps

- **Camera hardware**: Planning to use ZED X One cameras (GMSL2, requires Jetson). Camera capture code currently uses OpenCV V4L2 — will need ZED SDK integration.
- **No collision checking**: The operator joint box is the only thing preventing arm-arm or arm-table collisions.
- **Dashboard is view-only**: No controls for starting/stopping recording from the browser yet.
- **Teleop smoothness**: User reported tuning was not right. Config values reverted to originals — further tuning TBD.
- **LeRobot v3.0**: Current export targets v2.1. XPolicyLab also supports v3.0 (different parquet layout). A `lerobot.scripts.convert_dataset_v21_to_v30` migration script exists.

## i2rt dependency

This project depends on the `i2rt` library at `/home/nico/i2rt` (or installed from GitHub). Key interfaces used:

- `get_yam_robot(sim=True/False, ...)` — factory that returns a `Robot` (either `MotorChainRobot` or `SimRobot`)
- `Robot` protocol: `get_joint_pos()`, `command_joint_pos()`, `get_observations()`, etc.
- `enter_gravity_comp_idle()` — zero-gravity mode for safe shutdown
- `ArmType` / `GripperType` enums from `i2rt.robots.utils`
- `combine_arm_and_gripper_xml()` for merging arm + gripper MJCF at runtime
- MuJoCo models under `i2rt/robot_models/`

The i2rt repo has its own CLAUDE.md at `/home/nico/i2rt/CLAUDE.md` with full architecture docs.
