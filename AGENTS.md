# AGENTS.md

Agent handoff document for the `yam_vr_teleop` project. Read this first.

## What this is

VR teleoperation for YAM 6-DOF robot arms using Meta Quest 3 controllers. One or two arms in a single process, driven by Quest controllers over USB (`adb logcat`). The operator holds grip to clutch, moves their hand, and the arm follows via IK. Trigger controls the gripper proportionally (squeeze = close). `--record` arms joystick-delimited episode recording from two ZED X wrist cameras and an overhead ZED X, timestamp-matched against joint state/action ticks; any number of episodes per run, each exportable to gap-separated HDF5 segments or a LeRobot v2 dataset for pi0.5/OpenPI fine-tuning.

## Environment

```bash
# Python venv (NOT uv-managed — uses requirements.txt)
cd /home/nico/yam_teleop
source .venv/bin/activate

# Run teleop in sim (no hardware needed)
.venv/bin/python -m deployment.quest_teleop --backend sim --gripper

# Both arms, recording armed + live dashboard (ZED cameras required for --record)
.venv/bin/python -m deployment.quest_teleop --config deployment/config.yaml --second-config deployment/config_left.yaml --gripper --record episodes/ --dashboard

# Inspect a raw episode's integrity/timing before exporting
.venv/bin/python -m deployment.inspect_episode episodes/<episode_dir>

# Export one episode into gap-separated HDF5 segments
.venv/bin/python -m deployment.export_dataset episodes/<episode_dir> --output out/segments

# ... and into a LeRobot v2 dataset for XPolicyLab/pi0.5
.venv/bin/python -m deployment.export_dataset episodes/<episode_dir> --output out/segments --lerobot-root out/yam_dataset --repo-id local/yam
```

Test suite: `pytest tests/ --ignore=tests/test_backform.py` (or run `test_backform.py` directly with `python`, not pytest — it mutates `sys.path` at import time and will shadow the real `i2rt` for anything collected after it). `test_teleop_stack.py` needs the real i2rt/mujoco/mink stack; everything else runs against the stubs in `tests/stubs/`. The venv is at `/home/nico/yam_teleop/.venv/bin/python` — the system has `python3` but no `python` alias.

## Dependencies

- `requirements.txt` lists: numpy, mujoco, mink, quadprog, pyyaml, h5py, pyarrow, opencv-python-headless
- `i2rt` is installed separately (`uv pip install "i2rt @ git+https://github.com/i2rt-robotics/i2rt"` or editable from `/home/nico/i2rt`). i2rt provides the Robot protocol, motor drivers, MuJoCo sim, URDF/MJCF models, gravity comp. Even sim mode needs i2rt because IK builds from its MJCF.
- `pyzed` (ZED SDK Python API) is required only for `--record`; imported lazily so everything else runs without it. No PyPI wheel — see `requirements.txt`.
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
              ┌─────┼──────┬───────────────┐
              ▼     ▼      ▼               ▼
          EpisodeRecorder  Dashboard  Console  ZedPairCapture
          (CSV+JPEG,      (SSE@10Hz) (stdout)  (threaded, 3 ZED
           its own thread)                      cameras + broker)
```

Control loop runs at 100 Hz on a single thread. Each tick: read joints (timestamped) → filter Quest pose → map hand motion to tray frame → solve IK (mink QP) → rate-limit and command joints → hand a `ControlSample` to the recorder, which matches it against camera frames off the control thread. `ZedPairCapture` runs its own grab threads and a frame-pair broker entirely independent of the control loop; nothing about capture can stall a command tick.

**Timing**: Nothing blocks waiting for data at source rates. Quest (~70 Hz) and the ZED cameras (~30 Hz) run on their own threads; matching happens in `EpisodeRecorder` by nearest causal timestamp (`deployment/recording.py`), not by the control loop calling `latest()`. Joint reads are synchronous CAN calls (~μs). If a tick overruns, missed ticks are skipped rather than sprinted.

## Key files

All code lives in `deployment/`:

| File | What it does |
|------|-------------|
| `quest_teleop.py` | Everything: QuestReader, TeleopIK, TrayTarget, ArmChannel, TeleopSession, CLI main. ~1850 lines. |
| `robot.py` | Arm backends: `I2rtArm` (real CAN hardware), `SimArm` (MuJoCo), `MockArm` (first-order servo). `enter_gravity_comp()` for safe shutdown. |
| `zed_capture.py` | `ZedPairCapture` — threaded capture across the two wrist ZED X cameras and the overhead ZED X, running continuously once started. `begin_recording(on_frame, on_fail)` and `end_recording()` atomically swap the callback to route/discard frames without stopping the cameras, so arming a new episode never re-pays ZED init cost. `FramePairBroker` matches wrist frames into pairs and emits explicit orphan events. `pyzed` imported lazily. |
| `recording.py` | `EpisodeRecorder` — bounded async writer for one episode, no robot/SDK imports. `add_control()`/`add_capture()` feed a background thread that matches control samples to camera frames by earliest-read timestamp and streams `data.csv`/`controls.jsonl`/images to disk. `fail()` marks the episode invalid without stopping the control loop; `finish_controls()` then `close()` end it. A new episode is a new `EpisodeRecorder` instance. |
| `export_dataset.py` | `segments()` splits a raw episode into gap-separated runs (mode/fault/staleness breaks a segment); `export()` writes each as its own HDF5 file; `to_lerobot()` converts HDF5 segments into a LeRobot v2 dataset. |
| `inspect_episode.py` | CLI to check a raw episode's integrity (missing images, invalid rows) and measured timing (control/camera Hz, skew) before exporting it. |
| `yam_policy.py` / `openpi_config.py` | OpenPI `YamInputs`/`YamOutputs` transforms (14D state, three camera views) and the `YamDataConfig`/`make_config()` used by `scripts/openpi_yam.py` in the training environment. |
| `dashboard.py` | `Dashboard` class runs an HTTP server on a background thread. SSE at `/events`, HTML at `/`, episode list at `/demos`. |
| `dashboard.html` | Single-page dark-theme dashboard. `EventSource('/events')` auto-reconnects. |
| `config.py` | YAML loader with validation. |
| `config.yaml` | Right arm config. |
| `config_left.yaml` | Left arm config. |
| `recording.yaml` | ZED camera serials and `EpisodeRecorder` writer settings for `--record`. |
| `record_pose.py` | Hand-guide arm to a pose and capture it for `deploy.reset_joint_position_rad`. |
| `preflight.py` | Hardware pre-flight checks (CAN, motors, limits). |
| `camera.py` | One Euro filter (NOT a camera — just the filter math). |

## Config structure (config.yaml)

- **`teleop`**: `hand`, `control_hz`, `position_scale`, `max_tray_speed_m_s`, filter params, `watchdog_s`, `move_s`, `settle_s`
- **`robot`**: `backend` (i2rt/sim/mock), `channel` (can0/can1), `arm_type`, `gripper_type`, `adapter_serial`
- **`safety`**: `max_command_velocity_rad_s`, `max_command_offset_rad`, `max_joint_velocity_rad_s`, `joint_position_min/max_rad`
- **`deploy`**: `home_joint_position_rad` (park pose), `reset_joint_position_rad` (homing pose)

Camera serials and writer settings for `--record` live in `deployment/recording.yaml`, not in `config.yaml` — see its inline comments.

## Data recording

`--record DIR` arms the session; it does not start capturing. Either
joystick click (`TeleopSession._apply_buttons`, debounced by
`_record_release_seen` so a stick held through startup can't fire) sets
`_record_requested_by`, which `main()` polls via `take_record_request()`
each loop iteration. The first click after arming creates an `EpisodeRecorder`
(prompting for a task name via `prompt_episode_task()`, or taking it from
`--task`) and calls `session.begin_episode(recorder)`; the next click calls
`session.end_episode(recorder)` and hands the recorder to a background
`episode-finalizer-N` thread, so closing/renaming the directory never blocks
teleop or the next episode. `session.recorder`/`_record_tick` are guarded by
`TeleopSession._recorder_lock` since they can change while the control loop
is running, not just at startup.

Each episode is its own directory (named from the task and start time):
`data.csv` (one row per camera capture: image paths, 14D state, 14D action,
control tick/timestamp, validity), `debug_timing.csv` (per-frame ZED/host
timestamps), `controls.jsonl` (every control tick, matched or not), and
`images/<side>/*.jpg`. State/action layout is `left arm(6), left gripper,
right arm(6), right gripper`. `metadata.json` carries `status`
(`recording`/`complete`/`invalid`), `errors`, `counts`, and the configs used.

Within one episode there is still no pause: B/Y, a fault, or lost input mark
ticks invalid (`ControlSample.valid=False`) rather than stopping capture —
`export_dataset.segments()` splits the raw stream into gap-separated segments
at those boundaries.

## Export formats

**Per-segment HDF5** (`export_dataset.export()`): each gap-separated segment
of a complete episode becomes its own HDF5 file — original 14D state/action
(no interpolation or resampling) plus RGB uint8 images per camera view.

**LeRobot v2** (`export_dataset.to_lerobot()`): converts a list of those HDF5
segments into a LeRobot v2 dataset for XPolicyLab/pi0.5, one file per camera
view under `observation.images.<name>`. Run in XPolicyLab's LeRobot
environment, not the Jetson capture environment.

## Controls

| Action | Button |
|--------|--------|
| Clutch (engage arm tracking) | Grip button (side) |
| Gripper (proportional) | Trigger while clutched (squeeze = close) |
| Open gripper | Trigger while NOT clutched |
| Pause both arms | B (right) / Y (left) |
| Home to start position | A (right) / X (left) |
| Start/stop an episode | Joystick click, either hand (requires `--record`) |

**Shutdown**: B/Y parks arms → Ctrl-C → arms enter gravity comp (float but don't fall) → operator supports arms → Enter to disable motors.

## CLI flags (quest_teleop.py)

| Flag | Effect |
|------|--------|
| `--config <path>` | Primary arm config YAML |
| `--second-config <path>` | Second arm for bimanual |
| `--backend sim` | MuJoCo sim (no hardware) |
| `--gripper` | Enable proportional gripper control |
| `--record <dir>` | Arm joystick-delimited episode recording into directory (needs `--second-config`, `--gripper`) |
| `--task <text>` | Task instruction for an episode; repeat to pre-supply names in order, or omit to be prompted |
| `--record-config <path>` | Camera/writer settings for `--record` (default `deployment/recording.yaml`) |
| `--dashboard [PORT]` | Start live dashboard (default 8080) |
| `--calibrate` | Calibrate operator frame and exit |
| `--dump` | Print raw controller stream and exit |

## Known issues and next steps

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
