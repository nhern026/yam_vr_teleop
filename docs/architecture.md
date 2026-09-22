# Architecture & Reference

Detailed documentation for yam_vr_teleop. For running commands, see [README.md](../README.md).

## System overview

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
                           │
                    ┌──────┼──────┬───────────────┐
                    ▼      ▼      ▼               ▼
              EpisodeRecorder  Dashboard  Console  ZedPairCapture
              (CSV + JPEG)    (SSE@10Hz) (stdout)  (own threads, 3 ZED cams)
```

## Control path (per tick, 100 Hz)

1. **Read** — `I2rtArm.read_state()` gets joint positions/velocities from the motor chain. Trip immediately if any joint exceeds `max_joint_velocity_rad_s`.
2. **Filter** — One Euro filter smooths the raw Quest pose. Jitter on a resting hand is filtered hard; fast motion passes through nearly 1:1. Pose teleports (tracking glitches) are rejected outright.
3. **Map** — While the clutch is held, filtered hand motion since the press is rotated into the world frame (using the calibrated operator heading) and scaled by `position_scale`. The tray pose is slew-capped at `max_tray_speed_m_s` / `max_tray_omega_rad_s`.
4. **IK** — `TeleopIK` solves one damped QP (via `mink`) to turn the target end-effector pose into joint angles, clamped to the intersection of mechanical limits and the operator's joint box.
5. **Command** — Joint targets are rate-limited (`max_command_velocity_rad_s`) and offset-clamped (`max_command_offset_rad`) before being sent to the motor chain.

Releasing the clutch freezes the arm. Re-pressing re-anchors without jumping — the arm picks up from where it is, not where the controller is.

## Data recording

`--record DIR` arms the session; it does not start capturing. Either
joystick click begins an episode (prompting for a task name, or taking one
from `--task`), the same click ends it, and any number of episodes can run
one after another in a single process. Two independent pieces cooperate,
neither able to stall the other:

- **`ZedPairCapture`** (`deployment/zed_capture.py`) owns one grab thread per
  camera (two wrist ZED X, one overhead ZED X) plus a `FramePairBroker` that
  matches the wrist pair by timestamp and emits explicit orphan events for
  frames whose mate never arrives — nothing is silently shifted onto the next
  exposure. The cameras run continuously from startup to shutdown;
  `begin_recording(on_frame, on_fail)` / `end_recording()` atomically swap
  the frame callback between "route to this episode's recorder" and
  "discard", so starting the next episode never re-pays ZED init cost.
- **`EpisodeRecorder`** (`deployment/recording.py`) is constructed fresh per
  episode. It receives a `ControlSample` from the control loop on every tick
  (14D state, 14D action, mode, and the read/command nanosecond intervals
  used for causal matching) and a capture event from `ZedPairCapture` on
  every routed camera frame. A background thread matches each frame to the
  most recent usable control sample within `max_skew_ms`, JPEG-encodes it,
  and streams rows to `data.csv`.

`TeleopSession.begin_episode()`/`end_episode()` attach/detach the active
recorder under `_recorder_lock` while the control loop keeps running;
`take_record_request()` lets `main()` poll for the joystick click
(`_apply_buttons` requires seeing the stick released once after startup
before it can arm a request, so a stick already held at boot can't start an
episode by accident). Ending an episode hands it to a background
`episode-finalizer-N` thread — `finish_controls()` then `close()` — so
closing/renaming the directory never blocks teleop or the next episode.

Each episode is its own directory (named from the task and start time):

| File | Contents |
|------|----------|
| `data.csv` | One row per camera capture: image paths, validity, matched control tick/timestamp, 14D state, 14D action. |
| `debug_timing.csv` | Per-frame ZED/host/grab/retrieve timestamps, for diagnosing skew. |
| `controls.jsonl` | Every control tick as JSON, matched to an image or not. |
| `images/<side>/*.jpg` | Raw frames, named `<sequence>_<capture_index>.jpg`. |
| `metadata.json` | `status` (`recording`/`complete`/`invalid`), `errors`, `counts`, `task`, `episode_index`, `record_trigger`, the configs used, `startup_joint_position_rad`. |

State/action layout is fixed: `left arm(6), left gripper, right arm(6), right gripper`.

`EpisodeRecorder.fail(reason)` marks the episode `invalid` in `metadata.json`
without stopping the control loop or the session — B/Y, a control-loop
exception, a stalled writer queue, or low disk space all route through it.
`deployment.quest_teleop` also marks individual ticks invalid (transitional
mode, stale/missing controller input, a >25 ms read span) without failing
the whole episode; those become segment boundaries at export time, not lost
data.

### Inspecting and exporting episodes

`deployment/inspect_episode.py DIR` reports raw/valid capture counts, missing
images, control/camera Hz, and camera-to-control skew — run it before
exporting to catch a bad episode early.

`deployment/export_dataset.py` turns one **complete** raw episode into
training data:

1. **`segments()`** splits the raw stream at any invalid row or a timestamp
   gap outside `[0.5, 1.5] × 1/fps` — no interpolation or resampling, so a
   segment is exactly the original ticks.
2. **`export()`** writes each segment as its own HDF5 file: `state`/`action`
   split into `left_arm_joint_states` (6), `left_ee_joint_states` (1),
   `right_arm_joint_states` (6), `right_ee_joint_states` (1), plus one RGB
   uint8 `vision/<camera_name>/colors` dataset per view.
3. **`to_lerobot()`** (run in XPolicyLab's LeRobot v2 environment, not the
   Jetson capture environment) converts a list of those HDF5 segments into a
   LeRobot v2 dataset for pi0.5/OpenPI training.

## Live dashboard

The dashboard (`deployment/dashboard.py` + `deployment/dashboard.html`) streams live telemetry to the browser via Server-Sent Events (SSE) at ~10 Hz. Zero external dependencies — just Python stdlib `http.server`.

**What you see:**

- **Recording status** — armed/active/finalizing, current task, episode count, duration and tick count, or dim IDLE when `--record` wasn't given
- **Session card** — mode badge (idle/engaged/parked/fault), loop Hz, arm count, connected clients
- **Per-arm cards** — 6 joint position bars (orange = pinned joint), position vs target values, gripper bar, end-effector XYZ, controller XYZ/trigger/grip, hand/tray travel vectors, lag in mm
- **Saved episodes** — auto-refreshing list of raw episode directories with task, status, duration, ticks, arms

**Internals:**

- Served from a daemon thread; pushed from the main thread's status-poll loop in `quest_teleop.main()`, not the control loop, so JSON serialization and network I/O never compete with real-time commands
- `_Broker` fan-out: one `publish()` call fans out to per-client queues; stale frames are dropped, dead clients are auto-removed; a late-joining client gets the last snapshot immediately
- `_ReusableHTTPServer` binds IPv6 (falling back to IPv4) with `allow_reuse_address` and `daemon_threads`
- `serve_forever()` + `shutdown()` for clean exit
- Keepalive comments every 15s prevent browser SSE timeout
- `/demos` endpoint reads each episode directory's `metadata.json`
- Multiple browser tabs can connect simultaneously without interfering with arm control

## Terminology

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

## Key classes

| Class | File | Role |
|-------|------|------|
| `QuestReader` | `quest_teleop.py` | Reads Quest controller poses via `adb logcat`. No network, no browser. |
| `TeleopIK` | `quest_teleop.py` | Builds a MuJoCo model from i2rt's YAM MJCF, solves IK with `mink`, enforces joint limits. |
| `TrayTarget` | `quest_teleop.py` | Clutch-anchored mapping from controller space to world-frame tray pose, with slew limiting. |
| `ArmChannel` | `quest_teleop.py` | One hand driving one arm: filters, clutch, gripper, IK — no mode of its own. |
| `TeleopSession` | `quest_teleop.py` | State machine (idle/engaged/homing/parked/fault) over one or two `ArmChannel`s. Single loop, single clock. |
| `I2rtArm` | `robot.py` | Real hardware backend. Claims the CAN bus (fcntl lock + USB serial verification), wraps `get_yam_robot(zero_gravity_mode=True)`. |
| `SimArm` | `robot.py` | i2rt's MuJoCo `SimRobot` — same `Robot` protocol as hardware. Preferred for development. |
| `EpisodeRecorder` | `recording.py` | Bounded async raw writer, one instance per episode. `add_control()`/`add_capture()` feed a background thread that timestamp-matches ticks to frames and streams CSV/JPEG to disk. `fail()` marks the episode invalid without stopping control. |
| `ZedPairCapture` | `zed_capture.py` | Threaded capture across the two wrist ZED X cameras and the overhead ZED X, running continuously across episodes. `begin_recording()`/`end_recording()` swap the frame callback without restarting the cameras. `FramePairBroker` matches wrist frames and emits explicit orphan events. |
| `Dashboard` | `dashboard.py` | SSE server for live telemetry. `push(snapshot)` fans out to all connected browsers via `_Broker`. |
| `export_dataset` | `export_dataset.py` | `segments()`/`export()`/`to_lerobot()` — raw episode to gap-separated HDF5 segments to a LeRobot v2 dataset. |

## Files

```
deployment/
  quest_teleop.py    Main teleop script (QuestReader, TeleopIK, TrayTarget, ArmChannel, TeleopSession)
  robot.py           Arm backends: I2rtArm (CAN), SimArm (MuJoCo), MockArm (first-order servo)
  zed_capture.py     Threaded ZED wrist/overhead capture (ZedPairCapture, FramePairBroker)
  recording.py       Bounded async raw episode writer (EpisodeRecorder, ControlSample)
  recording.yaml     ZED camera serials and EpisodeRecorder writer settings
  export_dataset.py  Raw episode -> gap-separated HDF5 segments -> LeRobot v2
  inspect_episode.py CLI tool to check a raw episode's integrity and timing
  yam_policy.py      OpenPI YamInputs/YamOutputs transforms (14D state, 3 camera views)
  openpi_config.py   OpenPI YamDataConfig / make_config() for training
  dashboard.py       Live SSE dashboard server (Dashboard, _Broker, zero deps)
  dashboard.html     Single-page dashboard UI (dark theme, dynamic arm cards)
  config.py          YAML config loader with validation
  config.yaml        Right arm config (hand, channel, backend, safety limits, tuning)
  config_left.yaml   Left arm config
  preflight.py       Pre-flight checks for real hardware
  camera.py          One Euro filter implementation
  calibration/       Saved operator frame calibration files
  record_pose.py     Hand-guide a pose and save it
scripts/
  openpi_yam.py      Entry point for stats/batch/train against openpi_config.make_config()
  recording_smoke.py Sim end-to-end smoke test for the recording pipeline
  bootstrap.sh        Create the project-local conda env (Jetson/aarch64 path)
docs/
  architecture.md    This file
balancing_act/
  assets.py          Reset pose loader (yam_home.json)
```

## Configuration

Each arm has its own YAML config (`deployment/config.yaml`, `deployment/config_left.yaml`). Key sections:

**`teleop`** — Control feel: `position_scale` (hand-to-tray ratio), `max_tray_speed_m_s` (slew cap), `filter_min_cutoff_hz` / `filter_beta` (One Euro filter), `orientation` (wrist tracking on/off).

**`robot`** — Hardware: `backend` (i2rt/sim/mock), `channel` (can0/can1), `arm_type`, `gripper_type`, `adapter_serial`, `mapping_verified`.

**`safety`** — Limits: `max_command_velocity_rad_s`, `max_command_offset_rad`, `max_joint_velocity_rad_s`, `joint_position_min/max_rad` (operator box).

**`deploy`** — Poses: `home_joint_position_rad` (park pose), `reset_joint_position_rad` (startup pose).

Camera serials and writer tuning for `--record` live in `deployment/recording.yaml`, separate from the arm configs — see its inline comments for what each field does.

### Tuning tips

- **Arm feels sluggish:** Raise `position_scale` (1.0 = 1:1), `max_tray_speed_m_s`, `max_command_velocity_rad_s`.
- **Arm jitters at rest:** Lower `filter_min_cutoff_hz`.
- **Arm lags behind fast movements:** Raise `filter_beta`, `max_tray_speed_m_s`.
- **Console shows `lag=` values:** The slew cap is binding. Raise `max_tray_speed_m_s`.
- **Console shows `pinned=j2,j3`:** IK is against a joint limit. Tighten the operator box or adjust your approach angle.

## Lab setup notes

The configs ship with the correct CAN adapter serials and channel mappings for our lab's two YAM arms (`mapping_verified: true`). The operator frame calibration files are also included — they're valid as long as you stand in the same spot as the original calibration. If you move to a different position, re-run `--calibrate` for each arm.

**CAN mapping:** Right arm is on `can1`, left arm is on `can0`. This is already set in the configs.

## Known gaps

- **No camera feed into the headset.** The ZED cameras record to disk, but there is no live preview in the Quest.
- **No collision checking.** The operator joint box is the only thing keeping arms apart.
- **Dashboard is view-only.** Episodes start/stop from the joystick, not the browser — the dashboard only reflects state.
