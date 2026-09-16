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
                    ┌──────┼──────┐
                    ▼      ▼      ▼
                Recorder  Dashboard  Console
                (HDF5)    (SSE@10Hz) (stdout)
```

## Control path (per tick, 100 Hz)

1. **Read** — `I2rtArm.read_state()` gets joint positions/velocities from the motor chain. Trip immediately if any joint exceeds `max_joint_velocity_rad_s`.
2. **Filter** — One Euro filter smooths the raw Quest pose. Jitter on a resting hand is filtered hard; fast motion passes through nearly 1:1. Pose teleports (tracking glitches) are rejected outright.
3. **Map** — While the clutch is held, filtered hand motion since the press is rotated into the world frame (using the calibrated operator heading) and scaled by `position_scale`. The tray pose is slew-capped at `max_tray_speed_m_s` / `max_tray_omega_rad_s`.
4. **IK** — `TeleopIK` solves one damped QP (via `mink`) to turn the target end-effector pose into joint angles, clamped to the intersection of mechanical limits and the operator's joint box.
5. **Command** — Joint targets are rate-limited (`max_command_velocity_rad_s`) and offset-clamped (`max_command_offset_rad`) before being sent to the motor chain.

Releasing the clutch freezes the arm. Re-pressing re-anchors without jumping — the arm picks up from where it is, not where the controller is.

## Data recording

The recorder (`deployment/recorder.py`) captures every signal in the system at the control loop rate (~100 Hz). Writing is decoupled from the loop — `tick()` appends to plain lists (cheap), `detach()` swaps them out in O(1), and `write_snapshot()` serializes to HDF5 on a background thread so the control loop never stalls.

Each HDF5 file contains per-arm datasets (keyed by hand: `/right/...`, `/left/...`):

| Dataset | Shape | Description |
|---------|-------|-------------|
| `joint_position` | (N, 6) | Measured joint angles [rad] |
| `joint_velocity` | (N, 6) | Measured joint velocities [rad/s] |
| `joint_target` | (N, 6) | Commanded joint targets [rad] |
| `gripper_position` | (N,) | Measured gripper opening [0-1] |
| `gripper_command` | (N,) | Commanded gripper target [0-1] |
| `ee_position` | (N, 3) | End-effector position [m] |
| `ee_quaternion` | (N, 4) | End-effector orientation [xyzw] |
| `controller_position` | (N, 3) | Quest controller position [m] |
| `controller_quaternion` | (N, 4) | Quest controller orientation [xyzw] |
| `controller_trigger` | (N,) | Trigger value [0-1] |
| `controller_grip` | (N,) | Grip value [0-1] |
| `controller_clutch` | (N,) | Clutch engaged [0/1] |

Plus session-level data: `/timestamps` (N,) seconds since first tick, `/mode` (N,) session mode per tick, and file-level attrs (`start_time`, `hz`, `arms`, `num_ticks`, `duration_s`).

## Live dashboard

The dashboard (`deployment/dashboard.py` + `deployment/dashboard.html`) streams live telemetry to the browser via Server-Sent Events (SSE) at ~10 Hz. Zero external dependencies — just Python stdlib `http.server`.

**What you see:**

- **Recording status** — pulsing red REC indicator with timer and tick count, or dim IDLE
- **Session card** — mode badge (idle/engaged/parked/fault), loop Hz, arm count, connected clients
- **Per-arm cards** — 6 joint position bars (orange = pinned joint), position vs target values, gripper bar, end-effector XYZ, controller XYZ/trigger/grip, hand/tray travel vectors, lag in mm
- **Saved demos** — auto-refreshing list of HDF5 files with duration, ticks, arms, file size

**Internals:**

- Served from a daemon thread alongside the control loop
- `_Broker` fan-out: one `publish()` call fans out to per-client queues; stale frames are dropped, dead clients are auto-removed
- `_ReusableHTTPServer` = `ThreadingMixIn + HTTPServer` with `allow_reuse_address` and `daemon_threads`
- `serve_forever()` + `shutdown()` for clean exit
- Keepalive comments every 15s prevent browser SSE timeout
- `/demos` endpoint reads HDF5 file attrs via h5py
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
| `DemoRecorder` | `recorder.py` | Appends per-tick data to in-memory buffers. `detach()` hands off a snapshot for background HDF5 writing. |
| `Dashboard` | `dashboard.py` | SSE server for live telemetry. `push(snapshot)` fans out to all connected browsers via `_Broker`. |

## Files

```
deployment/
  quest_teleop.py    Main teleop script (QuestReader, TeleopIK, TrayTarget, ArmChannel, TeleopSession)
  robot.py           Arm backends: I2rtArm (CAN), SimArm (MuJoCo), MockArm (first-order servo)
  recorder.py        HDF5 demonstration recorder (DemoRecorder, write_snapshot)
  dashboard.py       Live SSE dashboard server (Dashboard, _Broker, zero deps)
  dashboard.html     Single-page dashboard UI (dark theme, dynamic arm cards)
  inspect_demo.py    CLI tool to inspect recorded HDF5 demos
  config.py          YAML config loader with validation
  config.yaml        Right arm config (hand, channel, backend, safety limits, tuning)
  config_left.yaml   Left arm config
  preflight.py       Pre-flight checks for real hardware
  camera.py          One Euro filter implementation
  calibration/       Saved operator frame calibration files
  record_pose.py     Hand-guide a pose and save it
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

- **No cameras.** `camera.py` is just the One Euro filter. No camera feed into the headset yet.
- **No collision checking.** The operator joint box is the only thing keeping arms apart.
- **Dashboard is view-only.** No controls for starting/stopping recording from the browser yet.
