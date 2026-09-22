# Quest → two YAM arms → synchronized three-camera demonstrations

The implemented path uses the existing USB/ADB Quest application and one
100 Hz arm loop, with three ZED capture threads at 960×600, 30 FPS. The camera
set is left wrist ZED X One `301058360`, right wrist ZED X One `306353224`, and
overhead ZED X `41925345` (left RGB sensor). ZED X Mini `53655029` is not part
of capture. The GMSL cameras share the capture card's hardware frame trigger;
the recorder then validates all three SDK timestamps and associates the bundle
to arm/controller samples in software. There is no shared camera/encoder
trigger, and the driver may return cached encoder readings.

The USB CAN adapters are persistently named by adapter serial: right arm
`can_right`, left arm `can_left`. Do not replace these with enumeration-dependent
`can0`/`can1` names.

## 1. Environment

From a terminal on the Jetson, in this repo's root:

```bash
cd /path/to/yam_vr_teleop
# Skip on a Jetson that already has the env; bootstrap is for a fresh install.
bash scripts/bootstrap.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./.conda-env
python scripts/smoke.py
python tests/test_backform.py
python -m pytest tests/test_recording.py -q
python tests/test_teleop_stack.py
```

Bootstrap creates a project-local conda environment and writes exact conda/pip
lock files after successful installation. It retains the installed ZED SDK and
installs the supplied CPython 3.13 vendor wheel. Run bootstrap once; activate
the environment in each subsequent terminal. Training runs in XPolicyLab's
separate environment, not this Jetson capture environment.

## 2. Connect and verify the Quest (no robot motion)

Enable developer mode for your headset and USB debugging. Connect it to the
Jetson with a USB data cable, put it on, and accept the USB debugging prompt.
Keep both controllers visible to the headset's tracking cameras.

```bash
adb devices
python -m deployment.quest_teleop --dump
```

The device must show `device`, not `unauthorized`. The existing reader installs
and launches `com.rail.oculus.teleop`, downloading its SHA256-pinned APK if
necessary. If more than one Android device is present, pass `--serial SERIAL`.
Both controller poses must change smoothly as you move. Ctrl-C ends this test.
If `adb` is missing, install the OS `android-tools-adb` package. USB permissions
and the existing vendor-2833 udev rule are described in HANDOVER.md.

## 3. Verify cameras separately

Close the standalone camera test scripts before opening the recorder.

```bash
python -m deployment.zed_capture --frames 300
```

The three independent workers copy RGB frames from SDK buffers. The wrist
cameras use `CameraOne`; the overhead ZED X uses `Camera` with `VIEW.LEFT`.
The bundle gate is 12 ms, below half a 30 Hz period, to avoid accepting
adjacent exposures. Every accepted event must contain all three views. A large
number of orphans is a failed synchronization test, not a reason to widen the
gate to a whole frame. A wall-clock offset change above 2 ms invalidates
capture. Edit bounds in `deployment/recording.yaml` only after diagnostics.

On 2026-09-21 the live three-camera soak produced 301 bundles at approximately
30 FPS, zero grab errors/long intervals, 0.038 ms median and 0.558 ms maximum
three-camera SDK timestamp spread. Re-run this check after moving a FAKRA cable,
changing a port, updating the driver, or rebooting into a different device tree.

## 4. Configure and check the arms

Use `deployment/config.yaml` for the right arm and `config_left.yaml` for the
left. Preserve the established `can_right`/`can_left` mapping and 100 Hz rate. The shipped files
still have `robot.gripper_type: no_gripper`; set this existing option to the
actual mounted model before using `--gripper`. Recording refuses a missing
measured gripper rather than fabricating the seventh value. The optional
`teleop.gripper_closed` and `teleop.gripper_open` map trigger travel into driver
units (defaults 0 and 1); confirm their meaning on the hardware.

Once both physical CAN adapters are present, configure each interface:

```bash
sudo ip link set can_right down
sudo ip link set can_right type can bitrate 1000000
sudo ip link set can_right txqueuelen 1000
sudo ip link set can_right up
sudo ip link set can_left down
sudo ip link set can_left type can bitrate 1000000
sudo ip link set can_left txqueuelen 1000
sudo ip link set can_left up
```

Confirm which physical arm each interface addresses; persist adapter naming
before collecting. Complete HANDOVER.md sections 5–6 to
verify each arm independently, record per-arm reset poses, and tighten
the joint limits for your table and bimanual workspace. The shipped poses and
limits are placeholders, and there is no collision checker. Then run:

```bash
python -m deployment.preflight --config deployment/config.yaml --second-config deployment/config_left.yaml
python -m deployment.quest_teleop --calibrate --config deployment/config.yaml
python -m deployment.quest_teleop --calibrate --config deployment/config_left.yaml
```

Calibration uses the trigger: hold it, push the controller about 30 cm towards
the arm, then release. Repeat whenever your standing position/tracking origin
changes. Calibration itself does not command the arms.

## 5. Test motion before collecting

First test the Quest driving simulated arms:

```bash
python -m deployment.quest_teleop --backend sim --second-config deployment/config_left.yaml --gripper
```

After the hardware configuration and preflight pass, test each physical arm
separately at reduced speed, with the other arm powered off:

```bash
python -m deployment.quest_teleop --backend i2rt --position-scale 0.3 --max-speed 0.1 --lock-orientation --gripper
python -m deployment.quest_teleop --backend i2rt --config deployment/config_left.yaml --position-scale 0.3 --max-speed 0.1 --lock-orientation --gripper
```

Startup holds each arm at its measured starting joint pose; it does not home.
With `--gripper`, **grip** is the arm clutch. Hold the **trigger** while grip is
held to close the gripper; hold the trigger without grip to open it. Releasing
the trigger holds the current gripper opening. Release grip to hold the arm.
**A/X** requests the configured ready pose from idle/parked (only if verified);
**B/Y** parks both arms. **Ctrl-C** parks and exits. Parking is a commanded
return to the positions captured when that session opened the arms, not an
encoder re-zero. Ctrl-C removes motor torque after the return; keep the
hardware stop accessible.

## 6. Collect one task episode

```bash
python -m deployment.quest_teleop --backend i2rt \
  --second-config deployment/config_left.yaml --gripper \
  --record episodes --task "place vial" --dashboard
```

Open `http://localhost:8080` on the Jetson while that command is running (or
`http://JETSON_IP:8080` from another computer on the same network). The
dashboard is a live view of controller, arm, gripper, loop and recording state;
it does not start/stop recording and it is not a camera-video viewer. The
Python file `deployment/dashboard.py` is the small web server and telemetry
bridge. `deployment/dashboard.html` is the page that server sends to the
browser. `quest_teleop.py --dashboard` starts the server and feeds it live
snapshots. Stop the demonstration with Ctrl-C in the teleop terminal.

The dashboard also shows maximum target-versus-measured joint error and whether
the 1.5 rad/s command slew cap is active. Persistent large error or a speed cap
that stays active explains lag; stop and inspect rather than increasing the
0.10 rad measured-position offset guard.

Cameras open and warm up before the CAN arm drivers are opened, so SDK startup
cannot starve the motor communication loops. Recording and the arm control
thread then start together. Perform the demonstration,
then Ctrl-C to end the episode and park. One invocation creates one episode;
repeat for subsequent demonstrations. Homing, parking, stale Quest input and
fault modes are excluded from training but retained in diagnostics.

The printed episode path contains:

- `metadata.json`: task, effective configs, timing policy, validity and errors.
- `controls.jsonl`: all sampled control ticks, left-first state/action order,
  read/command intervals, controller receive timestamps and validity.
- `data.csv`: every emitted three-view bundle/orphan, raw image paths, selected 14-D state
  and command, timing offsets and rejection reason.
- `debug_timing.csv`: each frame's serial, sequence and SDK/host timestamps.
- `images/left`, `images/right`, and `images/overhead`: JPEGs, including orphan frames.

After shutdown, copy the printed path and run:

```bash
python -m deployment.inspect_episode episodes/episode_TIMESTAMP
eog episodes/episode_TIMESTAMP/images/left/*.jpg
eog episodes/episode_TIMESTAMP/images/right/*.jpg
eog episodes/episode_TIMESTAMP/images/overhead/*.jpg
```

The inspector is the integrity/timing check. The three `eog` commands let you
step through each camera stream as captured; the matching row for every usable
three-camera instant is in `data.csv`.

A bundle selects the first valid control read **after all three image timestamps**,
within 20 ms; no interpolation. State and action come from that same tick.
Five invalid captures within 30 active captures, queue overflow, low disk
space or camera/writer failure makes the episode
invalid and visibly reports it, while teleoperation remains available. An
invalid or interrupted recording keeps its `.partial` suffix. Successfully
flushed raw episodes are atomically renamed. The exporter separately checks
whether there are any valid training rows.

A process/power failure may lose buffered data or the last unfinished image.
Keep `.partial` directories for forensic recovery; do not rename them manually
and feed them to training. Complete raw retention cannot be guaranteed when the
disk is full or acquisition outruns a bounded queue; these cases are explicit
failures, never silently valid demonstrations.

## 7. Export and train

```bash
python -m deployment.export_dataset episodes/episode_TIMESTAMP --output exports
```

This creates one XPolicyLab-style HDF5 trajectory per uninterrupted valid
segment. Orphans, invalid controls and timestamp gaps create boundaries. Images
are RGB at their captured resolution, frequency is 30, and `action/` holds the
actual issued commands, never `state[t+1]`. Raw data is left intact. Camera
names are `cam_left_wrist`, `cam_right_wrist`, and `cam_overhead`; change
`camera_names` in recording.yaml only to reflect the physical mounting.

To create LeRobot data, run the exporter from an XPolicyLab environment with
LeRobot installed, adding `--lerobot-root DATASET_PATH --repo-id local/yam`.
Every HDF5 segment is saved as a separate LeRobot episode, so action windows
cannot cross a missing frame. The converter uses actual image dimensions and
frequency, and creates 14-D state and action features. It does not upload data.

For OpenPI, repack the resulting features to `state`, `images/left`,
`images/right`, `images/base`, `actions`, `prompt`, and use `deployment.yam_policy.YamInputs`
and `YamOutputs` instead of ALOHA's joint-unit transforms. Configure action
horizon 50 and dataset frequency 30. The transform maps the wrist views to the
two wrist slots and the overhead view to `base_0_rgb`. OpenPI's model transforms must still perform
resizing, normalization and action padding. `action_chunk` in export_dataset
provides repeat-last padding and a validity mask for offline inspection.

The full XPolicyLab training configuration and a real loader batch must be
validated in that project's environment before training. This repository does
not include XPolicyLab or a policy-to-motor deployment controller; do not run
50 predicted actions open-loop on the arms.

### Training commands

With the converted dataset under LeRobot's configured data home, run these
from XPolicyLab's `policy/Pi_05/openpi` directory in its training environment:

```bash
export PYTHONPATH=/path/to/yam_vr_teleop:$PYTHONPATH
python /path/to/yam_vr_teleop/scripts/openpi_yam.py stats --repo-id local/yam
python /path/to/yam_vr_teleop/scripts/openpi_yam.py batch --repo-id local/yam
python /path/to/yam_vr_teleop/scripts/openpi_yam.py train --repo-id local/yam --exp-name place-vial
```

`deployment/openpi_config.py` provides the YAM data factory and π0.5 config,
using absolute recorded actions and a 50-step horizon. The `batch` command is
the required loader check before training. These entrypoints target XPolicyLab
revision `bb9a0b5f5136a74503b679af830bfd0a3a837d5c`; execute the batch check again
when changing versions. Each HDF5 also has an adjacent 17-column CSV (three
image paths plus 14 joint observations), with image paths relative to that CSV.

### Jetson dependency adjustment

The pinned i2rt revision declares Raspberry Pi GPIO as a dependency on all
Linux aarch64 machines. Its `lgpio` wrapper does not build on Python 3.13, and
Jetson CAN teleoperation does not use it. `scripts/jetson_i2rt_wheel.py` makes
that dependency an explicit optional `raspberry-pi` extra in the local wheel.
Robot code is unchanged. Original and adjusted wheels, plus the metadata edit,
are retained in `.vendor-wheels`; `pip check` still validates the installed
dependency set. The environment does not support Raspberry Pi GPIO peripherals.

### Timing and integration checks

```bash
python -m deployment.inspect_episode episodes/episode_TIMESTAMP
python scripts/recording_smoke.py --seconds 10 --output /tmp/yam-synthetic
python scripts/recording_smoke.py --zed --seconds 10 --output /tmp/yam-with-cameras
```

Both smoke commands use simulated robots and synthetic controller poses; the
second uses the actual ZED cameras. Neither opens CAN. Metadata labels this
output as synthetic. The inspector reports missing files, camera/control rates,
read spans, latency and matching skew. SDK CURRENT validates the host clock
mapping; IMAGE supplies frame timestamps, so delivery latency is measured
separately. Homing starts after camera warmup, and transitional rows stay raw.
