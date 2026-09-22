# Handover: Quest → YAM teleop

Historical hardware bring-up notes. For the installed conda environment, camera
recording, exports and current validation, use [OPERATING.md](OPERATING.md) and
[validation/README.md](validation/README.md).

You have the Linux machine and the arms. This document is what you need to take
this repo from "runs in simulation" to "drives two real YAM arms".

Written 2026-09-10. Everything below was developed and tested on a Windows
laptop against simulated arms, because that machine has no CAN. **No line of
the hardware path has ever run.** That is the single most important thing to
know before you start.

---

## 1. What this does

A Meta Quest 3 streams both controllers' 6-DoF poses over USB. The teleop reads
them off `adb logcat` — no network, no browser — filters them, maps clutched
hand motion into each arm's frame, solves IK against i2rt's own YAM model, and
commands joint positions. Two arms run in **one process** with one clock, one
stop button and one fault domain.

```
Quest controller pose  ->  One Euro filter  ->  clutch-anchored world mapping
   ->  slew cap  ->  mink QP IK on i2rt's YAM MJCF  ->  joint clamp  ->  CAN
```

Per tick, at `teleop.control_hz` (100 Hz):

1. Read every arm's state; trip on a runaway (measured joint speed).
2. Filter the raw controller pose; declutch outright if it teleports.
3. While the clutch is held, add filtered motion since the press to the
   end-effector pose measured at that press. Release freezes and re-anchors.
4. Slew-cap the commanded pose as a final guard.
5. Solve IK, clamp into the safety envelope, command.

Normal stop and Ctrl-C walk both arms back to the joint poses captured when the
session opened them rather than dropping them at the teleoperated position.

## 2. What is tested, and what is not

| Component | Status |
|---|---|
| Quest pose streaming over adb | **verified** on a real Quest 3 (Windows) |
| `mink` IK on i2rt's YAM model | **verified** — tracks a 37 mm move to 0.000 mm |
| Two arms, one loop, 100 Hz | **verified** in sim |
| Shared stop, shared fault domain | **verified** — either controller parks both |
| Gripper mapping (trigger → gripper) | **verified** in sim |
| Runaway velocity trip | **verified** in sim |
| `deployment/robot.py` `I2rtArm` (CAN) | **NEVER RUN.** API calls audited against i2rt's real `MotorChainRobot` and they match, but that is all |
| Home poses, joint box, operator frame | **placeholders.** Yours to record |

File provenance, since it matters for how much to trust each:

- `deployment/quest_teleop.py` — originally from a working setup; the session
  layer was restructured for two arms, but the safety logic is unchanged.
- `deployment/robot.py`, `config.py`, `camera.py`, `balancing_act/assets.py` —
  reconstructed from the original script's interfaces. Expect bugs.
- `deployment/preflight.py`, `record_pose.py`, `tests/test_teleop_stack.py` —
  new, written 2026-09-10.
- Anything marked `TUNE` in the YAML is a guess.

## 3. Install

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install "i2rt @ git+https://github.com/i2rt-robotics/i2rt"
```

On Linux the plain i2rt install should work. (It fails on Windows — `ruckig`
won't build — hence the `--no-deps` recipe in `requirements.txt`, which you can
ignore.)

You also need `adb`:

```bash
sudo apt install android-tools-adb
```

and a udev rule so adb can open the Quest, in `/etc/udev/rules.d/51-quest.rules`:

```
SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0664", GROUP="plugdev"
```

Verify the whole stack with no hardware at all:

```bash
python tests/test_backform.py        # stubbed, no heavy deps
python tests/test_teleop_stack.py    # real mujoco/mink/i2rt, sim arms
python -m deployment.quest_teleop --backend sim   # needs the Quest
```

## 4. The Quest

The app (`com.rail.oculus.teleop`) may already be installed; if not, teleop
installs it on first run from `teleop.apk_path`.

1. Headset → Settings → System → Developer → **enable USB Debugging**.
2. Plug in over USB. Put the headset on and accept "Allow USB debugging"
   (tick *Always allow*).
3. `adb devices` should show `device`, not `unauthorized`.

Two gotchas that look exactly like software bugs:

- **The headset sleeps when off your head**, freezing poses and eventually
  killing the logcat stream. You'll see `age=` climb into the thousands of ms.
  Fix: `adb shell am broadcast -a com.oculus.vrpowermanager.prox_close`
- **Controllers are tracked by the headset's cameras.** A controller the
  headset cannot see reports a *frozen pose several metres from the origin*
  while its buttons still work fine. Both controllers must stay in view.

Check the stream before anything else:

```bash
python -m deployment.quest_teleop --dump
```

You want both hands showing plausible positions (tens of centimetres) that
change as you move. Metres-from-origin values mean untracked, not broken code.

## 5. CAN

Two channels, 1 Mbit/s:

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 up
```

**Pin the channels by adapter serial in a udev rule.** If `can0`/`can1` swap on
reboot, the left controller drives the right arm. With no collision checking
anywhere in this system (see §7), that is the expensive kind of mistake.

## 6. Bring-up, in this order

Do not skip ahead. Each step assumes the previous one passed.

### 6.1 Pre-flight — nothing is energised

```bash
python -m deployment.preflight --config deployment/config.yaml \
                               --second-config deployment/config_left.yaml
```

Checks the CAN interfaces are up at the right bitrate, surveys each motor chain
*without building a chain or enabling a motor*, and audits the configs: gripper
type, pose validity, joint-box width, frame calibration. Exits non-zero while
anything is unsafe. **Right now it fails deliberately** — see §7.1.

The chain survey it runs is i2rt's own, which you can also invoke directly:

```bash
python -m i2rt.motor_drivers.dm_driver --channel can0 --survey-only --check-motor-types
```

If a motor is missing, misidentified, or on the wrong ID, this is where you find
out, with everything still limp.

### 6.2 Confirm encoders, arm limp

```bash
python -m i2rt.robots.motor_chain_robot --arm yam --gripper linear_4310 \
       --channel can0 --operation-mode gravity_comp
```

The arm floats. Hand-guide each joint through its range and confirm nothing
fights you and nothing is stuck.

### 6.3 Record the poses — this clears the pre-flight failure

```bash
python -m deployment.record_pose --channel can0 --gripper linear_4310 --save-home
```

The arm comes up in gravity compensation and is **never commanded**. Guide it to
a comfortable ready pose over the table, press Enter. It writes
`balancing_act/yam_home.json` and sets `verified_on_hardware: true`.

For the **second arm**, if it is mounted differently, record it separately and
put the result in that arm's config as `deploy.reset_joint_position_rad` rather
than overwriting the shared file:

```bash
python -m deployment.record_pose --channel can1 --gripper linear_4310
```

There is no configured park pose. When teleop opens each arm, it immediately
captures the measured six-joint pose before camera warm-up or commanded motion.
B/Y and shutdown return each arm to its own captured session-start pose. This
does not re-zero the encoders.

### 6.4 Tighten the joint box

In each config, `safety.joint_position_min_rad` / `max_rad`. Shipped at ±3.0,
which constrains nothing. See §7.3 — this matters more than it looks.

### 6.5 Calibrate the operator frame, per arm

```bash
python -m deployment.quest_teleop --calibrate
python -m deployment.quest_teleop --calibrate --config deployment/config_left.yaml
```

Stand where you will teleoperate, hold the trigger, push the controller ~30 cm
straight at the arm, release. This measures your heading in the headset's
tracking frame — without it, hand motion drives the arm along the wrong axes.
Keep the controller in front of the headset; the tool rejects a glitched push
but can only do that if you notice the message.

### 6.6 First motion — one arm, deliberately timid

Second arm powered **off**. Hand on the power switch.

```bash
python -m deployment.quest_teleop --backend i2rt \
       --position-scale 0.3 --max-speed 0.1 --lock-orientation
```

`--lock-orientation` drives position only, which halves what can go wrong.
Squeeze the trigger, move a few centimetres, release. Then build up.

### 6.7 Both arms

```bash
python -m deployment.quest_teleop --backend i2rt \
       --second-config deployment/config_left.yaml --gripper
```

Controls: **grip** clutches (or the trigger, when `--gripper` is off), **trigger**
drives the gripper, **A/X** re-homes, **B/Y** parks. Either controller stops both
arms.

## 7. The five things that will bite you

### 7.1 The reset pose is not yours

`balancing_act/yam_home.json` holds a pose derived from *geometry in simulation*,
not measured on an arm. Homing is a 3-second smoothstep the clutch **cannot
interrupt**, so an unverified pose is an arm driving somewhere nobody checked.

The i2rt backend therefore **refuses to start** until you either record the pose
(§6.3) or set a per-arm `deploy.reset_joint_position_rad`. This is intentional.
Clear it by recording, not by flipping the flag.

### 7.2 `gripper_type` is `no_gripper`

If grippers are mounted, this is wrong twice over: i2rt builds a **6-motor chain
for a 7-motor arm**, so the gripper motor is never commanded and hangs limp; and
IK's `grasp_site` sits at the flange, roughly 10 cm behind the fingers, so the
operator drives the wrong point. Set it in both configs.

### 7.3 There is no collision checking. Anywhere.

Not between the arms, not with the table, not with themselves. The operator
joint box in `safety.joint_position_*` is the **only** thing keeping two arms
out of each other, and it ships wide open at ±3.0 rad. Tighten it before you
ever run both arms, and test the limits with one arm first.

### 7.4 `I2rtArm` has never run

`deployment/robot.py` is reconstructed. I audited its calls against i2rt's real
`MotorChainRobot` — `get_joint_pos()`, `get_observations()["joint_vel"]`,
`num_dofs()` all match, and the gripper-in-vector handling looks right — but
audited is not tested. Watch the first run closely.

One known subtlety already handled: i2rt's `SimRobot` zeroes its own `qvel` when
commanded, which would silently disable the runaway trip, so the sim backend
finite-differences velocity instead. Real hardware reports true velocity and
uses it directly.

### 7.5 The arm goes limp when the process exits

i2rt's `MotorChainRobot.close()` sets all torques to zero. Teleop first returns
each arm to the measured pose captured when that session opened the arm and
then closes. The startup pose therefore has to be one the arm can physically
rest in **unpowered**. If it is not, the arm sags or drops when Ctrl-C completes.

### 7.6 i2rt refuses to open an arm resting outside its limits

`MotorChainRobot` runs `_check_current_qpos_in_joint_limits` at construction
(0.1 rad of buffer) and raises if the arm is parked outside its joint range.
This surfaces as teleop's "Could not open the arm". It is not a CAN fault —
hand-guide the arm back inside its range and retry.

### 7.7 Loop rate vs stream rate

`control_hz` is 100 but the headset app logs at ~70 Hz, so ~30% of ticks see a
repeated sample. The One Euro filter reads that as zero velocity and over-
smooths. If it feels laggy, try `control_hz: 70`.

## 8. Tuning, once it moves

Read the console line — it is built for exactly this:

```
engaged  100.6Hz | R clutch=1 hand=(-0.13,-0.01,-0.05) tray=(-0.13,-0.01,-0.05) q=[...] lag=  4mm
```

- **`hand` vs `tray`** — the two vectors side by side. Different *direction*
  means the operator frame is wrong: recalibrate. Different *magnitude* means
  `position_scale`.
- **`lag=`** — how far the commanded pose is behind your hand after slew
  limiting. If it stays up while you move, `max_tray_speed_m_s` is binding;
  raise it. Standing lag is what "under-responsive" feels like.
- **`pinned=j4`** — that joint is against a limit and the arm cannot travel
  further that way however far your hand goes. Either the joint box is too
  tight or the ready pose is badly placed.
- **Jitter at rest** — lower `filter_min_cutoff_hz`. **Lag when moving fast** —
  raise `filter_beta`.

## 9. Known gaps

- **No recording.** Nothing writes data. This is the biggest missing piece for
  the actual goal of collecting demonstrations. i2rt bundles a
  `RobotMcapRecorder` (`robot.start_mcap_recording()`), which is the obvious
  starting point; a LeRobot-format episode writer is the alternative.
- **No cameras.** `deployment/camera.py` is a One Euro filter despite the name;
  there is no camera code anywhere.
- **No base-frame relationship between the arms.** Each arm's IK runs in its own
  base frame, which is correct for control, but nothing knows where the two
  bases sit relative to each other. Recorded data will need that transform.
- **Gripper feedback is not recorded or displayed** beyond the commanded value.

## 10. Quick reference

```bash
# no hardware
python tests/test_teleop_stack.py
python -m deployment.quest_teleop --backend sim

# check before touching arms
python -m deployment.preflight --config deployment/config.yaml \
                               --second-config deployment/config_left.yaml

# controller stream only, touches nothing
python -m deployment.quest_teleop --dump

# record a verified ready pose
python -m deployment.record_pose --channel can0 --gripper linear_4310 --save-home

# operator frame
python -m deployment.quest_teleop --calibrate

# drive, one arm, timid
python -m deployment.quest_teleop --backend i2rt \
       --position-scale 0.3 --max-speed 0.1 --lock-orientation

# drive, both arms
python -m deployment.quest_teleop --backend i2rt \
       --second-config deployment/config_left.yaml --gripper
```

Note `robot.backend` in the shipped configs is `sim`, so real runs need an
explicit `--backend i2rt`. That default is deliberate.
