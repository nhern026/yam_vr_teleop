# Validation on the Jetson, 2026-09-18

## Three-camera extension, 2026-09-21

The capture/record/export path now includes overhead ZED X `41925345` (left
RGB view) with wrist ZED X Ones `301058360` and `306353224`; ZED X Mini
`53655029` is excluded. A 301-bundle live soak measured approximately 30 FPS
on all three streams, zero grab errors/long intervals, and 0.038/0.558 ms
median/maximum SDK timestamp spread. A separate 10-second real-camera plus
simulated-robot smoke produced 300 valid training rows, no missing images or
fatal errors, and an HDF5 export containing all three 960×600 RGB datasets.

The Quest is ADB-authorized and its app emits data after the proximity wake
broadcast. At validation time only the right controller appeared, frozen about
5.5 m from the tracking origin; put on the headset and keep both controllers
visible before calibration or teleoperation.

The two persistent USB CAN interfaces are `can_right` and `can_left`, but both
were down. Physical-arm validation remains blocked by unverified gripper types,
reset/park poses, operator frames, and unconstrained joint boxes. No arm was
energized or commanded during this extension.

## Original two-camera acceptance, 2026-09-18

Installed project conda environment: `.conda-env`; import smoke and `pip check`
passed. Exact packages are in `conda-linux-aarch64.lock` and
`requirements.lock.txt`. i2rt's Raspberry Pi GPIO dependency was made optional
in the locally built wheel; see OPERATING.md.

Real camera capture plus **simulated** YAM arms/controllers, 10-second active
observation interval, with startup/homing/parking included in the raw episode:

| Measurement | Result |
|---|---:|
| Left/right camera frequency | 30.0002 / 30.0002 FPS |
| Control frequency over the complete loop | 99.97 Hz |
| Valid 30 Hz training rows | 300 |
| Total raw capture events | 512 |
| Orphans retained | 1 |
| Transitional/unmatched pairs retained | 211 |
| Maximum selected camera→control offset | 11.54 ms |
| Maximum two-camera timestamp difference | 0.384 ms |
| Maximum arm-read span | 1.67 ms |
| Maximum control period | 20.80 ms |
| Maximum writer queue depth | 2 / 8 |
| Missing images / recorded fatal errors | 0 / 0 |

See `camera_sim_metrics.json` for percentiles. These are measured software read
and image timestamps; cached motor encoder age remains unknown. Python is not
a hard-real-time controller. The 100 Hz target does not assert 100 fresh Quest
packets or encoder updates per second.

Artifacts are in `/tmp/yam-recording-acceptance/episode_20260918_203304_795367456`
and the adjacent `exports/` directory: one complete HDF5 segment and its
16-column observation CSV. These use real images but synthetic robot/controller
states and must not be mixed with demonstrations for training.

**12 focused tests passed**, along with the existing low-level and full simulation
suites. Unit coverage includes causal selection and bounds, orphan retention, timestamp
ordering, missing-frame segmentation, RGB channel correctness, recorded-action
preservation, writer failure, repeated invalid captures, action padding/masks,
per-controller button edges and the homing timer after camera startup.
The real MuJoCo/mink/i2rt simulation suite covers IK, velocity trip, both arms,
grippers, independent clutch release and shared parking.

Not validated on physical arms: CAN command/state freshness, gripper model and
range, calibrated reset/park poses, operator frame and workspace limits.
`adb devices` showed no connected Quest. `can0` was down; `can1` was absent.
The read-only preflight reported unverified reset poses and missing frame
calibrations. No physical arm movement was commanded during this work.

HDF5/CSV export was exercised. LeRobot conversion and OpenPI config/loader
entrypoints are implemented against the referenced XPolicyLab API, but an
actual XPolicyLab training-loader batch has not been run in this environment.
Run the documented `stats` and `batch` commands in the training environment
before launching training.
