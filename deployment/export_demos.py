"""Export recorded HDF5 demos to CSV or LeRobot v2.1 format.

CSV needs no extra dependencies. LeRobot format needs pyarrow:
    uv pip install pyarrow

Usage:
    # Single demo to CSV
    python -m deployment.export_demos demos/demo_20260916_182708.hdf5 --csv out/

    # All demos to CSV
    python -m deployment.export_demos demos/ --csv out/

    # All demos to LeRobot dataset (for policy training)
    python -m deployment.export_demos demos/ --lerobot out/yam_teleop_dataset

    # LeRobot at 30 Hz (resample to camera rate)
    python -m deployment.export_demos demos/ --lerobot out/dataset --fps 30

    # Both at once
    python -m deployment.export_demos demos/ --csv out/csv --lerobot out/lerobot
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ARM_DOF = 6

# CSV gets the full 13-dim state (positions + velocities + gripper).
CSV_STATE_NAMES = [f"pos_j{i+1}" for i in range(ARM_DOF)] + [f"vel_j{i+1}" for i in range(ARM_DOF)] + ["gripper"]
CSV_ACTION_NAMES = [f"j{i+1}" for i in range(ARM_DOF)] + ["gripper"]

# LeRobot/XPolicyLab gets 7-dim state (positions + gripper) to match
# the canonical format and enable relative-action mode.
LR_STATE_NAMES = [f"j{i+1}" for i in range(ARM_DOF)] + ["gripper"]
LR_ACTION_NAMES = LR_STATE_NAMES


def load_demo(path: Path) -> dict:
    """Load joint data from one HDF5 demo."""
    with h5py.File(path, "r") as f:
        hands = list(f.attrs.get("arms", []))
        hz = float(f.attrs.get("hz", 100.0))
        timestamps = f["timestamps"][:].astype(np.float64)
        arms = {}
        for hand in hands:
            g = f[hand]
            jp = g["joint_position"][:].astype(np.float32)
            jv = g["joint_velocity"][:].astype(np.float32)
            jt = g["joint_target"][:].astype(np.float32)
            gp = g["gripper_position"][:].astype(np.float32).reshape(-1, 1)
            gc = g["gripper_command"][:].astype(np.float32).reshape(-1, 1)
            arms[hand] = {
                "joint_position": jp,
                "joint_velocity": jv,
                "joint_target": jt,
                "gripper_position": gp,
                "gripper_command": gc,
                "csv_state": np.hstack([jp, jv, gp]),
                "csv_action": np.hstack([jt, gc]),
                "lr_state": np.hstack([jp, gp]),
                "lr_action": np.hstack([jt, gc]),
            }
        camera_names = []
        camera_frames = {}
        camera_resolutions = {}
        if "cameras" in f:
            for cam_name in f["cameras"]:
                cg = f[f"cameras/{cam_name}"]
                camera_names.append(cam_name)
                camera_frames[cam_name] = [bytes(cg["frames"][i]) for i in range(len(cg["frames"]))]
                camera_resolutions[cam_name] = (int(cg.attrs.get("width", 0)), int(cg.attrs.get("height", 0)))
    return {
        "path": path, "hands": hands, "hz": hz, "timestamps": timestamps, "arms": arms,
        "camera_names": camera_names, "camera_frames": camera_frames,
        "camera_resolutions": camera_resolutions,
    }


def collect_demos(source: Path) -> list[dict]:
    if source.is_file():
        return [load_demo(source)]
    paths = sorted(source.glob("*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No .hdf5 files in {source}")
    return [load_demo(p) for p in paths]


def _resample_indices(timestamps: np.ndarray, target_fps: float) -> np.ndarray:
    """Pick indices from timestamps that are closest to a uniform grid at target_fps."""
    duration = timestamps[-1] - timestamps[0]
    n_out = max(1, int(round(duration * target_fps)))
    grid = np.linspace(timestamps[0], timestamps[-1], n_out)
    indices = np.searchsorted(timestamps, grid, side="right") - 1
    return np.clip(indices, 0, len(timestamps) - 1)


# -------------------------------------------------------------------- CSV

def write_csv(demos: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for demo in demos:
        for hand in demo["hands"]:
            arm = demo["arms"][hand]
            n = len(demo["timestamps"])
            suffix = f"_{hand}" if len(demo["hands"]) > 1 else ""
            name = demo["path"].stem + suffix + ".csv"
            path = out_dir / name
            state_cols = [f"state_{s}" for s in CSV_STATE_NAMES]
            action_cols = [f"action_{s}" for s in CSV_ACTION_NAMES]
            header = ["timestamp"] + state_cols + action_cols
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(n):
                    row = [f"{demo['timestamps'][i]:.4f}"]
                    row += [f"{v:.6f}" for v in arm["csv_state"][i]]
                    row += [f"{v:.6f}" for v in arm["csv_action"][i]]
                    writer.writerow(row)
            written.append(path)
            print(f"  {path} ({n} rows)")
    return written


# ----------------------------------------------------------------- LeRobot

def write_lerobot(demos: list[dict], out_dir: Path, task: str = "teleop",
                  target_fps: float | None = None) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow is required for LeRobot format:", file=sys.stderr)
        print("  uv pip install pyarrow", file=sys.stderr)
        raise SystemExit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    ref = demos[0]
    all_hands = ref["hands"]
    state_dim = len(LR_STATE_NAMES) * len(all_hands)
    action_dim = len(LR_ACTION_NAMES) * len(all_hands)
    source_fps = ref["hz"]
    export_fps = target_fps if target_fps is not None else source_fps
    camera_names = ref.get("camera_names", [])
    resampling = target_fps is not None and target_fps < source_fps

    if len(all_hands) == 1:
        s_names = list(LR_STATE_NAMES)
        a_names = list(LR_ACTION_NAMES)
    else:
        s_names = [f"{h}_{s}" for h in all_hands for s in LR_STATE_NAMES]
        a_names = [f"{h}_{s}" for h in all_hands for s in LR_ACTION_NAMES]

    # tasks.jsonl
    (meta_dir / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")

    # episodes.jsonl + parquet files
    global_index = 0
    episodes = []
    all_stats_state = []
    all_stats_action = []

    for ep_idx, demo in enumerate(demos):
        timestamps = demo["timestamps"]

        if resampling:
            sel = _resample_indices(timestamps, target_fps)
        else:
            sel = np.arange(len(timestamps))

        n = len(sel)
        states = np.hstack([demo["arms"][h]["lr_state"][sel] for h in all_hands])
        actions = np.hstack([demo["arms"][h]["lr_action"][sel] for h in all_hands])
        ts_out = timestamps[sel]
        all_stats_state.append(states)
        all_stats_action.append(actions)

        indices = list(range(global_index, global_index + n))
        frame_indices = list(range(n))
        ts_list = (ts_out - ts_out[0]).tolist()
        episode_indices = [ep_idx] * n
        task_indices = [0] * n
        done = [False] * n
        done[-1] = True

        columns = {
            "observation.state": [states[i].tolist() for i in range(n)],
            "action": [actions[i].tolist() for i in range(n)],
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "timestamp": pa.array(ts_list, type=pa.float64()),
            "next.done": pa.array(done, type=pa.bool_()),
            "index": pa.array(indices, type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
        }

        for cam_name in camera_names:
            frames = demo.get("camera_frames", {}).get(cam_name, [])
            if frames:
                vid_dir = out_dir / "videos" / "chunk-000" / cam_name
                vid_dir.mkdir(parents=True, exist_ok=True)
                vid_path = vid_dir / f"episode_{ep_idx:06d}.mp4"
                selected_frames = [frames[i] for i in sel] if resampling else frames
                _write_video(selected_frames, vid_path, fps=int(export_fps))
                columns[f"observation.images.{cam_name}"] = [
                    {"path": f"videos/chunk-000/{cam_name}/episode_{ep_idx:06d}.mp4", "timestamp": t}
                    for t in ts_list
                ]

        table = pa.table(columns)
        ep_path = data_dir / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(table, ep_path)

        episodes.append({
            "episode_index": ep_idx,
            "tasks": [task],
            "length": n,
        })
        global_index += n
        print(f"  episode {ep_idx}: {n} frames ({demo['path'].name})")

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    # stats.json
    all_state = np.vstack(all_stats_state)
    all_action = np.vstack(all_stats_action)
    stats = {
        "observation.state": _compute_stats(all_state),
        "action": _compute_stats(all_action),
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    # info.json
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": s_names,
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": a_names,
        },
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "timestamp": {"dtype": "float64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    for cam_name in camera_names:
        res = ref.get("camera_resolutions", {}).get(cam_name, (640, 480))
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [res[1], res[0], 3],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.fps": int(export_fps),
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "has_audio": False,
            },
        }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "yam",
        "fps": int(export_fps),
        "features": features,
        "total_episodes": len(demos),
        "total_frames": global_index,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")

    resample_note = f" (resampled from {int(source_fps)} Hz)" if resampling else ""
    print(f"\n  LeRobot v2.1 dataset: {len(demos)} episodes, {global_index} frames @ {int(export_fps)} Hz{resample_note}")
    print(f"  observation.state: {state_dim}D ({', '.join(s_names)})")
    print(f"  action: {action_dim}D ({', '.join(a_names)})")
    if camera_names:
        print(f"  cameras: {', '.join(camera_names)}")
    return out_dir


def _write_video(jpeg_frames: list[bytes], path: Path, fps: int = 30) -> None:
    """Decode JPEG frames and write an mp4 video via OpenCV."""
    import cv2
    writer = None
    for jpg in jpeg_frames:
        if not jpg:
            continue
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
        writer.write(frame)
    if writer is not None:
        writer.release()


def _compute_stats(data: np.ndarray) -> dict:
    return {
        "mean": [round(float(v), 6) for v in data.mean(axis=0)],
        "std": [round(float(v), 6) for v in data.std(axis=0)],
        "min": [round(float(v), 6) for v in data.min(axis=0)],
        "max": [round(float(v), 6) for v in data.max(axis=0)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m deployment.export_demos",
        description="Export HDF5 demos to CSV or LeRobot format.",
    )
    parser.add_argument("source", type=Path, help="Single .hdf5 file or directory of them")
    parser.add_argument("--csv", type=Path, default=None, metavar="DIR", help="Write CSV files to DIR")
    parser.add_argument("--lerobot", type=Path, default=None, metavar="DIR", help="Write LeRobot v2.1 dataset to DIR")
    parser.add_argument("--task", default="teleop", help="Task name for LeRobot metadata (default: teleop)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Resample LeRobot output to this rate (e.g. 30 to match camera Hz). "
                             "Default: native recording rate. CSV is always full rate.")
    args = parser.parse_args()

    if args.csv is None and args.lerobot is None:
        parser.error("Specify at least one of --csv or --lerobot")

    demos = collect_demos(args.source)
    print(f"Loaded {len(demos)} demo(s)\n")

    if args.csv:
        print("CSV:")
        write_csv(demos, args.csv)
        print()

    if args.lerobot:
        print("LeRobot:")
        write_lerobot(demos, args.lerobot, task=args.task, target_fps=args.fps)


if __name__ == "__main__":
    main()
