"""Export complete raw episodes into gap-separated XPolicyLab trajectories.

No adjacent-state action synthesis, interpolation or time compression. Each
segment becomes an independent training episode. HDF5 images are RGB uint8.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

PARTS = {"left_arm_joint_states": slice(0, 6), "left_ee_joint_states": slice(6, 7),
         "right_arm_joint_states": slice(7, 13), "right_ee_joint_states": slice(13, 14)}


def camera_sides(meta):
    sides = tuple(meta.get("camera_sides", ("left", "right")))
    if (len(sides) < 2 or len(set(sides)) != len(sides)
            or not set(sides) <= {"left", "right", "overhead"}):
        raise ValueError("invalid camera_sides in episode metadata")
    return sides


def segments(episode):
    episode = Path(episode)
    meta = json.loads((episode / "metadata.json").read_text())
    sides = camera_sides(meta)
    if episode.suffix == ".partial" or meta["status"] != "complete" or meta.get("errors"):
        raise ValueError("incomplete/invalid episode; inspect raw files, do not export for training")
    with (episode / "data.csv").open() as f:
        rows = sorted(csv.DictReader(f), key=lambda r: (int(r["image_timestamp_ns"]), int(r["capture_index"])))
    current, previous = [], None
    period = 1e9 / meta["fps"]
    for row in rows:
        valid = row["valid"] == "1" and all(row[f"{side}_image"] for side in sides)
        if valid:
            values = np.array([float(row[f"{kind}_{i}"]) for kind in ("state", "action") for i in range(14)])
            valid = np.isfinite(values).all()
        timestamp = int(row["image_timestamp_ns"])
        contiguous = (previous is None or
                      (.5 * period <= timestamp - previous <= 1.5 * period))
        if not valid or not contiguous:
            if current:
                yield current
            current = []
        if valid:
            current.append(row)
            previous = timestamp
        else:
            previous = None
    if current:
        yield current


def action_chunk(actions, index, horizon=50):
    if horizon <= 0 or not 0 <= index < len(actions):
        raise ValueError("invalid horizon/index")
    indices = np.arange(index, index + horizon)
    mask = indices < len(actions)
    return actions[np.minimum(indices, len(actions) - 1)].copy(), mask


def export(episode, destination):
    import cv2
    import h5py
    episode, destination = Path(episode), Path(destination)
    meta = json.loads((episode / "metadata.json").read_text())
    sides = camera_sides(meta)
    names = meta.get("capture", {}).get("camera_names", ["cam_left_wrist", "cam_right_wrist"])
    if len(names) != len(sides) or len(set(names)) != len(names) or any("/" in n for n in names):
        raise ValueError("one unique simple camera name is required for each captured view")
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for number, rows in enumerate(segments(episode)):
        path = destination / f"{episode.name}_segment_{number:04d}.hdf5"
        if path.exists() or path.with_suffix(".partial").exists():
            raise FileExistsError(path)
        with h5py.File(path.with_suffix(".partial"), "w") as f:
            f.create_dataset("additional_info/frequency", data=meta["fps"])
            f.attrs["task"] = meta["task"]
            f.attrs["source_episode"] = episode.name
            f.attrs["color_order"] = "RGB"
            f.create_dataset("timing/image_timestamp_ns", data=[int(r["image_timestamp_ns"]) for r in rows])
            f.create_dataset("timing/capture_index", data=[int(r["capture_index"]) for r in rows])
            for kind in ("state", "action"):
                array = np.asarray([[float(r[f"{kind}_{i}"]) for i in range(14)] for r in rows], dtype=np.float32)
                for name, columns in PARTS.items():
                    f.create_dataset(f"{kind}/{name}", data=array[:, columns])
            for side, name in zip(sides, names):
                dataset = None
                for i, row in enumerate(rows):
                    frame = cv2.imread(str(episode / row[f"{side}_image"]))
                    if frame is None:
                        raise ValueError(f'missing/unreadable image in capture {row["capture_index"]}')
                    frame = frame[..., ::-1]
                    if dataset is None:
                        dataset = f.create_dataset(f"vision/{name}/colors", (len(rows), *frame.shape),
                                                   dtype="u1", chunks=(1, *frame.shape), compression="lzf")
                    if frame.shape != dataset.shape[1:]:
                        raise ValueError("image dimensions changed within segment")
                    dataset[i] = frame
        path.with_suffix(".partial").rename(path)
        # Image paths plus the exact original 14 joint observations, one file per segment.
        # Paths are relative to this CSV so it remains directly inspectable.
        import os
        with path.with_suffix(".csv").open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([f"{side}_image" for side in sides] +
                            [f"{side}_q{i}" for side in ("left", "right") for i in range(1, 8)])
            for row in rows:
                writer.writerow([os.path.relpath(episode / row[f"{side}_image"], destination)
                                 for side in sides] +
                                [row[f"state_{i}"] for i in range(14)])
        outputs.append(path)
    if not outputs:
        raise ValueError("episode contains no valid training segments")
    return outputs


def to_lerobot(paths, repo_id, root):
    """Run in XPolicyLab's LeRobot v2 environment, not the Jetson capture env."""
    import h5py
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    with h5py.File(paths[0]) as f:
        fps = int(f["additional_info/frequency"][()])
        camera_shapes = {name: f[f"vision/{name}/colors"].shape[1:] for name in f["vision"]}
    features = {key: {"dtype": "float32", "shape": (14,), "names": None}
                for key in ("observation.state", "action")}
    features.update({f"observation.images.{name}": {"dtype": "image", "shape": shape,
                     "names": ["height", "width", "channels"]} for name, shape in camera_shapes.items()})
    dataset = LeRobotDataset.create(repo_id=repo_id, root=Path(root), fps=fps, robot_type="yam", features=features)
    for path in paths:
        with h5py.File(path) as f:
            if int(f["additional_info/frequency"][()]) != fps or set(f["vision"]) != set(camera_shapes):
                raise ValueError("inconsistent frequency/camera schema")
            arrays = {kind: np.concatenate([f[f"{kind}/{name}"][:] for name in PARTS], axis=1)
                      for kind in ("state", "action")}
            for i in range(len(arrays["state"])):
                frame = {"observation.state": arrays["state"][i], "action": arrays["action"][i], "task": f.attrs["task"]}
                for name, shape in camera_shapes.items():
                    image = f[f"vision/{name}/colors"][i]
                    if image.shape != shape:
                        raise ValueError("inconsistent camera dimensions")
                    frame[f"observation.images.{name}"] = image
                dataset.add_frame(frame)
            dataset.save_episode()
    finalize = getattr(dataset, "finalize", None)
    if callable(finalize):
        finalize()
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lerobot-root", type=Path)
    parser.add_argument("--repo-id", default="local/yam")
    args = parser.parse_args()
    paths = export(args.episode, args.output)
    if args.lerobot_root:
        to_lerobot(paths, args.repo_id, args.lerobot_root)
    print("\n".join(map(str, paths)))

if __name__ == "__main__":
    main()
