"""Inspect a recorded demonstration HDF5 file.

    python -m deployment.inspect_demo demos/demo_20260916_143022.hdf5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
  parser = argparse.ArgumentParser(prog="python -m deployment.inspect_demo")
  parser.add_argument("file", type=Path, help="Path to a demo .hdf5 file")
  args = parser.parse_args()

  if not args.file.exists():
    print(f"File not found: {args.file}", file=sys.stderr)
    raise SystemExit(1)

  with h5py.File(args.file, "r") as f:
    print(f"=== {args.file.name} ===")
    print(f"  start_time:  {f.attrs.get('start_time', '?')}")
    print(f"  hz:          {f.attrs.get('hz', '?')}")
    print(f"  arms:        {list(f.attrs.get('arms', []))}")
    print(f"  num_ticks:   {f.attrs.get('num_ticks', '?')}")
    print(f"  duration:    {f.attrs.get('duration_s', '?')} s")
    print()

    ts = f["timestamps"][:]
    print(f"  timestamps:  {len(ts)} samples, {ts[-1] - ts[0]:.2f}s span")
    if len(ts) > 1:
      dt = np.diff(ts)
      print(f"  tick rate:   {1.0 / np.mean(dt):.1f} Hz (mean), {1.0 / np.median(dt):.1f} Hz (median)")
      print(f"  tick jitter: {np.std(dt) * 1000:.2f} ms std")
    print()

    for arm in f.attrs.get("arms", []):
      if arm not in f:
        continue
      g = f[arm]
      print(f"  [{arm}]")
      for key in sorted(g.keys()):
        ds = g[key]
        print(f"    {key:30s}  shape={str(ds.shape):15s}  dtype={ds.dtype}")
      jp = g["joint_position"][:]
      jt = g["joint_target"][:]
      gp = g["gripper_position"][:]
      gc = g["gripper_command"][:]
      ee = g["ee_position"][:]
      clutch = g["controller_clutch"][:]
      print(f"    joint pos range:  [{jp.min():.3f}, {jp.max():.3f}] rad")
      print(f"    joint target err: {np.abs(jp - jt).mean():.4f} rad (mean)")
      print(f"    gripper range:    [{gp.min():.3f}, {gp.max():.3f}]")
      print(f"    ee workspace:     x=[{ee[:,0].min():.3f},{ee[:,0].max():.3f}] "
            f"y=[{ee[:,1].min():.3f},{ee[:,1].max():.3f}] "
            f"z=[{ee[:,2].min():.3f},{ee[:,2].max():.3f}] m")
      engaged_pct = 100.0 * clutch.sum() / max(len(clutch), 1)
      print(f"    clutch engaged:   {engaged_pct:.0f}% of ticks")
      print()


if __name__ == "__main__":
  main()
