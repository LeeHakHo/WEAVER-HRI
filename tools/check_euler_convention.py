#!/usr/bin/env python3
"""Verify that ``obs_right_ee_euler_xyz`` is scipy's extrinsic-"xyz" Euler.

The hand orientation must be expressed in exactly the same convention as the
robot's, otherwise the two halves of the 14-D state mean different things.
This reads a few rows of one episode's parquet straight from the Hub (the
numeric columns are ~0.3 MB each, so nothing large is downloaded) and compares
``as_euler("xyz")`` of the recorded quaternion against the recorded Euler.

    .venv/bin/python tools/check_euler_convention.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from hri_common import PARQUET_PATH, matrix_to_euler_xyz, quat_xyzw_to_matrix  # noqa: E402


def main() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    repo = sys.argv[1] if len(sys.argv) > 1 else "LeeHakHo/play_hri_insertion_2_eef"
    fs = HfFileSystem()
    with fs.open(f"datasets/{repo}/{PARQUET_PATH}", "rb") as fh:
        pf = pq.ParquetFile(fh)
        tbl = pf.read_row_group(
            0, columns=["obs_right_ee_quat_xyzw", "obs_right_ee_euler_xyz",
                        "obs_right_ee_position", "obs_right_gripper"]
        )

    quat = np.stack(tbl["obs_right_ee_quat_xyzw"].to_pylist())
    eul = np.stack(tbl["obs_right_ee_euler_xyz"].to_pylist())
    pos = np.stack(tbl["obs_right_ee_position"].to_pylist())
    grip = np.asarray(tbl["obs_right_gripper"].to_pylist())

    from scipy.spatial.transform import Rotation
    R = quat_xyzw_to_matrix(quat)
    for conv in ["xyz", "XYZ", "zyx", "ZYX"]:
        got = Rotation.from_matrix(R).as_euler(conv)
        err = np.abs(np.arctan2(np.sin(got - eul), np.cos(got - eul))).max()
        print(f"  as_euler({conv!r:5s}) max angular error = {err:.3e} rad")

    print()
    print("extrinsic-xyz round trip (helper):",
          np.abs(matrix_to_euler_xyz(R) - eul).max())
    print(f"ee position  min={pos.min(0).round(3)}  max={pos.max(0).round(3)}")
    print(f"gripper      min={grip.min():.4f}  max={grip.max():.4f}  "
          f"mean={grip.mean():.4f}")


if __name__ == "__main__":
    main()
