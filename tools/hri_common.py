#!/usr/bin/env python3
"""Shared helpers for the HRI-insertion → WEAVER pipeline.

Camera geometry follows ``meta/camera_calibration.json`` of the
``LeeHakHo/play_hri_insertion_*_eef`` datasets, whose ``how_to_use`` field
states that the base frame is the one the episodes' FK/EEF features live in —
so robot and hand coordinates are directly comparable once a pixel has been
lifted to 3D.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- dataset layout ----------------------------------------------------------

TRAIN_REPOS = [f"LeeHakHo/play_hri_insertion_{i}_eef" for i in range(1, 7)]
EVAL_REPO = "LeeHakHo/play_hri_insertion_eval_eef"
ALL_REPOS = TRAIN_REPOS + [EVAL_REPO]

# LeRobot camera key -> the WEAVER/DROID camera slot it is written into.
# weaver/datasets/droid.py:17 fixes the order of the concatenated video panels.
CAM_ORDER = [
    ("observation.images.cam_high", "exterior_1_left"),
    ("observation.images.cam_low", "exterior_2_left"),
    ("observation.images.cam_right_wrist", "wrist_left"),
]

# Crop applied before the resize to 320x192, per camera, as (x0, y0, x1, y1).
# Projecting all 116k agent points into cam_high shows them spanning u[48, 471]
# and v[62, 321] at p1..p99, so the bottom half of the table and the dark strip
# above it never contain anything. This box keeps 99.55% of them, fixes the 4:3
# -> 5:3 squash the resize would otherwise apply, and spends 1.23x more pixels
# on the subject.
CAMERA_CROP = {
    "observation.images.cam_high": (10, 40, 530, 352),
}

VIDEO_PATH_TMPL = "videos/chunk-000/{cam}/episode_000000.mp4"
PARQUET_PATH = "data/chunk-000/episode_000000.parquet"

SRC_FPS = 30
STRIDE = 6                 # 30 fps -> 5 fps, the rate WEAVER was pretrained at
TARGET_FPS = SRC_FPS / STRIDE

# --- 18-D unified state/action layout ---------------------------------------
#
# Both agents are described by the *same* three points, in the robot base frame
# and in metres: an origin and the two surfaces that close on an object.  The
# gripper and the hand therefore have structurally identical 9-D blocks —
# no Euler angles, no per-agent conventions, no normalisation constants:
#
#     opening     = ||contact_a - contact_b||
#     where it is = origin
#     how it is posed = the triangle the three points span
#
#               origin          contact_a    contact_b
#   robot       ee_position     jaw tip +    jaw tip -
#   hand        wrist           thumb tip    index tip

STATE_DIM = 18
ROBOT_SLICE = slice(0, 9)
HAND_SLICE = slice(9, 18)
POINT_ROLES = ["origin", "contact_a", "contact_b"]
STATE_NAMES = [
    f"{agent}_{role}_{axis}"
    for agent in ("robot", "hand")
    for role in POINT_ROLES
    for axis in "xyz"
]
# The action is the *commanded* pose of the same three points, not a delta of
# the observed ones: that is what a policy or a teleoperator emits, and what a
# real controller would accept. ``action_right_gripper`` dips to -0.65 mm when
# the jaws are told to shut, which the three-point form cannot express -- a
# negative half-separation simply swaps contact_a and contact_b and flips the
# recovered jaw axis by 180 degrees. Those negatives carry no grip-force
# information (they appear in 80.6% of holding frames but also 67.3% of frames
# that shut on empty air, and 98% of them sit within 0.07 mm of zero), so
# clipping at zero loses nothing and keeps the mapping invertible.
ACTION_DIM = 18
ACTION_NAMES = [n.replace("robot_", "robot_cmd_").replace("hand_", "hand_cmd_")
                for n in STATE_NAMES]
GRIPPER_CMD_FLOOR = 0.0

# --- MANO / OpenPose hand joint indices -------------------------------------
# wilor_mini.models.mano_wrapper remaps MANO to the OpenPose hand order.
J_WRIST = 0
J_THUMB_TIP = 4
J_INDEX_TIP = 8
J_MIDDLE_MCP = 9      # only to seed the metric scale, see hand_scale_depth

# The hand's three points, in the order POINT_ROLES expects.
HAND_POINT_JOINTS = [J_WRIST, J_THUMB_TIP, J_INDEX_TIP]


# --- robot gripper geometry --------------------------------------------------
#
# The dataset records the end-effector pose and a single `obs_right_gripper`
# number, not the three points.  Overlaying candidates on cam_high (see
# tools/viz_hand_tracks.py) pins the geometry down:
#
#   contacts  the jaws separate along the EEF frame's y axis, their tips sit at
#             the origin's x, and `obs_right_gripper` is the *half* separation —
#             at 0.0363 the two projected points land exactly on the two
#             fingertips, and at 0 they collapse onto the closed gripper.
#
#   origin    must NOT be `ee_position`: the jaws open symmetrically about it, so
#             it is exactly the midpoint of the two contacts and carries no
#             information — the triangle degenerates to a line and the pose is
#             lost.  Setting it back along the approach axis puts it on the
#             gripper body, the counterpart of the wrist sitting behind a thumb
#             and index finger.  0.12 m both lands on the gripper's mount and
#             matches the hand's own 11.6 cm wrist-to-contact-midpoint offset, so
#             the two agents' triangles come out the same size.
JAW_AXIS = 1              # column of R(ee_euler) the jaws open along
APPROACH_AXIS = 0         # column of R(ee_euler) the fingers point along
ORIGIN_SETBACK = 0.12     # metres back along the approach axis


def robot_points(ee_position: np.ndarray, ee_euler_xyz: np.ndarray,
                 gripper: np.ndarray) -> np.ndarray:
    """(T,3), (T,3), (T,) -> (T, 3, 3): origin, contact_a, contact_b in base frame."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("xyz", np.asarray(ee_euler_xyz, dtype=np.float64)).as_matrix()
    pos = np.asarray(ee_position, dtype=np.float64)
    half = np.asarray(gripper, dtype=np.float64).reshape(-1, 1)
    jaw = R[:, :, JAW_AXIS]
    origin = pos - R[:, :, APPROACH_AXIS] * ORIGIN_SETBACK
    return np.stack([origin, pos + jaw * half, pos - jaw * half], axis=1)


# --- geometry ----------------------------------------------------------------

def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw quaternion -> (..., 3, 3) rotation matrix."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.asarray(q, dtype=np.float64)).as_matrix()


def matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 3) extrinsic-xyz Euler angles.

    Matches ``obs_right_ee_euler_xyz`` in the dataset; verified against
    ``obs_right_ee_quat_xyzw`` by ``tools/check_euler_convention.py``.
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_euler("xyz")


@dataclass
class CameraCalib:
    """One camera's colour + depth intrinsics and extrinsics w.r.t. the base."""

    name: str
    K: np.ndarray            # (3, 3) colour intrinsics
    R_cb: np.ndarray         # (3, 3) colour -> base rotation
    t_cb: np.ndarray         # (3,)   colour -> base translation
    K_depth: np.ndarray | None
    R_db: np.ndarray | None
    t_db: np.ndarray | None
    depth_scale: float | None
    depth_key: str | None

    @staticmethod
    def _intrinsics(d: dict) -> np.ndarray:
        return np.array([[d["fx"], 0.0, d["ppx"]],
                         [0.0, d["fy"], d["ppy"]],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @classmethod
    def from_json(cls, calib: dict, cam_key: str) -> "CameraCalib":
        c = calib["cameras"][cam_key]
        K = cls._intrinsics(c["color_intrinsics"])
        R_cb = quat_xyzw_to_matrix(c["color_to_base"]["quat_xyzw"])
        t_cb = np.asarray(c["color_to_base"]["position"], dtype=np.float64)

        dep = c.get("depth") or {}
        if dep:
            K_d = cls._intrinsics(dep["depth_intrinsics"])
            R_db = quat_xyzw_to_matrix(dep["depth_to_base"]["quat_xyzw"])
            t_db = np.asarray(dep["depth_to_base"]["position"], dtype=np.float64)
            scale = float(dep["depth_scale_meters_per_unit"])
            key = dep["feature_key"]
        else:
            K_d = R_db = t_db = scale = key = None

        return cls(cam_key, K, R_cb, t_cb, K_d, R_db, t_db, scale, key)

    # -- colour camera ------------------------------------------------------

    def unproject(self, uv: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Colour pixels (..., 2) at depth z (...) -> points in the colour frame."""
        uv = np.asarray(uv, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        x = (uv[..., 0] - self.K[0, 2]) / self.K[0, 0] * z
        y = (uv[..., 1] - self.K[1, 2]) / self.K[1, 1] * z
        return np.stack([x, y, z], axis=-1)

    def project(self, p_cam: np.ndarray) -> np.ndarray:
        """Points in the colour frame (..., 3) -> colour pixels (..., 2)."""
        p = np.asarray(p_cam, dtype=np.float64)
        z = np.clip(p[..., 2], 1e-6, None)
        u = p[..., 0] / z * self.K[0, 0] + self.K[0, 2]
        v = p[..., 1] / z * self.K[1, 1] + self.K[1, 2]
        return np.stack([u, v], axis=-1)

    def cam_to_base(self, p_cam: np.ndarray) -> np.ndarray:
        return np.asarray(p_cam, dtype=np.float64) @ self.R_cb.T + self.t_cb

    def base_to_cam(self, p_base: np.ndarray) -> np.ndarray:
        return (np.asarray(p_base, dtype=np.float64) - self.t_cb) @ self.R_cb

    def rot_cam_to_base(self, R_cam: np.ndarray) -> np.ndarray:
        return self.R_cb @ R_cam

    # -- depth camera -------------------------------------------------------

    def base_to_depth_px(self, p_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Base points -> (depth pixel (...,2), depth-frame z (...))."""
        p = (np.asarray(p_base, dtype=np.float64) - self.t_db) @ self.R_db
        z = np.clip(p[..., 2], 1e-6, None)
        u = p[..., 0] / z * self.K_depth[0, 0] + self.K_depth[0, 2]
        v = p[..., 1] / z * self.K_depth[1, 1] + self.K_depth[1, 2]
        return np.stack([u, v], axis=-1), p[..., 2]

    def depth_px_to_base(self, uv: np.ndarray, z: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        x = (uv[..., 0] - self.K_depth[0, 2]) / self.K_depth[0, 0] * z
        y = (uv[..., 1] - self.K_depth[1, 2]) / self.K_depth[1, 1] * z
        return np.stack([x, y, z], axis=-1) @ self.R_db.T + self.t_db


def load_calibration(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# --- dataset readers ---------------------------------------------------------

ROBOT_COLUMNS = [
    "obs_right_ee_position", "obs_right_ee_euler_xyz", "obs_right_gripper",
    "action_right_ee_position", "action_right_ee_euler_xyz", "action_right_gripper",
]


def read_robot_states(repo: str, columns: list[str] | None = None,
                      n_workers: int = 16) -> dict[str, np.ndarray]:
    """Read the small numeric columns of one episode straight from the Hub.

    The depth image columns dominate these parquet files (~2 GB each), so a
    column projection keeps the payload to a couple of megabytes — but it is
    spread over ~180 row groups, and one serial HTTPS range request per row
    group costs about 100 ms, so the read is entirely latency-bound. Fetching
    the row groups in parallel (each thread with its own handle) turns ~105 s
    into a few seconds; ``pre_buffer`` alone does not help, since it only
    coalesces reads *within* a row group.
    """
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    columns = columns or ROBOT_COLUMNS
    path = f"datasets/{repo}/{PARQUET_PATH}"
    fs = HfFileSystem()
    with fs.open(path, "rb") as fh:
        n_groups = pq.ParquetFile(fh).metadata.num_row_groups

    def read_group(rg: int):
        with fs.open(path, "rb") as fh:
            return pq.ParquetFile(fh).read_row_group(rg, columns=columns)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        tables = list(pool.map(read_group, range(n_groups)))
    tbl = pa.concat_tables(tables)

    return {name: np.asarray(tbl.column(name).to_pylist(), dtype=np.float32)
            for name in columns}


# --- hand geometry -----------------------------------------------------------


def hand_scale_depth(kp3d: np.ndarray, kp2d: np.ndarray, fx: float) -> np.ndarray:
    """Depth of the wrist from MANO's metric hand size and its pixel size.

    Uses the wrist -> middle-MCP segment, which is rigid and almost always
    visible.  Independent of WiLoR's assumed focal length (its ``pred_cam_t``
    is computed with FOCAL_LENGTH=5000, ~20x this camera's real focal), so this
    is the reliable monocular scale cue.
    """
    kp3 = np.asarray(kp3d, dtype=np.float64)
    kp2 = np.asarray(kp2d, dtype=np.float64)
    s_m = np.linalg.norm(kp3[..., J_MIDDLE_MCP, :] - kp3[..., J_WRIST, :], axis=-1)
    s_px = np.linalg.norm(kp2[..., J_MIDDLE_MCP, :] - kp2[..., J_WRIST, :], axis=-1)
    return fx * s_m / np.clip(s_px, 1e-6, None)


# --- signal cleanup ----------------------------------------------------------

def interpolate_invalid(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid rows of (T, D) by linear interpolation, edges held."""
    x = np.array(x, dtype=np.float64, copy=True)
    valid = np.asarray(valid, dtype=bool)
    if valid.all():
        return x
    if not valid.any():
        return np.zeros_like(x)
    t = np.arange(len(x))
    for d in range(x.shape[1]):
        x[:, d] = np.interp(t, t[valid], x[valid, d])
    return x


def smooth_savgol(x: np.ndarray, window: int = 9, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing along time for (T, D) signals."""
    from scipy.signal import savgol_filter
    window = min(window, len(x) if len(x) % 2 else len(x) - 1)
    if window < poly + 2:
        return np.asarray(x, dtype=np.float64)
    return savgol_filter(np.asarray(x, dtype=np.float64), window, poly, axis=0)

