#!/usr/bin/env python3
"""Stage 2 of hand-track extraction: WiLoR output -> 3 hand points in base frame.

Consumes the ``.npz`` written by ``tools/run_wilor.py`` (which runs in the
separate ``.venv-wilor``) and produces the hand half of the unified 18-D state —
the same three-point description the robot gripper gets (see ``hri_common``):

    points   (T, 3, 3)  wrist, thumb tip, index tip in the robot base frame, metres
    valid    (T,)       all three points were measured in this frame

Every point is obtained the same way: WiLoR's projected 2-D keypoint, lifted by
the depth camera.  Nothing depends on MANO's 3-D pose fit, which matters because
WiLoR's left/right classification flickers and a mirrored fit corrupts the 3-D
joints while leaving the 2-D keypoints essentially intact.

Depth handling:

1.  **MANO scale** seeds the lookup — the wrist->middle-MCP segment has a known
    metric length and a measured pixel length, so ``z = fx * s_m / s_px``.
    WiLoR's own ``pred_cam_t`` is unusable here: it assumes focal 5000 against
    this camera's real 604, putting the hand ~18 m away.
2.  **The depth sensor** then measures each point.  Colour and depth have
    different intrinsics *and* extrinsics, so each colour ray is walked into the
    depth camera, read there, and re-intersected with the ray rather than
    indexing the depth image at the colour pixel.  A point whose own reading is
    missing or implausible falls back to the wrist's depth, which costs at most
    the hand's few-centimetre depth extent under this top-down camera.

Example
-------
    .venv/bin/python tools/extract_hand_tracks.py \
        --repo LeeHakHo/play_hri_insertion_2_eef \
        --wilor data/hand_raw/ds2.npz \
        --out data/hand_tracks/ds2.npz --use-depth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hri_common import (  # noqa: E402
    HAND_POINT_JOINTS, PARQUET_PATH, POINT_ROLES, CameraCalib, hand_scale_depth,
    interpolate_invalid, smooth_savgol,
)

CAM_KEY = "observation.images.cam_high"
DEPTH_WINDOW = 5          # half-width of the depth median patch, pixels
N_POINTS = len(HAND_POINT_JOINTS)

# Plausible region for a hand in the robot base frame, in metres.  The robot's
# own end-effector spans x[-0.13, 0.21] y[-0.11, 0.21] z[0.06, 0.26] and the
# human works to its +y side, so this box is generous but still rejects the
# metres-off positions a bad monocular scale estimate produces.
WORKSPACE = ((-0.25, 0.75), (-0.25, 0.95), (-0.08, 0.55))

# A fingertip reading further than this from the wrist's is the table or the
# grasped object showing through, not the fingertip.  Under this top-down camera
# a hand spans only a few centimetres in depth, and MANO — which measures the
# thumb-index distance without the depth camera — says the true opening never
# exceeds ~16 cm.  At 20 cm the depth-derived opening reached 26 cm; at 6 cm it
# tops out at 17 cm and the disagreement with MANO halves at the tail
# (p95 4.0 -> 1.9 cm), for 14% of fingertip reads falling back to the wrist's.
MAX_POINT_DEPTH_SPREAD = 0.06


def _gap_lengths(valid: np.ndarray) -> np.ndarray:
    """Lengths of the runs of invalid frames."""
    out, run = [], 0
    for v in valid:
        if v:
            if run:
                out.append(run)
            run = 0
        else:
            run += 1
    if run:
        out.append(run)
    return np.asarray(out, dtype=int)


def fetch_calibration(repo: str) -> dict:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo, repo_type="dataset",
                           filename="meta/camera_calibration.json")
    from hri_common import load_calibration
    return load_calibration(path)


def read_depth_at(repo: str, frame_indices: np.ndarray, probe,
                  depth_column: str, n_points: int) -> np.ndarray:
    """Stream one depth column and sample it where ``probe`` asks.

    ``probe(frame_index, depth_image_uint16) -> (n_points,)`` returns depths in
    raw units (0 = no reading).  Only row groups holding a wanted frame are
    fetched and each decoded image is dropped immediately, so peak memory stays
    at one 640x480 frame.
    """
    import cv2
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    wanted = set(int(i) for i in frame_indices)
    out: dict[int, np.ndarray] = {}

    fs = HfFileSystem()
    with fs.open(f"datasets/{repo}/{PARQUET_PATH}", "rb") as fh:
        pf = pq.ParquetFile(fh)
        row0 = 0
        for rg in range(pf.metadata.num_row_groups):
            n_rows = pf.metadata.row_group(rg).num_rows
            rows_here = [i for i in range(row0, row0 + n_rows) if i in wanted]
            if rows_here:
                tbl = pf.read_row_group(rg, columns=[depth_column])
                col = tbl.column(depth_column).to_pylist()
                for i in rows_here:
                    rec = col[i - row0]
                    if rec is None or rec.get("bytes") is None:
                        continue
                    img = cv2.imdecode(np.frombuffer(rec["bytes"], np.uint8),
                                       cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    out[i] = probe(i, img)
                print(f"  depth row-group {rg + 1}/{pf.metadata.num_row_groups}"
                      f"  ({len(out)}/{len(wanted)} sampled)", end="\r", flush=True)
            row0 += n_rows
    print()
    zero = np.zeros(n_points)
    return np.stack([out.get(int(i), zero) for i in frame_indices], axis=0)


def sample_patch(img: np.ndarray, u: float, v: float) -> float:
    """Median of the non-zero depth in a small window around a pixel."""
    h, w = img.shape[:2]
    u0, v0 = int(round(u)), int(round(v))
    if not (0 <= u0 < w and 0 <= v0 < h):
        return 0.0
    patch = img[max(0, v0 - DEPTH_WINDOW):v0 + DEPTH_WINDOW + 1,
                max(0, u0 - DEPTH_WINDOW):u0 + DEPTH_WINDOW + 1]
    nz = patch[patch > 0]
    return float(np.median(nz)) if nz.size else 0.0


def refine_with_depth(repo: str, cam: CameraCalib, uv: np.ndarray, z_seed: np.ndarray,
                      frame_indices: np.ndarray, valid: np.ndarray,
                      cache: Path | None = None, max_spread: float = MAX_POINT_DEPTH_SPREAD,
                      iters: int = 3, anchor_spread: bool = True,
                      labels: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Measure each point's depth with the depth camera.

    ``uv`` is (T, P, 2) colour pixels and ``z_seed`` (T,) the monocular estimate
    that starts every point's lookup.  Returns (T, P) colour-frame depths and a
    (T, P) mask of which ones the sensor actually provided.
    """
    T, P = uv.shape[:2]
    z = np.repeat(z_seed[:, None], P, axis=1)
    used = np.zeros((T, P), bool)

    def depth_pixels(z_cur: np.ndarray) -> np.ndarray:
        p_base = cam.cam_to_base(cam.unproject(uv, z_cur))
        uvd, _ = cam.base_to_depth_px(p_base)
        return uvd                                      # (T, P, 2)

    uvd = depth_pixels(z)
    lookup = {int(f): uvd[i] for i, f in enumerate(frame_indices)}

    def probe(frame_index: int, img: np.ndarray) -> np.ndarray:
        return np.array([sample_patch(img, u, v) for u, v in lookup[frame_index]])

    # Reading the depth column costs ~2 GB per episode, so cache the samples.
    # They are a pure function of the WiLoR output, hence the uvd check.
    raw = None
    if cache is not None and cache.exists():
        c = np.load(cache)
        if (c["raw"].shape == (T, P)
                and np.array_equal(c["frame_index"], frame_indices)
                and np.allclose(c["uvd"], uvd, atol=0.5)):
            raw = c["raw"]
            print(f"  reusing cached depth samples from {cache.name}")
    if raw is None:
        raw = read_depth_at(repo, frame_indices, probe, cam.depth_key, P)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, raw=raw, uvd=uvd, frame_index=frame_indices)

    z_sensor = raw * cam.depth_scale                    # depth-camera frame
    ok = valid[:, None] & (z_sensor > 0)
    if not ok.any():
        print("  no usable depth readings; keeping the MANO-scale estimate")
        return z, used

    # Lift each reading into the base frame, then re-intersect with the colour
    # ray through its own detected pixel so the result stays consistent with the
    # 2-D detection.  The sensor is trusted on its own merits; the monocular cue
    # only seeds the first depth-pixel lookup.
    for _ in range(iters):
        p_base = cam.depth_px_to_base(uvd, z_sensor)
        z_new = cam.base_to_cam(p_base)[..., 2]
        z = np.where(ok, z_new, z)
        used = ok.copy()
        uvd = depth_pixels(z)

    # A fingertip whose reading sits far from the wrist's is the table or the
    # grasped object showing through; fall back to the wrist's depth there.
    # This assumes every probe belongs to one object a few centimetres deep,
    # which is true of a hand and false of probes spread over several separate
    # objects -- there, point 0 is not an anchor for anything else, so callers
    # tracking more than one object pass anchor_spread=False.
    stray = np.zeros_like(used)
    if anchor_spread:
        wrist_z = z[:, 0]
        stray = used & (np.abs(z - wrist_z[:, None]) > max_spread)
        stray[:, 0] = False
        if stray.any():
            z = np.where(stray, wrist_z[:, None], z)
            used &= ~stray
    names = labels if labels is not None else POINT_ROLES
    for p, role in enumerate(names):
        extra = f", {stray[:, p].sum()} fell back to the anchor" if stray[:, p].any() else ""
        print(f"  {role:10s}: depth measured in "
              f"{used[valid, p].mean():5.1%} of valid frames{extra}")
    return z, used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="HuggingFace dataset repo id")
    ap.add_argument("--wilor", type=Path, required=True, help="npz from run_wilor.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--use-depth", action="store_true",
                    help="measure depth with the dataset's depth stream (~2 GB read)")
    ap.add_argument("--smooth-window", type=int, default=9)
    ap.add_argument("--max-depth-spread", type=float, default=MAX_POINT_DEPTH_SPREAD,
                    help="metres; a fingertip depth further than this from the "
                         "wrist's falls back to it")
    ap.add_argument("--max-hand-jump", type=float, default=0.05,
                    help="metres; a minority-handedness frame whose points sit further "
                         "than this from the tracked path is treated as the other hand")
    args = ap.parse_args()

    raw = np.load(args.wilor)
    frame_index = raw["frame_index"]
    valid = raw["found"].copy()
    kp2d, kp3d = raw["kp2d"], raw["kp3d"]
    T = len(frame_index)
    print(f"{args.wilor.name}: {T} frames, hand detected in {valid.mean():.1%}")

    cam = CameraCalib.from_json(fetch_calibration(args.repo), CAM_KEY)

    # --- the three pixels we track ------------------------------------------
    uv = kp2d[:, HAND_POINT_JOINTS].astype(np.float64)        # (T, 3, 2)

    # --- seed depth, from MANO's metric hand size ---------------------------
    z_seed = hand_scale_depth(kp3d, kp2d, cam.K[0, 0])
    valid &= np.isfinite(z_seed) & (z_seed > 0.05) & (z_seed < 5.0)
    z_seed = np.where(valid, z_seed, 1.0)
    print(f"  MANO-scale wrist depth: median {np.median(z_seed[valid]):.3f} m, "
          f"range [{z_seed[valid].min():.3f}, {z_seed[valid].max():.3f}]")

    used_depth = np.zeros((T, N_POINTS), bool)
    z = np.repeat(z_seed[:, None], N_POINTS, axis=1)
    if args.use_depth:
        if cam.depth_key is None:
            print("  calibration has no depth block; skipping --use-depth")
        else:
            cache = args.wilor.with_name(args.wilor.stem + "_depth3.npz")
            z, used_depth = refine_with_depth(args.repo, cam, uv, z_seed, frame_index,
                                              valid, cache=cache,
                                              max_spread=args.max_depth_spread)

    # --- lift to the base frame ---------------------------------------------
    points = cam.cam_to_base(cam.unproject(uv, z))             # (T, 3, 3)

    # Cross-check against MANO, which knows the metric thumb-index distance
    # independently of the depth camera.
    sep = np.linalg.norm(points[:, 1] - points[:, 2], axis=-1)
    mano_sep = np.linalg.norm(
        kp3d[:, HAND_POINT_JOINTS[1]] - kp3d[:, HAND_POINT_JOINTS[2]], axis=-1)
    if valid.any():
        d = np.abs(sep - mano_sep)[valid]
        print(f"  thumb-index separation: measured median {np.median(sep[valid])*100:.1f} cm, "
              f"MANO median {np.median(mano_sep[valid])*100:.1f} cm, "
              f"median disagreement {np.median(d)*100:.1f} cm")

    # --- reject implausible frames ------------------------------------------
    in_box = np.ones(T, bool)
    for ax, (lo_a, hi_a) in enumerate(WORKSPACE):
        in_box &= np.all((points[:, :, ax] >= lo_a) & (points[:, :, ax] <= hi_a), axis=1)
    dropped = valid & ~in_box
    if dropped.any():
        print(f"  dropped {dropped.sum()} frames with a point outside the workspace")
    valid &= in_box

    # WiLoR's left/right call flickers on 10-20% of frames.  A mirrored fit can
    # change which finger a joint index lands on, so those frames survive only
    # if all three points stay continuous with the confidently-tracked path.
    is_right = raw["is_right"]
    if valid.any():
        majority = int(np.bincount(is_right[valid].astype(int), minlength=2).argmax())
        confident = valid & (is_right == majority)
        suspect = valid & ~confident
        if suspect.any() and confident.any():
            ref = np.stack([interpolate_invalid(points[:, p], confident)
                            for p in range(N_POINTS)], axis=1)
            jump = np.linalg.norm(points - ref, axis=-1).max(axis=1)
            too_far = suspect & (jump > args.max_hand_jump)
            print(f"  minority handedness ({'right' if majority else 'left'}-handed episode): "
                  f"kept {(suspect & ~too_far).sum()}, rejected {too_far.sum()} "
                  f"of {suspect.sum()} frames (threshold {args.max_hand_jump*100:.0f} cm)")
            valid &= ~too_far

    gaps = _gap_lengths(valid)
    if gaps.size:
        print(f"  {len(gaps)} gaps to interpolate, longest {gaps.max()} frames "
              f"({gaps.max() / 5.0:.1f} s), median {int(np.median(gaps))}")
    print(f"  final valid: {valid.mean():.1%} of frames")

    # --- fill gaps, smooth ---------------------------------------------------
    for p in range(N_POINTS):
        points[:, p] = interpolate_invalid(points[:, p], valid)
        if args.smooth_window > 1:
            points[:, p] = smooth_savgol(points[:, p], args.smooth_window)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        frame_index=frame_index,
        points=points.astype(np.float32),
        point_roles=np.array(POINT_ROLES),
        valid=valid,
        used_depth=used_depth,
        uv=uv.astype(np.float32),
        depth=z.astype(np.float32),
    )
    for p, role in enumerate(POINT_ROLES):
        q = points[valid, p]
        print(f"  {role:10s} base-frame  x[{q[:,0].min():+.3f},{q[:,0].max():+.3f}]"
              f"  y[{q[:,1].min():+.3f},{q[:,1].max():+.3f}]"
              f"  z[{q[:,2].min():+.3f},{q[:,2].max():+.3f}]")
    sep_v = np.linalg.norm(points[valid, 1] - points[valid, 2], axis=-1)
    print(f"  opening (thumb-index): p5 {np.percentile(sep_v, 5)*100:.1f} cm, "
          f"median {np.median(sep_v)*100:.1f} cm, "
          f"p95 {np.percentile(sep_v, 95)*100:.1f} cm")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
