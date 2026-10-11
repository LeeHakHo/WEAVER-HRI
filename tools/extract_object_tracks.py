#!/usr/bin/env python3
"""Cube and bowl positions in the robot's base frame, one 3-D point each.

The world model has no channel describing the objects, so over a long rollout
the cube fades out of the image -- at step 20,000 it was gone 13.4 s in and
never came back, while the hand, which does have state and action channels,
stayed put for all 179 frames.  An explicit object position gives the training
loss something to hold onto.

Detection is SAM 3 with a text prompt.  The colour-threshold version this
replaces did not work: human skin (82, 45, 29) passes a "red" test, so what it
tracked as the bowl was the person's hand, split into two components, while the
actual bowl -- grey, (53, 56, 58) -- was never detected at all.

Depth comes from the sensor, sampled at several points inside the mask rather
than at its centroid.  The bowl is hollow, so a centroid can fall in the
opening and the depth camera then measures the table underneath.

    .venv/bin/python tools/extract_object_tracks.py \
        --repo LeeHakHo/play_hri_insertion_1_eef --hand data/hand_tracks/ds1.npz \
        --video-dir data/weaver_hri/videos/train --episode-base 1000 \
        --out data/object_tracks/ds1.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_hand_tracks import fetch_calibration, refine_with_depth  # noqa: E402
from hri_common import (  # noqa: E402
    CAMERA_CROP, CameraCalib, interpolate_invalid, read_robot_states, robot_points,
)

CAM_KEY = "observation.images.cam_high"
PROMPTS = {"cube": "red cube", "bowl": "metal bowl"}
N_SAMPLES = 9             # depth probes per mask
SCORE_THRESHOLD = 0.25
WORKSPACE = ((-0.35, 0.85), (-0.35, 1.05), (-0.12, 0.60))


def mask_points(mask: np.ndarray, k: int = N_SAMPLES) -> np.ndarray | None:
    """Evenly spread interior points of a mask, eroded away from the boundary."""
    import cv2
    m = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), 1)
    if m.sum() < k:
        m = mask.astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    order = np.argsort(xs * 1000 + ys)
    idx = np.linspace(0, len(xs) - 1, k).astype(int)
    return np.stack([xs[order][idx], ys[order][idx]], 1).astype(float)


def detect(video_paths: list[Path], device: str) -> tuple[np.ndarray, np.ndarray]:
    """(T, n_objects * N_SAMPLES, 2) crop-frame points and (T, n_objects) seen."""
    import imageio.v3 as iio
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor

    proc = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3",
                                      dtype=torch.bfloat16).to(device).eval()
    frames = np.concatenate([iio.imread(p, plugin="pyav") for p in video_paths])
    T, n = len(frames), len(PROMPTS)
    pts = np.full((T, n * N_SAMPLES, 2), np.nan)
    seen = np.zeros((T, n), bool)
    for t in range(T):
        pil = Image.fromarray(frames[t])
        for pi, prompt in enumerate(PROMPTS.values()):
            inp = proc(images=pil, text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp)
            r = proc.post_process_instance_segmentation(
                out, threshold=SCORE_THRESHOLD, mask_threshold=0.5,
                target_sizes=inp["original_sizes"].tolist())[0]
            sc = r["scores"].float().cpu().numpy()
            if len(sc) == 0:
                continue
            m = r["masks"][int(np.argmax(sc))].cpu().numpy().astype(bool)
            g = mask_points(m)
            if g is None:
                continue
            pts[t, pi * N_SAMPLES:(pi + 1) * N_SAMPLES] = g
            seen[t, pi] = True
        if t % 500 == 0:
            print(f"    {t}/{T}", flush=True)
    del model
    import torch as _t
    _t.cuda.empty_cache()
    return pts, seen


def to_original_pixels(uv: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = CAMERA_CROP[CAM_KEY]
    out = np.empty_like(uv)
    out[..., 0] = x0 + uv[..., 0] * (x1 - x0) / 320.0
    out[..., 1] = y0 + uv[..., 1] * (y1 - y0) / 192.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--hand", type=Path, required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--episode-base", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    vids = sorted((v for v in args.video_dir.glob("*.mp4")
                   if args.episode_base <= int(v.stem) < args.episode_base + 1000),
                  key=lambda q: int(q.stem))
    if not vids:
        raise SystemExit(f"no segment videos under {args.video_dir}")
    print(f"{args.repo}: {len(vids)} segments", flush=True)

    pts, seen = detect(vids, args.device)
    T = len(pts)
    names = list(PROMPTS)
    for i, nm in enumerate(names):
        print(f"  {nm} detected in {seen[:, i].mean() * 100:.1f}% of {T} frames")

    hand = dict(np.load(args.hand))
    fi = hand["frame_index"][:T]
    cam = CameraCalib.from_json(fetch_calibration(args.repo), CAM_KEY)
    robot = read_robot_states(args.repo)
    f = np.clip(fi, 0, len(robot["obs_right_ee_position"]) - 1)
    rp = robot_points(robot["obs_right_ee_position"][f],
                      robot["obs_right_ee_euler_xyz"][f],
                      robot["obs_right_gripper"][f].reshape(-1))
    tcp = (rp[:, 1] + rp[:, 2]) / 2

    uv = np.where(np.isfinite(pts), to_original_pixels(pts), 0.0)
    # anchor_spread is the hand pipeline's "every probe is within a few cm of
    # probe 0" rule. Here probe 0 is a point on the cube and probes 9-17 are on
    # the bowl, which is usually further than that, so the rule would throw the
    # bowl away -- it cost 14% of its frames before this was turned off.
    labels = [f"{nm}[{k}]" for nm in names for k in range(N_SAMPLES)]
    z, used = refine_with_depth(args.repo, cam, uv, cam.base_to_cam(tcp)[:, 2],
                                fi, np.ones(T, bool),
                                cache=args.out.with_name(args.out.stem + "_depth.npz"),
                                anchor_spread=False, labels=labels)
    p_base = cam.cam_to_base(cam.unproject(uv, z))

    res, valid = {}, {}
    for i, nm in enumerate(names):
        sl = slice(i * N_SAMPLES, (i + 1) * N_SAMPLES)
        q = np.where(used[:, sl, None], p_base[:, sl], np.nan)
        with np.errstate(all="ignore"):
            pos = np.nanmedian(q, axis=1)
        ok = seen[:, i] & np.isfinite(pos[:, 0])
        for k, (lo, hi) in enumerate(WORKSPACE):
            ok &= (pos[:, k] > lo) & (pos[:, k] < hi)
        pos[~ok] = np.nan
        res[nm], valid[nm] = interpolate_invalid(pos, ok), ok
        print(f"  {nm}: 3-D valid {ok.mean() * 100:.1f}%")

    grasp = ((robot["action_right_gripper"][f].reshape(-1) < 0.0097)
             & (robot["obs_right_gripper"][f].reshape(-1) > 0.018))
    d = np.linalg.norm(res["cube"] - tcp, axis=1) * 100
    m = grasp & valid["cube"]
    if m.any():
        print(f"  check: cube-to-jaw while the sensor says held -- "
              f"median {np.median(d[m]):.2f} cm, p90 {np.percentile(d[m], 90):.2f} "
              f"(jaw gap is 5.28 cm); not held {np.median(d[grasp == 0]):.2f} cm")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, cube=res["cube"], bowl=res["bowl"],
             cube_valid=valid["cube"], bowl_valid=valid["bowl"],
             frame_index=fi, grasp=grasp)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
