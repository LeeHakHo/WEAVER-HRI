#!/usr/bin/env python3
"""Stage 1 of hand-track extraction: run WiLoR on one LeRobot camera video.

Runs in the dedicated ``.venv-wilor`` environment (python 3.10, torch 2.5,
ultralytics, smplx/chumpy).  It only touches the video — no calibration, no
depth, no parquet — so it stays independent of the WEAVER environment.

Output ``.npz`` holds one row per *sampled* frame (``frame_index`` gives the
index in the original 30 fps video):

    frame_index   (N,)      int32   index into the source video
    found         (N,)      bool    a hand was accepted for this frame
    bbox          (N, 4)    float32 x1 y1 x2 y2 in source pixels
    is_right      (N,)      int8    1 = right hand, 0 = left hand
    kp3d          (N, 21, 3) float32 MANO joints, metres, wrist-relative
    kp2d          (N, 21, 2) float32 projected joints, source pixels
    global_orient (N, 3)    float32 MANO root rotation (rotvec, camera frame)
    cam_t         (N, 3)    float32 weak-perspective translation, camera frame
    focal         ()        float32 scaled focal length used for the projection
    n_det         (N,)      int8    how many hands the detector returned
    propagated    (N,)      bool    the box came from a neighbouring frame, not
                                    from the detector (see propagate_boxes)

Joints follow the OpenPose hand order (see ``mano_wrapper.MANO``):
0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.

Almost 90% of the frames that come out empty do so because the *detector*
returned no box at all — typically where the hand is wrapped around the object
it holds, so it no longer looks hand-shaped. The pose model is not the
bottleneck. A second pass therefore reuses a neighbouring frame's box and runs
the pose model directly on it, skipping the detector entirely.

Example
-------
    .venv-wilor/bin/python tools/run_wilor.py \
        --video /path/cam_high.mp4 --out data/hand_raw/ds2.npz --stride 6
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

N_JOINTS = 21


def probe_video(path: Path) -> tuple[int, int, int]:
    """Return (width, height, nb_frames) via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"]), int(s["nb_frames"])


def iter_frames(path: Path, stride: int, width: int, height: int):
    """Yield (frame_index, RGB uint8 array) for every ``stride``-th frame.

    Decoding goes through a raw ffmpeg pipe rather than OpenCV: these videos are
    AV1 and the bundled OpenCV ffmpeg does not reliably carry an AV1 decoder.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", rf"select=not(mod(n\,{stride}))",
        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=frame_bytes * 4)
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield idx * stride, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            idx += 1
    finally:
        # The caller may stop early (--max-frames); kill ffmpeg rather than
        # letting it write into a closed pipe and complain.
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.wait()


def pick_hand(dets: list[dict], width: int, roi_max_x: float,
              min_area: float) -> dict | None:
    """Choose the human hand among the detections.

    The detector runs at a low confidence threshold to catch the frames where
    the hand is partly hidden behind the object it holds, so the spurious boxes
    that come with that have to be screened out here:

    * the robot gripper reads as a hand often enough, and it lives on the right
      of the frame, so only boxes centred in the left ``roi_max_x`` fraction count;
    * low-confidence junk boxes are much smaller than a real hand at this
      camera distance, hence ``min_area``.

    Among the survivors the largest box wins.
    """
    best, best_area = None, -1.0
    for det in dets:
        if "wilor_preds" not in det:
            continue
        x1, y1, x2, y2 = det["hand_bbox"]
        cx = 0.5 * (x1 + x2)
        if cx > roi_max_x * width:
            continue
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < min_area:
            continue
        if area > best_area:
            best, best_area = det, area
    return best


def propagate_boxes(pipe, video: Path, stride: int, width: int, height: int,
                    rec: dict, n: int, max_distance: int, rescale: float) -> int:
    """Second pass: re-run the pose model on missed frames with a borrowed box.

    The hand is continuously present, so a frame the detector skipped is almost
    always one where the hand is there but not recognisable as a hand. Taking
    the nearest accepted frame's box and calling ``predict_with_bboxes`` runs the
    pose model on the right image region regardless.

    Only frames within ``max_distance`` of an accepted one are attempted, so a
    genuinely hand-free stretch is not filled with a stale box. Recovered frames
    are flagged in ``rec['propagated']``; the downstream continuity test in
    extract_hand_tracks.py screens them like any other suspect detection.
    """
    accepted = np.flatnonzero(rec["found"][:n])
    if accepted.size == 0:
        return 0
    missing = np.flatnonzero(~rec["found"][:n])
    donor = accepted[np.clip(np.searchsorted(accepted, missing), 0, accepted.size - 1)]
    prev = accepted[np.clip(np.searchsorted(accepted, missing) - 1, 0, accepted.size - 1)]
    pick = np.where(np.abs(prev - missing) < np.abs(donor - missing), prev, donor)
    close = np.abs(pick - missing) <= max_distance
    targets = {int(m): int(d) for m, d in zip(missing[close], pick[close])}
    if not targets:
        return 0

    print(f"second pass: {len(targets)} missed frames have an accepted frame "
          f"within {max_distance}; borrowing its box")
    recovered = 0
    for i, (_, rgb) in enumerate(iter_frames(video, stride, width, height)):
        if i >= n:
            break
        d = targets.get(i)
        if d is None:
            continue
        box = rec["bbox"][d].astype(np.float64)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        half_w, half_h = (box[2] - box[0]) / 2 * rescale, (box[3] - box[1]) / 2 * rescale
        grown = np.array([[cx - half_w, cy - half_h, cx + half_w, cy + half_h]])
        dets = pipe.predict_with_bboxes(rgb, grown, [int(rec["is_right"][d])])
        if not dets or "wilor_preds" not in dets[0]:
            continue
        w = dets[0]["wilor_preds"]
        rec["found"][i] = True
        rec["propagated"][i] = True
        rec["bbox"][i] = grown[0]
        rec["is_right"][i] = int(rec["is_right"][d])
        rec["kp3d"][i] = w["pred_keypoints_3d"][0]
        rec["kp2d"][i] = w["pred_keypoints_2d"][0]
        rec["global_orient"][i] = np.asarray(w["global_orient"]).reshape(-1)[:3]
        rec["cam_t"][i] = w["pred_cam_t_full"][0]
        recovered += 1
    return recovered


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=6,
                    help="keep every Nth frame (30 fps / 6 = 5 fps, WEAVER's rate)")
    ap.add_argument("--hand-conf", type=float, default=0.05,
                    help="detector threshold; low on purpose, see pick_hand()")
    ap.add_argument("--min-bbox-area", type=float, default=4000.0,
                    help="reject hand boxes smaller than this, in square pixels")
    ap.add_argument("--roi-max-x", type=float, default=0.6,
                    help="reject detections whose centre is right of this fraction of the width")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after this many sampled frames (smoke tests)")
    ap.add_argument("--propagate-boxes", action="store_true", default=True,
                    help="second pass over missed frames with a neighbour's box")
    ap.add_argument("--no-propagate-boxes", dest="propagate_boxes", action="store_false")
    ap.add_argument("--propagate-max-distance", type=int, default=5,
                    help="frames; how far a borrowed box may travel")
    ap.add_argument("--propagate-rescale", type=float, default=1.3,
                    help="grow the borrowed box by this factor to allow for motion")
    ap.add_argument("--force-hand", choices=["auto", "right", "left"], default="right",
                    help="handedness to reconstruct with. These episodes contain one "
                         "person's one hand, but the detector's left/right call flickers "
                         "on 10-20%% of frames and a wrong call mirrors the whole "
                         "reconstruction. Pinning it keeps every frame in the same "
                         "convention; 'auto' restores the detector's own call.")
    args = ap.parse_args()

    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    width, height, nb_frames = probe_video(args.video)
    n_expected = (nb_frames + args.stride - 1) // args.stride
    if args.max_frames:
        n_expected = min(n_expected, args.max_frames)
    print(f"{args.video.name}: {width}x{height}, {nb_frames} frames "
          f"-> {n_expected} sampled (stride {args.stride})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = WiLorHandPose3dEstimationPipeline(
        device=device, dtype=torch.float16, verbose=False,
    )

    rec = {
        "frame_index": np.zeros(n_expected, np.int32),
        "found": np.zeros(n_expected, bool),
        "bbox": np.zeros((n_expected, 4), np.float32),
        "is_right": np.zeros(n_expected, np.int8),
        "kp3d": np.zeros((n_expected, N_JOINTS, 3), np.float32),
        "kp2d": np.zeros((n_expected, N_JOINTS, 2), np.float32),
        "global_orient": np.zeros((n_expected, 3), np.float32),
        "cam_t": np.zeros((n_expected, 3), np.float32),
        "n_det": np.zeros(n_expected, np.int8),
        "propagated": np.zeros(n_expected, bool),
    }
    focal = np.float32(0.0)

    forced = {"right": 1, "left": 0}.get(args.force_hand)
    n_forced = 0

    import time
    t0 = time.time()
    n = 0
    for frame_index, rgb in iter_frames(args.video, args.stride, width, height):
        if n >= n_expected:
            break
        rec["frame_index"][n] = frame_index
        dets = pipe.predict(rgb, hand_conf=args.hand_conf)
        rec["n_det"][n] = min(len(dets), 127)
        det = pick_hand(dets, width, args.roi_max_x, args.min_bbox_area)
        if det is not None and forced is not None and int(det["is_right"]) != forced:
            # Re-run the pose model on the same box with the handedness pinned,
            # so the joint-to-finger assignment never flips mid-episode.
            redo = pipe.predict_with_bboxes(
                rgb, np.asarray([det["hand_bbox"]], dtype=np.float64), [forced])
            if redo and "wilor_preds" in redo[0]:
                det = redo[0]
                n_forced += 1
        if det is not None:
            w = det["wilor_preds"]
            rec["found"][n] = True
            rec["bbox"][n] = det["hand_bbox"]
            rec["is_right"][n] = int(det["is_right"])
            rec["kp3d"][n] = w["pred_keypoints_3d"][0]
            rec["kp2d"][n] = w["pred_keypoints_2d"][0]
            rec["global_orient"][n] = np.asarray(w["global_orient"]).reshape(-1)[:3]
            rec["cam_t"][n] = w["pred_cam_t_full"][0]
            focal = np.float32(w["scaled_focal_length"])
        n += 1
        if n % 200 == 0:
            rate = n / (time.time() - t0)
            print(f"  {n}/{n_expected}  found={rec['found'][:n].mean():.1%}  "
                  f"{rate:.1f} fps  eta {(n_expected - n) / rate / 60:.1f} min", flush=True)

    if forced is not None:
        print(f"  handedness pinned to {args.force_hand}: "
              f"{n_forced} frames re-run with the mirrored call corrected")
    first_pass = rec["found"][:n].mean()
    if args.propagate_boxes:
        recovered = propagate_boxes(pipe, args.video, args.stride, width, height,
                                    rec, n, args.propagate_max_distance,
                                    args.propagate_rescale)
        print(f"  recovered {recovered} frames "
              f"({first_pass:.1%} -> {rec['found'][:n].mean():.1%})")

    rec = {k: v[:n] for k, v in rec.items()}
    rec["focal"] = focal
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **rec)
    print(f"wrote {args.out}  ({n} frames, hand found in {rec['found'].mean():.1%})")


if __name__ == "__main__":
    main()
