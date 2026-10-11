#!/usr/bin/env python3
"""Draw both agents' three points on cam_high, in the one representation.

Each agent is reduced to the same triple — an origin and the two surfaces that
close on an object — and both are projected through ``camera_calibration.json``
from the same base frame:

    role        robot                      hand
    origin      gripper mount              wrist
    contact_a   jaw tip +                  thumb tip
    contact_b   jaw tip -                  index tip

The robot's origin is set back along the approach axis rather than being
``ee_position``: the jaws open symmetrically about that point, so it would be
exactly the midpoint of the two contacts and the triangle would be degenerate.

The robot overlay is what validates the whole chain.  Its coordinates never pass
through the hand pipeline, so when the two jaw points land on the two fingertips
and collapse as the gripper closes, the calibration, the base-frame convention,
the projection code *and* the jaw geometry are confirmed at once — and the hand,
which takes the identical path, is then trustworthy too.

Pass ``--wilor`` to also draw the 21-joint skeleton the hand points come from.

    .venv/bin/python tools/viz_hand_tracks.py \
        --repo LeeHakHo/play_hri_insertion_2_eef \
        --video data/raw/ds2_cam_high.mp4 \
        --tracks data/hand_tracks/ds2.npz \
        --wilor data/hand_raw/ds2.npz \
        --out data/viz/ds2_overlay.mp4 --max-frames 400
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hri_common import (  # noqa: E402
    POINT_ROLES, TARGET_FPS, CameraCalib, read_robot_states, robot_points,
)
from extract_hand_tracks import CAM_KEY, fetch_calibration  # noqa: E402

# BGR. The origin is white; the two contacts share a colour per agent so that
# "these two points are what closes" reads at a glance.
ROBOT_COLOR = (90, 180, 255)       # amber
HAND_COLOR = (120, 255, 140)       # green

# OpenPose hand topology (the order wilor_mini.models.mano_wrapper remaps to):
# 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.
FINGERS = [
    ([0, 1, 2, 3, 4], (80, 80, 255)),        # thumb   - red
    ([0, 5, 6, 7, 8], (80, 200, 255)),       # index   - orange
    ([0, 9, 10, 11, 12], (80, 255, 120)),    # middle  - green
    ([0, 13, 14, 15, 16], (255, 200, 80)),   # ring    - cyan
    ([0, 17, 18, 19, 20], (255, 120, 220)),  # pinky   - magenta
]
PALM_ARC = [5, 9, 13, 17]


def draw_skeleton(img, kp2d):
    """Draw the 21-joint hand skeleton from WiLoR's projected keypoints."""
    import cv2
    pts = [(int(round(x)), int(round(y))) for x, y in kp2d]
    for a, b in zip(PALM_ARC, PALM_ARC[1:]):
        cv2.line(img, pts[a], pts[b], (190, 190, 190), 1, cv2.LINE_AA)
    for chain, color in FINGERS:
        for a, b in zip(chain, chain[1:]):
            cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for j in chain[1:]:
            cv2.circle(img, pts[j], 2, color, -1, cv2.LINE_AA)


def draw_agent(img, cam: CameraCalib, points, color, label):
    """Draw one agent's origin + two contacts, with the opening between them."""
    import cv2
    uv = [cam.project(cam.base_to_cam(np.asarray(p, dtype=np.float64))) for p in points]
    if not all(np.isfinite(u).all() for u in uv):
        return
    o, a, b = [tuple(int(round(v)) for v in u) for u in uv]

    cv2.line(img, a, b, color, 2, cv2.LINE_AA)          # what closes on the object
    cv2.line(img, o, a, color, 1, cv2.LINE_AA)
    cv2.line(img, o, b, color, 1, cv2.LINE_AA)
    for p in (a, b):
        cv2.circle(img, p, 5, color, -1, cv2.LINE_AA)
    cv2.circle(img, o, 4, (255, 255, 255), -1, cv2.LINE_AA)

    opening = float(np.linalg.norm(np.asarray(points[1]) - np.asarray(points[2])))
    cv2.putText(img, f"{label} {opening * 100:.1f}cm", (o[0] + 8, o[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--video", type=Path, required=True, help="source cam_high mp4")
    ap.add_argument("--tracks", type=Path, required=True,
                    help="npz from extract_hand_tracks.py")
    ap.add_argument("--wilor", type=Path, default=None,
                    help="npz from run_wilor.py; adds the 21-joint hand skeleton")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--brighten", type=float, default=1.0,
                    help="contrast gain; the scene is dim, 1.8 makes it readable")
    args = ap.parse_args()

    tr = np.load(args.tracks)
    wl = np.load(args.wilor) if args.wilor else None
    frame_index = tr["frame_index"]
    hand_points = tr["points"]
    n = min(len(frame_index), args.max_frames)

    cam = CameraCalib.from_json(fetch_calibration(args.repo), CAM_KEY)
    robot = read_robot_states(args.repo)
    r_pts = robot_points(robot["obs_right_ee_position"],
                         robot["obs_right_ee_euler_xyz"],
                         robot["obs_right_gripper"].reshape(-1))

    from run_wilor import iter_frames, probe_video
    width, height, _ = probe_video(args.video)
    stride = int(frame_index[1] - frame_index[0]) if len(frame_index) > 1 else 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", str(TARGET_FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(args.out)],
        stdin=subprocess.PIPE,
    )

    try:
        for i, (fidx, rgb) in enumerate(iter_frames(args.video, stride, width, height)):
            if i >= n:
                break
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
            if args.brighten != 1.0:
                img = cv2.convertScaleAbs(img, alpha=args.brighten, beta=12)

            if fidx < len(r_pts):
                draw_agent(img, cam, r_pts[fidx], ROBOT_COLOR, "robot")

            if wl is not None and wl["found"][i]:
                draw_skeleton(img, wl["kp2d"][i])
            if tr["valid"][i]:
                draw_agent(img, cam, hand_points[i], HAND_COLOR, "hand")
            else:
                cv2.putText(img, "hand: not detected", (8, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 255), 1, cv2.LINE_AA)

            cv2.putText(img, f"frame {fidx}   roles: {', '.join(POINT_ROLES)}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            writer.stdin.write(img.tobytes())
    finally:
        writer.stdin.close()
        writer.wait()

    print(f"wrote {args.out} ({n} frames)")


if __name__ == "__main__":
    main()
