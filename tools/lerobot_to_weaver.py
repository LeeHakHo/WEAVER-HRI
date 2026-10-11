#!/usr/bin/env python3
"""Phase 2: LeRobot HRI-insertion episode -> WEAVER preprocessed dataset.

Produces exactly the layout ``datasets/README.md`` specifies, so the stock
dataloader (``weaver/datasets/droid.py``) can read it:

    <out_root>/
      annotation_rewards/{train,val}/<id>.json
      videos/{train,val}/<id>.mp4           192x960, cam_high|cam_low|cam_wrist, 5 fps
      latents/{train,val}/<id>_sd3.npz      'latents' (3, T, 60, 256) float16

Each 30 fps episode is subsampled by ``--stride`` (6 -> 5 fps, the rate WEAVER
was pretrained at) and cut into fixed-length segments, each becoming its own
trajectory.  One ~3000-frame episode makes a poor training set of size one; the
segments give the sampler something to shuffle and keep random access cheap.

States and actions are written into the annotation directly, in the unified
18-D layout from ``hri_common``: three base-frame points per agent, so the
gripper and the hand are described identically.  Actions are state deltas
between consecutive *video* frames, matching the semantics of WEAVER's
``relabel_actions`` path.

Frames are resized (not cropped) from 640x480 to 320x192, the same squash
``datasets/preprocess_droid_ood.py`` applies, so the whole workspace stays in
view.

Example
-------
    .venv/bin/python tools/lerobot_to_weaver.py \
        --repo LeeHakHo/play_hri_insertion_2_eef \
        --hand data/hand_tracks/ds2.npz \
        --out-root data/weaver_hri --split train --episode-base 2000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hri_common import (  # noqa: E402
    ACTION_DIM, ACTION_NAMES, CAM_ORDER, CAMERA_CROP, GRIPPER_CMD_FLOOR,
    STATE_DIM, STATE_NAMES, TARGET_FPS, VIDEO_PATH_TMPL,
    read_robot_states, robot_points,
)

CAM_H, CAM_W = 192, 320
LATENT_SPATIAL_SIZE = 4      # saved at 60x256; WEAVER._reshape_features re-pools to the model's


def download_video(repo: str, cam_key: str, dest: Path) -> Path:
    """Fetch one camera's episode video, unless it is already on disk.

    Downloads straight to ``dest`` rather than through ``hf_hub_download``: the
    hub cache would keep a second ~400 MB copy of every episode video, and these
    files are read once and deleted.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = (f"https://huggingface.co/datasets/{repo}/resolve/main/"
           + VIDEO_PATH_TMPL.format(cam=cam_key))
    tmp = dest.with_suffix(".part")
    subprocess.run(["curl", "-fsSL", "-o", str(tmp), url], check=True)
    tmp.replace(dest)
    return dest


def decode_resized(path: Path, stride: int, n_expect: int | None = None,
                   max_frames: int | None = None,
                   crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Decode every ``stride``-th frame, cropped then resized to (CAM_H, CAM_W).

    Cropping first means the resize no longer has to squash 4:3 into 5:3, and
    the pixels it does keep land on the part of the table where something
    happens. -> (T, H, W, 3) uint8.
    """
    vf = rf"select=not(mod(n\,{stride}))"
    if crop:
        x0, y0, x1, y1 = crop
        vf += f",crop={x1 - x0}:{y1 - y0}:{x0}:{y0}"
    vf += f",scale={CAM_W}:{CAM_H}:flags=bilinear"
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", vf, "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24",
    ]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(out) // (CAM_H * CAM_W * 3)
    frames = np.frombuffer(out[: n * CAM_H * CAM_W * 3], np.uint8)
    frames = frames.reshape(n, CAM_H, CAM_W, 3)
    if n_expect is not None and n != n_expect:
        print(f"    note: {path.name} decoded {n} frames, expected {n_expect}")
    return frames


def write_panel_video(panels: list[np.ndarray], dest: Path, fps: float) -> int:
    """Concatenate camera panels horizontally and encode to mp4. Returns frame count."""
    T = min(len(p) for p in panels)
    stacked = np.concatenate([p[:T] for p in panels], axis=2)   # (T, H, 3W, 3)
    dest.parent.mkdir(parents=True, exist_ok=True)
    h, w = stacked.shape[1:3]
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(dest)],
        stdin=subprocess.PIPE,
    )
    proc.communicate(stacked.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {dest}")
    return T


@torch.no_grad()
def encode_latents(encoder, frames: np.ndarray, batch: int, device: str) -> np.ndarray:
    """(T,H,W,3) uint8 -> (T, 60, 256) float16 SD3 latents."""
    out = []
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch].astype(np.float32) / 255.0
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device)
        out.append(encoder(x).cpu().numpy().astype(np.float16))
    return np.concatenate(out, axis=0)


def build_states(robot: dict, hand: dict,
                 frame_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the 18-D state track and the per-frame hand-validity mask.

    Both agents become the same three points in the base frame — origin,
    contact A, contact B — so the two 9-D blocks are structurally identical.
    The robot's three points are derived from the recorded EEF pose and gripper
    value by ``robot_points``; the hand's come straight from the tracker.
    """
    n_raw = len(robot["obs_right_ee_position"])
    f = np.clip(frame_index, 0, n_raw - 1)

    rp = robot_points(robot["obs_right_ee_position"][f],
                      robot["obs_right_ee_euler_xyz"][f],
                      robot["obs_right_gripper"][f].reshape(-1))

    states = np.zeros((len(f), STATE_DIM), np.float64)
    states[:, 0:9] = rp.reshape(len(f), 9)
    states[:, 9:18] = hand["points"].reshape(len(f), 9)
    return states, hand["valid"].astype(bool)


def build_actions(robot: dict, states: np.ndarray,
                  frame_index: np.ndarray) -> np.ndarray:
    """The commanded pose of both agents' three points, (T, 18) in metres.

    Not a difference of observed states. A delta of observations says how far
    the arm *went*, which is feedback; a policy or a teleoperator emits where
    the arm should *go*. Only the second can drive the model, and only the
    second carries contact: the jaws are commanded shut while the object holds
    them at ~2.6 cm, and that gap is the grasp.

    The robot's command is the recorded ``action_right_ee_*`` pose, run through
    the same ``robot_points`` as the observation so the two are comparable. The
    gripper command is clipped at zero (see GRIPPER_CMD_FLOOR).

    A human arm has no servo loop between intent and motion, so the hand's
    command is simply where it is on the next frame.
    """
    n_raw = len(robot["action_right_ee_position"])
    f = np.clip(frame_index, 0, n_raw - 1)
    cmd_grip = np.maximum(robot["action_right_gripper"][f].reshape(-1),
                          GRIPPER_CMD_FLOOR)
    rp = robot_points(robot["action_right_ee_position"][f],
                      robot["action_right_ee_euler_xyz"][f], cmd_grip)

    actions = np.zeros((len(f), ACTION_DIM), np.float64)
    actions[:, 0:9] = rp.reshape(len(f), 9)
    actions[:, 9:18] = np.concatenate([states[1:, 9:18], states[-1:, 9:18]], axis=0)
    return actions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--hand", type=Path, required=True, help="npz from extract_hand_tracks.py")
    ap.add_argument("--objects", type=Path, default=None,
                    help="npz from extract_object_tracks.py; adds the cube and "
                         "bowl positions as an auxiliary prediction target")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--episode-base", type=int, required=True,
                    help="first trajectory id for this episode's segments")
    ap.add_argument("--video-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--segment-len", type=int, default=200)
    ap.add_argument("--min-segment-len", type=int, default=40)
    ap.add_argument("--task", default="insert cube")
    ap.add_argument("--encode-batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep-videos", action="store_true",
                    help="keep the downloaded source videos (default: delete cam_low/wrist)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="process only the first N video-rate frames (smoke tests)")
    ap.add_argument("--annotations-only", action="store_true",
                    help="rewrite the state/action tracks of an existing conversion, "
                         "reusing its videos and SD3 latents. Each annotation's "
                         "source_frame_index pins the alignment, so the state "
                         "representation can change without re-encoding anything.")
    ap.add_argument("--cameras", nargs="*", default=["exterior_1_left"],
                    help="which WEAVER camera slots to build. Training uses the "
                         "top-down view alone, so the default skips downloading "
                         "and encoding the other two.")
    ap.add_argument("--tag", default=None,
                    help="short name used for the cached video files "
                         "(default: the episode suffix of --repo, e.g. '2' or 'eval')")
    args = ap.parse_args()

    # "LeeHakHo/play_hri_insertion_3_eef" -> "3"; "..._eval_eef" -> "eval"
    tag = args.tag or args.repo.rsplit("_", 2)[-2]
    cams = [(k, slot) for k, slot in CAM_ORDER if slot in args.cameras]
    if not cams:
        raise SystemExit(f"--cameras {args.cameras} matches none of "
                         f"{[s for _, s in CAM_ORDER]}")
    hand = dict(np.load(args.hand))
    frame_index = hand["frame_index"]
    print(f"{args.repo}: {len(frame_index)} sampled frames, "
          f"hand valid {hand['valid'].mean():.1%}")

    # --- states / actions ----------------------------------------------------
    robot = read_robot_states(args.repo)
    grip = robot["obs_right_gripper"].reshape(-1)
    print(f"  robot gripper over the episode: min {grip.min():.3f}  max {grip.max():.3f}  "
          f"mean {grip.mean():.3f}")

    states, hand_valid = build_states(robot, hand, frame_index)
    actions = build_actions(robot, states, frame_index)

    # Object positions never enter the state or the action: the model predicts
    # them from a head of its own and never reads them back, so it has to
    # recover them from the image rather than copying the previous value.
    objects = obj_valid = None
    if args.objects is not None:
        o = np.load(args.objects)
        n = min(len(states), len(o["cube"]))
        objects = np.concatenate([o["cube"][:n], o["bowl"][:n]], axis=1)
        obj_valid = np.stack([o["cube_valid"][:n], o["bowl_valid"][:n]], axis=1)
        if n < len(states):
            pad = len(states) - n
            objects = np.concatenate([objects, np.repeat(objects[-1:], pad, 0)])
            obj_valid = np.concatenate([obj_valid, np.zeros((pad, 2), bool)])
        print(f"  objects: cube valid {obj_valid[:, 0].mean():.1%}, "
              f"bowl valid {obj_valid[:, 1].mean():.1%}")

    out_root = args.out_root
    anno_dir = out_root / "annotation_rewards" / args.split
    vid_dir = out_root / "videos" / args.split
    lat_dir = out_root / "latents" / args.split

    if args.annotations_only:
        panels, cam_paths, latents = None, [], None
        existing = sorted(anno_dir.glob("*.json"),
                          key=lambda q: int(q.stem))
        existing = [q for q in existing
                    if args.episode_base <= int(q.stem) < args.episode_base + 1000]
        if not existing:
            raise SystemExit(f"--annotations-only: nothing to rewrite under {anno_dir} "
                             f"for episode base {args.episode_base}")
        old = json.loads(existing[0].read_text())
        text_features = old["text_features"]
        T = sum(json.loads(q.read_text())["video_length"] for q in existing)
        print(f"  reusing {len(existing)} existing segments ({T} frames) "
              f"with their videos and latents")
    else:
        # --- video panels ----------------------------------------------------
        panels, cam_paths = [], []
        for cam_key, slot in cams:
            dest = args.video_dir / f"ds{tag}_{cam_key.rsplit('.', 1)[-1]}.mp4"
            download_video(args.repo, cam_key, dest)
            cam_paths.append((dest, slot))
            panels.append(decode_resized(dest, args.stride, len(frame_index),
                                         args.max_frames, CAMERA_CROP.get(cam_key)))
            crop = CAMERA_CROP.get(cam_key)
            print(f"  decoded {slot:16s} {panels[-1].shape}"
                  f"{'  crop ' + str(crop) if crop else ''}")

        T = min(len(p) for p in panels + [states])
        if args.max_frames:
            T = min(T, args.max_frames)
        panels = [p[:T] for p in panels]

        # --- encoders --------------------------------------------------------
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from weaver.wm.encoders import SD3Encoder, get_task_encoder
        print("  loading SD3 VAE + CLIP ...")
        img_encoder = SD3Encoder(
            model_name="stabilityai/stable-diffusion-3-medium-diffusers",
            image_size=(CAM_H, CAM_W), spatial_size=LATENT_SPATIAL_SIZE,
            device=args.device,
        ).eval()
        txt_encoder = get_task_encoder(config={}, device=args.device).to(args.device).eval()
        with torch.no_grad():
            text_features = txt_encoder([args.task]).cpu().numpy()[0].tolist()

        latents = np.stack(
            [encode_latents(img_encoder, p, args.encode_batch, args.device) for p in panels],
            axis=0,
        )   # (3, T, 60, 256)
        print(f"  latents {latents.shape} {latents.dtype}")

    states, actions, hand_valid = states[:T], actions[:T], hand_valid[:T]

    # --- segment and write ---------------------------------------------------
    for d in (anno_dir, vid_dir, lat_dir):
        d.mkdir(parents=True, exist_ok=True)

    written = []
    for seg_i, start in enumerate(range(0, T, args.segment_len)):
        end = min(start + args.segment_len, T)
        if end - start < args.min_segment_len:
            break
        tid = args.episode_base + seg_i
        if args.annotations_only:
            n = json.loads((anno_dir / f"{tid}.json").read_text())["video_length"]
        else:
            n = write_panel_video([p[start:end] for p in panels],
                                  vid_dir / f"{tid}.mp4", TARGET_FPS)
            np.savez_compressed(lat_dir / f"{tid}_sd3.npz", latents=latents[:, start:end])

        seg_states = states[start:end]
        seg_actions = actions[start:end].copy()
        # The last frame of a segment has no successor. Zero is not "no command"
        # any more -- it would order both agents to the base-frame origin -- so
        # repeat the previous command, which reads as "hold".
        if len(seg_actions) > 1:
            seg_actions[-1] = seg_actions[-2]

        annotation = {
            "texts": [args.task],
            "text_features": text_features,
            "episode_id": tid,
            "episode_id_orig": f"{args.repo}#seg{seg_i}",
            "success": 0,
            "video_length": n,
            "state_length": int(end - start),
            "raw_length": int(end - start),
            "num_cameras": len(cams),
            "video_path": f"videos/{args.split}/{tid}.mp4",
            "latent_path": f"latents/{args.split}/{tid}.npz",
            "sd3_latent_path": f"latents/{args.split}/{tid}_sd3.npz",
            # Unified 18-D track, consumed directly by the dataloader.
            "state_names": STATE_NAMES,
            "action_names": ACTION_NAMES,
            "states": seg_states.astype(np.float32).tolist(),
            "actions": seg_actions.astype(np.float32).tolist(),
            "hand_valid": hand_valid[start:end].astype(int).tolist(),
            **({"objects": objects[start:end].astype(np.float32).tolist(),
                "object_valid": obj_valid[start:end].astype(int).tolist()}
               if objects is not None else {}),
            # No success labels exist for this play data and the reward/critic
            # heads are disabled during finetuning, so this is a constant.
            "reward_progress": [0.0] * (end - start),
            "source_frame_index": frame_index[start:end].astype(int).tolist(),
        }
        with (anno_dir / f"{tid}.json").open("w") as fh:
            json.dump(annotation, fh)
        written.append(tid)

    print(f"  wrote {len(written)} segments: ids {written[0]}..{written[-1]}")

    if not args.keep_videos:
        for dest, slot in cam_paths:
            if "cam_high" not in dest.name:     # cam_high is reused by the viz tool
                dest.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
