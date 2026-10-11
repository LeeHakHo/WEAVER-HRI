#!/usr/bin/env python3
"""Does the world model actually listen to each agent separately?

FID/FVD on replayed ground-truth actions cannot answer that: a model that
ignored the hand channels entirely could still score well by predicting the
most likely continuation.  So this replays the *same* validation clip four
times from the *same* seed, changing only which half of the 18-D action vector
is live:

    full    both agents follow the recorded commands
    robot   robot follows the recording, the hand is commanded to hold still
    hand    the hand follows the recording, the robot is commanded to hold still
    frozen  both commanded to hold still

The action is the commanded *pose* of the three points, so holding still means
commanding the pose the agent currently occupies -- read back from the rollout
each chunk. A zero action would instead order the agent to the base-frame
origin.

The output is a labelled grid video, ground truth on top.  If the hand moves in
"robot" or the arm moves in "hand", the conditioning did not separate.

    .venv/bin/python tools/counterfactual_rollout.py \
        --checkpoint runs/weaver_hri/logs/chkpts_hri \
        --overrides dataset.path=data/weaver_hri
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weaver.finetune_utils import load_finetuned  # noqa: E402
from weaver.generate_views import build_model, load_eval_config  # noqa: E402
from weaver.interactive import Normaliser, load_seed_clip  # noqa: E402
from weaver.wm.streaming import StreamingRollout  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hri_common import HAND_SLICE, ROBOT_SLICE  # noqa: E402

# Each variant sets a mode per agent: the recorded action, a true physical
# zero ("hold still"), or the recorded action permuted in time.
#
# A shuffled channel is the noise floor for that channel: identical empirical
# distribution and per-frame magnitudes, including the hand's heavy tails, but
# no correspondence with the video. It must be applied *per agent* — shuffling
# both at once mostly measures what a scrambled robot arm does to the image,
# which says nothing about the hand.
LIVE, HOLD, SHUF = "live", "hold", "shuffle"
VARIANTS = {
    "full": (LIVE, LIVE),
    "robot": (LIVE, HOLD),
    "hand": (HOLD, LIVE),
    "frozen": (HOLD, HOLD),
    "robot shuf": (SHUF, HOLD),
    "hand shuf": (HOLD, SHUF),
}


def gt_actions(cfg, img_keys, traj_index: int, n_seed: int, n_steps: int, device: str):
    """Recorded actions following the seed window, normalised, (1, n_steps, 14)."""
    from weaver.datasets.droid import PrecomputedDroid
    encoder_type = "sd3" if "stable-diffusion-3" in cfg.im_encoder.name else "svd"
    ds = PrecomputedDroid(
        root=cfg.dataset.path, split="val", horizon=cfg.horizon, img_keys=img_keys,
        relabel_actions=cfg.dataset.relabel_actions, normalize=cfg.dataset.normalize,
        norm_stats_path=getattr(cfg.dataset, "norm_stats_path", None),
        cache_trajectories=False, return_language=True, return_video_frames=True,
        encoder_type=encoder_type, n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory, n_history=cfg.n_history, eval_mode=True,
        annotation_dir=cfg.dataset.annotation_dir,
    )
    ti = ds.valid_trajectories[traj_index % len(ds.valid_trajectories)]
    traj = ds.trajectories[ti]
    acts = ds._actions[ti]
    nd = ds.norm_dict if ds.normalize else None

    # Action at frame t produces frame t+1, so the actions driving frames
    # n_seed .. n_seed+n_steps-1 start at index n_seed-1.
    lo = max(0, n_seed - 1)
    hi = min(len(acts), lo + n_steps)
    a = acts[lo:hi].clone()
    if len(a) < n_steps:
        a = torch.cat([a, a[-1:].repeat(n_steps - len(a), 1)], dim=0)
    if nd:
        a = (a - nd["actions"]["mean"]) / nd["actions"]["std"]

    frames = traj.get_video_frames(n_seed, n_seed + n_steps)
    return a.unsqueeze(0).to(device), frames


def label(img: np.ndarray, text: str) -> np.ndarray:
    import cv2
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--checkpoint-name", default="checkpoint.pt")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--traj-index", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=40, help="frames to imagine (5 fps)")
    ap.add_argument("--val-steps", type=int, default=16)
    ap.add_argument("--camera", default=None, help="camera to render (default: first)")
    ap.add_argument("--out", type=Path, default=Path("data/viz/counterfactual.mp4"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_eval_config(args.checkpoint, list(args.overrides) +
                           [f"model.val_steps={args.val_steps}"])
    model, img_keys = build_model(cfg, args.device)
    load_finetuned(model, args.checkpoint, cfg,
                   checkpoint_name=args.checkpoint_name, device=args.device)
    model.eval()
    model._use_precomputed_features = True

    norm = Normaliser(cfg.dataset.norm_stats_path, args.device, cfg.dataset.path)
    n_seed = max(cfg.n_history, cfg.n_memory_frames * cfg.t_memory + 1)
    obs, seed_actions, traj_id = load_seed_clip(cfg, img_keys, args.traj_index,
                                                n_seed, args.device)
    actions, gt_frames = gt_actions(cfg, img_keys, args.traj_index, n_seed,
                                    args.n_steps, args.device)
    cam = args.camera or img_keys[0]
    print(f"val trajectory {traj_id}, {n_seed} seed frames, "
          f"{args.n_steps} imagined frames, camera {cam}")

    # The action is a commanded pose now, not a delta, so a zero action is not
    # "hold still" -- it orders both agents to the base-frame origin. Holding
    # still means commanding the pose the agent is already in, which is only
    # known once the rollout has produced it, so the held channels have to be
    # filled in chunk by chunk rather than up front.
    phys = actions * norm.a_std + norm.a_mean

    renders = {}
    perm = torch.randperm(phys.shape[1], generator=torch.Generator().manual_seed(0))
    shuffled = phys[:, perm]
    for name, (robot_mode, hand_mode) in VARIANTS.items():
        roll = StreamingRollout(model, horizon=cfg.horizon)
        roll.seed(obs, seed_actions)
        out = []
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for i in range(0, args.n_steps, roll.bootstrap):
                chunk = phys[:, i:i + roll.bootstrap].clone()
                shuf = shuffled[:, i:i + roll.bootstrap]
                if chunk.shape[1] < roll.bootstrap:
                    pad = roll.bootstrap - chunk.shape[1]
                    chunk = torch.nn.functional.pad(chunk, (0, 0, 0, pad))
                    shuf = torch.nn.functional.pad(shuf, (0, 0, 0, pad))
                here = norm.denorm_state(roll.latest_states(1))[:, 0]   # (B, 18)
                for mode, sl in ((robot_mode, ROBOT_SLICE), (hand_mode, HAND_SLICE)):
                    if mode == HOLD:
                        chunk[..., sl] = here[:, None, sl]
                    elif mode == SHUF:
                        chunk[..., sl] = shuf[..., sl]
                out.append(roll.step(norm.norm_action(chunk))[cam][0].float().cpu())
        renders[name] = torch.cat(out, dim=0)[:args.n_steps]
        print(f"  {name:10s} done")

    T = min(len(v) for v in renders.values())
    gt = gt_frames[cam][:T] if cam in gt_frames else None

    def mad(x: torch.Tensor, y: torch.Tensor) -> float:
        """Mean absolute pixel difference, 0-255."""
        return float((x[:T] - y[:T]).abs().mean() * 255)

    gt_t = (torch.from_numpy(gt).permute(0, 3, 1, 2).float() / 255.0
            if gt is not None else None)
    print(f"\n{'variant':<11} {'vs frozen':>10} {'vs GT':>9}")
    print("-" * 32)
    for name in VARIANTS:
        vf = mad(renders[name], renders["frozen"])
        vg = f"{mad(renders[name], gt_t):9.2f}" if gt_t is not None else "        -"
        print(f"{name:<11} {vf:>10.2f} {vg}")

    # The judgement. Magnitude ("does this channel move the image more than
    # noise would") and direction ("does it move the image toward the truth")
    # are different questions, and only the second one is the goal: the hand is
    # a small object, so a correct hand prediction need not move many pixels.
    if gt_t is not None:
        print(f"\n{'channel':<8} {'live err':>9} {'shuf err':>9} {'gain':>7} "
              f"{'move':>7} {'floor':>7} {'x':>6}")
        print("-" * 58)
        for agent, live, shuf in (("robot", "robot", "robot shuf"),
                                  ("hand", "hand", "hand shuf")):
            le, se = mad(renders[live], gt_t), mad(renders[shuf], gt_t)
            lm = mad(renders[live], renders["frozen"])
            sm = mad(renders[shuf], renders["frozen"])
            print(f"{agent:<8} {le:>9.2f} {se:>9.2f} {se - le:>+7.2f} "
                  f"{lm:>7.2f} {sm:>7.2f} {lm / sm:>5.2f}x")
        print("\ngain > 0: the real action beats its own shuffled floor at "
              "predicting the video -> that channel is being read.")

    rows = []
    if gt is not None:
        rows.append(np.stack([label(f, "ground truth") for f in gt[:T]]))
    for name in VARIANTS:
        arr = (renders[name][:T].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        rows.append(np.stack([label(f, name) for f in arr]))

    pairs = [
        np.concatenate(rows[i:i + 2], axis=2) if i + 1 < len(rows)
        else np.concatenate([rows[i], np.zeros_like(rows[i])], axis=2)
        for i in range(0, len(rows), 2)
    ]
    grid = np.concatenate(pairs, axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import imageio
    imageio.mimwrite(args.out, grid, fps=5)
    print(f"wrote {args.out}  ({T} frames, {grid.shape[2]}x{grid.shape[1]})")


if __name__ == "__main__":
    main()
