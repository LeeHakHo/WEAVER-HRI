#!/usr/bin/env python3
"""Does the imagined video actually follow a *commanded* action?

``counterfactual_rollout.py`` answers this for actions replayed from the
recording. The interactive demo asks something harder: a constant command held
for many frames, which the training set never contains — it moves an origin by a
median of 0.3-0.4 cm per frame and never sustains more than ~1 cm/frame for long.

So this drives the same seed with a constant command in each direction and
measures two things per variant:

  state   how far the model's own predicted state travelled, against what was
          commanded — the conditioning reaching the state read-out
  pixels  mean absolute difference from the do-nothing rollout — the
          conditioning reaching the image

A variant that moves the state but not the pixels means the action is being
registered and then ignored by the generator.

    .venv/bin/python tools/test_action_following.py \
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
from weaver.interactive import (  # noqa: E402
    AGENTS, STEP_OPEN, STEP_POS, Normaliser, accumulate, apply_controls,
    load_rest_offset, load_seed_clip,
)
from weaver.wm.streaming import StreamingRollout  # noqa: E402

VARIANTS = {
    "still": [],
    "hand +y": ["ArrowLeft"],
    "hand -y": ["ArrowRight"],
    "hand +x": ["ArrowUp"],
    "hand open": ["["],
    "robot +x": ["w"],
    "robot +y": ["a"],
    "robot close": ["2"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--traj-index", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=5, help="8-frame chunks per variant")
    ap.add_argument("--val-steps", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("data/viz/action_following.mp4"))
    ap.add_argument("--no-rest-offset", dest="rest_offset", action="store_false",
                    help="command the observed pose exactly, as this used to. "
                         "The rig holds a standing commanded-minus-observed "
                         "bias even at rest, so a zero gap is not 'stay put'")
    ap.add_argument("--step-scale", type=float, default=1.0,
                    help="scale the per-frame command. The default 0.8 cm held "
                         "for 48 frames asks for 38.4 cm, past the 99th "
                         "percentile of what the arm covers in that window "
                         "during training (29.5 cm); 0.4 lands near the median")
    args = ap.parse_args()

    if args.step_scale != 1.0:
        import weaver.interactive as _inter
        for _n in ("STEP_POS", "STEP_OPEN", "STEP_TURN"):
            setattr(_inter, _n, getattr(_inter, _n) * args.step_scale)

    cfg = load_eval_config(args.checkpoint, list(args.overrides) +
                           [f"model.val_steps={args.val_steps}",
                            "inference.pyramid_stagger_width=0"])
    model, img_keys = build_model(cfg, args.device)
    load_finetuned(model, args.checkpoint, cfg, device=args.device)
    model.eval()
    model._use_precomputed_features = True

    norm = Normaliser(cfg.dataset.norm_stats_path, args.device, cfg.dataset.path)
    rest = load_rest_offset(cfg.dataset.path, "cpu").numpy() if args.rest_offset else np.zeros(18)
    n_seed = max(cfg.n_history, cfg.n_memory_frames * cfg.t_memory + 1)
    obs, seed_actions, traj_id = load_seed_clip(cfg, img_keys, args.traj_index,
                                                n_seed, args.device)
    cam = "exterior_1_left" if "exterior_1_left" in img_keys else img_keys[0]
    print(f"seed: val trajectory {traj_id}, {n_seed} frames, camera {cam}")
    import weaver.interactive as _inter
    print(f"commanding {_inter.STEP_POS * 100:.2f} cm/frame "
          f"(open {_inter.STEP_OPEN * 100:.2f} cm) "
          f"for {args.chunks} chunks = {args.chunks * 8} frames\n")

    renders, moved = {}, {}
    for name, keys in VARIANTS.items():
        roll = StreamingRollout(model, horizon=cfg.horizon)
        roll.seed(obs, seed_actions)
        start = norm.denorm_state(roll.latest_states(1))[0, 0].cpu().numpy()
        frames, commanded = [], np.zeros(len(start))
        # The command is an absolute pose, so the target is ours to hold: start
        # it at the seed and move it by the key alone. Re-reading the model's
        # own state each chunk, as this used to, makes "no key" mean "go to
        # wherever you just drifted" -- a random walk with nothing pulling it
        # back, which showed up as 14 cm of travel under an empty command.
        target = start.copy()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for _ in range(args.chunks):
                cmd = accumulate(keys)
                # One command per frame, ramped across the chunk: a repeated
                # vector would ask for a single step and then a hold, not the
                # continuous motion a held key should mean.
                step = np.zeros(len(target))
                for agent, sl in AGENTS.items():
                    pts = target[sl].reshape(3, 3)
                    tgt = apply_controls(pts, cmd[agent]["move"], cmd[agent]["open"],
                                         cmd[agent]["turn"])
                    step[sl] = (tgt - pts).reshape(9)
                ramp = np.arange(1, roll.bootstrap + 1)[:, None] * step[None, :]
                chunk_cmd = target[None, :] + ramp           # (bootstrap, 18)
                target = chunk_cmd[-1].copy()
                chunk_cmd = chunk_cmd + rest[None, :]
                commanded += step * roll.bootstrap
                a = torch.tensor(chunk_cmd, dtype=torch.float32, device=args.device)
                a = norm.norm_action(a).view(1, roll.bootstrap, -1)
                frames.append(roll.step(a)[cam][0].float().cpu())

        end = norm.denorm_state(roll.latest_states(1))[0, 0].cpu().numpy()
        renders[name] = torch.cat(frames, dim=0)
        moved[name] = (start, end, commanded)

    # Split the travel into the part along the command and the part across it.
    # The arm wanders ~6 cm under an empty command, so a bare |end - start|
    # mixes obedience with drift, flattering or penalising a variant depending
    # on which way the drift happened to point.
    print(f"{'variant':<12} {'agent':<6} {'commanded':>10} {'along cmd':>10} "
          f"{'ratio':>7} {'across':>8} {'pixels vs still':>16}")
    print("-" * 80)
    still = renders["still"]
    for name in VARIANTS:
        start, end, commanded = moved[name]
        px = float((renders[name] - still).abs().mean() * 255)
        for i, (agent, sl) in enumerate(AGENTS.items()):
            cvec = commanded[sl][:3]
            mvec = (end - start)[sl][:3]
            c = np.linalg.norm(cvec) * 100
            if c > 1e-6:
                u = cvec / np.linalg.norm(cvec)
                along = float(mvec @ u) * 100
                across = float(np.linalg.norm(mvec - (mvec @ u) * u)) * 100
                ratio = f"{along / c:6.2f}"
            else:
                along = np.linalg.norm(mvec) * 100
                across, ratio = 0.0, "     -"
            print(f"{name if i == 0 else '':<12} {agent:<6} {c:>9.1f}cm "
                  f"{along:>9.1f}cm {ratio:>7} {across:>7.1f}cm "
                  f"{px if i == 0 else '':>16}")

    T = min(len(v) for v in renders.values())
    import cv2
    tiles = []
    for name in VARIANTS:
        arr = (renders[name][:T].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        arr = np.stack([cv2.putText(f.copy(), name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45, (255, 255, 255), 1, cv2.LINE_AA) for f in arr])
        tiles.append(arr)
    rows = [np.concatenate(tiles[i:i + 4], axis=2) for i in range(0, len(tiles), 4)]
    grid = np.concatenate(rows, axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import imageio
    imageio.mimwrite(args.out, grid, fps=5)
    print(f"\nwrote {args.out}  ({T} frames, {grid.shape[2]}x{grid.shape[1]})")


if __name__ == "__main__":
    main()
