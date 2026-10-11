#!/usr/bin/env python3
"""Does the predicted skeleton actually follow the commanded action?

Nothing in the code adds the action to the state.  The action is an input, the
state is a separate output, and the relation ``state[t+1] = state[t] +
action[t]`` -- which holds by construction in the data -- is only ever learned,
never enforced.  So in a rollout the skeleton can quietly drift away from what
was commanded while the action channel is ignored.

This replays the recorded actions and compares three trajectories:

    predicted    what out_prj['states'] produces, fed back each chunk
    commanded    the action itself, which is already an absolute target pose
    recorded     the ground truth

The real arm lags its own command by about 1.2 cm, so ``commanded`` is not a
trajectory anyone reaches exactly.  What matters is whether the model lags it
by a similar amount or drifts away entirely.

    .venv/bin/python tools/state_vs_action.py \
        --checkpoint runs/weaver_hri/logs/chkpts_hri \
        --checkpoint-name checkpoint40000.pt \
        --overrides dataset.path=data/weaver_hri
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weaver.finetune_utils import load_finetuned          # noqa: E402
from weaver.generate_views import build_model, load_eval_config  # noqa: E402
from weaver.interactive import Normaliser, load_seed_clip  # noqa: E402
from weaver.wm.streaming import StreamingRollout           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hri_common import HAND_SLICE, ROBOT_SLICE             # noqa: E402

AGENTS = {"robot": ROBOT_SLICE, "hand": HAND_SLICE}


def recorded(cfg, img_keys, traj_index, n_seed, n_steps):
    """Recorded actions and states for the window after the seed."""
    from weaver.datasets.droid import PrecomputedDroid
    enc = "sd3" if "stable-diffusion-3" in cfg.im_encoder.name else "svd"
    ds = PrecomputedDroid(
        root=cfg.dataset.path, split="val", horizon=cfg.horizon, img_keys=img_keys,
        relabel_actions=cfg.dataset.relabel_actions, normalize=cfg.dataset.normalize,
        norm_stats_path=getattr(cfg.dataset, "norm_stats_path", None),
        cache_trajectories=False, return_language=True, return_video_frames=False,
        encoder_type=enc, n_memory_frames=cfg.n_memory_frames, t_memory=cfg.t_memory,
        n_history=cfg.n_history, eval_mode=True, annotation_dir=cfg.dataset.annotation_dir,
    )
    ti = ds.valid_trajectories[traj_index % len(ds.valid_trajectories)]
    acts, sts = ds._actions[ti], ds._states[ti]
    nd = ds.norm_dict if ds.normalize else None
    if nd:   # _states/_actions are stored normalised; bring them back to metres
        sts = sts * nd["states"]["std"] + nd["states"]["mean"]
        acts = acts * nd["actions"]["std"] + nd["actions"]["mean"]
    lo = max(0, n_seed - 1)
    a = acts[lo:lo + n_steps].clone()
    s = sts[n_seed:n_seed + n_steps].clone()
    return a.numpy(), s.numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--checkpoint-name", default="checkpoint.pt")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--traj-index", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--val-steps", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_eval_config(args.checkpoint,
                           list(args.overrides) + [f"model.val_steps={args.val_steps}"])
    model, img_keys = build_model(cfg, args.device)
    load_finetuned(model, args.checkpoint, cfg,
                   checkpoint_name=args.checkpoint_name, device=args.device)
    model.eval()
    model._use_precomputed_features = True

    norm = Normaliser(cfg.dataset.norm_stats_path, args.device, cfg.dataset.path)
    n_seed = max(cfg.n_history, cfg.n_memory_frames * cfg.t_memory + 1)
    obs, seed_actions, traj_id = load_seed_clip(cfg, img_keys, args.traj_index,
                                                n_seed, args.device)
    phys_a, rec_s = recorded(cfg, img_keys, args.traj_index, n_seed, args.n_steps)
    print(f"val trajectory {traj_id}, {args.n_steps} imagined frames\n")

    roll = StreamingRollout(model, horizon=cfg.horizon)
    roll.seed(obs, seed_actions)
    s0 = norm.denorm_state(roll.latest_states(1))[0, 0].cpu().numpy()

    a = norm.norm_action(torch.tensor(phys_a, dtype=torch.float32,
                                      device=args.device)).unsqueeze(0)
    pred = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(0, args.n_steps, roll.bootstrap):
            ch = a[:, i:i + roll.bootstrap]
            if ch.shape[1] < roll.bootstrap:
                ch = torch.nn.functional.pad(ch, (0, 0, 0, roll.bootstrap - ch.shape[1]))
            roll.step(ch)
            pred.append(norm.denorm_state(roll.latest_states(roll.bootstrap))[0].cpu().numpy())
    pred = np.concatenate(pred)[:args.n_steps]
    # The action is the commanded pose, so the trajectory the commands ask for
    # IS the action sequence -- no integration needed.
    integ = phys_a[:args.n_steps]
    T = min(len(pred), len(integ), len(rec_s))
    pred, integ, rec = pred[:T], integ[:T], rec_s[:T]

    print(f"{'frames':<10} {'agent':<6} {'pred vs commanded':>19} {'pred vs recorded':>18} "
          f"{'cmd vs recorded':>18}")
    print("-" * 76)
    for lo, hi in ((0, 8), (8, 16), (16, 24), (24, 32), (32, T)):
        if lo >= T:
            break
        for ag, sl in AGENTS.items():
            def e(x, y):
                d = (x[lo:hi, sl] - y[lo:hi, sl]).reshape(hi - lo, 3, 3)
                return np.linalg.norm(d, axis=2).mean() * 100
            print(f"{f'{lo}-{min(hi,T)}':<10} {ag:<6} {e(pred,integ):>18.2f}cm "
                  f"{e(pred,rec):>17.2f}cm {e(integ,rec):>17.2f}cm")
    print("\ncm, mean over the three points of each agent.")
    print("'cmd vs recorded' is the floor -- the real arm also lags its command by")
    print("about 1.2 cm -- so whatever 'pred vs commanded' adds on top is the model")
    print("tracking the command worse than the real robot does.")


if __name__ == "__main__":
    main()
