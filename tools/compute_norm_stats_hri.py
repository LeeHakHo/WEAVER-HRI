#!/usr/bin/env python3
"""Normalisation statistics for the 14-D HRI state/action space.

``datasets/compute_norm_stats.py`` rebuilds states from DROID's joint-position
columns and reads ``annotations/``; this dataset carries its states and actions
on the annotation directly, under ``annotation_rewards/``.  Output format is
identical, so ``dataset.norm_stats_path`` consumes it unchanged.

    .venv/bin/python tools/compute_norm_stats_hri.py --data-root data/weaver_hri
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hri_common import STATE_NAMES  # noqa: E402

MIN_STD = 1e-3   # floor so a constant channel cannot blow up after normalisation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    anno_dir = args.data_root / "annotation_rewards" / args.split
    files = sorted(anno_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no annotations in {anno_dir}")

    states, actions = [], []
    for f in files:
        a = json.loads(f.read_text())
        states.append(np.asarray(a["states"], dtype=np.float64))
        actions.append(np.asarray(a["actions"], dtype=np.float64))
    S = np.concatenate(states, axis=0)
    A = np.concatenate(actions, axis=0)
    print(f"{len(files)} trajectories, {len(S)} frames, {S.shape[1]} dims")

    s_std = np.maximum(S.std(axis=0), MIN_STD)
    a_std = np.maximum(A.std(axis=0), MIN_STD)

    print(f"\n{'dim':<18} {'state mean':>11} {'state std':>11} "
          f"{'act mean':>11} {'act std':>11}")
    for i, name in enumerate(STATE_NAMES[:S.shape[1]]):
        print(f"{name:<18} {S[:, i].mean():>11.4f} {s_std[i]:>11.4f} "
              f"{A[:, i].mean():>11.5f} {a_std[i]:>11.5f}")

    floored = [STATE_NAMES[i] for i in range(S.shape[1])
               if S[:, i].std() < MIN_STD or A[:, i].std() < MIN_STD]
    if floored:
        print(f"\nnote: near-constant channels, std floored to {MIN_STD}: {floored}")

    out = args.out or args.data_root / "norm_stats_relabel.json"
    out.write_text(json.dumps({
        "norm_stats": {
            "state": {"mean": S.mean(axis=0).tolist(), "std": s_std.tolist()},
            "actions": {"mean": A.mean(axis=0).tolist(), "std": a_std.tolist()},
        }
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
