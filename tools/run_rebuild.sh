#!/usr/bin/env bash
# Rebuild the hand tracks end to end after a change to the detection stage:
# WiLoR (with box propagation) -> base-frame points -> annotations -> norm stats.
# Videos and SD3 latents are untouched, so Phase 2 only rewrites annotations.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

for s in "$@"; do
    echo "=============================================================="
    echo "[$s] WiLoR"
    echo "=============================================================="
    "$ROOT/.venv-wilor/bin/python" tools/run_wilor.py \
        --video "data/raw/ds${s}_cam_high.mp4" \
        --out "data/hand_raw/ds${s}.npz" --stride 6
done

for s in "$@"; do
    echo "=============================================================="
    echo "[$s] base-frame points"
    echo "=============================================================="
    "$ROOT/.venv/bin/python" tools/extract_hand_tracks.py \
        --repo "LeeHakHo/play_hri_insertion_${s}_eef" \
        --wilor "data/hand_raw/ds${s}.npz" \
        --out "data/hand_tracks/ds${s}.npz" --use-depth
done

echo "=============================================================="
echo "annotations + norm stats"
echo "=============================================================="
bash tools/run_phase2.sh
