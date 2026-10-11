#!/usr/bin/env bash
# Re-run stage 2 of hand extraction only (the WiLoR output is unchanged).
# Needed whenever the point set changes: the depth cache is keyed on the pixels
# being sampled, so a new set of points means a fresh ~2 GB read per episode.
set -euo pipefail
cd "$(dirname "$0")/.."
for s in "$@"; do
    echo "=============================================================="
    echo "[$s]"
    echo "=============================================================="
    ./.venv/bin/python tools/extract_hand_tracks.py \
        --repo "LeeHakHo/play_hri_insertion_${s}_eef" \
        --wilor "data/hand_raw/ds${s}.npz" \
        --out "data/hand_tracks/ds${s}.npz" --use-depth
done
echo "stage 2 re-run done"
