#!/usr/bin/env bash
# Phase 1 driver: cam_high video -> WiLoR -> 7-D hand state in the robot base frame.
#
# Runs the two stages in their own environments (.venv-wilor for WiLoR, .venv
# for the geometry + depth work) over every episode of the HRI insertion set.
# cam_high videos are kept; Phase 2 reuses them.
#
#   bash tools/run_phase1.sh            # all repos
#   bash tools/run_phase1.sh 2 eval     # just these suffixes
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
STRIDE=${STRIDE:-6}

SUFFIXES=("$@")
if [ ${#SUFFIXES[@]} -eq 0 ]; then
    SUFFIXES=(1 2 3 4 5 6 eval)
fi

mkdir -p data/raw data/hand_raw data/hand_tracks data/viz

for s in "${SUFFIXES[@]}"; do
    repo="LeeHakHo/play_hri_insertion_${s}_eef"
    vid="data/raw/ds${s}_cam_high.mp4"
    raw="data/hand_raw/ds${s}.npz"
    out="data/hand_tracks/ds${s}.npz"

    echo "=============================================================="
    echo "[$s] $repo"
    echo "=============================================================="

    if [ ! -f "$vid" ]; then
        echo "--- downloading cam_high"
        curl -fsSL -o "$vid.part" \
          "https://huggingface.co/datasets/${repo}/resolve/main/videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
        mv "$vid.part" "$vid"
    fi
    ls -lh "$vid"

    if [ ! -f "$raw" ]; then
        echo "--- WiLoR"
        "$ROOT/.venv-wilor/bin/python" tools/run_wilor.py \
            --video "$vid" --out "$raw" --stride "$STRIDE"
    fi

    echo "--- base-frame hand state (depth-refined)"
    "$ROOT/.venv/bin/python" tools/extract_hand_tracks.py \
        --repo "$repo" --wilor "$raw" --out "$out" --use-depth

    echo "--- overlay (first 400 frames)"
    "$ROOT/.venv/bin/python" tools/viz_hand_tracks.py \
        --repo "$repo" --video "$vid" --tracks "$out" \
        --out "data/viz/ds${s}_overlay.mp4" --max-frames 400
done

echo
echo "Phase 1 done. Review data/viz/*_overlay.mp4 before Phase 2."
