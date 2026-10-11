#!/usr/bin/env bash
# Phase 2 driver: LeRobot episodes + hand tracks -> a WEAVER preprocessed dataset.
#
# Episodes 1-6 go to train, the eval episode to val.  Trajectory ids are spaced
# per episode so segments never collide.
#
# Idempotent: an episode whose SD3 latents are already on disk only gets its
# annotations rewritten.  The videos and latents do not depend on the
# state/action representation, so changing that costs nothing here.
#
#   bash tools/run_phase2.sh              # all episodes
#   bash tools/run_phase2.sh 5 6 eval     # just these
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
PY="$ROOT/.venv/bin/python -u"
OUT=${OUT:-data/weaver_hri}
SEGMENT_LEN=${SEGMENT_LEN:-200}

SUFFIXES=("$@")
if [ ${#SUFFIXES[@]} -eq 0 ]; then
    SUFFIXES=(1 2 3 4 5 6 eval)
fi

split_of () { [ "$1" = eval ] && echo val || echo train; }
base_of ()  { [ "$1" = eval ] && echo 9000 || echo $(( $1 * 1000 )); }

for s in "${SUFFIXES[@]}"; do
    tracks="data/hand_tracks/ds${s}.npz"
    split=$(split_of "$s")
    base=$(base_of "$s")

    if [ ! -f "$tracks" ]; then
        echo "!! missing $tracks — run tools/run_phase1.sh first" >&2
        exit 1
    fi

    extra=""
    if [ -f "$OUT/latents/$split/${base}_sd3.npz" ]; then
        extra="--annotations-only"
    fi

    echo "=============================================================="
    echo "[$s] -> $split (ids from $base) ${extra:-full conversion}"
    echo "=============================================================="
    $PY tools/lerobot_to_weaver.py \
        --repo "LeeHakHo/play_hri_insertion_${s}_eef" \
        --hand "$tracks" \
        --out-root "$OUT" \
        --split "$split" \
        --episode-base "$base" \
        --segment-len "$SEGMENT_LEN" \
        $extra
done

echo
echo "--- normalisation statistics ---"
$PY tools/compute_norm_stats_hri.py --data-root "$OUT"

echo
echo "Phase 2 done:"
echo "  train trajectories: $(ls "$OUT/annotation_rewards/train" | wc -l)"
echo "  val   trajectories: $(ls "$OUT/annotation_rewards/val" | wc -l)"
du -sh "$OUT"
