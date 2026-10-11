#!/bin/bash
# Phase 4: evaluate the finetuned world model on the HRI validation episode.
#
# Same two steps as scripts/evaluate_wm.sh + scripts/compute_eval_metrics.sh,
# minus Slurm and the array sharding — one GPU, one process.
#
#   bash scripts/eval_local.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
PY="$ROOT/.venv/bin/python"

CHECKPOINT=${CHECKPOINT:-"$ROOT/runs/weaver_hri/logs/chkpts_hri"}
DATASET_PATH=${DATASET_PATH:-"$ROOT/data/weaver_hri"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/runs/weaver_hri/eval"}
SPLIT=${SPLIT:-val}
START_IDX=${START_IDX:-20}
NUM_VIDEOS=${NUM_VIDEOS:-4}
VAL_STEPS=${VAL_STEPS:-27}
EVAL_HORIZON=${EVAL_HORIZON:-5}
EVAL_BOOTSTRAP=${EVAL_BOOTSTRAP:-5}
NUM_FRAMES=${NUM_FRAMES:-50}
# One camera slot per entry in dataset.img_keys; the run trains on the top-down
# view alone, so asking for view1 as well makes the metric script find nothing.
CAMERAS=${CAMERAS:-view0}

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

anno_dir="$DATASET_PATH/annotation_rewards/$SPLIT"
num_samples=${NUM_SAMPLES:-$(find "$anno_dir" -maxdepth 1 -name '*.json' | wc -l)}
echo "checkpoint : $CHECKPOINT"
echo "dataset    : $DATASET_PATH ($num_samples $SPLIT trajectories)"
echo "output     : $OUTPUT_DIR"

echo
echo "=== 1/3  rollouts ==="
"$PY" -m weaver.generate_views \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  --num-samples "$num_samples" \
  --num-videos "$NUM_VIDEOS" \
  --start-idx "$START_IDX" \
  --use-real-history \
  --use-ema \
  --use-kv-cache \
  --overrides \
    dataset.path="$DATASET_PATH" \
    model.val_steps="$VAL_STEPS" \
    eval_horizon="$EVAL_HORIZON" \
    eval_bootstrap="$EVAL_BOOTSTRAP" \
    inference.pyramid_stagger_width=1 \
    inference.pyramid_schedule=cosine

echo
echo "=== 2/3  FID / FVD / LPIPS ==="
views_dir="$OUTPUT_DIR/views"
[ -d "$views_dir" ] || views_dir="$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/metrics"
"$PY" -m weaver.utils.metrics \
  --views_dir "$views_dir" \
  --cameras $CAMERAS \
  --start_frame 0 \
  --num_frames "$NUM_FRAMES" \
  --fvd_sliding_window --fvd_window 16 --fvd_stride 8 \
  --pad_short_clips --skip_psnr --skip_ssim \
  --lpips_impl torchmetrics \
  --output "$OUTPUT_DIR/metrics/fid_fvd_lpips.json"

echo
echo "=== 3/3  agent-separation counterfactual ==="
"$PY" tools/counterfactual_rollout.py \
  --checkpoint "$CHECKPOINT" \
  --out "$OUTPUT_DIR/counterfactual.mp4" \
  --overrides dataset.path="$DATASET_PATH"

echo
echo "done:"
echo "  metrics        $OUTPUT_DIR/metrics/fid_fvd_lpips.json"
echo "  counterfactual $OUTPUT_DIR/counterfactual.mp4"
