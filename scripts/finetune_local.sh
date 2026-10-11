#!/bin/bash
# Finetune WEAVER on a single consumer GPU (developed against a 16 GB RTX 4080).
#
# Differences from scripts/finetune.sh, which assumes 4xH100 under Slurm:
#   * one process, no torchrun / DDP
#   * LoRA on the transformer blocks (LORA_RANK), so only ~35M of the ~915M
#     world-model parameters carry optimiser state
#   * torch.compile off — it interacts badly with the swapped-in LoRA modules
#     and buys little at batch size 2
#   * reward and critic losses off: this play data has no success labels
#   * an 18-D state/action space: three base-frame points per agent, where
#     the action is the COMMANDED pose of those points rather than a delta
#     of the observed ones -- that is what a policy emits, and the gap
#     between commanded and observed jaws is the grasp
#
#   bash scripts/finetune_local.sh
set -euo pipefail

cd "$(dirname "$0")/.."
WEAVER_DIR=$PWD
PY="$WEAVER_DIR/.venv/bin/python"

PRETRAINED_DIR=${PRETRAINED_DIR:-"$WEAVER_DIR/checkpoints/WEAVER"}
DATASET_PATH=${DATASET_PATH:-"$WEAVER_DIR/data/weaver_hri"}
EXP_NAME=${EXP_NAME:-weaver_hri}
FINETUNE_SUFFIX=${FINETUNE_SUFFIX:-hri}
TRAIN_STEPS=${TRAIN_STEPS:-20000}
BATCH_SIZE=${BATCH_SIZE:-2}
# 4, not 8: the run this resumes from used 4 (effective batch 8) at 1.79 s/step.
# Changing it mid-run would double the step time and shift the effective batch.
GRAD_ACCUM=${GRAD_ACCUM:-4}
LORA_RANK=${LORA_RANK:-32}
# The state/action projections are re-initialised because the action space
# changed, so they start much further from their target than the LoRA adapters.
PROJ_LR_MULT=${PROJ_LR_MULT:-10}
# Only the top-down view. The wrist camera is mounted on the gripper and rarely
# shows the person's hand at all, so half the image tokens had nothing for the
# hand action to explain.
IMG_KEYS=${IMG_KEYS:-"['exterior_1_left']"}
LR=${LR:-3e-5}
# Cube and bowl, predicted by a head of their own and never fed back in,
# so the latent has to carry where they are. 0 disables the head.
N_OBJECTS=${N_OBJECTS:-2}
OBJECT_LOSS_SCALE=${OBJECT_LOSS_SCALE:-0.1}
N_WORKERS=${N_WORKERS:-4}
# Diagnostics (val metrics + rollout clips) cost ~10 min each, so they run a few
# times over the run rather than every couple of thousand steps.
DIAG_FREQ=${DIAG_FREQ:-4000}
CKPT_FREQ=${CKPT_FREQ:-1000}
USE_WANDB=${USE_WANDB:-False}
# No API key is configured on this machine and wandb.init() hard-fails
# without one. Offline is what the previous run used; `wandb sync <run-dir>`
# uploads the history later if a key is added.
export WANDB_MODE=${WANDB_MODE:-offline}
WANDB_PROJECT=${WANDB_PROJECT:-weaver-hri}
WANDB_ENTITY=${WANDB_ENTITY:-None}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}

export PYTHONPATH="$WEAVER_DIR:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONUNBUFFERED=1

echo "pretrained : $PRETRAINED_DIR"
echo "dataset    : $DATASET_PATH"
echo "lora rank  : $LORA_RANK"
echo "batch/accum: $BATCH_SIZE / $GRAD_ACCUM"
echo "steps      : $TRAIN_STEPS"
echo "cameras    : $IMG_KEYS"
echo "proj lr x  : $PROJ_LR_MULT"
echo "diagnostics: every $DIAG_FREQ steps   checkpoints: every $CKPT_FREQ"
echo "wandb      : $USE_WANDB ${USE_WANDB:+($WANDB_PROJECT)}"

# -u: stdout is a redirected file here, and block buffering would hide
# ~40 minutes of step logs behind a 4 KB buffer.
exec "$PY" -u -m weaver.finetune \
  --mode defaults \
  --pretrained_dir "$PRETRAINED_DIR" \
  --finetune_suffix "$FINETUNE_SUFFIX" \
  --lora_rank "$LORA_RANK" \
  --projection_lr_mult "$PROJ_LR_MULT" \
  scratch_dir="$WEAVER_DIR/runs/${EXP_NAME}" \
  exp_name="$EXP_NAME" \
  use_wandb=$USE_WANDB \
  wandb.project="$WANDB_PROJECT" \
  wandb.entity=$WANDB_ENTITY \
  use_compile=False \
  use_activation_checkpointing=True \
  save_model=True \
  ckpt_freq=$CKPT_FREQ \
  valid_log_freq=$DIAG_FREQ \
  video_log_freq=$DIAG_FREQ \
  log_freq=${LOG_FREQ:-10} \
  dataset.name=DROID \
  dataset.path="$DATASET_PATH" \
  dataset.img_keys="$IMG_KEYS" \
  dataset.n_states=18 \
  dataset.n_actions=18 \
  dataset.relabel_actions=False \
  dataset.normalize=True \
  dataset.norm_stats_path="$DATASET_PATH/norm_stats_cmd.json" \
  dataset.n_workers="$N_WORKERS" \
  model.rm_loss_coeff=0.0 \
  model.critic_loss_coeff=0.0 \
  model.n_objects=$N_OBJECTS \
  model.loss_scale.objects=$OBJECT_LOSS_SCALE \
  finetune_cfg.batch_size="$BATCH_SIZE" \
  finetune_cfg.gradient_accumulation_steps="$GRAD_ACCUM" \
  finetune_cfg.max_steps="$TRAIN_STEPS" \
  finetune_cfg.max_lr="$LR" \
  finetune_cfg.min_lr="$LR" \
  $EXTRA_OVERRIDES
