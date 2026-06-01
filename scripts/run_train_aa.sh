#!/usr/bin/env bash
# Training script: aa_emb baseline (no ESM2)
# Usage: bash scripts/run_train_aa.sh

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR="/path/to/data.h5"
TRAIN_SPLIT="splits/train.csv"
VAL_SPLIT="splits/val.csv"
# Optional: resume from a checkpoint
CKPT=""   # leave empty to train from scratch

# ── Run settings ───────────────────────────────────────────────────────────
# Which GPU(s) to use. Examples:
#   single GPU:  "0"
#   multi-GPU:   "0,1,2,3"
GPU_IDS="0"
NUM_GPUS=1        # must match the number of IDs in GPU_IDS
RUN_NAME="baseline_aa_emb"

# ── Hyperparameters ────────────────────────────────────────────────────────
EPOCHS=1000
BATCH_SIZE=1
LR=1e-4
NUM_FRAMES=50
CROP=256
NUM_WORKERS=4
GRAD_CLIP=1.0
ACCUMULATE_GRAD=1
PRECISION="32-true"

# Logging
PRINT_FREQ=100   # log every N optimizer steps
CKPT_FREQ=10     # save checkpoint every N epochs
VAL_EPOCH_FREQ=1 # validate every N epochs

# ── Build optional args ────────────────────────────────────────────────────
CKPT_ARG=""
if [ -n "${CKPT}" ]; then
    CKPT_ARG="--ckpt ${CKPT}"
fi

# ── Launch ─────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="${GPU_IDS}" torchrun --nproc_per_node "${NUM_GPUS}" scripts/train.py \
    ${CKPT_ARG} \
    --run_name   "${RUN_NAME}" \
    --data_dir   "${DATA_DIR}" \
    --train_split "${TRAIN_SPLIT}" \
    --val_split   "${VAL_SPLIT}" \
    \
    --epochs          "${EPOCHS}" \
    --batch_size      "${BATCH_SIZE}" \
    --lr              "${LR}" \
    --num_frames      "${NUM_FRAMES}" \
    --crop            "${CROP}" \
    --num_workers     "${NUM_WORKERS}" \
    --grad_clip       "${GRAD_CLIP}" \
    --accumulate_grad "${ACCUMULATE_GRAD}" \
    --precision       "${PRECISION}" \
    \
    --prepend_ipa \
    --ema \
    --adamW \
    \
    --print_freq      "${PRINT_FREQ}" \
    --ckpt_freq       "${CKPT_FREQ}" \
    --val_epoch_freq  "${VAL_EPOCH_FREQ}"
