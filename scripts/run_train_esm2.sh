#!/usr/bin/env bash
# Training script: ESM2 embedding conditioning
#
# Step 1 — pre-compute ESM2 embeddings (run once):
#   python scripts/precompute_esm2.py \
#       --splits splits/train.csv splits/val.csv \
#       --fasta_dir /path/to/fastas \
#       --model esm2_t33_650M_UR50D \
#       --output /path/to/esm2_embeddings.pkl
#
# Step 2 — run this script.
#
# Usage: bash scripts/run_train_esm2.sh

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR="/path/to/data.h5"
TRAIN_SPLIT="splits/train.csv"
VAL_SPLIT="splits/val.csv"
ESM2_EMB_PATH="/path/to/esm2_embeddings.pkl"
# Optional: start from aa_emb checkpoint (recommended)
CKPT=""   # e.g. "workdir/baseline_aa_emb/checkpoints/backup_epoch99.ckpt"

# ── Run settings ───────────────────────────────────────────────────────────
# Which GPU(s) to use. Examples:
#   single GPU:  "0"
#   multi-GPU:   "0,1,2,3"
GPU_IDS="0"
NUM_GPUS=1        # must match the number of IDs in GPU_IDS
RUN_NAME="esm2_650M"

# ── ESM2 settings ──────────────────────────────────────────────────────────
# Match esm2_dim to the model used in precompute_esm2.py:
#   esm2_t6_8M_UR50D   → 320
#   esm2_t12_35M_UR50D → 480
#   esm2_t33_650M_UR50D → 1280
#   esm2_t36_3B_UR50D  → 2560
ESM2_DIM=1280
# Hidden dim of the 2-layer projection MLP: (esm2_dim + embed_dim) // 2 by default.
# Set to 0 to use a single linear layer instead.
ESM2_PROJ_DIM=""   # leave empty to use default

# ── Hyperparameters (keep identical to run_train_aa.sh for fair comparison) ─
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
PRINT_FREQ=100
CKPT_FREQ=10
VAL_EPOCH_FREQ=1

# ── Build optional args ────────────────────────────────────────────────────
CKPT_ARG=""
if [ -n "${CKPT}" ]; then
    CKPT_ARG="--ckpt ${CKPT}"
fi

PROJ_DIM_ARG=""
if [ -n "${ESM2_PROJ_DIM}" ]; then
    PROJ_DIM_ARG="--esm2_proj_dim ${ESM2_PROJ_DIM}"
fi

# ── Launch ─────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="${GPU_IDS}" torchrun --nproc_per_node "${NUM_GPUS}" scripts/train.py \
    ${CKPT_ARG} \
    --run_name   "${RUN_NAME}" \
    --data_dir   "${DATA_DIR}" \
    --train_split "${TRAIN_SPLIT}" \
    --val_split   "${VAL_SPLIT}" \
    \
    --esm2_emb_path "${ESM2_EMB_PATH}" \
    --esm2_dim      "${ESM2_DIM}" \
    ${PROJ_DIM_ARG} \
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
