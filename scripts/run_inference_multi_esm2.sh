#!/usr/bin/env bash
# esm2_multi (ATLAS+MoDEL mixed) inference -- /teams machine, explicit paths.
# Use the data_dir/split/ESM2 pkl matching whichever dataset you evaluate. Change EVAL_DATASET to switch atlas/model.
# Usage: bash scripts/run_inference_multi_esm2.sh

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

GPU_IDS="0,1,2,3"
NUM_GPUS=4
MASTER_PORT="${MASTER_PORT:-29524}"

MODEL="esm2_multi"
EVAL_DATASET="model"     # atlas or model

# ── Select data_dir / split / ESM2 pkl on /teams per dataset ─────────────────
case "${EVAL_DATASET}" in
  atlas)
    DATA_DIR="/path/to/data/Data/Atlas/atlas_100ps.h5"
    SPLIT="/path/to/data/splits/atlas_test.csv"
    ESM2_EMB="/path/to/data/esm2_t33_650m/atlas_esm2_embeddings.pkl" ;;
  model)
    DATA_DIR="/path/to/data/ProteinFlux_20260610/ProteinFlux_dev/Dataset/data_model_final_1393.h5"
    SPLIT="/path/to/data/ProteinFlux_20260610/ProteinFlux_dev/Dataset/model_test_1393.csv"
    ESM2_EMB="/path/to/data/ProteinFlux_20260610/ProteinFlux_dev/Dataset/model_esm2_t33_650m_1393.pkl" ;;
  *) echo "unknown EVAL_DATASET=${EVAL_DATASET}"; exit 1 ;;
esac

NUM_FRAMES=100
NUM_REPLICAS=5
SEED_BASE=42
PDB_IDS=""

CUDA_VISIBLE_DEVICES="${GPU_IDS}" torchrun \
    --nproc_per_node "${NUM_GPUS}" \
    --master_port "${MASTER_PORT}" \
    scripts/inference.py \
    --model         "${MODEL}" \
    --eval_dataset  "${EVAL_DATASET}" \
    --data_dir      "${DATA_DIR}" \
    --split         "${SPLIT}" \
    --esm2_emb_path "${ESM2_EMB}" \
    --num_frames    "${NUM_FRAMES}" \
    --num_replicas  "${NUM_REPLICAS}" \
    --seed_base     "${SEED_BASE}" \
    --skip_existing \
    --suffix ema \
    ${PDB_IDS:+--pdb_id ${PDB_IDS}}
