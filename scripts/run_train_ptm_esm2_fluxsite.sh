#!/usr/bin/env bash
# =============================================================================
# ESM2-base PTM finetune  +  ptm_channel  +  FluxSite(prelogit256) features
#   base:   esm2_multi (best ckpt from ATLAS+MoDEL mixed pretraining, esm2_proj_dim=512)
#   phospho signal: ptm_channel (hard marker, reads seqres 21/22/23) + ptm_emb (byhead 256-dim FluxSite features)
#   full finetune by default; pass lora as first arg to switch to LoRA
#
# Usage:  bash scripts/run_train_ptm_esm2_fluxsite.sh [full|lora] [GPU]
#   single GPU:        ... full 0
#   multi-GPU (auto DDP):  ... full 0,1,2,3     # effective batch = batch_size × num GPUs
# =============================================================================
set -euo pipefail

MODE="${1:-full}"       # full | lora
GPU="${2:-0}"           # single id or comma list (e.g. 0,1,2,3) for multi-GPU DDP

# repo root (script lives under scripts/)
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# If system libstdc++ lacks CXXABI (scipy/torch import fails), uncomment the next line pointing to the conda env's lib:
# export LD_LIBRARY_PATH="$(dirname "$(dirname "$(command -v "$PY")")")/lib:${LD_LIBRARY_PATH:-}"

# ---- Data paths (set to your actual paths) ---------------------------------
CKPT="${CKPT:-/path/to/data/workdir/esm2_multi/checkpoints/best.ckpt}"   # esm2 pretrained base
H5="${H5:-/path/to/data/ptm_data.h5}"                                    # DynaMo-phos trajectory H5
ESM2="${ESM2:-/path/to/data/dynamo_phos_esm2_embeddings.pkl}"            # full-length ESM2 embedding
FEAT="${FEAT:-/path/to/data/dynamo_prelogit_features_256_byhead.pkl}"    # FluxSite prelogit 256
TRAIN="${TRAIN:-/path/to/data/splits/dynamo_phos/train.csv}"
VAL="${VAL:-/path/to/data/splits/dynamo_phos/val.csv}"

RUN="ptm_dynamo_esm2_prelogit256_${MODE}"

# ---- Precheck: all input files must exist ----------------------------------
for f in "$CKPT" "$H5" "$ESM2" "$FEAT" "$TRAIN" "$VAL"; do
  [ -f "$f" ] || { echo "[FATAL] missing: $f"; exit 1; }
done
echo "[ok] all 6 inputs present.  MODE=$MODE  GPU=$GPU  RUN=$RUN"

# ---- LoRA switch -------------------------------------------------------------
LORA_ARGS=()
if [ "$MODE" = "lora" ]; then
  LORA_ARGS=(--use_lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --expected_finetune_mode lora)
else
  LORA_ARGS=(--expected_finetune_mode full)
fi

cd "$REPO"
mkdir -p "workdir/$RUN"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train_finetune_PTMfeature.py \
  --ckpt "$CKPT" \
  --data_dir "$H5" \
  --esm2_emb_path "$ESM2" --esm2_dim 1280 --esm2_proj_dim 512 \
  --train_split "$TRAIN" --val_split "$VAL" \
  --ptm_feat_path "$FEAT" --ptm_feat_dim 256 \
  --use_ptm_channel --num_ptm_types 3 \
  --embed_dim 384 --num_layers 5 --mha_heads 16 \
  --ipa_heads 4 --ipa_head_dim 32 --ipa_qk 8 --ipa_v 8 --prepend_ipa \
  --num_frames 100 --max_train_frames 100 --crop 256 --fixed_ptm_crop \
  --path-type GVP --prediction velocity \
  --batch_size 2 --lr 1e-5 --min_lr 1e-6 --epochs 200 \
  --ema --ema_decay 0.999 --grad_clip 1.0 \
  --num_workers 4 --val_epoch_freq 1 --ckpt_freq 5 \
  --run_name "$RUN" \
  "${LORA_ARGS[@]}"

echo "✅ PTM finetune done. ckpt → workdir/$RUN/checkpoints/"
