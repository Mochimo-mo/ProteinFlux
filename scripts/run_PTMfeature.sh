#!/bin/bash

# PTM Feature Fine-tuning Launch Script

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# --- Paths (update to match your deployment) ---
CKPT_PATH="/path/to/pretrained/atlas.ckpt"
DATA_DIR="/path/to/data/data.h5"
TRAIN_SPLIT="/path/to/splits/train.csv"
VAL_SPLIT="/path/to/splits/val.csv"
PTM_FEAT_PATH="/path/to/ptm_features/residue_features_256.pkl"
RUN_NAME="PTM_Finetune_Feature_Fusion"

echo "Checkpoint : $CKPT_PATH"
echo "Data       : $DATA_DIR"
echo "Train split: $TRAIN_SPLIT"
echo "Val split  : $VAL_SPLIT"
echo "PTM feats  : $PTM_FEAT_PATH"

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: checkpoint not found: $CKPT_PATH"; exit 1
fi
if [ ! -f "$DATA_DIR" ]; then
    echo "Error: data file not found: $DATA_DIR"; exit 1
fi

torchrun \
  --nproc_per_node 8 \
  scripts/train_finetune_PTMfeature.py \
  --sim_condition \
  --run_name "$RUN_NAME" \
  --ckpt "$CKPT_PATH" \
  --data_dir "$DATA_DIR" \
  --train_split "$TRAIN_SPLIT" \
  --val_split "$VAL_SPLIT" \
  --ptm_feat_path "$PTM_FEAT_PATH" \
  --batch_size 1 \
  --lr 5e-5 \
  --prepend_ipa \
  --num_frames 100 \
  --frame_interval 1 \
  --crop 384 \
  --repeat 1 \
  --fixed_ptm_crop \
  --epochs 200 \
  --print_freq 100 \
  --check_grad \
  --ema \
  --adamW \
  --validate \
  --swanlab

if [ $? -eq 0 ]; then
    echo "Training finished successfully."
else
    echo "Training failed. Check the logs above."
fi
