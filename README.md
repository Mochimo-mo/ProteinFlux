# ProteinFlux-PTM

Flow matching model for protein MD trajectory generation with Post-Translational Modification (PTM) support.

Supports three PTM types: **SEP** (phosphoserine), **TPO** (phosphothreonine), **PTR** (phosphotyrosine).

## Project Structure

```
ProteinFlux_PTM/
├── core/          # Geometry, rigid body utils, residue constants, tensor ops
├── model/         # LatentModel (Transformer + IPA), attention layers
├── transport/     # Flow matching transport (ODE/SDE integrators, path types)
├── datasets/      # ProteinDataset, PTMDataset, DataModule
├── training/      # ProteinWrapper (Lightning module), EMA, train utilities, argparse
├── scripts/       # Training entry points and launch scripts
└── splits/        # Dataset split utilities
```

## Installation

```bash
pip install -r requirements.txt
```

For SwanLab logging (optional):
```bash
pip install swanlab
```

## Data Preparation

1. Prepare your MD trajectory data in HDF5 format (shape: `[frames, residues, 14, 3]`).
2. Create train/val/test splits using the cluster-based split script:

```bash
python splits/split.py \
    --fasta /path/to/all_sequences.fasta \
    --cluster_tsv /path/to/mmseqs2_clusters.tsv \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

3. (Optional) Prepare per-residue PTM feature files in `.pkl` format with shape `[residues, feat_dim]` per protein.

## Training

### Without PTM features (base model fine-tuning)

```bash
torchrun --nproc_per_node n scripts/train_finetune.py \
    --sim_condition \
    --ckpt /path/to/pretrained.ckpt \
    --data_dir /path/to/data.h5 \
    --train_split splits/train.csv \
    --val_split splits/val.csv \
    --batch_size 1 --lr 5e-5 \
    --prepend_ipa --num_frames 100 --crop 512 \
    --ema --adamW --epochs 1000
```

### With PTM features

```bash
bash scripts/run_PTMfeature.sh
```

Edit the path variables at the top of `scripts/run_PTMfeature.sh` before running:

```bash
CKPT_PATH="/path/to/pretrained/atlas.ckpt"
DATA_DIR="/path/to/data/data.h5"
TRAIN_SPLIT="/path/to/splits/train.csv"
VAL_SPLIT="/path/to/splits/val.csv"
PTM_FEAT_PATH="/path/to/ptm_features/residue_features_256.pkl"
```

### Key training arguments

| Argument | Default | Description |
|---|---|---|
| `--num_frames` | 100 | Number of trajectory frames per sample |
| `--crop` | 512 | Residue crop size |
| `--fixed_ptm_crop` | off | Center crop window on PTM site |
| `--ptm_feat_path` | None | Path to PTM feature pkl; enables PTMDataset |
| `--ema` | off | Exponential moving average of weights |
| `--swanlab` | off | Enable SwanLab logging |

## PTM Residue Encoding

PTM residues are encoded as additional amino acid types beyond the standard 20:

| Code | Index | PTM type |
|------|-------|----------|
| B | 21 | SEP (phosphoserine) |
| J | 22 | TPO (phosphothreonine) |
| Z | 23 | PTR (phosphotyrosine) |

---

## FluxSite — PTM Site Prediction

FluxSite is a deep learning module for predicting post-translational modification (PTM) sites from protein sequence and structure features. It lives in the `fluxsite/` sub-package and can be used independently of the flow-matching backbone.

### Supported PTM types

| PTM | Target residues |
|-----|-----------------|
| phosphorylation | S, T, Y |
| acetylation | K |
| methylation | K, R |
| ubiquitination | K |
| sumoylation | K |
| n-linked glycosylation | N |
| o-linked glycosylation | S, T |
| palmitoylation / nitrosylation | C |
| malonylation / crotonylation | K |
| succinylation | K |
| amidation | C-terminal / N-terminal (Gly) |
| glutathionylation | C |
| sulfoxidation | M |
| hydroxylation | P, K, W, Y |

### Module structure

```
fluxsite/
├── data/
│   └── unified_data_processor.py   # Dataset & DataLoader for PTM data (HDF5/CSV)
├── models/
│   ├── acetylation_predictor.py    # Main model: AcetylationPredictor / DualBranchFusionPredictor
│   ├── dual_branch_modules.py      # Sequence & structure towers, gated fusion
│   ├── encoders.py                 # ESM2 sequence encoder, ESM-IF1 structure encoder
│   ├── enhanced_components.py      # Multi-scale fusion, dynamic feature selector
│   ├── cnn_model.py                # CNN dual-stream predictor (lightweight baseline)
│   ├── mixtures.py                 # Mixture-of-Experts routing
│   ├── reinforcement.py            # RL-guided classifier head
│   └── attention_pooling.py        # Attention pooling layer
├── utils/
│   ├── config_manager.py           # JSON config loading & CLI merging
│   ├── training_utils_enhanced.py  # Core training loop with early stopping & EMA
│   ├── rl_training_manager.py      # RL-enhanced training manager
│   ├── kfold_validator.py          # Stratified group K-fold cross-validation
│   ├── metrics.py                  # AUROC, AUPRC, MCC, F1, optimal threshold
│   ├── enhanced_loss.py            # Focal loss, supervised contrastive loss
│   ├── feature_normalization.py    # Per-feature normalizer with persistence
│   ├── feature_reduction.py        # Incremental PCA for high-dim features
│   ├── data_loading_optimizer.py   # LMDB / lazy feature store for large datasets
│   └── ptm_residue_info.py         # PTM → expected residue catalogue
└── train.py                        # Training entry point
```

### Data format

FluxSite expects tabular input (CSV/TSV) with at minimum:

| Column | Description |
|--------|-------------|
| `uniprot_id` | Protein identifier (used for group-aware splitting) |
| `sequence` | Full amino-acid sequence |
| `site_position` | 1-based index of the candidate PTM residue |
| `label` | 1 = positive site, 0 = negative |

Pre-computed ESM embeddings and structural features can be stored in HDF5 files and loaded on demand via the lazy feature store to avoid memory bottlenecks.

### Training

Run from the project root:

```bash
python -m fluxsite.train \
    --data_path /path/to/data.csv \
    --features_path /path/to/features.h5 \
    --target_ptm_type phosphorylation \
    --output_dir ./outputs/phospho_run \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 2e-4
```

Enable K-fold cross-validation (default: 5-fold, protein-group stratified):

```bash
python -m fluxsite.train \
    --data_path /path/to/data.csv \
    --features_path /path/to/features.h5 \
    --target_ptm_type acetylation \
    --use_kfold --kfold_splits 5 \
    --output_dir ./outputs/acetyl_kfold
```

Use a JSON config file and override individual arguments on the command line:

```bash
python -m fluxsite.train \
    --config_path configs/fluxsite_default.json \
    --target_ptm_type methylation \
    --epochs 200 --learning_rate 1e-4
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--target_ptm_type` | `phosphorylation` | PTM type to train for |
| `--core_model_type` | `dual_branch_fusion` | Model architecture (`dual_branch_fusion`, `ptm_bert_bilstm`) |
| `--seq_encoder_type` | `esm2_t33_650M_UR50D` | ESM2 variant for sequence encoding |
| `--window_size` | 61 | Global context window around the candidate site |
| `--local_window_size` | 31 | Local context window |
| `--use_kfold` | off | Enable K-fold cross-validation |
| `--kfold_splits` | 5 | Number of folds |
| `--use_moe` | off | Enable Mixture-of-Experts routing |
| `--use_reinforcement` | off | Enable RL hyperparameter controller |
| `--integration_method` | `serial` | Module integration strategy (`serial`, `ensemble`, `dynamic`) |
| `--early_stopping_patience` | 15 | Epochs without improvement before stopping |
| `--adaptive_dropout` | on | Adaptive dropout schedule during training |

### Advanced modules

**Mixture-of-Experts (MoE)** — routes each sample through a learned subset of expert sub-networks:

```bash
python -m fluxsite.train ... --use_moe --moe_num_experts 8 --moe_top_k 2
```

**RL hyperparameter controller** — adjusts learning rate and regularization based on training dynamics:

```bash
python -m fluxsite.train ... --use_reinforcement --enable_rl_controller \
    --rl_reward_type performance_based --rl_exploration_epsilon 0.1
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Portions adapted from OpenFold and AlphaFold2 (Apache 2.0), and Hyena/SiT (MIT).
