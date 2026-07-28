# ProteinFlux-PTM

SE(3) flow-matching model for generating protein molecular-dynamics (MD)
conformational ensembles, with support for phosphorylation post-translational
modifications (PTMs).

The release contains three components:

1. **FluxSite** — a PTM-site feature extractor (`fluxsite/`). Predicts
   phosphorylation propensity per residue and produces per-residue feature
   vectors used to condition the PTM dynamics model.
2. **Dynamics model — pretrain** — an ESM2-conditioned flow-matching backbone
   trained on unmodified MD trajectories (ATLAS + MoDEL).
3. **Dynamics model — PTM** — the pretrained ESM2 backbone fine-tuned on
   phosphorylated trajectories (DynaMo-phos), conditioned on an explicit PTM
   channel plus FluxSite features.

Three phospho PTM types are supported: **SEP** (phosphoserine),
**TPO** (phosphothreonine), **PTR** (phosphotyrosine).

## Pretrained weights

All checkpoints are hosted on the Hugging Face Hub:
**[clab-qqt/ProteinFlux](https://huggingface.co/clab-qqt/ProteinFlux)**

### Dynamics models

| Model | File | Description |
|---|---|---|
| Pretrain (ESM2) | [`esm2_multi.ckpt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/esm2_multi.ckpt) | ESM2-conditioned backbone, trained on ATLAS + MoDEL |
| PTM (ESM2 + FluxSite) | [`ptm_esm2_fluxsite.ckpt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/ptm_esm2_fluxsite.ckpt) | Phospho fine-tune: PTM channel + FluxSite prelogit-256, full fine-tune |

### FluxSite site-prediction models

Per-PTM-type checkpoints under [`fluxsite/`](https://huggingface.co/clab-qqt/ProteinFlux/tree/main/fluxsite). The phospho heads (`pho_st_model.pt`, `phos_y_model.pt`) are the ones used to produce the PTM dynamics features.

| PTM type | File |
|---|---|
| phosphorylation (S/T) | [`fluxsite/pho_st_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/pho_st_model.pt) |
| phosphorylation (Y) | [`fluxsite/phos_y_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/phos_y_model.pt) |
| acetylation (K) | [`fluxsite/Acetylation_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/Acetylation_model.pt) |
| methylation (K) | [`fluxsite/met_K_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/met_K_model.pt) |
| methylation (R) | [`fluxsite/met_R_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/met_R_model.pt) |
| ubiquitination (K) | [`fluxsite/Ubiqu_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/Ubiqu_model.pt) |
| sumoylation (K) | [`fluxsite/SUMO_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/SUMO_model.pt) |
| N-linked glycosylation (N) | [`fluxsite/n-linked_glycosylation_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/n-linked_glycosylation_model.pt) |
| O-linked glycosylation (S/T) | [`fluxsite/o_linked_glycosylation_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/o_linked_glycosylation_model.pt) |
| palmitoylation (C) | [`fluxsite/s_palmitoylation_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/s_palmitoylation_model.pt) |
| nitrosylation (C) | [`fluxsite/s-nitrosylation_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/s-nitrosylation_model.pt) |
| malonylation (K) | [`fluxsite/malonylation_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/malonylation_model.pt) |
| crotonylation (K) | [`fluxsite/Crotonylation_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/Crotonylation_model.pt) |
| succinylation (K) | [`fluxsite/succinylation_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/succinylation_model.pt) |
| amidation (C-term) | [`fluxsite/amidation_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/amidation_model.pt) |
| glutathionylation (C) | [`fluxsite/glutathionylation_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/glutathionylation_model.pt) |
| sulfoxidation (M) | [`fluxsite/sulfoxidation_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/sulfoxidation_model.pt) |
| hydroxylation (P/K/W/Y) | [`fluxsite/hydroxylation_model.pt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/hydroxylation_model.pt) |

Download example:

```bash
pip install huggingface_hub
huggingface-cli download clab-qqt/ProteinFlux esm2_multi.ckpt --local-dir ./weights
huggingface-cli download clab-qqt/ProteinFlux --include "fluxsite/*" --local-dir ./weights
```

## Project structure

```
proteinflux-release/
├── core/       # Geometry, rigid-body ops, residue constants, PTM utils, tensor ops
├── model/      # Backbone (Transformer + IPA), attention, embeddings, Hyena, LoRA
├── flow/       # Flow-matching paths, matching objective, ODE/SDE solvers
├── trainer/    # Lightning module, EMA, ensemble loss, checkpointing, argparse config
├── data/       # DataModule, trajectory dataset (single- and multi-dataset, PTM)
├── fluxsite/   # PTM-site prediction / feature extraction (standalone sub-package)
├── scripts/    # Training, inference, precompute and feature-extraction entry points
└── splits/     # Dataset splits (name,seqres) for ATLAS / MoDEL / DynaMo-phos
```

## Installation

```bash
pip install -r requirements.txt
```

Optional experiment logging (SwanLab):

```bash
pip install swanlab
```

## Data preparation

MD trajectories are stored in HDF5, one entry per protein, with heavy-atom
coordinates of shape `[frames, residues, 14, 3]`.

### 1. Splits

Splits are provided under `splits/`, one directory per dataset, each with
`train.csv` / `val.csv` / `test.csv` (columns: `name,seqres`):

```
splits/
├── atlas/         # ATLAS      (1550 / 194 / 194)
├── model/         # MoDEL      (1114 / 139 / 140)
└── dynamo_phos/   # DynaMo-phos (977 / 114 / 91)
```

To build new splits by sequence-cluster (avoids train/test leakage):

```bash
python splits/split.py \
    --fasta /path/to/all_sequences.fasta \
    --cluster_tsv /path/to/mmseqs2_clusters.tsv \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

### 2. ESM2 embeddings (for the ESM2-conditioned models)

Precompute per-protein ESM2 embeddings (keyed by the split's `name` column):

```bash
python scripts/precompute_esm2.py \
    --split splits/atlas/train.csv \
    --model esm2_t33_650M_UR50D \
    --out /path/to/atlas_esm2_embeddings.pkl
```

### 3. FluxSite features (for the PTM model only)

The PTM model is conditioned on **per-residue FluxSite features** (256-dim): the
pre-logit representation of the FluxSite phospho-site predictor (see the FluxSite
section below), extracted per head — the S/T head for Ser/Thr systems and the Y
head for Tyr systems — and merged into a single per-residue feature array. The
released model was trained on `dynamo_prelogit_features_256_byhead.pkl`.

These features are treated as a precomputed input: extract them from your trained
FluxSite model (`pho_st_model.pt`, `phos_y_model.pt`) and pass the resulting pkl
via `--ptm_feat_path` (with `--ptm_feat_dim 256`).

> The FluxSite checkpoints are large and are not included in this release; train
> them via `fluxsite/` (see below) or obtain them separately.

## Component 2 — Dynamics model (pretrain, ESM2)

Multi-dataset ESM2-conditioned training on ATLAS + MoDEL. Edit the path
variables at the top of the launch script, then run:

```bash
bash scripts/run_train_multi_esm2.sh
```

Key settings (see the script header): `--esm2_dim 1280 --esm2_proj_dim 512`,
`--num_frames 100 --crop 256`, `--prepend_ipa --ema`, mixed sampling via
`--dataset_weights`.

## Component 3 — Dynamics model (PTM, ESM2 + FluxSite)

Fine-tune the pretrained ESM2 backbone on phosphorylated trajectories. The
phospho signal enters through both an explicit PTM channel (`--use_ptm_channel`,
hard markers read from `seqres` codes 21/22/23) and the FluxSite feature adapter
(`--ptm_feat_path ... --ptm_feat_dim 256`). Full fine-tuning by default; pass
`lora` as the first argument to switch to LoRA.

```bash
# full fine-tune on GPUs 0-3
bash scripts/run_train_ptm_esm2_fluxsite.sh full 0,1,2,3
```

This reproduces the released PTM model
(`esm2_multi` base → + PTM channel + FluxSite prelogit-256, full fine-tune).

### PTM residue encoding

PTM residues are encoded as amino-acid types beyond the standard 20:

| Code | Index | PTM |
|------|-------|-----|
| B | 21 | SEP (phosphoserine) |
| J | 22 | TPO (phosphothreonine) |
| Z | 23 | PTR (phosphotyrosine) |

## Inference

Multi-GPU, multi-replica trajectory generation. The same entry point serves
both the pretrained and the PTM models.

### Pretrained (ESM2) model

```bash
bash scripts/run_inference_multi_esm2.sh          # set EVAL_DATASET=atlas|model
# or directly:
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 scripts/inference.py \
    --model esm2_multi --eval_dataset atlas \
    --data_dir /path/to/atlas_100ps.h5 \
    --split    splits/atlas/test.csv \
    --esm2_emb_path /path/to/atlas_esm2_embeddings.pkl \
    --num_frames 100 --num_replicas 5
```

### PTM model

The `ptm_dynamo` entry in `inference.py` already points at the released PTM
model (ESM2 base + PTM channel + FluxSite prelogit-256, full fine-tune); fill in
the `/path/to/...` placeholders in `MODEL_DEFAULTS` / `EVAL_DEFAULTS` (or pass
`--ckpt` / `--esm2_emb_path` / `--ptm_feat_path` to override) and run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 scripts/inference.py \
    --model ptm_dynamo --eval_dataset dynamo \
    --num_frames 100 --num_replicas 5
```

`inference.py` loads FluxSite features, activates the PTM channel from the
`seqres` PTM codes, and supports `--ptm_ablation {zero,wrong_site}` for
counterfactual (phospho vs. control) comparisons.

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--num_frames` | 100 | Trajectory frames generated per replica |
| `--num_replicas` | 5 | Independent replicas per protein |
| `--crop` | 256 | Residue crop size |
| `--fixed_ptm_crop` | off | Center the crop window on the PTM site |
| `--ptm_feat_path` | None | Per-residue PTM feature pkl (enables the FluxSite adapter) |
| `--ptm_feat_dim` | 256 | PTM feature dimension |
| `--use_ptm_channel` | off | Enable the explicit PTM marker channel |
| `--ptm_ablation` | none | `zero` (remove PTM signal) or `wrong_site` (misplace it) |
| `--ema` | off | Use EMA weights |

---

## FluxSite — PTM site prediction

FluxSite is a deep-learning module for predicting PTM sites from protein
sequence and structure features. It lives in `fluxsite/` and can be used
independently of the flow-matching backbone. In this project it also serves as
the feature extractor for the PTM dynamics model (see Data preparation §3).

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
│   ├── attention_pooling.py        # Attention pooling layer
│   └── evaluation.py               # Evaluation utilities
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

Pre-computed ESM embeddings and structural features can be stored in HDF5 and
loaded on demand via the lazy feature store to avoid memory bottlenecks.

### Training

Run from the project root:

```bash
python -m fluxsite.train \
    --data_path /path/to/data.csv \
    --features_path /path/to/features.h5 \
    --target_ptm_type phosphorylation \
    --output_dir ./outputs/phospho_run \
    --epochs 100 --batch_size 32 --learning_rate 2e-4
```

Protein-group-stratified K-fold cross-validation:

```bash
python -m fluxsite.train ... --use_kfold --kfold_splits 5
```

Use a JSON config and override individual arguments on the command line:

```bash
python -m fluxsite.train --config_path configs/fluxsite_default.json \
    --target_ptm_type methylation --epochs 200 --learning_rate 1e-4
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
| `--use_moe` | off | Enable Mixture-of-Experts routing |
| `--use_reinforcement` | off | Enable RL hyperparameter controller |
| `--integration_method` | `serial` | Module integration (`serial`, `ensemble`, `dynamic`) |
| `--early_stopping_patience` | 15 | Epochs without improvement before stopping |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Portions adapted from OpenFold and AlphaFold2 (Apache 2.0), and Hyena / SiT (MIT).
