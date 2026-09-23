#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/fluxsite/pho_st_model.pt}"
OUT_DIR="${OUT_DIR:-$DEMO_DIR/out}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT_DIR"
cd "$DEMO_DIR"

python "$DEMO_DIR/predict_phos.py" \
    --config_path "$DEMO_DIR/phosphorylation_st_config2.json" \
    --model_path "$MODEL_PATH" \
    --pos_file "$DEMO_DIR/Q08460.positions.csv" \
    --feature_h5 "$DEMO_DIR/Q08460esm_features.h5" \
    --pdb_dir "$DEMO_DIR" \
    --fasta_dir "$DEMO_DIR" \
    --out_dir "$OUT_DIR" \
    "$@"
