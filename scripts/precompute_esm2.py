"""Pre-compute per-residue ESM2 embeddings for all proteins in the dataset splits.

Usage:
    python scripts/precompute_esm2.py \
        --fasta_dir /path/to/fastas \
        --splits splits/train.csv splits/val.csv splits/test.csv \
        --output esm2_embeddings.pkl \
        --model esm2_t33_650M_UR50D \
        --batch_size 4 \
        --device cuda

Output pkl format:
    { protein_name (str): np.ndarray [L, esm2_dim] }
"""

import os
import argparse
import pickle
import numpy as np
from pathlib import Path

import torch
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_dir', type=str, required=True,
                        help='Directory containing per-protein FASTA files (<name>.fasta) '
                             'OR a single multi-sequence FASTA file.')
    parser.add_argument('--splits', type=str, nargs='+', required=True,
                        help='One or more CSV split files with a "name" index and "seqres" column.')
    parser.add_argument('--output', type=str, default='esm2_embeddings.pkl',
                        help='Output pkl file path.')
    parser.add_argument('--model', type=str, default='esm2_t33_650M_UR50D',
                        choices=[
                            'esm2_t6_8M_UR50D',
                            'esm2_t12_35M_UR50D',
                            'esm2_t30_150M_UR50D',
                            'esm2_t33_650M_UR50D',
                            'esm2_t36_3B_UR50D',
                        ],
                        help='ESM2 model variant. Larger = better but slower.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of proteins per forward pass.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--repr_layer', type=int, default=None,
                        help='ESM2 layer to extract representations from. '
                             'Defaults to the last layer.')
    return parser.parse_args()


def load_sequences_from_splits(split_files):
    """Return {protein_name: sequence_str} from all CSV split files."""
    sequences = {}
    for path in split_files:
        df = pd.read_csv(path, index_col='name')
        for name, row in df.iterrows():
            if name not in sequences:
                sequences[str(name)] = str(row['seqres'])
    print(f"Found {len(sequences)} unique proteins across {len(split_files)} split file(s).")
    return sequences


def compute_esm2_embeddings(sequences, model_name, batch_size, device, repr_layer=None):
    """Run ESM2 and return {protein_name: np.ndarray [L, esm2_dim]}."""
    try:
        import esm
    except ImportError:
        raise ImportError("ESM not installed. Run: pip install fair-esm")

    print(f"Loading ESM2 model: {model_name}")
    model, alphabet = esm.pretrained.__dict__[model_name]()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    if repr_layer is None:
        repr_layer = model.num_layers
    print(f"Extracting representations from layer {repr_layer} / {model.num_layers}")

    names = list(sequences.keys())
    results = {}

    for i in range(0, len(names), batch_size):
        batch_names = names[i: i + batch_size]
        batch_data = [(n, sequences[n]) for n in batch_names]

        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            out = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)

        token_representations = out['representations'][repr_layer]

        for j, name in enumerate(batch_names):
            seq_len = len(sequences[name])
            # ESM2 prepends <cls> and appends <eos>; slice [1:1+seq_len] to remove them
            emb = token_representations[j, 1: 1 + seq_len].cpu().numpy()
            results[name] = emb

        done = min(i + batch_size, len(names))
        print(f"  {done}/{len(names)} proteins processed", end='\r')

    print()
    return results


def main():
    args = parse_args()

    sequences = load_sequences_from_splits(args.splits)

    embeddings = compute_esm2_embeddings(
        sequences,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        repr_layer=args.repr_layer,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    sample_name = next(iter(embeddings))
    sample_dim = embeddings[sample_name].shape[-1]
    print(f"Saved {len(embeddings)} embeddings → {output_path}")
    print(f"ESM2 dim: {sample_dim}  (pass --esm2_dim {sample_dim} to train.py)")


if __name__ == '__main__':
    main()
