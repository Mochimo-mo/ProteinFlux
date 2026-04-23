import argparse
import pandas as pd
import numpy as np
import random
from Bio import SeqIO
import os


def split_and_export_csv(fasta_path, cluster_tsv, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Cluster-based dataset splitting. Reads sequences from FASTA and exports
    train/val/test CSVs with columns: name, seqres."""
    random.seed(seed)
    np.random.seed(seed)

    print(f"Reading sequences: {fasta_path}")
    seq_dict = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        clean_id = str(record.id).replace('.pdb', '').split()[0].strip()
        seq_dict[clean_id] = str(record.seq)
    print(f"  Loaded {len(seq_dict)} sequences.")

    print(f"Reading cluster results: {cluster_tsv}")
    df_cluster = pd.read_csv(cluster_tsv, sep='\t', names=['representative', 'member'])
    df_cluster['representative'] = df_cluster['representative'].astype(str).str.replace('.pdb', '', regex=False).str.split().str[0]
    df_cluster['member'] = df_cluster['member'].astype(str).str.replace('.pdb', '', regex=False).str.split().str[0]

    cluster_groups = df_cluster.groupby('representative')['member'].apply(list).to_dict()
    all_reps = list(cluster_groups.keys())
    random.shuffle(all_reps)

    num_clusters = len(all_reps)
    train_end = int(num_clusters * train_ratio)
    val_end = train_end + int(num_clusters * val_ratio)

    partitions = {
        'train': all_reps[:train_end],
        'val': all_reps[train_end:val_end],
        'test': all_reps[val_end:]
    }

    print("\nExporting CSV splits...")
    for split_name, reps in partitions.items():
        rows = []
        for rep in reps:
            for member in cluster_groups[rep]:
                if member in seq_dict:
                    rows.append({'name': member, 'seqres': seq_dict[member]})

        df_out = pd.DataFrame(rows)
        output_name = f"{split_name}.csv"
        df_out.to_csv(output_name, index=False)

        print(f"  {split_name}: {len(reps)} clusters, {len(df_out)} records -> {output_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split PTM dataset by MMseqs2 clusters.")
    parser.add_argument('--fasta', required=True, help='Path to the FASTA file with all sequences')
    parser.add_argument('--cluster_tsv', required=True, help='Path to MMseqs2 cluster TSV result')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    split_and_export_csv(
        fasta_path=args.fasta,
        cluster_tsv=args.cluster_tsv,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
