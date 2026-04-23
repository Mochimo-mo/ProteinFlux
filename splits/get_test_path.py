"""
Match entries in test.csv against raw trajectory directories to build
test_inference_list.csv, which includes pdb_path, xtc_path, ptm_idx, and ptm_type.

Usage:
    python get_test_path.py --dirs /data/split_a /data/split_b --test_csv test.csv
"""
import argparse
import os
import re
import pandas as pd


PTM_TYPE_MAP = {'S': 'SEP', 'T': 'TPO', 'Y': 'PTR'}


def extract_info_from_filename(filename):
    """Parse PTM site and type from filenames like '1FPJ_244Y_chainA.pdb' -> (244, 'PTR')."""
    match = re.search(r'_(\d+)([STY])_', filename)
    if match:
        ptm_idx = int(match.group(1))
        ptm_char = match.group(2)
        return ptm_idx, PTM_TYPE_MAP.get(ptm_char, 'SEP')
    return None, None


def get_master_data(trajectory_dirs):
    """Scan all directories and build a DataFrame of all available trajectories."""
    all_records = []
    for directory in trajectory_dirs:
        if not os.path.exists(directory):
            print(f"Warning: directory not found, skipping: {directory}")
            continue
        for f in os.listdir(directory):
            if not f.endswith('.pdb'):
                continue
            ptm_idx, ptm_type = extract_info_from_filename(f)
            if ptm_idx is None:
                continue
            name_without_ext = f.replace('.pdb', '')
            pdb_path = os.path.join(directory, f)
            xtc_path = os.path.join(directory, f.replace('.pdb', '.xtc'))
            all_records.append({
                'name': name_without_ext,
                'pdb_path': pdb_path,
                'xtc_path': xtc_path if os.path.exists(xtc_path) else None,
                'ptm_idx': ptm_idx,
                'ptm_type': ptm_type,
            })
    return pd.DataFrame(all_records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build test inference list from raw trajectory directories.")
    parser.add_argument('--dirs', nargs='+', required=True, help='One or more raw data directories to scan')
    parser.add_argument('--test_csv', default='test.csv', help='Path to test.csv (default: test.csv)')
    parser.add_argument('--output', default='test_inference_list.csv', help='Output CSV path')
    args = parser.parse_args()

    try:
        test_df = pd.read_csv(args.test_csv)
        print(f"Loaded {args.test_csv}: {len(test_df)} rows")
    except FileNotFoundError:
        print(f"Error: {args.test_csv} not found.")
        raise SystemExit(1)

    print("Scanning trajectory directories...")
    master_df = get_master_data(args.dirs)

    if 'name' not in test_df.columns:
        print("Error: test.csv has no 'name' column.")
        raise SystemExit(1)

    final_test_list = pd.merge(test_df[['name']], master_df, on='name', how='inner')
    final_test_list.to_csv(args.output, index=False)

    print(f"Test entries: {len(test_df)}")
    print(f"Matched: {len(final_test_list)}")
    if len(final_test_list) < len(test_df):
        print(f"Warning: {len(test_df) - len(final_test_list)} entries not found in the given directories.")
    print(f"Saved to: {args.output}")
