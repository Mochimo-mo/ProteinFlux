#!/usr/bin/env python3
"""Export phosphorylation model pre-classifier residue features for QT datasets."""

import argparse
import csv
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from extract_esm_features_optimized import ExtractionConfig, SequenceFeatureExtractor
from predict_phos import PredictNormalizer, adjust_cross_attention_hyperparams
from src.data.unified_data_processor import UnifiedDataProcessor, UnifiedPTMDataset
from src.models.phosphorylation_checkpoint_compat import (
    PhosphorylationCheckpointCompatPredictor,
    build_compat_config,
    load_phosphorylation_checkpoint,
)
from src.utils.common_utils import custom_collate_fn


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("extract_qt_phos_prelogit_features")


CANONICAL_RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")
RESIDUE_NORMALIZATION = {
    # Phosphorylated residues.
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
    "PSE": "S",
    # Protonation-state / force-field aliases.
    "HSD": "H",
    "HSE": "H",
    "HSP": "H",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "ASH": "D",
    "GLH": "E",
    "LYN": "K",
    "CYX": "C",
    "CYM": "C",
    # Common modified amino-acid names that preserve the parent residue.
    "MSE": "M",
    "SEC": "C",
    "PYL": "K",
}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    return obj




def normalize_residue(value: object) -> str:
    """Map PDB/force-field residue aliases to one-letter residues for ESM."""
    raw = "" if value is None else str(value).strip().upper()
    if not raw or raw == "NAN":
        return "X"
    if len(raw) == 1:
        return raw if raw in CANONICAL_RESIDUES else RESIDUE_NORMALIZATION.get(raw, "X")
    if raw in RESIDUE_NORMALIZATION:
        return RESIDUE_NORMALIZATION[raw]
    return "X"


def residue_name_to_one_letter(resname: str) -> str:
    resname = str(resname).strip().upper()
    return THREE_TO_ONE.get(resname) or RESIDUE_NORMALIZATION.get(resname, "X")


def pdb_ca_residues(pdb_path: Path) -> List[Tuple[str, int, str]]:
    residues = []
    seen = set()
    with pdb_path.open(errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain = line[21].strip() or "_"
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26].strip()
            resname = line[17:20].strip()
            key = (chain, resseq, icode, resname)
            if key in seen:
                continue
            seen.add(key)
            residues.append((chain, resseq, residue_name_to_one_letter(resname)))
    return residues


def write_fasta(path: Path, protein_id: str, sequence: str) -> None:
    with path.open("w") as handle:
        handle.write(f">{protein_id}\n")
        for start in range(0, len(sequence), 60):
            handle.write(sequence[start:start + 60] + "\n")


def build_pdb_based_qt_inputs(dataset_dir: Path, work_dir: Path) -> Tuple[Path, Path, Path]:
    pdb_dir = dataset_dir / "pdb"
    fasta_dir = work_dir / "pdb_based_fasta"
    esm_dir = work_dir / "pdb_based_esm_features"
    csv_path = work_dir / "pdb_based_merged_csv_files.csv"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    esm_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pdb_path in sorted(pdb_dir.glob("*.pdb")):
        protein_id = pdb_path.stem
        residues = pdb_ca_residues(pdb_path)
        if not residues:
            logger.warning("No CA residues found in %s", pdb_path)
            continue
        write_fasta(fasta_dir / f"{protein_id}.fasta", protein_id, "".join(r for _, _, r in residues))
        for chain, resseq, residue in residues:
            rows.append({"uniprot_id": protein_id, "residue": residue, "position": resseq, "chain": chain})

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, fasta_dir, esm_dir / "esm_features.h5"


def build_sequence_h5_from_fasta(
    *,
    fasta_dir: Path,
    h5_path: Path,
    output_dir: Path,
    gpu_id: int,
    batch_size: int,
) -> Path:
    fasta_paths = sorted(fasta_dir.glob("*.fasta"))
    if not fasta_paths:
        raise FileNotFoundError(f"No FASTA files found in {fasta_dir}")

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    config = ExtractionConfig(
        fasta_dir=str(fasta_dir),
        pdb_dir=None,
        output_dir=str(output_dir),
        mode="sequence",
        batch_size=batch_size,
        use_cache=False,
        force_recompute=True,
    )
    extractor = SequenceFeatureExtractor(config, gpu_id=gpu_id)

    with h5py.File(h5_path, "w") as handle:
        handle.create_group("metadata")
        proteins_group = handle.create_group("proteins")
        for fasta_path in tqdm(fasta_paths, desc="Dynamo PDB-based ESM"):
            protein_id = fasta_path.stem
            sequence = "".join(
                line.strip()
                for line in fasta_path.read_text().splitlines()
                if line and not line.startswith(">")
            )
            result = extractor.extract_features({"id": protein_id, "sequence": sequence})
            if result.sequence_features is None or result.sequence_mean is None:
                raise RuntimeError(f"Failed to extract ESM features for {protein_id}: {result.error_message}")
            group = proteins_group.create_group(protein_id)
            group.attrs["sequence"] = sequence
            group.create_dataset("sequence_features", data=result.sequence_features, compression="gzip")
            group.create_dataset("sequence_mean", data=result.sequence_mean)
            if result.sequence_cls is not None:
                group.create_dataset("sequence_cls", data=result.sequence_cls)
    return h5_path


def load_manifest_lengths(csv_path: Path, dataset_name: str) -> Dict[str, int]:
    manifest_path = csv_path.parent / f"{dataset_name}_manifest.csv"
    if not manifest_path.exists():
        return {}
    manifest = pd.read_csv(manifest_path)
    if "name" not in manifest.columns or "seqres_len" not in manifest.columns:
        return {}
    lengths = {}
    for _, row in manifest.iterrows():
        try:
            lengths[str(row["name"])] = int(row["seqres_len"])
        except Exception:
            continue
    return lengths


def write_alignment_summary(
    *,
    output_path: Path,
    dataset_name: str,
    original_df: pd.DataFrame,
    residue_feature_dict: Dict[str, np.ndarray],
    manifest_lengths: Dict[str, int],
) -> None:
    csv_lengths = original_df.groupby("uniprot_id", sort=False).size().to_dict()
    chain_counts = (
        original_df.assign(chain=original_df.get("chain", "").fillna("").astype(str))
        .groupby("uniprot_id")["chain"]
        .nunique()
        .to_dict()
        if "chain" in original_df.columns
        else {}
    )

    rows = []
    for system, csv_len in csv_lengths.items():
        system = str(system)
        feature_key = system
        feature_len: Optional[int] = None
        if system in residue_feature_dict:
            feature_len = int(residue_feature_dict[system].shape[0])
        elif int(chain_counts.get(system, 0)) == 1 and "chain" in original_df.columns:
            chain = str(original_df.loc[original_df["uniprot_id"].astype(str) == system, "chain"].iloc[0])
            candidate = f"{system}_{chain}" if chain and chain != "nan" else system
            if candidate in residue_feature_dict:
                feature_key = candidate
                feature_len = int(residue_feature_dict[candidate].shape[0])

        seqres_len = manifest_lengths.get(system)
        feature_vs_csv = None if feature_len is None else feature_len - int(csv_len)
        feature_vs_seqres = None if feature_len is None or seqres_len is None else feature_len - int(seqres_len)
        if feature_len is None:
            status = "missing_features"
            note = "No protein-level feature matrix was written for this system."
        elif feature_vs_csv == 0:
            status = "ok_structure_residue_aligned"
            note = "Feature length matches merged_csv_files.csv / resolved structure residues."
        else:
            status = "feature_csv_mismatch"
            note = "Feature length differs from resolved structure residue table."
        if status == "ok_structure_residue_aligned" and feature_vs_seqres not in (None, 0):
            note += " Difference from manifest seqres_len reflects unresolved SEQRES residues, not feature loss."

        rows.append(
            {
                "dataset": dataset_name,
                "system": system,
                "feature_key": feature_key,
                "feature_len": feature_len,
                "csv_resolved_residue_len": int(csv_len),
                "manifest_seqres_len": seqres_len,
                "delta_feature_vs_csv": feature_vs_csv,
                "delta_feature_vs_seqres": feature_vs_seqres,
                "status": status,
                "note": note,
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

def load_model(config_path: Path, model_path: Path, device: torch.device):
    with config_path.open("r") as handle:
        config = json.load(handle)
    adjust_cross_attention_hyperparams(config)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }

    config = build_compat_config(config, state_dict)
    model = PhosphorylationCheckpointCompatPredictor(config=config)
    load_phosphorylation_checkpoint(model, state_dict, logger=logger)
    model.to(device)
    model.eval()

    normalizer = None
    if isinstance(checkpoint, dict) and checkpoint.get("normalization_stats"):
        normalizer = PredictNormalizer(checkpoint["normalization_stats"])

    return model, config, normalizer


def prepare_dataframe(csv_path: Path, residues: Iterable[str], output_dir: Path, filter_residues: bool) -> Tuple[pd.DataFrame, Path]:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["__source_row_id"] = np.arange(len(df), dtype=np.int64)
    df["original_position"] = df["position"]
    df["original_residue"] = df["residue"].astype(str)
    df["residue"] = df["residue"].map(normalize_residue)

    # ESM HDF5 features are indexed by sequence order, while this CSV keeps the
    # original PDB residue number. Use a temporary sequence index for model
    # window extraction, and keep original_position for output metadata.
    df["position"] = df.groupby("uniprot_id").cumcount() + 1

    if "ptm_type" not in df.columns:
        df["ptm_type"] = "phosphorylation"

    if filter_residues:
        df = df[df["residue"].astype(str).str.upper().isin(set(residues))].copy()

    residue_tag = "all" if not filter_residues else "_".join(residues)
    temp_path = output_dir / f"temp_{csv_path.stem}_{residue_tag}.csv"
    df.to_csv(temp_path, index=False)
    return df, temp_path


def extract_for_model(
    *,
    dataset_name: str,
    csv_path: Path,
    feature_h5: Path,
    pdb_dir: Path,
    fasta_dir: Path,
    output_dir: Path,
    model_label: str,
    residues: List[str],
    config_path: Path,
    model_path: Path,
    device: torch.device,
    batch_size: int,
    micro_env_mode: str,
    max_batches: int = None,
    filter_residues: bool = False,
) -> Dict[str, object]:
    model_out_dir = output_dir / model_label
    model_out_dir.mkdir(parents=True, exist_ok=True)
    original_df = pd.read_csv(csv_path)
    original_df = original_df.copy()
    if "residue" in original_df.columns:
        original_df["original_residue"] = original_df["residue"].astype(str)
        original_df["residue"] = original_df["residue"].map(normalize_residue)

    filtered_df, temp_csv = prepare_dataframe(csv_path, residues, model_out_dir, filter_residues)
    if filtered_df.empty:
        logger.warning("%s %s: no residues to process", dataset_name, model_label)
        return {"processed": 0, "saved": 0, "output": None}

    model, config, normalizer = load_model(config_path, model_path, device)
    window_size = int(config.get("window_size", 73))
    local_window_size = int(config.get("local_window_size", 21))

    try:
        processor = UnifiedDataProcessor(
            data_path=str(temp_csv),
            esm_features_path=str(feature_h5),
            pdb_dir=str(pdb_dir),
            fasta_dir=str(fasta_dir),
            window_size=window_size,
            local_window_size=local_window_size,
            target_ptm_type="phosphorylation",
        )
        processed_data = processor.prepare_dataset()
    finally:
        temp_csv.unlink(missing_ok=True)

    if micro_env_mode == "zero":
        for item in processed_data:
            item["micro_env_features"] = np.zeros(6, dtype=np.float32)

    dataset = UnifiedPTMDataset(
        processed_data,
        fixed_window_size=window_size,
        fixed_local_size=local_window_size,
        target_ptm_type="phosphorylation",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    captured: List[torch.Tensor] = []

    def hook_fn(_module, inputs, _output):
        captured.append(inputs[0].detach().cpu())

    hook = model.classifier.register_forward_hook(hook_fn)
    rows = []
    feature_by_protein_chain: Dict[str, Dict[int, np.ndarray]] = {}
    feature_by_site: Dict[Tuple[int, str, str, int, str], np.ndarray] = {}
    cursor = 0

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"{dataset_name}:{model_label}")):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                if normalizer:
                    normalizer.normalize(batch)
                batch = to_device(batch, device)

                captured.clear()
                output = model(batch)
                logits = output["logits"] if isinstance(output, dict) else output
                probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                if not captured:
                    raise RuntimeError("Classifier hook did not capture prelogit features")
                features = captured[0].numpy()
                if features.shape[1] != 256:
                    raise RuntimeError(f"Expected 256-d features, got shape {features.shape}")

                batch_size_actual = features.shape[0]
                batch_items = processed_data[cursor: cursor + batch_size_actual]
                cursor += batch_size_actual

                for item, prob, feature in zip(batch_items, probs, features):
                    protein_id = str(item["uniprot_id"])
                    sequence_position = int(item["position"])
                    position = sequence_position
                    residue = str(item["residue"])
                    original_residue = residue
                    chain = ""
                    source_row_id = int(item.get("__source_row_id", -1))
                    if source_row_id >= 0:
                        if "chain" in original_df.columns:
                            chain = str(original_df.iloc[source_row_id]["chain"])
                            if chain == "nan":
                                chain = ""
                        if "position" in original_df.columns:
                            position = int(original_df.iloc[source_row_id]["position"])
                        if "original_residue" in original_df.columns:
                            original_residue = str(original_df.iloc[source_row_id]["original_residue"])

                    key = f"{protein_id}_{chain}" if chain else protein_id
                    feature_by_protein_chain.setdefault(key, {})[source_row_id] = feature.astype(np.float32, copy=False)
                    site_key = (source_row_id, protein_id, chain, position, residue)
                    feature_by_site[site_key] = feature.astype(np.float32, copy=False)
                    rows.append(
                        {
                            "source_row_id": source_row_id,
                            "uniprot_id": protein_id,
                            "chain": chain,
                            "position": position,
                            "sequence_position": sequence_position,
                            "residue": residue,
                            "original_residue": original_residue,
                            "model": model_label,
                            "probability": float(prob),
                        }
                    )
    finally:
        hook.remove()

    residue_feature_dict = {}
    for protein_key, pos_map in feature_by_protein_chain.items():
        ordered_positions = sorted(pos_map)
        residue_feature_dict[protein_key] = np.stack([pos_map[pos] for pos in ordered_positions], axis=0)

    # Single-chain FluxSite systems are commonly referenced by manifest name
    # without the chain suffix. Keep the chain-specific key and add a compatible
    # alias when it cannot collide with another chain.
    if "chain" in original_df.columns:
        chain_counts = original_df.groupby("uniprot_id")["chain"].nunique()
        for protein_id, chain_count in chain_counts.items():
            protein_id = str(protein_id)
            if int(chain_count) != 1 or protein_id in residue_feature_dict:
                continue
            chain = str(original_df.loc[original_df["uniprot_id"].astype(str) == protein_id, "chain"].iloc[0])
            chain_key = f"{protein_id}_{chain}" if chain and chain != "nan" else protein_id
            if chain_key in residue_feature_dict:
                residue_feature_dict[protein_id] = residue_feature_dict[chain_key]

    site_pkl = model_out_dir / f"{dataset_name}_{model_label}_site_features_256.pkl"
    protein_pkl = model_out_dir / f"{dataset_name}_{model_label}_residue_features_256.pkl"
    meta_csv = model_out_dir / f"{dataset_name}_{model_label}_feature_metadata.csv"
    npz_path = model_out_dir / f"{dataset_name}_{model_label}_features_256.npz"

    with site_pkl.open("wb") as handle:
        pickle.dump(feature_by_site, handle)
    with protein_pkl.open("wb") as handle:
        pickle.dump(residue_feature_dict, handle)
    pd.DataFrame(rows).to_csv(meta_csv, index=False)
    ordered_features = (
        np.stack([feature_by_site[(r["source_row_id"], r["uniprot_id"], r["chain"], int(r["position"]), r["residue"])] for r in rows], axis=0)
        if rows
        else np.empty((0, 256), dtype=np.float32)
    )
    np.savez_compressed(
        npz_path,
        features=ordered_features,
        source_row_id=np.array([r["source_row_id"] for r in rows], dtype=np.int64),
        probability=np.array([r["probability"] for r in rows], dtype=np.float32),
    )
    alignment_csv = model_out_dir / f"{dataset_name}_{model_label}_alignment_summary.csv"
    write_alignment_summary(
        output_path=alignment_csv,
        dataset_name=dataset_name,
        original_df=original_df,
        residue_feature_dict=residue_feature_dict,
        manifest_lengths=load_manifest_lengths(csv_path, dataset_name),
    )

    logger.info(
        "%s %s: saved %d features to %s",
        dataset_name,
        model_label,
        len(rows),
        model_out_dir,
    )
    return {
        "processed": len(processed_data),
        "saved": len(rows),
        "site_pkl": str(site_pkl),
        "protein_pkl": str(protein_pkl),
        "metadata_csv": str(meta_csv),
        "npz": str(npz_path),
        "alignment_csv": str(alignment_csv),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output_dir", type=Path, default=Path("QT/fluxsite_pkg/prelogit_features_256"))
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_env_mode", choices=["zero", "proxy"], default="zero")
    parser.add_argument("--datasets", nargs="+", choices=["bio_validation", "dynamo"], default=["bio_validation", "dynamo"])
    parser.add_argument("--models", nargs="+", choices=["phos_st", "phos_y"], default=["phos_st", "phos_y"])
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional debug limit per dataset/model")
    parser.add_argument("--filter_target_residues", action="store_true", help="Legacy mode: only export S/T for phos_st and Y for phos_y")
    args = parser.parse_args()

    base = args.base_dir.resolve()
    output_dir = (base / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info("Using device: %s", device)

    datasets = {
        "bio_validation": base / "QT/fluxsite_pkg/bio_validation",
        "dynamo": base / "QT/fluxsite_pkg/dynamo",
    }
    model_specs = [
        {
            "model_label": "phos_st",
            "residues": ["S", "T"],
            "config_path": base / "phosphorylation_st_config2.json",
            "model_path": base / "phos_st_data_ptms_output_Dual_attention_win63-3-25_trans8/pho_st_model.pt",
        },
        {
            "model_label": "phos_y",
            "residues": ["Y"],
            "config_path": base / "phosphorylation_y_config2.json",
            "model_path": base / "phos_y_data_ptms_output_Dual_attention_win127-2-12--trans8/best_model.pt",
        },
    ]

    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        try:
            with summary_path.open("r") as handle:
                summary = json.load(handle)
        except Exception:
            summary = {}
    else:
        summary = {}
    for dataset_name, dataset_dir in datasets.items():
        if dataset_name not in args.datasets:
            continue
        summary[dataset_name] = {}
        for spec in model_specs:
            if spec["model_label"] not in args.models:
                continue
            csv_path = dataset_dir / "merged_csv_files.csv"
            feature_h5 = dataset_dir / "esm_features/esm_features.h5"
            fasta_dir = dataset_dir / "fasta"
            if dataset_name == "dynamo":
                work_dir = output_dir / dataset_name / "pdb_based_inputs"
                csv_path, fasta_dir, feature_h5 = build_pdb_based_qt_inputs(dataset_dir, work_dir)
                if not feature_h5.exists():
                    feature_h5 = build_sequence_h5_from_fasta(
                        fasta_dir=fasta_dir,
                        h5_path=feature_h5,
                        output_dir=work_dir,
                        gpu_id=args.gpu_id,
                        batch_size=args.batch_size,
                    )

            result = extract_for_model(
                dataset_name=dataset_name,
                csv_path=csv_path,
                feature_h5=feature_h5,
                pdb_dir=dataset_dir / "pdb",
                fasta_dir=fasta_dir,
                output_dir=output_dir / dataset_name,
                device=device,
                batch_size=args.batch_size,
                micro_env_mode=args.micro_env_mode,
                max_batches=args.max_batches,
                filter_residues=args.filter_target_residues,
                **spec,
            )
            summary[dataset_name][spec["model_label"]] = result

    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
