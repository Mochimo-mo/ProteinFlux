"""
scripts/inference.py
--------------------
Multi-GPU, multi-replica trajectory inference for ProteinFlux.

Usage:
    python scripts/inference.py --model esm2_multi --eval_dataset atlas
    python scripts/inference.py --model ptm_dynamo --eval_dataset dynamo

Multi-GPU (torchrun):
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \\
        scripts/inference.py --model esm2_multi --eval_dataset atlas
"""

import argparse
import sys
import os

# ════════════════════════════════════════════════════════════════════════════
# Model config: one entry per trained model
# ════════════════════════════════════════════════════════════════════════════

MODEL_DEFAULTS = {
    "esm2_multi": {  # Pretrained dynamics model: ESM2 trained on mixed ATLAS + MoDEL
        "ckpt":          "/path/to/data/workdir/esm2_multi_atlas_model1393/checkpoints/best-epoch=499-val_loss=0.3083.ckpt",
        "desc":          "ESM2 trained on ATLAS + MoDEL (multi-dataset)",
        # Note: use the ESM2 pkl matching the eval dataset — default atlas; for
        # MoDEL eval, override with --esm2_emb_path model_esm2_embeddings.pkl
        # (handled by the run script).
        "esm2_emb_path": "/path/to/data/atlas_esm2_embeddings.pkl",
    },
    "ptm_dynamo": {  # PTM model: esm2_multi base + PTM channel + FluxSite (prelogit-256), full finetune
        "ckpt":          "/path/to/data/workdir/ptm_dynamo_esm2_prelogit256_full/checkpoints/best-epoch=060-val_loss=0.2561.ckpt",
        "desc":          "ESM2 base + PTM channel + FluxSite (prelogit-256), full finetune",
        "esm2_emb_path": "/path/to/data/dynamo_phos_esm2_embeddings.pkl",
        "ptm_feat_path": "/path/to/data/dynamo_prelogit_features_256_byhead.pkl",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# Eval dataset config: one entry per test set
# ════════════════════════════════════════════════════════════════════════════

EVAL_DEFAULTS = {
    "atlas": {
        "data_dir":   "/path/to/data/atlas_100ps.h5",          # TODO: path to ATLAS HDF5
        "split":      "splits/atlas/test.csv",
        "h5_key_fmt": "{name}_R1",
    },
    "model": {  # MoDEL single-chain dataset (no replicas, one trajectory per protein)
        "data_dir":   "/path/to/data/data_model_final.h5",          # TODO: path to MoDEL HDF5
        "split":      "splits/model/test.csv",
        "h5_key_fmt": "{name}",
    },
    "dynamo": {  # PTM phospho dataset (H5 in group format, keys already carry _R{1,2,3})
        "data_dir":   "/path/to/data/ptm_data.h5",          # TODO: path to DynaMo-phos HDF5
        "split":      "splits/dynamo_phos/test.csv",
        "h5_key_fmt": "{name}",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# Default output root directory
# ════════════════════════════════════════════════════════════════════════════

OUT_DIR_BASE = "./inference"

# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()

# Core selection
parser.add_argument("--model",        choices=list(MODEL_DEFAULTS.keys()), required=True)
parser.add_argument("--eval_dataset", choices=list(EVAL_DEFAULTS.keys()),  required=True)

# Path overrides (fall back to the defaults above if unset)
parser.add_argument("--ckpt",          default=None)
parser.add_argument("--data_dir",      default=None)
parser.add_argument("--split",         default=None)
parser.add_argument("--out_dir",       default=None)
parser.add_argument("--esm2_emb_path", default=None,
                    help="override esm2_emb_path in MODEL_DEFAULTS")
parser.add_argument("--ptm_feat_path", default=None,
                    help="per-residue PTM feature pkl, e.g. FluxSite (overrides MODEL_DEFAULTS)")
parser.add_argument("--ptm_feat_dim", type=int, default=256,
                    help="PTM feature dim (must match training, FluxSite=256)")

# Inference parameters
parser.add_argument("--h5_key_fmt",   default=None,
                    help="override h5_key_fmt; falls back to EVAL_DEFAULTS if unset")
parser.add_argument("--num_frames",   type=int, default=100)
parser.add_argument("--num_replicas", type=int, default=5)
parser.add_argument("--seed_base",    type=int, default=42)
parser.add_argument("--pdb_id",       nargs="*", default=[])
parser.add_argument("--sde",          action="store_true", help="use SDE stochastic sampling instead of ODE (ablation)")
parser.add_argument("--churn",        type=float, default=1.0, help="SDE noise strength g=churn·σ_t; 0=ODE")
parser.add_argument("--sde_steps",    type=int, default=100, help="SDE Euler-Maruyama step count")
parser.add_argument("--skip_existing", action="store_true")
parser.add_argument("--no_ema", action="store_true",
                    help="disable applying EMA weights; run inference on raw weights (default uses EMA, matching training val)")
parser.add_argument(
    "--ptm_ablation",
    choices=["none", "zero", "wrong_site"],
    default="none",
    help=(
        "PTM condition ablation: zero removes the explicit PTM channel signal; "
        "wrong_site moves the same PTM type to a deterministic incorrect site. "
        "Sequence geometry and the initial structure are unchanged."
    ),
)
parser.add_argument("--xtc",          action="store_true")
parser.add_argument("--failed_log",   default="failed.txt")
parser.add_argument("--suffix",       default="",
                    help="appended to the output directory name to distinguish multiple inference runs of the same model")

args = parser.parse_args()

# ── Fill unspecified paths with defaults ─────────────────────────────────────
_model_cfg = MODEL_DEFAULTS[args.model]
_eval_cfg  = EVAL_DEFAULTS[args.eval_dataset]

if args.ckpt          is None: args.ckpt          = _model_cfg["ckpt"]
if args.data_dir      is None: args.data_dir      = _eval_cfg["data_dir"]
if args.split         is None: args.split         = _eval_cfg["split"]
if args.h5_key_fmt    is None: args.h5_key_fmt    = _eval_cfg["h5_key_fmt"]
if args.esm2_emb_path is None: args.esm2_emb_path = _model_cfg.get("esm2_emb_path")
if args.ptm_feat_path is None: args.ptm_feat_path = _model_cfg.get("ptm_feat_path")
if args.out_dir       is None:
    suffix = f"_{args.suffix}" if args.suffix else ""
    args.out_dir = os.path.join(
        OUT_DIR_BASE, args.model, f"inference_{args.eval_dataset}{suffix}"
    )

# ── Auto-locate ckpt: workdir/{model}/checkpoints/best-*.ckpt ──────────────────
if not args.ckpt:
    import glob, re
    pattern = os.path.join("workdir", args.model, "checkpoints", "best-*.ckpt")
    candidates = glob.glob(pattern)
    if candidates:
        def _val_loss(p):
            m = re.search(r'val_loss=(\d+\.\d+)', os.path.basename(p))
            return float(m.group(1)) if m else float('inf')
        args.ckpt = min(candidates, key=_val_loss)
        print(f"  [auto] ckpt → {args.ckpt}")
    else:
        print(f"❌  checkpoint not found: {pattern}")
        print(f"    set the path in MODEL_DEFAULTS, or specify it with --ckpt.")
        sys.exit(1)

if not args.data_dir or not args.split:
    print(f"❌  data_dir / split not set for dataset '{args.eval_dataset}'!")
    sys.exit(1)
if not args.data_dir or not args.split:
    print(f"❌  data_dir / split not set for dataset '{args.eval_dataset}'!")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"Model        : {args.model}  ({_model_cfg['desc']})")
print(f"Eval dataset : {args.eval_dataset}")
print(f"ckpt         : {args.ckpt}")
print(f"data_dir     : {args.data_dir}")
print(f"split        : {args.split}")
print(f"out_dir      : {args.out_dir}")
print(f"num_frames   : {args.num_frames}")
print(f"num_replicas : {args.num_replicas}")
print(f"PTM ablation : {args.ptm_ablation}")
print(f"{'='*60}\n")

# ════════════════════════════════════════════════════════════════════════════
import csv
import gc
import time
from collections import deque
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch

torch.serialization.add_safe_globals([argparse.Namespace])
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.makedirs(args.out_dir, exist_ok=True)

# ── ESM2 embeddings (loaded once, shared across all proteins) ─────────────────
ESM2_EMBEDDINGS = None
if args.esm2_emb_path:
    import pickle
    with open(args.esm2_emb_path, "rb") as _f:
        ESM2_EMBEDDINGS = pickle.load(_f)
    print(f"  [ESM2] loaded {len(ESM2_EMBEDDINGS)} embeddings from {args.esm2_emb_path}")

# ── FluxSite / PTM features (loaded once) — needed to reproduce the AA+FluxSite
#    conditioning at inference (otherwise the ptm_adapter signal is missing). ──
PTM_FEATURES = None
if args.ptm_feat_path:
    import pickle
    with open(args.ptm_feat_path, "rb") as _f:
        PTM_FEATURES = pickle.load(_f)
    print(f"  [PTM feat] loaded {len(PTM_FEATURES)} features from {args.ptm_feat_path}")


# ── Progress / ETA ────────────────────────────────────────────────────────────

def _fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class ETATracker:
    """Rolling-average ETA tracker.  Call .update(duration) after each job."""

    def __init__(self, total: int, window: int = 20):
        self.total    = total
        self.done     = 0
        self.t_start  = time.time()
        self._window  = deque(maxlen=window)   # recent job durations

    def update(self, duration: float):
        if duration > 0:
            self._window.append(duration)
        self.done += 1

    def _eta(self) -> float:
        remaining = self.total - self.done
        if not self._window or remaining <= 0:
            return 0.0
        return np.mean(self._window) * remaining

    def line(self, rank: int = 0) -> str:
        elapsed = time.time() - self.t_start
        pct     = self.done / max(self.total, 1) * 100
        return (
            f"[rank{rank}] {self.done}/{self.total} jobs  "
            f"({pct:.1f}%)  elapsed {_fmt_eta(elapsed)}  "
            f"ETA {_fmt_eta(self._eta())}"
        )


# ── Distributed helpers ───────────────────────────────────────────────────────

def get_dist_info():
    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, local_rank, world_size


# ── Data loading ──────────────────────────────────────────────────────────────

UNKNOWN_AA = {"X": "G", "U": "C", "B": "D", "Z": "E", "O": "K"}


def clean_seqres(s: str) -> str:
    cleaned  = "".join(UNKNOWN_AA.get(c, c) for c in s)
    replaced = [(c, UNKNOWN_AA[c]) for c in s if c in UNKNOWN_AA]
    if replaced:
        print(f"  ⚠️  Replaced non-standard AA: {replaced}")
    return cleaned


def load_item(h5file, name: str, seqres_str: str) -> dict:
    """Load the first H5 frame and convert to a tensor dict (no batch dim)."""
    from core.geometry import atom14_to_atom37, atom14_to_frames, atom37_to_torsions
    from core.residue_constants import restype_order
    from core.ptm_h5 import read_structure
    from core.ptm_utils import map_ptm_to_parent
    from core.ptm_ablation import ptm_condition_ids, ptm_condition_aatype

    # Resolve H5 key (with replica fallback)
    h5_key = args.h5_key_fmt.format(name=name)
    if h5_key not in h5file:
        for sfx in ("_R1", "_R2", "_R3"):
            if (name + sfx) in h5file:
                h5_key = name + sfx
                break
        else:
            raise KeyError(f"Key '{name}' (tried '{h5_key}') not found in H5 file.")

    # Read structure; PTM group → residue tokens + H5 seqres (carries 21/22/23
    # so the model's ptm_channel activates). Plain dataset → use csv seqres.
    coords_full, ptm_seq = read_structure(h5file[h5_key])
    if ptm_seq is not None:
        seqres_idx = ptm_seq
    else:
        seqres_str = clean_seqres(seqres_str)
        seqres_idx = np.array([restype_order.get(c, restype_order["G"]) for c in seqres_str])

    arr = np.copy(coords_full[0:1]).astype(np.float32)  # [1, L, 14, 3]

    # Align lengths
    h5_len, seq_len = arr.shape[1], len(seqres_idx)
    if h5_len != seq_len:
        mn = min(h5_len, seq_len)
        arr        = arr[:, :mn]
        seqres_idx = seqres_idx[:mn]
        print(f"  ⚠️  {name}: length mismatch h5={h5_len} seq={seq_len}, truncated to {mn}")

    frames   = atom14_to_frames(torch.from_numpy(arr))
    seqres_t = torch.from_numpy(seqres_idx.astype(np.int64))

    aatype_geom = map_ptm_to_parent(seqres_t)

    atom37             = torch.from_numpy(atom14_to_atom37(arr, aatype_geom[None])).float()
    torsions, tor_mask = atom37_to_torsions(atom37, aatype_geom[None])

    result = {
        "name":         name,
        "trans":        frames._trans,           # [1, L, 3]
        "rots":         frames._rots._rot_mats,  # [1, L, 3, 3]
        "torsions":     torsions,                # [1, L, 7, 2]
        "torsion_mask": tor_mask[0],             # [L, 7]
        "seqres":       seqres_t,                # [L]
        "mask":         torch.ones(frames.shape[1]),
    }

    if args.ptm_ablation != "none":
        ptm_ids, moves = ptm_condition_ids(seqres_idx, mode=args.ptm_ablation)
        result["ptm_ids_override"] = torch.from_numpy(ptm_ids)
        # AA-base models carry PTM via seqres tokens (21/22/23 → aatype_to_emb),
        # not the ptm_channel — also override the conditioning aatype. Geometry /
        # initial structure keep the real seqres (result["seqres"] unchanged).
        aa_cond = ptm_condition_aatype(seqres_idx, mode=args.ptm_ablation)
        result["aatype_override"] = torch.from_numpy(aa_cond.astype(np.int64))
        if args.ptm_ablation == "zero":
            print(f"  [PTM ablation] {name}: explicit PTM condition removed")
        else:
            move_text = ", ".join(
                f"id{ptm_id}:{source + 1}->{destination + 1}({selection_kind})"
                for source, destination, ptm_id, selection_kind in moves
            )
            print(f"  [PTM ablation] {name}: wrong-site condition {move_text}")

    if ESM2_EMBEDDINGS is not None:
        import re
        base_name = re.sub(r'_R\d+$', '', name)
        candidates = [name, base_name, name.lower(), base_name.lower()]
        emb_raw = next((ESM2_EMBEDDINGS[k] for k in candidates if k in ESM2_EMBEDDINGS), None)
        if emb_raw is None:
            raise KeyError(f"ESM2 embedding not found for '{name}'. Tried: {candidates}")
        emb = torch.from_numpy(emb_raw).float() if isinstance(emb_raw, np.ndarray) else emb_raw
        result["esm2_emb"] = emb[:seqres_t.shape[0]]  # [L, esm2_dim]

    if PTM_FEATURES is not None:
        import re
        base_name = re.sub(r'_R\d+$', '', name, flags=re.IGNORECASE)
        cands = [name, base_name, name.upper(), base_name.upper(), name.lower(), base_name.lower()]
        feat_raw = next((PTM_FEATURES[k] for k in cands if k in PTM_FEATURES), None)
        if feat_raw is None:
            raise KeyError(f"PTM feature not found for '{name}'. Tried: {cands}")
        feat = torch.from_numpy(np.asarray(feat_raw)).float()
        result["ptm_feat"] = feat[:seqres_t.shape[0]]  # [L, ptm_feat_dim]

    return result


def to_batch(item: dict, device: torch.device) -> dict:
    """Add batch dim, move to device, expand T=1 → T=num_frames."""
    batch = {}
    for k, v in item.items():
        if k == "name":
            continue
        batch[k] = (v if isinstance(v, torch.Tensor) else torch.from_numpy(v)).unsqueeze(0).to(device)

    T = args.num_frames
    batch["trans"]    = batch["trans"].expand(-1, T, -1, -1).contiguous()
    batch["rots"]     = batch["rots"].expand(-1, T, -1, -1, -1).contiguous()
    batch["torsions"] = batch["torsions"].expand(-1, T, -1, -1, -1).contiguous()

    if "esm2_emb" in item:
        batch["esm2_emb"] = item["esm2_emb"].unsqueeze(0).to(device)  # [1, L, esm2_dim]

    return batch


# ── Inference ─────────────────────────────────────────────────────────────────

def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@torch.no_grad()
def do_one_replica(model, item: dict, name: str, rep_idx: int) -> tuple:
    from core.io_utils import atom14_to_pdb
    from core.pdb_utils import is_valid_pdb

    out_path = os.path.join(args.out_dir, f"{name}_rep{rep_idx}.pdb")
    if args.skip_existing and is_valid_pdb(out_path):
        print(f"    ⏭️  rep{rep_idx} already exists, skipping")
        return True, None, 0.0
    if args.skip_existing and os.path.exists(out_path):
        print(f"    ♻️  rep{rep_idx} exists but is empty/incomplete; regenerating")

    seed = args.seed_base + rep_idx
    set_all_seeds(seed)

    device = next(model.parameters()).device
    batch  = to_batch(item, device)

    t0        = time.time()
    atom14, _ = model.inference(batch)
    duration  = time.time() - t0

    atom14_to_pdb(atom14[0].cpu().numpy(), item["seqres"].cpu().numpy(), out_path)

    if args.xtc:
        import mdtraj
        traj = mdtraj.load(out_path)
        traj.superpose(traj)
        traj.save(os.path.join(args.out_dir, f"{name}_rep{rep_idx}.xtc"))
        traj[0].save(out_path)

    gc.collect()
    torch.cuda.empty_cache()
    print(f"    ✓ rep{rep_idx} done  ({duration:.1f}s, seed={seed})")
    return True, None, duration


# ── Main ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    rank, local_rank, world = get_dist_info()
    is_main = (rank == 0)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    from trainer.module import ProteinWrapper
    from core.split_utils import filter_short_sequences
    if is_main:
        print(f"Loading checkpoint: {args.ckpt}")
    model = ProteinWrapper.load_from_checkpoint(args.ckpt, map_location=device)
    # SDE sampling ablation: override model.args (inference() reads self.args). churn=0 is equivalent to ODE.
    model.args.sde = args.sde
    model.args.churn = args.churn
    model.args.sde_steps = args.sde_steps
    if is_main and args.sde:
        print(f"  [SDE] stochastic sampling: churn={args.churn} steps={args.sde_steps}")
    # Apply EMA weights: during training, val_loss / best-checkpoint are computed
    # with EMA weights, but the checkpoint's model.state_dict() stores raw weights.
    # Without switching to EMA you'd use the single-epoch raw weights (noisier),
    # inconsistent with the criterion used to select the checkpoint.
    if getattr(model.args, 'ema', False) and not args.no_ema:
        model.ema.to(device)
        model.load_ema_weights()
        if is_main:
            print("  [EMA] applied EMA weights for inference")
    elif is_main:
        print("  [EMA] using RAW weights (ema off or --no_ema)")
    model.eval().to(device)

    df = pd.read_csv(args.split, index_col="name")
    df = filter_short_sequences(
        df, source=args.split, report=is_main,
    )
    if df.empty:
        raise ValueError(f"No valid protein sequences remain in split: {args.split}")
    names = [n for n in df.index if not args.pdb_id or n in args.pdb_id]
    my_names = names[rank::world]

    if is_main:
        print(f"Total proteins : {len(names)}  (GPUs: {world}, this rank: {len(my_names)})")
        print(f"Total jobs     : {len(names) * args.num_replicas}\n")

    timing_csv = os.path.join(args.out_dir, f"timing_rank{rank}.csv")
    with open(timing_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "index", "name", "length", "replica", "seed",
            "start_time", "end_time", "duration_seconds", "status", "error",
        ])

    stats          = {"total": 0, "success": 0, "failed": 0}
    failed_entries = []
    total_jobs     = len(my_names) * args.num_replicas
    tracker        = ETATracker(total_jobs)

    with h5py.File(args.data_dir, "r") as h5:
        for prot_idx, name in enumerate(my_names, 1):
            seqres  = str(df.loc[name, "seqres"])
            seq_len = len(seqres)
            print(f"\n[rank{rank}] [{prot_idx}/{len(my_names)} proteins | "
                  f"{tracker.done}/{total_jobs} jobs] {name}  (L={seq_len})")

            try:
                item = load_item(h5, name, seqres)
            except Exception as e:
                print(f"  [skip] load failed: {e}")
                for rep_i in range(args.num_replicas):
                    stats["total"]  += 1
                    stats["failed"] += 1
                    tracker.update(0.0)
                    with open(timing_csv, "a", newline="") as f:
                        csv.writer(f).writerow([
                            prot_idx, name, seq_len, rep_i, args.seed_base + rep_i,
                            "", "", 0.0, "Failed", str(e),
                        ])
                failed_entries.append({"name": name, "error": str(e),
                                       "length": seq_len, "replica": -1})
                continue

            oom_hit = False
            for rep_i in range(args.num_replicas):
                if oom_hit:
                    break

                stats["total"] += 1
                seed     = args.seed_base + rep_i
                start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                t0       = time.time()

                try:
                    success, error_msg, dur = do_one_replica(model, item, name, rep_i)
                except torch.cuda.OutOfMemoryError as e:
                    success, error_msg, dur = False, f"CUDA OOM: {e}", 0.0
                    oom_hit = True
                    gc.collect(); torch.cuda.empty_cache()
                    print(f"  ⚠️  OOM — skipping remaining replicas for {name}")
                except Exception as e:
                    success, error_msg, dur = False, str(e), 0.0
                    print(f"  ✗ rep{rep_i} failed: {type(e).__name__}: {e}")

                tracker.update(dur)
                print(f"  >> {tracker.line(rank)}")

                with open(timing_csv, "a", newline="") as f:
                    csv.writer(f).writerow([
                        prot_idx, name, seq_len, rep_i, seed,
                        start_dt, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        round(time.time() - t0, 2),
                        "Success" if success else "Failed",
                        error_msg or "",
                    ])

                if success:
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                    failed_entries.append({"name": name, "error": error_msg,
                                           "length": seq_len, "replica": rep_i})

    if failed_entries:
        failed_path = os.path.join(args.out_dir, f"rank{rank}_{args.failed_log}")
        with open(failed_path, "w") as f:
            f.write("Failed Proteins\n" + "=" * 60 + "\n\n")
            for e in failed_entries:
                f.write(
                    f"Name:    {e['name']}\n"
                    f"Length:  {e['length']}\n"
                    f"Replica: {e['replica']}\n"
                    f"Error:   {e['error']}\n"
                    + "-" * 60 + "\n\n"
                )

    timing_df  = pd.read_csv(timing_csv)
    total_time = timing_df["duration_seconds"].sum()

    print(f"\n{'='*60}")
    print(f"[rank{rank}]  success={stats['success']}  "
          f"failed={stats['failed']}  "
          f"total_time={total_time/3600:.2f}h  "
          f"avg={total_time/max(stats['total'],1):.1f}s/job")
    print(f"{'='*60}")

    if stats["failed"]:
        failed_path = os.path.join(args.out_dir, f"rank{rank}_{args.failed_log}")
        raise SystemExit(
            f"[rank{rank}] inference failed for {stats['failed']}/{stats['total']} jobs; "
            f"see {failed_path}"
        )


if __name__ == "__main__":
    main()
