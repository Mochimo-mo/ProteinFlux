#!/usr/bin/env python3
"""
ProteinFlux phospho-conditioned ensemble demo
=============================================

Generates two conformational ensembles for the same starting structure --
one without the phosphorylation condition, one with it -- and reports how the
per-residue flexibility changes around the phospho site.

System: human TAF9 (TFIID subunit 9, UniProt Q16594, chain from PDB 6F3T),
116 residues, phospho site pSer8.

    python demo/run_phospho_demo.py --ckpt /path/to/proteinflux_ptm.ckpt

Add --controls to also run the two negative controls reported in the paper.
Both drop the FluxSite site features and keep only the phospho token, which
isolates what the label itself does:

    token_only  phospho token at the real site, no site features
    wrong_site  phospho token moved to a different Ser/Thr/Tyr

Outputs: one multi-model PDB per replica per arm, under demo/out/<arm>/.
Requires only torch + numpy (no mdtraj / matplotlib).
"""

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")

NAME = "6F3T_TAF9"
PDB = os.path.join(DATA, f"{NAME}.pdb")
ESM2_PKL = os.path.join(DATA, f"{NAME}_esm2.pkl")
FLUXSITE_PKL = os.path.join(DATA, f"{NAME}_fluxsite256.pkl")

SITE = "S8"          # UniProt Q16594 numbering, as written in the demo PDB
SITE_SEQ_IDX = 3     # same residue, 0-based index into the sequence
WINDOW = 3           # +/- residues around the site for the local average


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True,
                   help="ProteinFlux PTM checkpoint (see demo/README.md for the download)")
    p.add_argument("--num_replicas", type=int, default=5,
                   help="trajectories per arm (default 5, one per seed)")
    p.add_argument("--num_frames", type=int, default=100,
                   help="frames per trajectory (default 100)")
    p.add_argument("--seed_base", type=int, default=42,
                   help="arms are paired: the same seeds are used in every arm")
    p.add_argument("--out_dir", default=os.path.join(HERE, "out"))
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--controls", action="store_true",
                   help="also run the zero / wrong_site negative controls")
    p.add_argument("--skip_existing", action="store_true",
                   help="reuse trajectories that are already on disk")
    return p.parse_args()


# ── running one arm ─────────────────────────────────────────────────────────
def run_arm(arm, args, phospho=None, ptm_feat=None, ablation="none"):
    out_dir = os.path.join(args.out_dir, arm)
    cmd = [
        sys.executable, os.path.join(HERE, "inference_single.py"),
        "--pdb", PDB, "--name", NAME, "--ckpt", args.ckpt,
        "--esm2_pkl", ESM2_PKL,
        "--num_replicas", str(args.num_replicas),
        "--num_frames", str(args.num_frames),
        "--seed_base", str(args.seed_base),
        "--out_dir", out_dir,
        "--no_eval",                      # the demo does its own RMSF below
    ]
    if phospho:
        cmd += ["--phospho", phospho]
    if ptm_feat:
        cmd += ["--ptm_feat_pkl", ptm_feat]
    if ablation != "none":
        cmd += ["--ptm_ablation", ablation]
    if args.device:
        cmd += ["--device", args.device]
    if args.skip_existing:
        cmd += ["--skip_existing"]

    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
    subprocess.run(cmd, check=True, cwd=REPO, env=env)
    return [os.path.join(out_dir, f"{NAME}_rep{i}.pdb") for i in range(args.num_replicas)]


# ── per-residue C-alpha RMSF, numpy only ────────────────────────────────────
def read_ca_models(pdb_path):
    """Multi-model PDB -> [n_frames, n_residues, 3] of C-alpha coordinates."""
    models, current = [], []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                current.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            elif line.startswith("ENDMDL") and current:
                models.append(current)
                current = []
    if current:
        models.append(current)
    if not models:
        raise ValueError(f"No C-alpha atoms found in {pdb_path}")
    return np.asarray(models, dtype=np.float64)


def rmsf(coords):
    """Superpose every frame on the mean structure (Kabsch), then per-residue RMSF."""
    x = coords - coords.mean(axis=1, keepdims=True)
    ref = x[0]
    for _ in range(3):                                    # iterate to a stable mean
        aligned = np.stack([kabsch(frame, ref) for frame in x])
        ref = aligned.mean(axis=0)
    dev = aligned - aligned.mean(axis=0)
    return np.sqrt((dev ** 2).sum(axis=-1).mean(axis=0))  # [n_residues]


def kabsch(P, Q):
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return P @ R.T


def arm_rmsf(pdb_paths):
    """[n_replicas, n_residues] of per-residue RMSF in Angstrom."""
    return np.stack([rmsf(read_ca_models(p)) for p in pdb_paths])


# ── reporting ───────────────────────────────────────────────────────────────
def report(apo, arms):
    n_res = apo.shape[1]
    site = np.zeros(n_res, dtype=bool)
    site[max(0, SITE_SEQ_IDX - WINDOW): SITE_SEQ_IDX + WINDOW + 1] = True

    print("\n" + "=" * 72)
    print(f"  TAF9 (PDB 6F3T), {n_res} residues, phospho site {SITE}")
    print(f"  Delta RMSF = arm - apo, paired per seed, averaged over residues "
          f"{SITE} +/- {WINDOW}")
    print("=" * 72)
    print(f"  {'arm':<12}{'dRMSF @site':>14}{'elsewhere':>12}{'site/else':>11}"
          f"{'same sign':>11}")
    print("  " + "-" * 68)
    for arm, values in arms.items():
        d = values - apo                                  # [n_replicas, n_residues]
        d_site = d[:, site].mean(axis=1)
        d_else = d[:, ~site].mean(axis=1)
        ratio = abs(d_site.mean()) / max(abs(d_else.mean()), 1e-9)
        same = bool(np.all(d_site > 0) or np.all(d_site < 0))
        print(f"  {arm:<12}{d_site.mean():>+9.3f} +/-{d_site.std():<5.3f}"
              f"{d_else.mean():>+12.3f}{ratio:>10.1f}x{str(same):>11}")
    print("  " + "-" * 68)
    print(f"  mean RMSF: apo {apo.mean():.3f} A"
          + "".join(f" | {a} {v.mean():.3f} A" for a, v in arms.items()))
    print("=" * 72)
    print("""
  How to read this:
    * a negative dRMSF means the phospho condition RIGIDIFIES the site region;
    * 'site/else' is how much larger the effect is at the site than over the
      rest of the chain -- this is what makes the effect local rather than a
      uniform shift;
    * with --controls: 'token_only' keeps the phospho token but drops the
      FluxSite site features, and 'wrong_site' moves that token elsewhere.
      Most of the effect comes from the site-feature channel; the token alone
      contributes a small but sharply localised part that disappears once the
      label is moved.
    * absolute effect sizes are tenths of an Angstrom. Read demo/README.md
      before drawing mechanistic conclusions from them.
""")


def main():
    args = parse_args()
    for path in (PDB, ESM2_PKL, FLUXSITE_PKL):
        if not os.path.exists(path):
            sys.exit(f"Missing demo input: {path}")
    if not os.path.exists(args.ckpt):
        sys.exit(f"Checkpoint not found: {args.ckpt}\nSee demo/README.md for the download.")

    plan = [("apo", dict()),
            ("phospho", dict(phospho=SITE, ptm_feat=FLUXSITE_PKL))]
    if args.controls:
        # Controls carry the phospho token but NOT the FluxSite site features,
        # which separates the label from the site-feature channel.
        plan += [("token_only", dict(phospho=SITE)),
                 ("wrong_site", dict(phospho=SITE, ablation="wrong_site"))]

    results = {}
    for i, (arm, kwargs) in enumerate(plan, 1):
        print(f"\n########## arm {i}/{len(plan)}: {arm} ##########")
        results[arm] = arm_rmsf(run_arm(arm, args, **kwargs))

    report(results.pop("apo"), results)
    print(f"Trajectories written to {args.out_dir}/<arm>/{NAME}_rep*.pdb")


if __name__ == "__main__":
    main()
