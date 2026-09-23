# Demo — phospho-conditioned ensemble generation

This demo generates two conformational ensembles for the same protein with
ProteinFlux: one **without** the phosphorylation condition and one **with** it,
then reports how per-residue flexibility changes around the phospho site.
Everything it needs except the model checkpoint ships with the repository.

## System

**Human TAF9** — TFIID subunit 9 (UniProt Q16594), the 116-residue chain of
PDB entry **6F3T**, phosphorylated at **Ser8**.

Residues are numbered in UniProt coordinates, so the site is written `S8`
exactly as it is annotated. The starting structure in `data/6F3T_TAF9.pdb`
carries the **unmodified** Ser8 — the phospho state is applied as a condition,
not baked into the input coordinates.

The protein is held out from PTM training: highest sequence identity to any
training system is 11.2%, and no structure with the same PDB ID appears in
the training set.

## Run it

```bash
python demo/run_phospho_demo.py --ckpt /path/to/proteinflux_ptm.ckpt
```

Add the negative controls from the paper:

```bash
python demo/run_phospho_demo.py --ckpt /path/to/proteinflux_ptm.ckpt --controls
```

Install the project requirements with `pip install -r requirements.txt`. The
demo computes RMSF itself, so no mdtraj or matplotlib is needed.

Runtime with the defaults (5 replicas x 100 frames per arm): about **30 s per
arm on one GPU** (6 s per trajectory). It also runs without a GPU, at about
**2.5 min per trajectory** on 8 CPU threads — so for a CPU run, ask for fewer
replicas:

```bash
python demo/run_phospho_demo.py --ckpt /path/to/proteinflux_ptm.ckpt \
    --device cpu --num_replicas 2
```

Fewer replicas means a noisier seed spread; the five-replica numbers below are
what the defaults reproduce.

**Checkpoint:** download [`ptm_esm2_fluxsite.ckpt`](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/ptm_esm2_fluxsite.ckpt)
and pass its local path to `--ckpt`.

## What it prints

```
  arm            dRMSF @site   elsewhere  site/else  same sign
  --------------------------------------------------------------------
  phospho        -0.366 +/-0.071      -0.083       4.4x       True
  token_only     -0.043 +/-0.015      -0.001      34.9x       True
  wrong_site     +0.005 +/-0.005      -0.003       1.4x      False
```

Reading the table:

- **dRMSF @site** — change in C-alpha RMSF, averaged over Ser8 ± 3 residues,
  relative to the unconditioned (apo) arm. Arms are paired: every arm uses the
  same five seeds, so the comparison is per-seed rather than between
  independent samples. The `+/-` is the spread across those five seeds.
- **Negative values mean rigidification.** Phosphorylation of Ser8 makes the
  N-terminal region of TAF9 less mobile in the generated ensemble.
- **site/else** — how much larger the effect is inside the site window than
  over the rest of the chain. This is what distinguishes a local response from
  a uniform shift in flexibility.
- **token_only** keeps the phospho token but drops the FluxSite site features;
  **wrong_site** moves that token to the most distant Ser in the chain.

## What the controls show — and what they do not

The controls decompose the effect, and the decomposition is worth stating
plainly:

- Most of the −0.366 Å comes from the **FluxSite site-feature channel**, not
  from the phospho token. With the features removed (`token_only`), what
  remains is −0.043 Å.
- That residual is small, but it is **sharply localised** — 35x larger inside
  the site window than elsewhere — and it **disappears when the label is moved**
  (`wrong_site`: +0.005 Å at the real site). So the model is responding to
  *where* the modification is, not merely to the presence of a label.
- Absolute effect sizes are **tenths of an Ångström**. TAF9/pSer8 is the
  largest phospho response among the 22 validation targets; typical responses
  are smaller, and for several targets the site effect is not distinguishable
  from the background. The paper reports the full distribution.

Please do not read a single run of this demo as evidence about phosphorylation
mechanism. It shows that the conditioning pathway is wired, reproducible, and
testable against its own negative controls.

## Running your own protein

`demo/inference_single.py` takes any PDB:

```bash
python demo/inference_single.py \
    --pdb your_protein.pdb --ckpt /path/to/proteinflux_ptm.ckpt \
    --phospho S65,T72 --num_replicas 5 --num_frames 100
```

- `--phospho` takes `S65` / `T72` / `Y19` style sites, and the residue letter is
  checked against the structure. Numbering follows the PDB file; pass
  `--site_numbering seq` to use 1-based sequence positions instead.
- ESM2 embeddings are computed on the fly (needs `pip install fair-esm`), or
  pass a precomputed `{name: [L, dim]}` pickle with `--esm2_pkl`.
- FluxSite site features are passed with `--ptm_feat_pkl`. Without them the
  phospho condition runs through the token channel alone — the weaker pathway
  shown in the table above.
- `--ptm_ablation zero|wrong_site` reproduces the negative controls for your
  own system.

## Files

| Path | Size | What it is |
|---|---|---|
| `data/6F3T_TAF9.pdb` | 74 KB | Starting structure, unmodified Ser8, UniProt numbering |
| `data/6F3T_TAF9_esm2.pkl` | 581 KB | Precomputed ESM2-650M embedding |
| `data/6F3T_TAF9_fluxsite256.pkl` | 117 KB | Precomputed FluxSite site features |
| `run_phospho_demo.py` | | Demo driver |
| `inference_single.py` | | Single-protein inference used by the demo |

Generated trajectories are written to `demo/out/<arm>/6F3T_TAF9_rep*.pdb` as
multi-model PDBs, ready to open in PyMOL or VMD.
