#!/usr/bin/env python3
"""
单蛋白推理 + 基础自参考分析（无需 MD 真值）。

输入一个外部 PDB 文件 → 从其结构取初始帧 → 生成 num_replicas 条轨迹 →
分析 RMSD(相对初始帧) / RMSF(逐残基) / Rg(回转半径),全部基于生成轨迹自身,
不与任何参考 MD 比较。

Usage:
  python demo/inference_single.py \
      --pdb target.pdb \
      --ckpt /path/to/model.ckpt \
      --num_replicas 8 --num_frames 100 \
      --out_dir ./inference_single/target

ESM2 基座:若 checkpoint 是 ESM2 条件模型,脚本会自动为该序列计算 ESM2 embedding
（需 `pip install fair-esm`),或用 --esm2_pkl 传入预先算好的 {name: [L,dim]} pkl。

磷酸化条件:用 --phospho 指定位点(默认按 PDB 残基编号),对应残基的 seqres token
被替换成 pSer/pThr/pTyr(21/22/23);几何/初始结构仍用 parent 残基,只有条件通道变化。
FluxSite 特征用 --ptm_feat_pkl 传入({name:[L,dim]});--ptm_ablation 做阴性对照。

  python demo/inference_single.py --pdb 6F3T.pdb --ckpt model.ckpt \\
      --phospho S8 --ptm_feat_pkl feats.pkl --ptm_ablation none
"""
import argparse
import csv
import os

import numpy as np
import torch

from core.ptm_utils import map_ptm_to_parent

# torch>=2.6 默认 weights_only=True,checkpoint 里的 argparse.Namespace 会被拒绝
try:
    torch.serialization.add_safe_globals([argparse.Namespace])
except Exception:  # 老版本 torch 没有这个 API
    pass

# 可磷酸化残基 → seqres token(与 core/ptm_utils.PTM_TO_PARENT 一致)
PHOSPHO_TOKEN = {"S": 21, "T": 22, "Y": 23}
PTM_NAME = {21: "pSer", 22: "pThr", 23: "pTyr"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb", required=True, help="外部 PDB 文件路径")
    p.add_argument("--ckpt", required=True, help="模型 checkpoint 路径")
    p.add_argument("--name", default=None, help="蛋白名(默认取 PDB 文件名)")
    p.add_argument("--chain_id", default=None, help="只取某条链(默认全部)")
    p.add_argument("--num_replicas", type=int, default=5)
    p.add_argument("--num_frames", type=int, default=100, help="每条轨迹帧数")
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--out_dir", default=None, help="默认 ./inference_single/<name>")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_ema", action="store_true", help="用 raw 权重而非 EMA")
    p.add_argument("--skip_existing", action="store_true")
    # ESM2
    p.add_argument("--esm2_pkl", default=None, help="预算 ESM2 embedding 的 pkl({name:[L,dim]})")
    p.add_argument("--esm2_model_name", default="esm2_t33_650M_UR50D")
    # 磷酸化条件
    p.add_argument("--phospho", default=None,
                   help="磷酸位点,逗号分隔,如 'S8' 或 'T183,Y185'(也接受裸编号 '8,183');"
                        "编号默认按 PDB 残基编号,见 --site_numbering")
    p.add_argument("--site_numbering", choices=("pdb", "seq"), default="pdb",
                   help="--phospho 的编号口径:pdb=PDB 残基编号(默认),seq=序列 1-based 位置")
    p.add_argument("--ptm_feat_pkl", default=None,
                   help="FluxSite 特征 pkl({name:[L,dim]}),AA-base+FluxSite 模型需要")
    p.add_argument(
        "--ptm_ablation", choices=("none", "zero", "wrong_site"), default="none",
        help="磷酸条件阴性对照:zero 去掉磷酸条件;wrong_site 把磷酸挪到最远的同类 S/T/Y")
    # analysis
    p.add_argument("--no_eval", action="store_true", help="只生成,不做分析")
    p.add_argument("--no_plot", action="store_true", help="不画图(只出 CSV)")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# 从外部 PDB 构建模型输入(对齐 inference.py 的 load_item)
# ════════════════════════════════════════════════════════════════════════════
def build_item_from_pdb(pdb_path, name, chain_id, num_frames, device,
                        need_esm2, esm2_pkl, esm2_model_name,
                        phospho=None, site_numbering="pdb",
                        ptm_feat_pkl=None, ptm_ablation="none"):
    from core.protein import from_pdb_string
    from core import residue_constants as rc
    from core.geometry import atom37_to_atom14, atom14_to_atom37, \
        atom14_to_frames, atom37_to_torsions

    with open(pdb_path) as f:
        prot = from_pdb_string(f.read(), chain_id=chain_id)

    atom37 = prot.atom_positions.astype(np.float32)      # [L, 37, 3]
    aa = prot.aatype.astype(np.int64)                    # [L]  (20 = UNK)

    # 序列(非标准/UNK → G),映射到 restype_order 索引,与 load_item 一致
    seqres_str = "".join(rc.restypes[a] if 0 <= a < 20 else "G" for a in aa)
    seqres_idx = np.array([rc.restype_order.get(c, rc.restype_order["G"])
                           for c in seqres_str], dtype=np.int64)
    L = len(seqres_idx)

    # 磷酸化位点 → seqres token 21/22/23(几何仍走 parent 残基)
    if phospho:
        sites = parse_phospho_sites(phospho, seqres_str, prot.residue_index,
                                    site_numbering)
        for idx, ptm_token in sites:
            seqres_idx[idx] = ptm_token
        site_text = ", ".join(
            f"{seqres_str[i]}{prot.residue_index[i]}→{PTM_NAME[t]}" for i, t in sites)
        print(f"  [PTM] 磷酸位点: {site_text}")

    # Geometry uses parent residues; PTM identity remains in the model input.
    aatype_geom = map_ptm_to_parent(seqres_idx)

    arr = atom37_to_atom14(atom37[None], aatype_geom[None]).astype(np.float32)  # [1,L,14,3]

    frames = atom14_to_frames(torch.from_numpy(arr))
    seqres_t = torch.from_numpy(seqres_idx)
    atom37b = torch.from_numpy(atom14_to_atom37(arr, aatype_geom[None])).float()
    torsions, tor_mask = atom37_to_torsions(atom37b, torch.from_numpy(aatype_geom[None]))

    item = {
        "name":         name,
        "trans":        frames._trans,
        "rots":         frames._rots._rot_mats,
        "torsions":     torsions,
        "torsion_mask": tor_mask[0],
        "seqres":       seqres_t,
        "mask":         torch.ones(L),
    }

    if need_esm2:
        emb = _get_esm2(name, seqres_str, esm2_pkl, esm2_model_name, device)
        item["esm2_emb"] = torch.from_numpy(emb).float()[:L]

    if ptm_feat_pkl:
        item["ptm_feat"] = torch.from_numpy(
            _get_ptm_feat(name, ptm_feat_pkl)).float()[:L]

    # 阴性对照:只改条件输入,seqres(几何/初始结构)保持不变
    if ptm_ablation != "none":
        from core.ptm_ablation import ptm_condition_ids, ptm_condition_aatype
        ptm_ids, moves = ptm_condition_ids(seqres_idx, mode=ptm_ablation)
        item["ptm_ids_override"] = torch.from_numpy(ptm_ids)
        item["aatype_override"] = torch.from_numpy(
            ptm_condition_aatype(seqres_idx, mode=ptm_ablation).astype(np.int64))
        if ptm_ablation == "zero":
            print("  [PTM ablation] 磷酸条件已清零(zero)")
        else:
            move_text = ", ".join(f"id{pid}:{src + 1}→{dst + 1}({kind})"
                                  for src, dst, pid, kind in moves)
            print(f"  [PTM ablation] 磷酸条件挪到错误位点(wrong_site): {move_text}")

    return item, L, seqres_str


def parse_phospho_sites(spec, seqres_str, residue_index, numbering):
    """'S8,T183' → [(序列 0-based 下标, PTM token), ...],并校验残基类型。"""
    sites = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        letter = token[0].upper() if token[0].isalpha() else None
        number = token[1:] if letter else token
        try:
            number = int(number)
        except ValueError:
            raise ValueError(f"无法解析磷酸位点 '{token}',格式应为 'S8' 或 '8'")

        if numbering == "pdb":
            matches = np.flatnonzero(np.asarray(residue_index) == number)
            if len(matches) == 0:
                raise ValueError(
                    f"位点 '{token}': PDB 里没有残基编号 {number}"
                    f"(范围 {residue_index.min()}-{residue_index.max()});"
                    f"若用的是序列位置请加 --site_numbering seq")
            idx = int(matches[0])
        else:
            idx = number - 1
            if not 0 <= idx < len(seqres_str):
                raise ValueError(f"位点 '{token}': 序列位置 {number} 超出长度 {len(seqres_str)}")

        actual = seqres_str[idx]
        if letter and actual != letter:
            raise ValueError(
                f"位点 '{token}': 该位置实际是 {actual}{residue_index[idx]},不是 {letter}")
        if actual not in PHOSPHO_TOKEN:
            raise ValueError(f"位点 '{token}': {actual} 不可磷酸化(只支持 S/T/Y)")
        sites.append((idx, PHOSPHO_TOKEN[actual]))
    return sites


def _get_ptm_feat(name, ptm_feat_pkl):
    import pickle
    with open(ptm_feat_pkl, "rb") as f:
        d = pickle.load(f)
    cands = [name, name.upper(), name.lower()]
    cands += [c + "_A" for c in cands]
    for k in cands:
        if k in d:
            return np.asarray(d[k], dtype=np.float32)
    raise KeyError(f"FluxSite 特征里没有 '{name}'。试过: {cands}")


def _get_esm2(name, seq, esm2_pkl, model_name, device):
    if esm2_pkl:
        import pickle
        with open(esm2_pkl, "rb") as f:
            d = pickle.load(f)
        for k in (name, name.lower(), name.upper()):
            if k in d:
                return np.asarray(d[k], dtype=np.float32)
        raise KeyError(f"ESM2 embedding for '{name}' not in {esm2_pkl}")
    # on-the-fly
    print(f"  [ESM2] computing embedding on-the-fly via {model_name} ...")
    from scripts.precompute_esm2 import compute_esm2_embeddings
    return compute_esm2_embeddings({name: seq}, model_name, 1, device)[name]


# ════════════════════════════════════════════════════════════════════════════
# 生成
# ════════════════════════════════════════════════════════════════════════════
def to_batch(item, num_frames, device):
    batch = {}
    for k, v in item.items():
        if k == "name":
            continue
        batch[k] = (v if torch.is_tensor(v) else torch.from_numpy(v)).unsqueeze(0).to(device)
    T = num_frames
    batch["trans"]    = batch["trans"].expand(-1, T, -1, -1).contiguous()
    batch["rots"]     = batch["rots"].expand(-1, T, -1, -1, -1).contiguous()
    batch["torsions"] = batch["torsions"].expand(-1, T, -1, -1, -1).contiguous()
    if "esm2_emb" in item:
        batch["esm2_emb"] = item["esm2_emb"].unsqueeze(0).to(device)
    return batch


@torch.no_grad()
def generate(model, item, args):
    from core.io_utils import atom14_to_pdb
    paths = []
    for rep in range(args.num_replicas):
        out_path = os.path.join(args.out_dir, f"{item['name']}_rep{rep}.pdb")
        if args.skip_existing and os.path.exists(out_path):
            print(f"  ⏭️  rep{rep} 已存在,跳过")
            paths.append(out_path)
            continue
        seed = args.seed_base + rep
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
        batch = to_batch(item, args.num_frames, args.device)
        import time
        t0 = time.time()
        atom14, _ = model.inference(batch)
        atom14_to_pdb(atom14[0].cpu().numpy(), item["seqres"].cpu().numpy(), out_path)
        print(f"  ✓ rep{rep} 完成 ({time.time()-t0:.1f}s, seed={seed}) → {out_path}")
        paths.append(out_path)
    return paths


# ════════════════════════════════════════════════════════════════════════════
# 基础自参考分析:RMSD(相对初始帧) / RMSF(逐残基) / Rg
# ════════════════════════════════════════════════════════════════════════════
def analyze(pdb_paths, out_dir, name, make_plot=True):
    import mdtraj as md
    ana = os.path.join(out_dir, "analysis")
    os.makedirs(ana, exist_ok=True)

    rmsd_reps, rmsf_reps, rg_reps = [], [], []
    for p in pdb_paths:
        traj = md.load(p)
        ca = traj.topology.select("name CA")
        traj.superpose(traj, 0, atom_indices=ca)
        rmsd_reps.append(md.rmsd(traj, traj, 0, atom_indices=ca) * 10.0)   # nm→Å
        rmsf_reps.append(md.rmsf(traj, traj, 0, atom_indices=ca) * 10.0)   # [n_ca]
        rg_reps.append(md.compute_rg(traj) * 10.0)                          # [T]

    R = len(pdb_paths)
    # 长度对齐(理论上一致)
    Tmin = min(len(x) for x in rmsd_reps)
    Lmin = min(len(x) for x in rmsf_reps)
    rmsd = np.stack([x[:Tmin] for x in rmsd_reps])    # [R, T]
    rmsf = np.stack([x[:Lmin] for x in rmsf_reps])    # [R, L]
    rg   = np.stack([x[:Tmin] for x in rg_reps])      # [R, T]

    # ── RMSD vs frame ──
    with open(os.path.join(ana, "rmsd_vs_frame.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame"] + [f"rep{i}" for i in range(R)] + ["mean"])
        for t in range(Tmin):
            col = rmsd[:, t]
            w.writerow([t] + [f"{v:.4f}" for v in col] + [f"{col.mean():.4f}"])

    # ── RMSF per residue ──
    with open(os.path.join(ana, "rmsf_per_residue.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["residue"] + [f"rep{i}" for i in range(R)] + ["mean"])
        for r in range(Lmin):
            col = rmsf[:, r]
            w.writerow([r + 1] + [f"{v:.4f}" for v in col] + [f"{col.mean():.4f}"])

    # ── Rg per frame ──
    with open(os.path.join(ana, "rg_per_frame.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame"] + [f"rep{i}" for i in range(R)] + ["mean"])
        for t in range(Tmin):
            col = rg[:, t]
            w.writerow([t] + [f"{v:.4f}" for v in col] + [f"{col.mean():.4f}"])

    # ── summary ──
    summary = {
        "n_replicas":        R,
        "n_frames":          Tmin,
        "n_residues":        Lmin,
        "RMSD_from_init_mean_A": float(rmsd.mean()),
        "RMSD_from_init_max_A":  float(rmsd.max()),
        "RMSF_mean_A":           float(rmsf.mean()),
        "RMSF_max_A":            float(rmsf.max()),
        "Rg_mean_A":             float(rg.mean()),
        "Rg_std_A":              float(rg.std()),
    }
    with open(os.path.join(ana, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, v])

    print("\n── 基础指标 summary ──")
    for k, v in summary.items():
        print(f"  {k:<24}{v}")
    print(f"\n✅ CSV 写到 {ana}/")

    if make_plot:
        _plot(rmsd, rmsf, rg, ana, name)
    return summary


def _plot(rmsd, rmsf, rg, ana, name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib 未安装,跳过画图)")
        return
    R = rmsd.shape[0]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(R):
        ax[0].plot(rmsd[i], alpha=0.5, lw=1)
        ax[2].plot(rg[i], alpha=0.5, lw=1)
    ax[0].plot(rmsd.mean(0), "k", lw=2, label="mean")
    ax[0].set(title="RMSD from initial", xlabel="frame", ylabel="RMSD (Å)"); ax[0].legend()
    ax[1].plot(rmsf.mean(0), "b", lw=2)
    ax[1].fill_between(range(rmsf.shape[1]), rmsf.min(0), rmsf.max(0), alpha=0.2)
    ax[1].set(title="RMSF per residue", xlabel="residue", ylabel="RMSF (Å)")
    ax[2].plot(rg.mean(0), "k", lw=2)
    ax[2].set(title="Radius of gyration", xlabel="frame", ylabel="Rg (Å)")
    fig.suptitle(name); fig.tight_layout()
    out = os.path.join(ana, "basic_metrics.png")
    fig.savefig(out, dpi=130); plt.close()
    print(f"✅ 图写到 {out}")


# ════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    name = args.name or os.path.splitext(os.path.basename(args.pdb))[0]
    args.name = name
    if args.out_dir is None:
        args.out_dir = os.path.join("inference_single", name)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.ckpt}")
    from trainer.module import ProteinWrapper
    model = ProteinWrapper.load_from_checkpoint(args.ckpt, map_location=device)
    if getattr(model.args, "ema", False) and not args.no_ema:
        model.ema.to(device); model.load_ema_weights()
        print("  [EMA] applied EMA weights")
    model.eval().to(device)

    need_esm2 = bool(getattr(model.model, "use_esm2", False))
    print(f"Model needs ESM2: {need_esm2}")

    uses_ptm_feat = bool(getattr(model, "use_ptm_feat", False))
    if uses_ptm_feat and not args.ptm_feat_pkl:
        print("  ⚠️  该 checkpoint 训练时带 FluxSite 特征,但没传 --ptm_feat_pkl,"
              "磷酸条件只会走 token 通道")
    if args.ptm_feat_pkl and not uses_ptm_feat:
        print("  ⚠️  该 checkpoint 没有 FluxSite adapter,--ptm_feat_pkl 会被忽略")
    if args.ptm_ablation != "none" and not args.phospho:
        print("  ⚠️  --ptm_ablation 对没有磷酸位点的输入无意义")

    item, L, seq = build_item_from_pdb(
        args.pdb, name, args.chain_id, args.num_frames, device,
        need_esm2, args.esm2_pkl, args.esm2_model_name,
        phospho=args.phospho, site_numbering=args.site_numbering,
        ptm_feat_pkl=args.ptm_feat_pkl, ptm_ablation=args.ptm_ablation)
    print(f"Protein {name}: L={L}, replicas={args.num_replicas}, frames={args.num_frames}")

    paths = generate(model, item, args)

    if not args.no_eval:
        analyze(paths, args.out_dir, name, make_plot=not args.no_plot)


if __name__ == "__main__":
    main()
