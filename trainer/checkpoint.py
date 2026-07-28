import os
import json
import argparse
import torch
import torch.nn as nn

torch.serialization.add_safe_globals([argparse.Namespace])

from trainer.ema import ExponentialMovingAverage


def smart_load_checkpoint(model, ckpt_path, filter_adapter=True):
    """Load pretrained weights, skipping layers with mismatched shapes.

    filter_adapter: skip ptm_adapter/feat_adapter keys absent from the checkpoint.
    """
    print(f"Loading pretrained weights: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    model_dict = model.state_dict()
    pretrained_dict = {}
    ignored_keys = []

    for k, v in state_dict.items():
        if filter_adapter and ("ptm_adapter" in k or "feat_adapter" in k):
            continue

        if k not in model_dict and k.startswith("model."):
            k_target = k[6:]
        elif k not in model_dict and ("model." + k) in model_dict:
            k_target = "model." + k
        else:
            k_target = k

        if k_target in model_dict:
            if v.shape == model_dict[k_target].shape:
                pretrained_dict[k_target] = v
            else:
                ignored_keys.append(f"{k_target} (ckpt {v.shape} != model {model_dict[k_target].shape})")

    model.load_state_dict(pretrained_dict, strict=False)

    if ignored_keys:
        print("Warning: the following layers were skipped due to shape mismatch:")
        for k in ignored_keys:
            print(f"  - {k}")
    else:
        print("All layer shapes matched.")

    return model


def perform_embedding_surgery(wrapper):
    """Expand the amino-acid Embedding from <=22 to 24 dims to accommodate PTM tokens,
    and synchronise the EMA shadow weights when EMA is enabled.
    """
    is_rank0 = (int(os.environ.get("LOCAL_RANK", 0)) == 0)

    if is_rank0:
        print("Searching for target Embedding layer...")

    target_module = None
    target_name = None

    for name, module in wrapper.model.named_modules():
        if isinstance(module, nn.Embedding) and module.num_embeddings in [20, 21, 22]:
            target_module = module
            target_name = name
            break

    if target_module is None:
        if is_rank0:
            print("No suitable Embedding layer found — skipping surgery.")
        return wrapper

    device = target_module.weight.device
    old_dim = target_module.num_embeddings
    emb_dim = target_module.embedding_dim

    if old_dim >= 24:
        if is_rank0:
            print(f"Embedding already at dim {old_dim} — skipping.")
        return wrapper

    IDX_S, IDX_T, IDX_Y = 15, 16, 18
    IDX_SEP, IDX_TPO, IDX_PTR = 21, 22, 23

    new_emb = nn.Embedding(24, emb_dim).to(device).to(target_module.weight.dtype)
    with torch.no_grad():
        new_emb.weight[:old_dim] = target_module.weight
        new_emb.weight[IDX_SEP] = target_module.weight[min(IDX_S, old_dim - 1)]
        new_emb.weight[IDX_TPO] = target_module.weight[min(IDX_T, old_dim - 1)]
        new_emb.weight[IDX_PTR] = target_module.weight[min(IDX_Y, old_dim - 1)]

    def _setattr_recursive(obj, path, value):
        if '.' in path:
            parent, child = path.rsplit('.', 1)
            for p in parent.split('.'):
                obj = getattr(obj, p)
            setattr(obj, child, value)
        else:
            setattr(obj, path, value)

    _setattr_recursive(wrapper.model, target_name, new_emb)
    if is_rank0:
        print(f"Embedding surgery done: {target_name} -> [24, {emb_dim}]")

    if getattr(wrapper.args, 'ema', False) and hasattr(wrapper, 'ema'):
        if is_rank0:
            print("Syncing EMA shadow weights...")
        old_ema_state = wrapper.ema.state_dict()
        wrapper.ema = ExponentialMovingAverage(wrapper.model, decay=wrapper.args.ema_decay)

        if "params" in old_ema_state:
            param_dict = old_ema_state["params"]
            emb_key = f"{target_name}.weight"
            if emb_key in param_dict:
                old_tensor = param_dict[emb_key]
                new_tensor = torch.zeros_like(new_emb.weight)
                new_tensor[:old_dim] = old_tensor
                new_tensor[IDX_SEP] = old_tensor[min(IDX_S, old_dim - 1)]
                new_tensor[IDX_TPO] = old_tensor[min(IDX_T, old_dim - 1)]
                new_tensor[IDX_PTR] = old_tensor[min(IDX_Y, old_dim - 1)]
                param_dict[emb_key] = new_tensor
                if is_rank0:
                    print(f"EMA weight expanded: {emb_key}")

        wrapper.ema.load_state_dict(old_ema_state)

    return wrapper


def warmstart_ptm_tokens(wrapper):
    """When the aa Embedding is already 24 (e.g. aa_multi base), the PTM rows
    21/22/23 were never trained (base data has no PTM tokens) → still at init.
    Warm-start them from their parent residues (S/T/Y = 15/16/18), like the
    surgery does. Safe for a fresh finetune; do NOT use when resuming a PTM run
    (it would clobber learned PTM rows). Syncs EMA too.
    """
    is_rank0 = (int(os.environ.get("LOCAL_RANK", 0)) == 0)
    IDX_S, IDX_T, IDX_Y = 15, 16, 18
    IDX_SEP, IDX_TPO, IDX_PTR = 21, 22, 23

    target_name = None
    for name, module in wrapper.model.named_modules():
        if isinstance(module, nn.Embedding) and module.num_embeddings == 24:
            target_name = name
            target_module = module
            break
    if target_name is None:
        if is_rank0:
            print("warmstart_ptm_tokens: no 24-dim aa embedding found — skipping.")
        return wrapper

    with torch.no_grad():
        for ptm, parent in ((IDX_SEP, IDX_S), (IDX_TPO, IDX_T), (IDX_PTR, IDX_Y)):
            target_module.weight[ptm] = target_module.weight[parent]
    if is_rank0:
        print(f"warmstart_ptm_tokens: {target_name} rows 21/22/23 ← 15/16/18 (S/T/Y)")

    if getattr(wrapper.args, 'ema', False) and hasattr(wrapper, 'ema'):
        state = wrapper.ema.state_dict()
        emb_key = f"{target_name}.weight"
        if "params" in state and emb_key in state["params"]:
            t = state["params"][emb_key]
            for ptm, parent in ((IDX_SEP, IDX_S), (IDX_TPO, IDX_T), (IDX_PTR, IDX_Y)):
                t[ptm] = t[parent]
            wrapper.ema.load_state_dict(state)
            if is_rank0:
                print(f"warmstart_ptm_tokens: EMA {emb_key} rows warm-started")
    return wrapper


def save_args(args, save_dir):
    """Persist training args to <save_dir>/config.json."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Args saved to: {save_dir}/config.json")
