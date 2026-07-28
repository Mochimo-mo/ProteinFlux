"""
Lightweight LoRA -- for PTM finetuning (freeze the backbone, low-rank adaptation of attention/FFN).

Key design: LoRALinear **reuses the original nn.Linear's weight/bias as its own parameters**
(rather than nesting them as a submodule), so the state_dict keys stay as `...weight`/`...bias`.
This way, even if apply_lora is called before loading a pretrained checkpoint, the base weights
still load correctly under their original keys.
The B matrix is zero-initialized -> ΔW=0 at start, equivalent to the original model, safe to finetune.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        # Reuse original params (keys stay 'weight'/'bias'), frozen
        self.weight = base.weight
        self.weight.requires_grad_(False)
        self.bias = base.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))   # B stays 0 -> equivalent to original model at start

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = out + self.scaling * (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t())
        return out


def apply_lora(model: nn.Module,
               targets=("q_proj", "v_proj", "out_proj", "fc1", "fc2"),
               r: int = 8, alpha: int = 16, dropout: float = 0.05) -> nn.Module:
    """In-place replace the nn.Linear layers in model named in targets with LoRALinear."""
    n = 0
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(mod, name, LoRALinear(child, r, alpha, dropout))
                n += 1
    print(f"[LoRA] wrapped {n} Linear layers (targets={targets}, r={r}, alpha={alpha})")
    return model


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """Merge LoRA back into a plain Linear for inference deployment (the inference path needs no LoRA structure)."""
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                lin = nn.Linear(child.in_features, child.out_features,
                                bias=child.bias is not None)
                lin.weight = nn.Parameter(
                    child.weight + child.scaling * (child.lora_B @ child.lora_A))
                if child.bias is not None:
                    lin.bias = nn.Parameter(child.bias)
                setattr(mod, name, lin)
    return model
