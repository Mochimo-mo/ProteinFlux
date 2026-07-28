"""
trainer/ensemble_loss.py
------------------------
Ensemble-level distribution-matching loss (standalone, toggleable, differentiable).

Motivation: per-frame flow loss only constrains the marginal of each frame, and
does not guarantee the whole ensemble distribution is correct, causing systematic
amplitude bias (RMSF under/over-sampling). This module adds, within a batch, a
distribution-matching term between the generated ensemble and the MD ensemble,
trained jointly with the flow loss via a weighting.

The "ensemble" comes from the T (frame) dimension of the tensor: in [B, T, L, C]
the T frames of the same protein form one conformational ensemble, so batch_size=1
still works.

Generated samples use endpoint prediction (x1-prediction), at nearly zero extra cost:
    x1_hat = (σ'·x_t − σ·v) / (α·σ' − α'·σ)        (works for GVP/Linear/VP)
where v=v_θ is the velocity the model already computed, and α,σ and their
derivatives come from path_sampler.

The feature space defaults to latent (offset+torsion are already SE(3)-invariant
internal coordinates, no alignment needed).

All losses are differentiable and lightweight (O(T²), negligible for T~100).
"""
import torch
from flow.paths import expand_t_like_x


# ════════════════════════════════════════════════════════════════════
# Endpoint prediction: recover x1 from velocity (path-agnostic)
# ════════════════════════════════════════════════════════════════════
def predict_x1_from_velocity(path_sampler, xt, t, v):
    """x1_hat = (σ'·xt − σ·v) / (α·σ' − α'·σ). Holds for Linear/GVP/VP."""
    te = expand_t_like_x(t, xt)
    alpha,  d_alpha = path_sampler.compute_alpha_t(te)
    sigma,  d_sigma = path_sampler.compute_sigma_t(te)
    denom = alpha * d_sigma - d_alpha * sigma           # GVP: constant -π/2
    x1_hat = (d_sigma * xt - sigma * v) / (denom + 1e-8)
    return x1_hat


# ════════════════════════════════════════════════════════════════════
# Distribution distances (two sets X,Y; batched form with last two dims [N, C])
# ════════════════════════════════════════════════════════════════════
def _energy_distance(X, Y):
    """Energy distance²: 2·E‖x−y‖ − E‖x−x'‖ − E‖y−y'‖.
    X,Y: [..., N, C] / [..., M, C], reduces over the last two dims; returns [...]. No hyperparameters."""
    dxy = torch.cdist(X, Y).mean(dim=(-2, -1))
    dxx = torch.cdist(X, X).mean(dim=(-2, -1))
    dyy = torch.cdist(Y, Y).mean(dim=(-2, -1))
    return 2.0 * dxy - dxx - dyy


def _mmd_rbf(X, Y, sigmas=(0.5, 1.0, 2.0)):
    """Multi-bandwidth Gaussian-kernel MMD². X,Y: [..., N, C]/[..., M, C] → [...]."""
    def _k(A, B):
        d2 = torch.cdist(A, B) ** 2
        out = 0.0
        for s in sigmas:
            out = out + torch.exp(-d2 / (2.0 * s * s))
        return out.mean(dim=(-2, -1)) / len(sigmas)
    return _k(X, X) + _k(Y, Y) - 2.0 * _k(X, Y)


def _sliced_wasserstein(X, Y, n_proj=64, generator=None):
    """Sliced-Wasserstein². X,Y: [..., N, C]/[..., M, C] (N==M) → [...]."""
    C = X.shape[-1]
    theta = torch.randn(C, n_proj, device=X.device, dtype=X.dtype, generator=generator)
    theta = theta / (theta.norm(dim=0, keepdim=True) + 1e-8)
    xp = (X @ theta).sort(dim=-2).values        # [..., N, n_proj]
    yp = (Y @ theta).sort(dim=-2).values
    return ((xp - yp) ** 2).mean(dim=(-2, -1))


_DIST = {"energy": _energy_distance, "mmd": _mmd_rbf, "sw": _sliced_wasserstein}


# ════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════
def ensemble_distribution_loss(x1_hat, x1, res_mask=None, *,
                               kind="energy", feature="per_residue",
                               t=None, tau_min=0.3):
    """
    x1_hat, x1 : [B, T, L, C]  generated / MD ensemble
    res_mask   : [B, L] bool   valid residues (None = all valid)
    kind       : 'energy' | 'mmd' | 'sw'
    feature    : 'per_residue' (match per-residue fluctuation distribution, robust, fixes amplitude)
                 'global' (flatten residues, match joint/correlated distribution)
    t          : [B] flow time; when provided, gate on t<tau_min (skip when x1_hat is too coarse)
    Returns a scalar.
    """
    B, T, L, C = x1_hat.shape
    dist_fn = _DIST[kind]

    if res_mask is None:
        res_mask = x1_hat.new_ones(B, L, dtype=torch.bool)

    if feature == "per_residue":
        # [B,L,T,C]: treat (B,L) as batch, compute set distance over the T frames
        Xr = x1_hat.permute(0, 2, 1, 3)
        Yr = x1.permute(0, 2, 1, 3)
        d = dist_fn(Xr, Yr)                       # [B, L]
        d = d.clamp(min=0.0)
        w = res_mask.float()
        per_b = (d * w).sum(dim=1) / (w.sum(dim=1) + 1e-8)   # [B]
    elif feature == "global":
        m = res_mask.float()[:, None, :, None]    # [B,1,L,1]
        Xg = (x1_hat * m).reshape(B, T, L * C)
        Yg = (x1 * m).reshape(B, T, L * C)
        per_b = dist_fn(Xg, Yg).clamp(min=0.0)    # [B]
    else:
        raise ValueError(f"unknown feature: {feature}")

    if t is not None:
        gate = (t.reshape(B) >= tau_min).float()
        denom = gate.sum().clamp(min=1.0)
        return (per_b * gate).sum() / denom

    return per_b.mean()
