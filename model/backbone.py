import torch
from torch.utils.checkpoint import checkpoint
import numpy as np
import torch.nn as nn

from .attention import LatentLayer, IPALayer
from .embeddings import TimestepEmbedder, FinalLayer


def grad_checkpoint(func, args, checkpointing=False):
    if checkpointing:
        return checkpoint(func, *args, use_reentrant=False)
    else:
        return func(*args)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class LatentModel(nn.Module):
    def __init__(self, args, latent_dim):
        super().__init__()
        self.args = args

        self.latent_to_emb = nn.Linear(latent_dim, args.embed_dim)
        self.cond_to_emb = nn.Linear(latent_dim, args.embed_dim)
        self.mask_to_emb = nn.Embedding(2, args.embed_dim)

        self.use_esm2 = getattr(args, 'esm2_emb_path', None) is not None
        if self.use_esm2:
            esm2_dim = getattr(args, 'esm2_dim', 1280)
            proj_dim = getattr(args, 'esm2_proj_dim', None)
            if proj_dim is None:
                proj_dim = (esm2_dim + args.embed_dim) // 2

            def _make_esm2_proj(in_dim, hidden_dim, out_dim):
                if hidden_dim == 0:
                    return nn.Linear(in_dim, out_dim)
                return nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, out_dim),
                )

            # IPA path: Xavier init (non-zero), ESM2 replaces aa_emb entirely
            self.esm2_proj_ipa = _make_esm2_proj(esm2_dim, proj_dim, args.embed_dim)
            # Main path: zero-init output layer so ESM2 contribution grows from zero
            self.esm2_proj_main = _make_esm2_proj(esm2_dim, proj_dim, args.embed_dim)

        ipa_args = {
            'c_s': args.embed_dim,
            'c_z': 0,
            'c_hidden': args.ipa_head_dim,
            'no_heads': args.ipa_heads,
            'no_qk_points': args.ipa_qk,
            'no_v_points': args.ipa_v,
            'dropout': args.dropout,
        }

        if args.prepend_ipa:
            if not self.args.no_aa_emb:
                self.aatype_to_emb = nn.Embedding(24, args.embed_dim)
            self.ipa_layers = nn.ModuleList([
                IPALayer(
                    embed_dim=args.embed_dim,
                    ffn_embed_dim=4 * args.embed_dim,
                    mha_heads=args.mha_heads,
                    dropout=args.dropout,
                    use_rotary_embeddings=not args.no_rope,
                    ipa_args=ipa_args
                )
                for _ in range(args.num_layers)
            ])

        self.layers = nn.ModuleList([
            LatentLayer(
                embed_dim=args.embed_dim,
                ffn_embed_dim=4 * args.embed_dim,
                mha_heads=args.mha_heads,
                dropout=args.dropout,
                hyena=args.hyena,
                num_frames=args.num_frames,
                use_rotary_embeddings=not args.no_rope,
                use_time_attention=True,
                ipa_args=ipa_args if args.interleave_ipa else None,
            )
            for _ in range(args.num_layers)
        ])

        self.emb_to_latent = FinalLayer(args.embed_dim, latent_dim)

        self.t_embedder = TimestepEmbedder(args.embed_dim)
        if args.abs_pos_emb:
            self.register_buffer('pos_embed',
                                 nn.Parameter(torch.zeros(1, args.crop, args.embed_dim), requires_grad=False))
        if args.abs_time_emb:
            self.register_buffer('time_embed',
                                 nn.Parameter(torch.zeros(1, args.num_frames, args.embed_dim), requires_grad=False))

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.args.prepend_ipa:
            for block in self.ipa_layers:
                nn.init.constant_(block.ipa.linear_out.weight, 0)
                nn.init.constant_(block.ipa.linear_out.bias, 0)

        if self.args.interleave_ipa:
            for block in self.layers:
                nn.init.constant_(block.ipa.linear_out.weight, 0)
                nn.init.constant_(block.ipa.linear_out.bias, 0)

        if self.args.abs_pos_emb:
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.args.crop))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.args.abs_time_emb:
            time_embed = get_1d_sincos_pos_embed_from_grid(self.time_embed.shape[-1], np.arange(self.args.num_frames))
            self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.emb_to_latent.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.emb_to_latent.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.emb_to_latent.linear.weight, 0)
        nn.init.constant_(self.emb_to_latent.linear.bias, 0)

        if self.use_esm2:
            # esm2_proj_ipa: Xavier (non-zero) — IPA needs a real signal from step 0
            # esm2_proj_main: zero-init output layer — main path ESM2 grows from zero
            last_main = self.esm2_proj_main if isinstance(self.esm2_proj_main, nn.Linear) \
                else self.esm2_proj_main[-1]
            nn.init.zeros_(last_main.weight)
            nn.init.zeros_(last_main.bias)

    def run_ipa(self, t, mask, start_frames, aatype, ptm_emb=None, esm2_emb=None):
        B, L = mask.shape
        x = torch.zeros(B, L, self.args.embed_dim, device=mask.device)
        if esm2_emb is not None and self.use_esm2:
            # ESM2 replaces aa_emb: richer contextual sequence repr subsumes simple type lookup
            x = x + self.esm2_proj_ipa(esm2_emb.to(x.device))
            if ptm_emb is not None:
                x = x + ptm_emb.to(x.device)
        elif aatype is not None and not self.args.no_aa_emb:
            # Fallback to aa_emb when ESM2 is not available
            aa_feat = self.aatype_to_emb(aatype)
            if ptm_emb is not None:
                aa_feat = aa_feat + ptm_emb.to(aa_feat.device)
            x = x + aa_feat
        for layer in self.ipa_layers:
            x = layer(x, t, mask, frames=start_frames)
        return x

    def forward(self, x, t, mask, start_frames=None, end_frames=None,
                x_cond=None, x_cond_mask=None, aatype=None, ptm_emb=None, esm2_emb=None, **kwargs):
        x = self.latent_to_emb(x)

        if self.args.abs_pos_emb:
            x = x + self.pos_embed
        if self.args.abs_time_emb:
            x = x + self.time_embed[:, :, None]
        if x_cond is not None:
            x = x + self.cond_to_emb(x_cond) + self.mask_to_emb(x_cond_mask)

        # Add ESM2 embeddings as residue-level condition (broadcast across all frames)
        if esm2_emb is not None and self.use_esm2:
            x = x + self.esm2_proj_main(esm2_emb.to(x.device))[:, None]  # [B,L,C] -> [B,1,L,C]

        t = self.t_embedder(t * self.args.time_multiplier)[:, None]

        if self.args.prepend_ipa:
            ipa_out = self.run_ipa(t[:, 0], mask[:, 0], start_frames, aatype,
                                   ptm_emb=ptm_emb, esm2_emb=esm2_emb)
            x = x + ipa_out[:, None]

        for layer_idx, layer in enumerate(self.layers):
            x = grad_checkpoint(layer, (x, t, mask, start_frames), self.args.grad_checkpointing)

        return self.emb_to_latent(x, t)

    def forward_inference(self, x, t, mask, start_frames=None, end_frames=None,
                          x_cond=None, x_cond_mask=None, aatype=None, ptm_emb=None, esm2_emb=None, **kwargs):
        return self.forward(x, t, mask, start_frames, end_frames, x_cond, x_cond_mask, aatype,
                            ptm_emb=ptm_emb, esm2_emb=esm2_emb)
