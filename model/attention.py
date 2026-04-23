import torch.nn as nn

from .mha import MultiheadAttention
from .ipa import InvariantPointAttention
from .embeddings import gelu, modulate
from .hyena import HyenaOperator


class AttentionWithRoPE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.attn = MultiheadAttention(*args, **kwargs)

    def forward(self, x, mask):
        x = x.transpose(0, 1)
        x, _ = self.attn(query=x, key=x, value=x, key_padding_mask=1 - mask)
        x = x.transpose(0, 1)
        return x


class IPALayer(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, mha_heads, dropout=0.0,
                 use_rotary_embeddings=False, ipa_args=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.mha_heads = mha_heads
        self.inf = 1e5
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv=True, dropout=dropout, ipa_args=ipa_args)

    def _init_submodules(self, add_bias_kv=False, dropout=0.0, ipa_args=None):
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embed_dim, 6 * self.embed_dim, bias=True)
        )
        self.ipa_norm = nn.LayerNorm(self.embed_dim)
        self.ipa = InvariantPointAttention(**ipa_args)

        self.mha_l = AttentionWithRoPE(
            self.embed_dim,
            self.mha_heads,
            add_bias_kv=add_bias_kv,
            dropout=dropout,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )

        self.mha_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, t, mask=None, frames=None):
        shift_msa_l, scale_msa_l, gate_msa_l, \
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=-1)
        x = x + self.ipa(self.ipa_norm(x), frames, frame_mask=mask)

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_l, scale_msa_l)
        x = self.mha_l(x, mask=mask)
        x = residual + gate_msa_l.unsqueeze(1) * x

        residual = x
        x = modulate(self.final_layer_norm(x), shift_mlp, scale_mlp)
        x = self.fc2(gelu(self.fc1(x)))
        x = residual + gate_mlp.unsqueeze(1) * x

        return x


class LatentLayer(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, mha_heads, dropout=0.0, num_frames=50, hyena=False,
                 use_rotary_embeddings=False, use_time_attention=True, ipa_args=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.hyena = hyena
        self.ffn_embed_dim = ffn_embed_dim
        self.mha_heads = mha_heads
        self.inf = 1e5
        self.use_time_attention = use_time_attention
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv=True, dropout=dropout, ipa_args=ipa_args)

    def _init_submodules(self, add_bias_kv=False, dropout=0.0, ipa_args=None):
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embed_dim, 9 * self.embed_dim, bias=True)
        )

        if ipa_args is not None:
            self.ipa_norm = nn.LayerNorm(self.embed_dim)
            self.ipa = InvariantPointAttention(**ipa_args)

        if self.hyena:
            self.mha_t = HyenaOperator(
                d_model=self.embed_dim,
                l_max=self.num_frames,
                order=2,
                filter_order=64,
            )
        else:
            self.mha_t = AttentionWithRoPE(
                self.embed_dim,
                self.mha_heads,
                add_bias_kv=add_bias_kv,
                dropout=dropout,
                use_rotary_embeddings=self.use_rotary_embeddings,
            )

        self.mha_l = AttentionWithRoPE(
            self.embed_dim,
            self.mha_heads,
            add_bias_kv=add_bias_kv,
            dropout=dropout,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )

        self.mha_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, t, mask=None, frames=None):
        B, T, L, C = x.shape

        shift_msa_l, scale_msa_l, gate_msa_l, \
            shift_msa_t, scale_msa_t, gate_msa_t, \
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(9, dim=-1)

        if hasattr(self, 'ipa'):
            x = x + self.ipa(self.ipa_norm(x), frames[:, None], frame_mask=mask)

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_l, scale_msa_l)
        x = self.mha_l(
            x.reshape(B * T, L, C),
            mask=mask.reshape(B * T, L),
        ).reshape(B, T, L, C)
        x = residual + gate_msa_l.unsqueeze(1) * x

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_t, scale_msa_t)
        if self.hyena:
            assert (mask - 1).sum() == 0
            x = self.mha_t(
                x.transpose(1, 2).reshape(B * L, T, C)
            ).reshape(B, L, T, C).transpose(1, 2)
        else:
            x = self.mha_t(
                x.transpose(1, 2).reshape(B * L, T, C),
                mask=mask.transpose(1, 2).reshape(B * L, T)
            ).reshape(B, L, T, C).transpose(1, 2)
        x = residual + gate_msa_t.unsqueeze(1) * x

        residual = x
        x = modulate(self.final_layer_norm(x), shift_mlp, scale_mlp)
        x = self.fc2(gelu(self.fc1(x)))
        x = residual + gate_mlp.unsqueeze(1) * x

        return x
