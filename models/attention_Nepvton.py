from typing import Any, Dict, Optional

import torch
from torch import nn

from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormZero


@maybe_allow_in_graph
class GatedSelfAttentionDense(nn.Module):
    """
    Gated attention unit designed for feature fusion.

    This is a candidate module for integrating cultural feature conditioning
    in the NepVTON pipeline. E.g., embeddings related to cultural context can
    be passed in `objs` to steer the attention towards regional garment attributes.
    """
    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()

        self.linear_proj = nn.Linear(context_dim, query_dim)
        self.attention = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.feedforward = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.alpha_attn = nn.Parameter(torch.tensor(0.0))
        self.alpha_ff = nn.Parameter(torch.tensor(0.0))
        self.enabled = True

    def forward(self, visual_feats: torch.Tensor, context_feats: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return visual_feats

        n_tokens = visual_feats.shape[1]
        context_feats = self.linear_proj(context_feats)

        combined = torch.cat([visual_feats, context_feats], dim=1)
        attention_output = self.attention(self.norm1(combined))[:, :n_tokens, :]
        fused = visual_feats + self.alpha_attn.tanh() * attention_output
        fused = fused + self.alpha_ff.tanh() * self.feedforward(self.norm2(fused))

        return fused


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    """
    Transformer block with optional GLIGEN control and advanced normalization.

    Ideal location to inject localized or ethnicity-aware embeddings for Nepalese traditional outfit guidance.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_zero = norm_type == "ada_norm_zero" and num_embeds_ada_norm is not None
        self.use_ada = norm_type == "ada_norm" and num_embeds_ada_norm is not None
        self.use_ada_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        norm_cls = AdaLayerNormZero if self.use_ada_zero else (
            AdaLayerNorm if self.use_ada else nn.LayerNorm
        )

        self.norm1 = norm_cls(dim, num_embeds_ada_norm) if norm_cls != nn.LayerNorm else norm_cls(
            dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
        )

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        if cross_attention_dim is not None or double_self_attention:
            norm2_cls = AdaLayerNorm if self.use_ada else nn.LayerNorm
            self.norm2 = norm2_cls(
                dim, num_embeds_ada_norm
            ) if self.use_ada else norm2_cls(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.norm2, self.attn2 = None, None

        if not self.use_ada_single:
            self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        if attention_type in {"gated", "gated-text-image"}:
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        if self.use_ada_single:
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        spatial_attn_inputs=[],
        spatial_attn_idx=0,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:

        batch_size = hidden_states.shape[0]
        spatial_attn_input = spatial_attn_inputs[spatial_attn_idx]
        spatial_attn_idx += 1
        hidden_states = torch.cat((hidden_states, spatial_attn_input), dim=1)

        # First attention block
        if self.use_ada:
            normed = self.norm1(hidden_states, timestep)
        elif self.use_ada_zero:
            normed, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_ada_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            normed = self.norm1(hidden_states) * (1 + scale_msa) + shift_msa
        else:
            normed = self.norm1(hidden_states)

        if self.pos_embed is not None:
            normed = self.pos_embed(normed)

        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs else 1.0
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs else {}
        gligen = cross_attention_kwargs.pop("gligen", None)

        attn_out = self.attn1(normed, encoder_hidden_states if self.only_cross_attention else None, attention_mask, **cross_attention_kwargs)
        if self.use_ada_zero:
            attn_out *= gate_msa.unsqueeze(1)
        elif self.use_ada_single:
            attn_out *= gate_msa

        hidden_states = attn_out + hidden_states
        hidden_states, _ = hidden_states.chunk(2, dim=1)

        if gligen is not None:
            hidden_states = self.fuser(hidden_states, gligen["objs"])

        # Cross attention
        if self.attn2 is not None:
            normed = self.norm2(hidden_states, timestep) if self.use_ada else self.norm2(hidden_states)
            if self.pos_embed is not None and not self.use_ada_single:
                normed = self.pos_embed(normed)

            hidden_states = self.attn2(normed, encoder_hidden_states, encoder_attention_mask, **cross_attention_kwargs) + hidden_states

        normed = self.norm3(hidden_states) if not self.use_ada_single else hidden_states
        if self.use_ada_zero:
            normed = normed * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        elif self.use_ada_single:
            normed = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            chunks = normed.chunk(normed.shape[self._chunk_dim] // self._chunk_size, dim=self._chunk_dim)
            ff_out = torch.cat([self.ff(chunk, scale=lora_scale) for chunk in chunks], dim=self._chunk_dim)
        else:
            ff_out = self.ff(normed, scale=lora_scale)

        if self.use_ada_zero:
            ff_out *= gate_mlp.unsqueeze(1)
        elif self.use_ada_single:
            ff_out *= gate_mlp

        hidden_states = ff_out + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states, spatial_attn_inputs, spatial_attn_idx


class FeedForward(nn.Module):
    """
    Lightweight feedforward block for vision transformer modules.
    """
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        linear = LoRACompatibleLinear if not USE_PEFT_BACKEND else nn.Linear

        activation = {
            "gelu": GELU,
            "gelu-approximate": lambda d, h: GELU(d, h, approximate="tanh"),
            "geglu": GEGLU,
            "geglu-approximate": ApproximateGELU,
        }[activation_fn](dim, inner_dim)

        self.layers = nn.ModuleList([
            activation,
            nn.Dropout(dropout),
            linear(inner_dim, dim_out),
        ])

        if final_dropout:
            self.layers.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, (GEGLU,) if USE_PEFT_BACKEND else (GEGLU, LoRACompatibleLinear)):
                hidden_states = layer(hidden_states, scale)
            else:
                hidden_states = layer(hidden_states)
        return hidden_states
