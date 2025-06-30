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
    Modified Gated Self-Attention Dense Layer for integration of cultural embeddings.

    \ud83d\udd2c Research Hook:
    - Add garment-style embeddings (e.g., daura, saree) into `objs`.
    - Use region-specific vectors for pose-aware garment control.
    """

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()
        self.visual_to_context_proj = nn.Linear(context_dim, query_dim)
        self.attn_layer = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff_block = FeedForward(query_dim, activation_fn="geglu")
        self.norm_attn = nn.LayerNorm(query_dim)
        self.norm_ff = nn.LayerNorm(query_dim)
        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_ff", nn.Parameter(torch.tensor(0.0)))
        self.enabled = True

    def forward(self, visual_feats: torch.Tensor, obj_feats: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return visual_feats

        num_visual_tokens = visual_feats.shape[1]
        obj_feats_proj = self.visual_to_context_proj(obj_feats)
        fused_input = torch.cat([visual_feats, obj_feats_proj], dim=1)
        attended = self.attn_layer(self.norm_attn(fused_input))[:, :num_visual_tokens, :]

        visual_feats = visual_feats + self.alpha_attn.tanh() * attended
        visual_feats = visual_feats + self.alpha_ff.tanh() * self.ff_block(self.norm_ff(visual_feats))

        return visual_feats


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    """
    Adapted Transformer Block with integration points for Nepali traditional clothing try-on.

    \ud83d\udd2c Research Hooks:
    - Insert cultural vectors (e.g. region, dress type) into attention/cross-attention.
    - Control norm/adaptive features with traditional identity embeddings.
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

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        self.norm1 = (
            AdaLayerNorm(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm else
            AdaLayerNormZero(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm_zero else
            nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
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

        self.attn2, self.norm2 = None, None
        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm else
                nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            )
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )

        self.norm3 = (
            None if self.use_ada_layer_norm_single
            else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        )
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        self.fuser = None
        if attention_type in {"gated", "gated-text-image"}:
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        if self.use_ada_layer_norm_single:
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
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:

        if self.pos_embed:
            hidden_states = self.pos_embed(hidden_states)

        spatial_attn_inputs.append(hidden_states)

        norm_h = self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        attn_out = self.attn1(norm_h, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_out

        if self.fuser:
            gligen = cross_attention_kwargs.get("gligen", None) if cross_attention_kwargs else None
            if gligen is not None:
                hidden_states = self.fuser(hidden_states, gligen["objs"])

        if self.attn2:
            norm_h = self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            hidden_states = hidden_states + self.attn2(norm_h, encoder_hidden_states=encoder_hidden_states)

        norm_h = self.norm3(hidden_states) if self.norm3 else hidden_states
        ff_out = self.ff(norm_h)
        hidden_states = hidden_states + ff_out

        return hidden_states, spatial_attn_inputs


class FeedForward(nn.Module):
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
        linear_cls = LoRACompatibleLinear if not USE_PEFT_BACKEND else nn.Linear

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)

        self.net = nn.ModuleList([
            act_fn,
            nn.Dropout(dropout),
            linear_cls(inner_dim, dim_out),
        ])
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        compatible_cls = (GEGLU,) if USE_PEFT_BACKEND else (GEGLU, LoRACompatibleLinear)
        for module in self.net:
            if isinstance(module, compatible_cls):
                hidden_states = module(hidden_states, scale)
            else:
                hidden_states = module(hidden_states)
        return hidden_states
