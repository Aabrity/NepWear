from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .attention_Nepvton import BasicTransformerBlock

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import (
    ImagePositionalEmbeddings,
    CaptionProjection,
    PatchEmbed,
)
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle


@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    Output container for Transformer2DModel, representing
    the final processed latent features or probability distributions.

    Attributes:
        sample (torch.FloatTensor):
            The output tensor from the transformer, either
            continuous latent features or discrete class logits.
    """

    sample: torch.FloatTensor


class Transformer2DModel(ModelMixin, ConfigMixin):
    """
    Modular 2D Transformer for NepVTON - Nepali Traditional Virtual Try-On,
    supporting continuous, discrete (vectorized), and patch-based inputs.

    This model is designed for image latent manipulation conditioned on pose, style,
    and timestep embeddings, enabling high-fidelity garment try-on with
    efficient fine-tuning capabilities (via LoRA layers when PEFT is disabled).

    Parameters are registered in the config for reproducibility and checkpointing.

    Input modes supported:
        - Continuous latent images (e.g., dress style latent maps)
        - Discrete token embeddings (quantized latent vectors)
        - Patch embeddings (localized patches for fine detail attention)
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: Optional[int] = None,
    ):
        super().__init__()

        # Initialize layer classes (Conv/Linear) respecting PEFT backend compatibility
        self.conv_cls, self.linear_cls = self._choose_layer_classes()

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.dropout = dropout

        # Input/output channel config and mode selection
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        self.sample_size = sample_size
        self.num_vector_embeds = num_vector_embeds
        self.patch_size = patch_size
        self.use_linear_projection = use_linear_projection

        # Modes flags
        self.is_continuous = self._check_continuous_mode()
        self.is_vectorized = self._check_vectorized_mode()
        self.is_patch_based = self._check_patch_mode()

        # Validate mode exclusivity
        self._validate_input_modes()

        # Setup input embeddings or projections per mode
        self._build_input_embeddings(norm_num_groups, norm_elementwise_affine, norm_eps)

        # Setup transformer blocks with conditioning options
        self.transformer_blocks = self._build_transformer_blocks(
            num_layers,
            cross_attention_dim,
            activation_fn,
            num_embeds_ada_norm,
            attention_bias,
            only_cross_attention,
            double_self_attention,
            upcast_attention,
            norm_type,
            norm_elementwise_affine,
            norm_eps,
            attention_type,
        )

        # Setup output layers to map back to expected output shape/format
        self._build_output_layers(norm_type, norm_eps)

        # Additional conditioning layers
        self._build_adaptive_layers(norm_type)
        self._build_caption_projection(caption_channels)

        self.gradient_checkpointing = False

    # ---------- Initialization helpers -----------

    def _choose_layer_classes(self):
        """Return compatible conv and linear layers based on PEFT backend."""
        if USE_PEFT_BACKEND:
            return nn.Conv2d, nn.Linear
        else:
            return LoRACompatibleConv, LoRACompatibleLinear

    def _check_continuous_mode(self) -> bool:
        return (self.in_channels is not None) and (self.patch_size is None)

    def _check_vectorized_mode(self) -> bool:
        return self.num_vector_embeds is not None

    def _check_patch_mode(self) -> bool:
        return (self.in_channels is not None) and (self.patch_size is not None)

    def _validate_input_modes(self):
        mode_sum = sum([self.is_continuous, self.is_vectorized, self.is_patch_based])
        if mode_sum != 1:
            raise ValueError(
                "Exactly one input mode must be specified among "
                "`in_channels` (continuous), `num_vector_embeds` (vectorized), or `patch_size` (patch-based)."
            )
        if self.is_vectorized and self.sample_size is None:
            raise ValueError("`sample_size` must be specified for vectorized input mode.")
        if self.is_patch_based and self.sample_size is None:
            raise ValueError("`sample_size` must be specified for patch input mode.")

    def _build_input_embeddings(
        self, norm_num_groups: int, norm_elementwise_affine: bool, norm_eps: float
    ):
        """Create input projection or embedding layers depending on mode."""
        if self.is_continuous:
            self.norm = nn.GroupNorm(
                num_groups=norm_num_groups, num_channels=self.in_channels, eps=1e-6, affine=True
            )
            if self.use_linear_projection:
                self.proj_in = self.linear_cls(self.in_channels, self.inner_dim)
            else:
                self.proj_in = self.conv_cls(self.in_channels, self.inner_dim, kernel_size=1, stride=1, padding=0)

        elif self.is_vectorized:
            self.height = self.width = self.sample_size
            self.num_latent_pixels = self.height * self.width
            self.latent_embedding = ImagePositionalEmbeddings(
                num_embed=self.num_vector_embeds, embed_dim=self.inner_dim, height=self.height, width=self.width
            )

        elif self.is_patch_based:
            self.height = self.width = self.sample_size
            interpolation_scale = max(self.sample_size // 64, 1)
            self.pos_embed = PatchEmbed(
                height=self.sample_size,
                width=self.sample_size,
                patch_size=self.patch_size,
                in_channels=self.in_channels,
                embed_dim=self.inner_dim,
                interpolation_scale=interpolation_scale,
            )

    def _build_transformer_blocks(
        self,
        num_layers: int,
        cross_attention_dim: Optional[int],
        activation_fn: str,
        num_embeds_ada_norm: Optional[int],
        attention_bias: bool,
        only_cross_attention: bool,
        double_self_attention: bool,
        upcast_attention: bool,
        norm_type: str,
        norm_elementwise_affine: bool,
        norm_eps: float,
        attention_type: str,
    ) -> nn.ModuleList:
        """Construct transformer blocks with the given parameters."""
        blocks = []
        for _ in range(num_layers):
            block = BasicTransformerBlock(
                dim=self.inner_dim,
                num_heads=self.num_attention_heads,
                head_dim=self.attention_head_dim,
                dropout=self.dropout,
                cross_attention_dim=cross_attention_dim,
                activation_fn=activation_fn,
                num_embeds_ada_norm=num_embeds_ada_norm,
                attention_bias=attention_bias,
                only_cross_attention=only_cross_attention,
                double_self_attention=double_self_attention,
                upcast_attention=upcast_attention,
                norm_type=norm_type,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=norm_eps,
                attention_type=attention_type,
            )
            blocks.append(block)
        return nn.ModuleList(blocks)

    def _build_output_layers(self, norm_type: str, norm_eps: float):
        """Create output layers for each mode to map back to expected output format."""
        if self.is_continuous:
            if self.use_linear_projection:
                self.proj_out = self.linear_cls(self.inner_dim, self.in_channels)
            else:
                self.proj_out = self.conv_cls(self.inner_dim, self.in_channels, kernel_size=1, stride=1, padding=0)

        elif self.is_vectorized:
            self.norm_out = nn.LayerNorm(self.inner_dim, eps=norm_eps)
            self.out = nn.Linear(self.inner_dim, self.num_vector_embeds - 1)

        elif self.is_patch_based:
            if norm_type != "ada_norm_single":
                self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
                self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
                self.proj_out_2 = nn.Linear(self.inner_dim, self.patch_size * self.patch_size * self.out_channels)
            else:
                self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
                # Initialize learnable scale_shift_table for adaptive layer norm
                self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / (self.inner_dim ** 0.5))
                self.proj_out = nn.Linear(self.inner_dim, self.patch_size * self.patch_size * self.out_channels)

    def _build_adaptive_layers(self, norm_type: str):
        """Set up AdaLayerNormSingle module for adaptive normalization when used."""
        self.adaln_single = None
        self.use_additional_conditions = False
        if norm_type == "ada_norm_single":
            self.use_additional_conditions = (self.sample_size == 128)
            self.adaln_single = AdaLayerNormSingle(
                self.inner_dim, use_additional_conditions=self.use_additional_conditions
            )

    def _build_caption_projection(self, caption_channels: Optional[int]):
        """Optional projection layer for textual caption embeddings (e.g., garment description)."""
        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = CaptionProjection(in_features=caption_channels, hidden_size=self.inner_dim)

    # ----------- Forward pass helpers -----------

    def _preprocess_inputs(
        self,
        hidden_states: torch.Tensor,
        lora_scale: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Preprocess input based on mode to produce transformer-friendly embeddings.

        Args:
            hidden_states: raw input tensor (continuous, discrete indices, or patches)
            lora_scale: LoRA scale factor for fine-tuning adjustment

        Returns:
            Tuple of:
                - processed hidden states tensor,
                - residual tensor for skip connection (if applicable)
        """
        residual = None
        if self.is_continuous:
            batch, channels, height, width = hidden_states.shape
            residual = hidden_states
            hidden_states = self.norm(hidden_states)

            if self.use_linear_projection:
                hidden_states = (
                    self.proj_in(hidden_states.view(batch, channels, -1).permute(0, 2, 1))
                    if USE_PEFT_BACKEND
                    else self.proj_in(hidden_states.view(batch, channels, -1).permute(0, 2, 1), scale=lora_scale)
                )
            else:
                hidden_states = (
                    self.proj_in(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_in(hidden_states)
                )
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, self.inner_dim)
        elif self.is_vectorized:
            hidden_states = self.latent_embedding(hidden_states)
        elif self.is_patch_based:
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            hidden_states = self.pos_embed(hidden_states)
        else:
            raise ValueError("Unknown input mode during preprocessing.")

        return hidden_states, residual

    def _postprocess_outputs(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Convert transformer output back to required output format per input mode.

        Args:
            hidden_states: transformer output tensor
            residual: skip connection tensor if any (for continuous inputs)
            timestep: current diffusion timestep embedding
            added_cond_kwargs: additional conditioning for AdaLayerNormSingle
            class_labels: class conditioning labels

        Returns:
            Final output tensor
        """
        if self.is_continuous:
            batch = residual.shape[0]
            height = residual.shape[2]
            width = residual.shape[3]

            if self.use_linear_projection:
                hidden_states = (
                    self.proj_out(hidden_states)
                    if USE_PEFT_BACKEND
                    else self.proj_out(hidden_states, scale=1.0)
                )
                hidden_states = hidden_states.reshape(batch, height, width, self.in_channels).permute(0, 3, 1, 2)
            else:
                hidden_states = hidden_states.reshape(batch, height, width, self.inner_dim).permute(0, 3, 1, 2)
                hidden_states = (
                    self.proj_out(hidden_states)
                    if USE_PEFT_BACKEND
                    else self.proj_out(hidden_states, scale=1.0)
                )

            return hidden_states + residual

        elif self.is_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            logits = logits.permute(0, 2, 1)  # batch x classes x pixels
            # Return log-softmax for stable discrete latent training
            return F.log_softmax(logits.double(), dim=1).float()

        elif self.is_patch_based:
            # AdaLayerNormSingle conditioning on timestep and additional conds
            if self.adaln_single is not None:
                if self.use_additional_conditions and added_cond_kwargs is None:
                    raise ValueError("`added_cond_kwargs` required for additional conditions in AdaLayerNormSingle.")
                batch_size = hidden_states.shape[0]
                timestep, embedded_timestep = self.adaln_single(
                    timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
                )

            if self.adaln_single is None:
                # Normal LayerNorm + shift and scale from projection for conditioning
                conditioning = self.transformer_blocks[0].norm1.emb(timestep, class_labels, hidden_dtype=hidden_states.dtype)
                shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                hidden_states = self.proj_out_2(hidden_states)
            else:
                # Adaptive norm with learnable scale_shift_table and embedded timestep
                shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states)
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = self.proj_out(hidden_states).squeeze(1)

            # Unpatchify to full image shape
            h = w = int(hidden_states.shape[1] ** 0.5) if self.adaln_single is None else self.height
            patches = hidden_states.view(
                -1, h, w, self.patch_size, self.patch_size, self.out_channels
            )
            patches = torch.einsum("nhwpqc->nchpwq", patches)
            return patches.reshape(-1, self.out_channels, h * self.patch_size, w * self.patch_size)

        else:
            raise ValueError("Unknown input mode during postprocessing.")

    # ----------- Forward method -----------

    def forward(
        self,
        hidden_states: torch.Tensor,
        spatial_attn_inputs: Optional[list] = None,
        spatial_attn_idx: int = 0,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, list, int], Tuple[Transformer2DModelOutput, list, int]]:
        """
        Forward pass of Transformer2DModel.

        Args:
            hidden_states: input tensor of shape depending on mode (continuous images, discrete tokens, or patches)
            spatial_attn_inputs: optional list for spatial attention inputs (tracking attention maps)
            spatial_attn_idx: current index for spatial attention inputs
            encoder_hidden_states: conditioning embeddings (e.g., pose, garment style)
            timestep: diffusion timestep embedding for conditioning normalization
            added_cond_kwargs: extra conditioning tensors for AdaLayerNormSingle
            class_labels: class conditioning for denoising step
            cross_attention_kwargs: additional kwargs for cross attention
            attention_mask: mask for self-attention
            encoder_attention_mask: mask for cross-attention
            return_dict: whether to return a dict output or a tuple

        Returns:
            Tuple of (output, spatial_attn_inputs, spatial_attn_idx) either wrapped in
            Transformer2DModelOutput or plain tensor.
        """
        spatial_attn_inputs = spatial_attn_inputs or []

        # Convert attention masks from boolean to additive bias
        attention_mask = self._convert_mask_to_bias(attention_mask)
        encoder_attention_mask = self._convert_mask_to_bias(encoder_attention_mask)

        # Preprocess input tensor into transformer embedding space
        hidden_states, residual = self._preprocess_inputs(hidden_states, lora_scale=1.0)

        # Apply transformer blocks sequentially
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                spatial_attn_inputs=spatial_attn_inputs,
                spatial_attn_idx=spatial_attn_idx,
            )
            spatial_attn_idx += 1

        # Postprocess outputs back to input domain
        output = self._postprocess_outputs(hidden_states, residual, timestep, added_cond_kwargs, class_labels)

        if return_dict:
            return Transformer2DModelOutput(sample=output), spatial_attn_inputs, spatial_attn_idx
        return output, spatial_attn_inputs, spatial_attn_idx

    # ----------- Utility -----------

    @staticmethod
    def _convert_mask_to_bias(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Convert boolean attention masks into additive bias masks
        for transformer compatibility. Transformer expects additive
        masks where masked positions have large negative values.

        Args:
            mask: boolean mask tensor with True for valid tokens

        Returns:
            Additive bias mask or None if input is None.
        """
        if mask is None:
            return None
        return torch.where(mask, torch.tensor(0.0, device=mask.device), torch.tensor(-1e9, device=mask.device))
