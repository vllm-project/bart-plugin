# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, TypedDict

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    BartConfig,
    BartTokenizer,
    BatchFeature,
    Florence2Config,
    Florence2Processor,
)
from vllm.config import CacheConfig, VllmConfig
from vllm.config.lora import LoRAConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.cross_attention import CrossAttention
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsQuant,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    cast_overflow_tensors,
    flatten_bn,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, ModalityData
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ModalityDataItems,
    ModalityDataParser,
    MultiModalDataItems,
    MultiModalDataParser,
    ProcessorBatchItems,
)
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.collection_utils import is_list_of
from vllm.v1.attention.backend import AttentionType

from vllm_bart_plugin.bart import (
    BartDecoder,
    BartEncoder,
    BartParallelLMHead,
    BartScaledWordEmbedding,
)


class Florence2ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: (batch_size, num_channel, height, width)"""


def _drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class Florence2VisionDropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class Florence2VisionMLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Florence2VisionConvEmbed(nn.Module):
    """Image-to-patch embedding via strided convolution (NCHW in, NCHW out)."""

    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        stride: int,
        padding: int,
        pre_norm: bool,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.conv = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        dim_norm = in_channels if pre_norm else embed_dim
        self.norm = nn.LayerNorm(dim_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv(x)
        if not self.pre_norm:
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class Florence2VisionChannelAttention(nn.Module):
    """Channel (group) attention — attends over the channel dimension."""

    def __init__(self, dim: int, groups: int, qkv_bias: bool = True):
        super().__init__()
        self.groups = groups
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Reshape: (B, N, 3, groups, C//groups) -> (3, B, groups, N, C//groups)
        qkv = self.qkv(x).reshape(B, N, 3, self.groups, C // self.groups)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, groups, N, C//groups)

        # Scale by sequence length and compute channel-to-channel attention
        q = q * (float(N) ** -0.5)
        attn = (q.transpose(-2, -1) @ k).softmax(dim=-1)  # (B, groups, C//g, C//g)
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)  # (B, groups, N, C//g)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class Florence2VisionChannelBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        groups: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.channel_attn = Florence2VisionChannelAttention(embed_dim, groups, qkv_bias)
        self.drop_path1 = (
            Florence2VisionDropPath(drop_path_rate)
            if drop_path_rate > 0
            else nn.Identity()
        )
        self.conv2 = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = Florence2VisionMLP(embed_dim, mlp_ratio)
        self.drop_path2 = (
            Florence2VisionDropPath(drop_path_rate)
            if drop_path_rate > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Sub-block 1: depthwise conv residual + channel attention
        x = self.conv1(x) + x
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        residual = x_flat
        x_flat = residual + self.drop_path1(self.channel_attn(self.norm1(x_flat)))
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        # Sub-block 2: depthwise conv residual + FFN
        x = self.conv2(x) + x
        x_flat = x.flatten(2).transpose(1, 2)
        residual = x_flat
        x_flat = residual + self.drop_path2(self.ffn(self.norm2(x_flat)))
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        return x


class Florence2VisionWindowAttention(nn.Module):
    """Window-based local spatial self-attention."""

    def __init__(
        self, dim: int, num_heads: int, window_size: int, qkv_bias: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C) BHWC
        B, H, W, C = x.shape
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x.shape[1], x.shape[2]

        # Partition into non-overlapping windows
        x = x.view(
            B,
            Hp // self.window_size,
            self.window_size,
            Wp // self.window_size,
            self.window_size,
            C,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size**2, C)

        Bw, Nw = x.shape[:2]
        qkv = (
            self.qkv(x)
            .reshape(Bw, Nw, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(Bw, Nw, C)
        x = self.proj(x)

        # Merge windows back
        x = x.view(-1, self.window_size, self.window_size, C)
        x = x.view(
            B,
            Hp // self.window_size,
            Wp // self.window_size,
            self.window_size,
            self.window_size,
            C,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        return x.view(B, H * W, C)


class Florence2VisionSpatialBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.window_attn = Florence2VisionWindowAttention(
            embed_dim, num_heads, window_size, qkv_bias
        )
        self.drop_path1 = (
            Florence2VisionDropPath(drop_path_rate)
            if drop_path_rate > 0
            else nn.Identity()
        )
        self.conv2 = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = Florence2VisionMLP(embed_dim, mlp_ratio)
        self.drop_path2 = (
            Florence2VisionDropPath(drop_path_rate)
            if drop_path_rate > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Sub-block 1: depthwise conv residual + window attention
        x = self.conv1(x) + x
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        residual = x_flat
        x_bhwc = self.norm1(x_flat).view(B, H, W, C)
        x_flat = residual + self.drop_path1(self.window_attn(x_bhwc))
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        # Sub-block 2: depthwise conv residual + FFN
        x = self.conv2(x) + x
        x_flat = x.flatten(2).transpose(1, 2)
        residual = x_flat
        x_flat = residual + self.drop_path2(self.ffn(self.norm2(x_flat)))
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        return x


class Florence2VisionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_groups: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        spatial_drop_path_rate: float = 0.0,
        channel_drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.spatial_block = Florence2VisionSpatialBlock(
            embed_dim,
            num_heads,
            window_size,
            mlp_ratio,
            qkv_bias,
            spatial_drop_path_rate,
        )
        self.channel_block = Florence2VisionChannelBlock(
            embed_dim, num_groups, mlp_ratio, qkv_bias, channel_drop_path_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_block(self.spatial_block(x))


class Florence2VisionBackbone(nn.Module):
    """
    DaViT-based vision backbone for the new Florence-2 architecture.
    Produces NCHW feature maps for the multi-modal projector.
    """

    def __init__(self, config):
        super().__init__()
        embed_dims = config.embed_dim
        num_stages = len(embed_dims)
        depths = config.depths
        mlp_ratio = getattr(config, "mlp_ratio", 4.0)
        qkv_bias = getattr(config, "qkv_bias", True)

        dpr = [
            x.item() for x in torch.linspace(0, config.drop_path_rate, sum(depths) * 2)
        ]
        depth_offset = 0

        convs = []
        blocks = []
        for stage_idx in range(num_stages):
            in_ch = config.in_channels if stage_idx == 0 else embed_dims[stage_idx - 1]
            convs.append(
                Florence2VisionConvEmbed(
                    patch_size=config.patch_size[stage_idx],
                    in_channels=in_ch,
                    embed_dim=embed_dims[stage_idx],
                    stride=config.patch_stride[stage_idx],
                    padding=config.patch_padding[stage_idx],
                    pre_norm=config.patch_prenorm[stage_idx],
                )
            )
            stage_blocks = nn.ModuleList(
                [
                    Florence2VisionBlock(
                        embed_dim=embed_dims[stage_idx],
                        num_heads=config.num_heads[stage_idx],
                        num_groups=config.num_groups[stage_idx],
                        window_size=config.window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        spatial_drop_path_rate=dpr[depth_offset + block_idx * 2],
                        channel_drop_path_rate=dpr[depth_offset + block_idx * 2 + 1],
                    )
                    for block_idx in range(depths[stage_idx])
                ]
            )
            blocks.append(stage_blocks)
            depth_offset += depths[stage_idx] * 2

        self.convs = nn.ModuleList(convs)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, 3, H, W). Returns: (B, C_last, H', W') NCHW feature map."""
        for conv, block_list in zip(self.convs, self.blocks):
            x = conv(x)
            for block in block_list:
                x = block(x)
        return x


class Florence2VisionLearnedAbsolutePositionEmbedding2D(nn.Module):
    """2D learned absolute position embedding (NCHW interface)."""

    def __init__(self, embedding_dim: int = 256, num_pos: int = 50):
        super().__init__()
        self.row_embeddings = nn.Embedding(num_pos, embedding_dim // 2)
        self.column_embeddings = nn.Embedding(
            num_pos, embedding_dim - (embedding_dim // 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) — returns positional embeddings of same shape."""
        height, width = x.shape[-2:]
        x_emb = self.column_embeddings(
            torch.arange(width, device=x.device)
        )  # (W, C//2)
        y_emb = self.row_embeddings(torch.arange(height, device=x.device))  # (H, C//2)
        pos = torch.cat(
            [
                x_emb.unsqueeze(0).expand(height, -1, -1),
                y_emb.unsqueeze(1).expand(-1, width, -1),
            ],
            dim=-1,
        )  # (H, W, C)
        return (
            pos.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        )  # (B, C, H, W)


class Florence2VisionPositionalEmbeddingCosine1D(nn.Module):
    """Sinusoidal temporal positional embedding; returns (T, C) without batch dim."""

    def __init__(self, embed_dim: int = 512, max_seq_len: int = 100) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        factor = math.log(10000)
        denominator = torch.exp(-factor * torch.arange(0, embed_dim, 2) / embed_dim)
        frequencies = torch.arange(0, max_seq_len).reshape(max_seq_len, 1) * denominator
        pos_idx_to_embed = torch.zeros((max_seq_len, embed_dim))
        pos_idx_to_embed[:, 0::2] = torch.sin(frequencies)
        pos_idx_to_embed[:, 1::2] = torch.cos(frequencies)
        self.pos_idx_to_embed = nn.Parameter(pos_idx_to_embed, requires_grad=False)

    def forward(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        """seq_embeds: (B, T, C) — returns (T, C) positional embeddings."""
        len_seq = seq_embeds.size(1)
        assert len_seq <= self.max_seq_len
        return self.pos_idx_to_embed[0:len_seq, :]  # (T, C)


class Florence2MultiModalProjector(nn.Module):
    """
    Projects vision backbone features into the language model's embedding space.
    Applies 2D spatial positional embeddings, a temporal embedding, pools to
    produce both a spatial-average and a per-token representation, then projects
    with a linear layer + layer norm.

    Input:  (B, C, H, W) NCHW feature map from Florence2VisionBackbone.
    Output: (B, 1 + H*W, projection_dim) token embeddings for the encoder.
    """

    def __init__(self, config):
        super().__init__()
        embed_dim = config.vision_config.embed_dim[-1]
        proj_dim = config.vision_config.projection_dim

        self.image_projection = nn.Linear(embed_dim, proj_dim, bias=False)
        self.image_proj_norm = nn.LayerNorm(proj_dim)
        self.image_position_embed = Florence2VisionLearnedAbsolutePositionEmbedding2D(
            embedding_dim=embed_dim,
            num_pos=config.vision_config.max_position_embeddings,
        )
        self.visual_temporal_embed = Florence2VisionPositionalEmbeddingCosine1D(
            embed_dim=embed_dim,
            max_seq_len=config.vision_config.max_temporal_embeddings,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        # image_features: (B, C, H, W)
        B, C, H, W = image_features.shape

        # 2D spatial positional embedding
        pos = self.image_position_embed(image_features)  # (B, C, H, W)
        x = (image_features + pos).flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Temporal positional embedding (T=1 for single-frame images)
        temporal_embed = self.visual_temporal_embed(x[:, :1, :])  # (1, C)
        x = x + temporal_embed  # broadcast over H*W tokens

        # Pool: spatial average (1 token) + all spatial tokens (H*W tokens)
        x_t = x.unsqueeze(1)  # (B, 1, H*W, C) — treat as T=1 video
        spatial_avg = x_t.mean(dim=2)  # (B, 1, C)
        temporal_avg = x_t.mean(dim=1)  # (B, H*W, C)
        x = torch.cat([spatial_avg, temporal_avg], dim=1)  # (B, 1+H*W, C)

        x = self.image_projection(x)  # (B, 1+H*W, proj_dim)
        x = self.image_proj_norm(x)
        return x


# Language backbone and processor implementation
class Florence2LanguageModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.vocab_size = config.vocab_size

        self.shared = BartScaledWordEmbedding(self.vocab_size, config.d_model)
        self.encoder = BartEncoder(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder",
        )
        self.decoder = BartDecoder(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.decoder",
        )

        if self.config.tie_word_embeddings:
            self.encoder.embed_tokens.weight = self.shared.weight
            self.decoder.embed_tokens.weight = self.shared.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # decoder outputs consists of
        # (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids=input_ids,
            decoder_positions=positions,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_outputs,
        )

        return decoder_outputs


class Florence2LanguageForConditionalGeneration(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config

        self.config = config
        self.model = Florence2LanguageModel(
            vllm_config=vllm_config, prefix=f"{prefix}.model"
        )
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.vocab_size = config.vocab_size
        self.lm_head = BartParallelLMHead(
            self.vocab_size, config.d_model, embed_scale=embed_scale
        )

        self.logits_processor = LogitsProcessor(self.vocab_size, config.vocab_size)
        if self.config.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.shared)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_outputs: torch.Tensor | None = None,
        # num_encoder_outputs: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            encoder_outputs=encoder_outputs,
        )

    def get_encoder_outputs(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_positions: torch.Tensor,
        inputs_embeds: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        # Run encoder attention if a non-zero number of encoder tokens
        # are provided as input
        encoder_hidden_states = self.model.encoder(
            input_ids=encoder_input_ids,
            positions=encoder_positions,
            inputs_embeds=inputs_embeds,
        )
        return encoder_hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.encoder.embed_tokens(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("encoder_attn.kv_proj", "encoder_attn.k_proj", "k"),
            ("encoder_attn.kv_proj", "encoder_attn.v_proj", "v"),
            ("self_attn.qkv_proj", "self_attn.q_proj", "q"),
            ("self_attn.qkv_proj", "self_attn.k_proj", "k"),
            ("self_attn.qkv_proj", "self_attn.v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "final_logits_bias" in name:
                    continue
                if self.config.tie_word_embeddings and "embed_tokens" in name:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Florence2ProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> Florence2Config:
        return self.ctx.get_hf_config()

    def get_hf_processor(self) -> Florence2Processor:
        return self.ctx.get_hf_processor()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_num_image_tokens(self) -> int:
        processor = self.get_hf_processor()
        return processor.num_image_tokens


class Florence2DummyInputsBuilder(BaseDummyInputsBuilder[Florence2ProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width = target_height = (
            self.info.get_hf_config().vision_config.projection_dim
        )

        return {
            "image": self._get_dummy_images(
                width=target_width, height=target_height, num_images=num_images
            )
        }


class Florence2MultiModalProcessor(EncDecMultiModalProcessor[Florence2ProcessingInfo]):

    def __init__(self, info, dummy_inputs, *, cache=None) -> None:
        super().__init__(info, dummy_inputs, cache=cache)
        # Florence2Config does not expose decoder_start_token_id at the
        # top level (it lives in text_config), so vLLM falls back to BOS
        # (token 0) and incorrectly prepends it to the decoder prompt.
        # Patch the top-level hf_config so vLLM's _prepare_decoder_input_ids
        # sees the real value (EOS / token 2) and leaves our prompt intact.
        hf_config = info.get_hf_config()
        if getattr(hf_config, "decoder_start_token_id", None) is None:
            hf_config.decoder_start_token_id = (
                hf_config.text_config.decoder_start_token_id
            )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # The Florence2Processor already inserts image_token_id placeholders
        # into the input_ids (577 tokens for a 768x768 image), so we tell
        # vllm to find those existing placeholders rather than insert new ones.
        return bool(mm_items.get_all_counts().get("image", 0))

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
    ) -> str | list[int]:
        return prompt

    def create_decoder_prompt(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
    ) -> str | list[int]:
        text_config = self.info.get_hf_config().text_config
        # Decoder prompt mirrors what transformers does before open-ended
        # generation: start with decoder_start_token_id (</s>, token 2),
        # then include forced_bos_token_id (<s>, token 0) so that vLLM
        # generates from the same position as transformers step 2.
        decoder_prompt = [text_config.decoder_start_token_id]
        forced_bos = getattr(text_config, "forced_bos_token_id", None)
        if forced_bos is not None:
            decoder_prompt.append(forced_bos)
        return decoder_prompt

    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:
        hf_processor = self.info.get_hf_processor()
        tokenizer: BartTokenizer = hf_processor.tokenizer
        prompt_text = tokenizer.decode(prompt_tokens)
        # convert task tokens to prompt
        prompt_text = hf_processor._construct_prompts([prompt_text])[0]
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        return prompt_tokens

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            processed_outputs = super()._call_hf_processor(
                prompt, mm_data, mm_kwargs, tok_kwargs
            )
        else:
            hf_processor = self.info.get_hf_processor()
            tokenizer = hf_processor.tokenizer
            prompt = hf_processor._construct_prompts([prompt])[0]
            processed_outputs = tokenizer(
                prompt, add_special_tokens=True, return_tensors="pt"
            )
        processed_outputs["encoder_input_ids"] = processed_outputs["input_ids"]
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            encoder_input_ids=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        # The placeholder must cover the FULL encoder input sequence (image
        # tokens + text/task tokens) so that vLLM's _get_encoder_seq_lens
        # computes the correct value for cross-attention KV cache allocation.
        # Using only the image token count (577) would cause cross-attention
        # to read only 577/590 K/V pairs, skipping the task-prompt tokens.
        #
        # With _hf_processor_applies_updates=True, vLLM detects the existing
        # token sequence rather than inserting new tokens. By setting the
        # insertion to the full encoder_input_ids sequence, the detected
        # placeholder range covers all 590 encoder tokens.
        insertion: list[int]
        image_items = out_mm_kwargs.get("image", [])
        if image_items:
            item_data = image_items[0].get_data()
            enc_ids = item_data.get("encoder_input_ids")
            if enc_ids is not None:
                insertion = enc_ids.tolist()
            else:
                # Cache hit: encoder_input_ids not available; fall back.
                hf_config = self.info.get_hf_config()
                insertion = (
                    [hf_config.image_token_id] * self.info.get_num_image_tokens()
                )
        else:
            hf_config = self.info.get_hf_config()
            insertion = (
                [hf_config.image_token_id] * self.info.get_num_image_tokens()
            )

        return [
            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.start(),
                insertion=insertion,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Florence2MultiModalProcessor,
    info=Florence2ProcessingInfo,
    dummy_inputs=Florence2DummyInputsBuilder,
)
class Florence2ForConditionalGeneration(nn.Module, SupportsMultiModal):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        processor_config = vllm_config.model_config.hf_image_processor_config

        self.config = config
        self.processor_config = processor_config
        assert config.vision_config.model_type == "florence_vision", (
            f"only Florence Vision is supported for now. "
            f"Received model type: {config.vision_config.model_type}"
        )
        self.vision_tower = Florence2VisionBackbone(config.vision_config)
        self.multi_modal_projector = Florence2MultiModalProjector(config)
        self.language_model = Florence2LanguageForConditionalGeneration(
            vllm_config=vllm_config.with_hf_config(config.text_config),
            prefix=f"{prefix}.language_model",
        )
        self.pad_token_id = config.text_config.pad_token_id

    def _validate_pixel_values(
        self, data: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor | list[torch.Tensor]:
        # The image processor config may use "size" or "crop_size"; fall back
        # to reading the actual tensor shape if neither key is available.
        cfg = self.processor_config
        size = cfg.get("size") or cfg.get("crop_size")
        if size is None:
            return data

        h, w = size["height"], size["width"]
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)
            if actual_dims != expected_dims:
                raise ValueError(
                    "The expected shape of pixel values per batch "
                    f"is {expected_dims}. You supplied {actual_dims}."
                )

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(self, **kwargs: object):
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None
        if pixel_values is not None and image_embeds is not None:
            raise ValueError("Both pixel values and image embeds are provided.")
        if pixel_values is not None:
            return Florence2ImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(pixel_values),
            )
        raise NotImplementedError("image_embeds not supported.")

    def _parse_and_validate_encoder_input(self, **kwargs: object) -> list[torch.Tensor]:
        encoder_input_ids = kwargs.get("encoder_input_ids", kwargs.get("input_ids"))
        if encoder_input_ids is None:
            return []
        if not isinstance(encoder_input_ids, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of encoder input_ids. Got type: {type(encoder_input_ids)}"
            )
        if isinstance(encoder_input_ids, list):
            return [
                item.unsqueeze(0) if item.dim() == 0 else item
                for item in encoder_input_ids
            ]
        return encoder_input_ids.unsqueeze(1).unbind(dim=0)

    def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(next(self.vision_tower.parameters()).dtype)
        return self.multi_modal_projector(self.vision_tower(pixel_values))

    def _process_image_input(
        self, image_input: Florence2ImagePixelInputs
    ) -> torch.Tensor:
        return self._encode_image(image_input["data"])

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        encoder_input_ids_list = self._parse_and_validate_encoder_input(**kwargs)
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            vision_embeddings = []
        else:
            vision_embeddings = self._process_image_input(image_input)

        if not encoder_input_ids_list:
            raise ValueError(
                "encoder_input_ids_list is empty - check multimodal data is being passed correctly."
            )

        # Batch encoder inputs (pad to max length if needed) and run a single forward pass.
        lengths = [t.numel() for t in encoder_input_ids_list]
        max_len = max(lengths) if lengths else 0
        assert max_len > 0, "Empty encoder_input_ids encountered."
        same_len = all(l == max_len for l in lengths)
        if len(encoder_input_ids_list) == 1:
            batch_encoder_input_ids = encoder_input_ids_list[0]
        elif same_len:
            batch_encoder_input_ids = torch.cat(encoder_input_ids_list, dim=0)
        else:
            batch_encoder_input_ids = torch.full(
                (len(encoder_input_ids_list), max_len),
                fill_value=self.pad_token_id,
                dtype=encoder_input_ids_list[0].dtype,
                device=encoder_input_ids_list[0].device,
            )
            for i, t in enumerate(encoder_input_ids_list):
                batch_encoder_input_ids[i, : t.numel()] = t.squeeze()
        inputs_embeds = self.language_model.model.encoder.embed_tokens(
            batch_encoder_input_ids
        )

        # Replace the leading image_token_id placeholders with vision features.
        if (
            isinstance(vision_embeddings, torch.Tensor)
            and vision_embeddings.numel() > 0
        ):
            num_vision = vision_embeddings.size(1)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[:, :num_vision, :] = vision_embeddings
        batch_encoder_positions = (
            torch.arange(
                inputs_embeds.size(1),
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            .unsqueeze(0)
            .expand(inputs_embeds.size(0), -1)
        )

        # Run encoder once on the batch, then split back per item.
        batch_encoder_output = self.language_model.model.encoder(
            input_ids=batch_encoder_input_ids,
            positions=batch_encoder_positions,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs: list[torch.Tensor] = batch_encoder_output.unbind(dim=0)
        if not same_len:
            encoder_outputs = [out[:l] for out, l in zip(encoder_outputs, lengths)]
        return encoder_outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_outputs: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_outputs is not None:
            # Assume same shape for all encoder outputs
            encoder_outputs = torch.cat(encoder_outputs, dim=0)

        hidden_states = self.language_model(
            input_ids,
            positions,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def _remap(weights: Iterable[tuple[str, torch.Tensor]]):
            for name, param in weights:
                # HF checkpoint layout (Florence2ForConditionalGeneration):
                #   model.vision_tower.*           -> vision_tower.*
                #   model.multi_modal_projector.*  -> multi_modal_projector.*
                #   model.language_model.*         -> language_model.model.*
                #       (HF uses BartModel directly; our wrapper adds .model)
                #   lm_head.*                      -> language_model.lm_head.*
                if name.startswith("model.vision_tower."):
                    name = name[len("model.") :]
                elif name.startswith("model.multi_modal_projector."):
                    name = name[len("model.") :]
                elif name.startswith("model.language_model."):
                    name = (
                        "language_model.model." + name[len("model.language_model.") :]
                    )
                elif name.startswith("lm_head."):
                    name = "language_model." + name
                yield name, param

        loader = AutoWeightsLoader(self)
        return loader.load_weights(_remap(weights))
