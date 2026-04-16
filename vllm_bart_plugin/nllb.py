# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Derived from M2M-100 / NLLB implementation in HuggingFace transformers.
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
"""PyTorch M2M-100 / NLLB model for vLLM.

Supports:
  - facebook/nllb-200-distilled-600M   (model_type=m2m_100, 6 enc/dec layers)
  - facebook/nllb-200-distilled-1.3B   (model_type=m2m_100, 24 enc/dec layers)
  - facebook/nllb-200-3.3B             (model_type=m2m_100, 36 enc/dec layers)

Architecture differences from BART (relevant for this implementation):
  - Sinusoidal (fixed) positional embeddings instead of learned
  - PRE-LayerNorm (norm before sublayer) instead of POST-LayerNorm
  - Extra layer_norm after all encoder / decoder layers (absent in BART)
  - ReLU activation instead of GELU
  - No final_logits_bias
  - Separate q/k/v projections in HF checkpoint → stacked by weight loader

Language handling:
  NllbTokenizer automatically appends the source-language token to encoder
  inputs.  Set forced_bos_token_id in SamplingParams to force the target
  language token as the first decoder output.
"""

import math
from collections.abc import Iterable, Mapping

import torch
from torch import nn
from transformers import M2M100Config
from transformers.utils import logging

from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
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
    cast_overflow_tensors,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

# Re-use BART attention classes — they only access config.d_model and
# config.*_attention_heads, which M2M100Config also provides.
from .bart import (
    BartCrossAttention,
    BartDummyInputsBuilder,
    BartEncoderAttention,
    BartDecoderSelfAttention,
    BartMultiModalProcessor,
    BartProcessingInfo,
)

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Sinusoidal positional embeddings
# ---------------------------------------------------------------------------

class M2M100SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional embeddings (non-learnable).

    Weights are pre-computed once and stored as a non-persistent buffer
    (not saved to / loaded from checkpoints).

    vLLM passes explicit ``positions`` tensors (0-indexed), so we index into
    the weight table using ``positions + offset`` to match HuggingFace's
    convention (positions start at padding_idx + 1 = 2 for M2M100).
    """

    def __init__(
        self,
        num_positions: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ):
        super().__init__()
        self.offset = 2  # matches HuggingFace M2M100 offset convention
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        weights = self._get_embedding(
            num_positions + self.offset, embedding_dim, padding_idx
        )
        # persistent=False: not saved to state_dict, re-computed at load time
        self.register_buffer("weights", weights, persistent=False)

    @staticmethod
    def _get_embedding(
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ) -> torch.Tensor:
        """Build sinusoidal embeddings matching HuggingFace M2M100."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.int64).float() * -emb
        )
        emb = (
            torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1)
            * emb.unsqueeze(0)
        )
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0.0
        # Match the default dtype (float32 normally, float16 in half-precision)
        return emb.to(torch.get_default_dtype())

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Return positional embeddings for the given position indices.

        Args:
            positions: integer tensor of shape (seq_len,) or (batch, seq_len).
        Returns:
            Float tensor of shape (*positions.shape, embedding_dim).
        """
        flat = (positions + self.offset).reshape(-1)
        embeds = self.weights.index_select(0, flat)
        return embeds.reshape(*positions.shape, self.embedding_dim)


# ---------------------------------------------------------------------------
# Scaled word embeddings
# ---------------------------------------------------------------------------

class M2M100ScaledWordEmbedding(VocabParallelEmbedding):
    """Word embeddings scaled by sqrt(d_model) when config.scale_embedding=True."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embed_scale: float = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale


# ---------------------------------------------------------------------------
# Encoder layer  (PRE-LayerNorm)
# ---------------------------------------------------------------------------

class M2M100EncoderLayer(nn.Module):
    """Single M2M100/NLLB encoder layer.

    Uses PRE-LayerNorm (norm applied before each sublayer), unlike BART
    which uses POST-LayerNorm.
    """

    def __init__(
        self,
        config: M2M100Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model

        # Re-use BART's self-attention implementation — config fields are
        # compatible (both expose d_model, encoder_attention_heads, etc.)
        self.self_attn = BartEncoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.encoder_ffn_dim,
            bias=True,
            quant_config=quant_config,
        )
        self.fc2 = RowParallelLinear(
            config.encoder_ffn_dim,
            self.embed_dim,
            bias=True,
            quant_config=quant_config,
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = get_act_fn(config.activation_function)  # relu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # PRE-norm self-attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        # PRE-norm FFN
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        fc1_out, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(fc1_out)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        return hidden_states


# ---------------------------------------------------------------------------
# Decoder layer  (PRE-LayerNorm)
# ---------------------------------------------------------------------------

class M2M100DecoderLayer(nn.Module):
    """Single M2M100/NLLB decoder layer with PRE-LayerNorm."""

    def __init__(
        self,
        config: M2M100Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartDecoderSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.encoder_attn = BartCrossAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder_attn",
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.decoder_ffn_dim,
            bias=True,
            quant_config=quant_config,
        )
        self.fc2 = RowParallelLinear(
            config.decoder_ffn_dim,
            self.embed_dim,
            bias=True,
            quant_config=quant_config,
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = get_act_fn(config.activation_function)  # relu

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # PRE-norm self-attention
        residual = decoder_hidden_states
        hidden_states = self.self_attn_layer_norm(decoder_hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        # PRE-norm cross-attention (only when encoder output is available)
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states = self.encoder_attn(
                decoder_hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states = residual + hidden_states

        # PRE-norm FFN
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        fc1_out, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(fc1_out)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class M2M100Encoder(nn.Module):
    """M2M100/NLLB encoder with sinusoidal positional embeddings."""

    def __init__(
        self,
        config: M2M100Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        embed_tokens: nn.Module | None = None,
        prefix: str = "",
    ):
        super().__init__()
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = M2M100ScaledWordEmbedding(
            config.vocab_size, config.d_model, embed_scale=embed_scale
        )
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = M2M100SinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [
                M2M100EncoderLayer(
                    config,
                    cache_config,
                    quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.encoder_layers)
            ]
        )
        # Final layer norm present in M2M100 but absent in BART
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(positions)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos

        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class M2M100Decoder(nn.Module):
    """M2M100/NLLB decoder with sinusoidal positional embeddings."""

    def __init__(
        self,
        config: M2M100Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        embed_tokens: nn.Module | None = None,
        prefix: str = "",
    ):
        super().__init__()
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = M2M100ScaledWordEmbedding(
            config.vocab_size, config.d_model, embed_scale=embed_scale
        )
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = M2M100SinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [
                M2M100DecoderLayer(
                    config,
                    cache_config,
                    quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.decoder_layers)
            ]
        )
        # Final layer norm present in M2M100 but absent in BART
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(decoder_input_ids)

        embed_pos = self.embed_positions(decoder_positions)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos

        for layer in self.layers:
            hidden_states = layer(
                decoder_hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class M2M100Model(nn.Module, SupportsQuant):
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: M2M100Config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.encoder = M2M100Encoder(
            config,
            cache_config,
            quant_config,
            prefix=f"{prefix}.encoder",
        )
        self.decoder = M2M100Decoder(
            config,
            cache_config,
            quant_config,
            prefix=f"{prefix}.decoder",
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        encoder_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        return self.decoder(
            decoder_input_ids=input_ids,
            decoder_positions=positions,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_outputs,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # M2M100 HF checkpoints store separate q/k/v projections.
        # Stack self-attention Q+K+V → qkv_proj (same strategy as BART).
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        # Stack cross-attention K+V → kv_proj; Q remains separate.
        cross_attn_stacked_params_mapping = [
            ("kv_proj", "k_proj", "k"),
            ("kv_proj", "v_proj", "v"),
        ]

        other_weights: list[tuple[str, torch.Tensor]] = []
        loaded_stacked_params: list[str] = []
        model_params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # embed_positions.weights is a non-persistent buffer — not present
            # in HF checkpoints, skip defensively if somehow encountered.
            if "embed_positions.weights" in name:
                continue

            # Cross-attention K/V stacking (only for encoder_attn layers)
            for param_name, weight_name, shard_id in cross_attn_stacked_params_mapping:
                if weight_name not in name or "encoder_attn" not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in model_params_dict:
                    continue
                param = model_params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_stacked_params.append(mapped)
                break
            else:
                # Self-attention Q/K/V stacking (skip cross-attn)
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name or "encoder_attn" in name:
                        continue
                    mapped = name.replace(weight_name, param_name)
                    if mapped not in model_params_dict:
                        continue
                    param = model_params_dict[mapped]
                    param.weight_loader(param, loaded_weight, shard_id)
                    loaded_stacked_params.append(mapped)
                    break
                else:
                    if name in model_params_dict:
                        other_weights.append((name, loaded_weight))

        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(other_weights)
        loaded_params.update(loaded_stacked_params)
        return loaded_params


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

class M2M100ProcessingInfo(BartProcessingInfo):
    """Processing info for M2M100 / NLLB models."""

    def get_hf_config(self) -> M2M100Config:
        return self.ctx.get_hf_config(M2M100Config)


class M2M100DummyInputsBuilder(BartDummyInputsBuilder):
    """Builds dummy inputs for profiling M2M100 / NLLB models."""
    pass


class M2M100MultiModalProcessor(BartMultiModalProcessor):
    """Multimodal processor for M2M100 / NLLB encoder-decoder models.

    Language routing for NLLB:

    **Decoder (target language):**
    Pass the FLORES-200 target language code as the ``decoder_prompt``
    (e.g. ``"fra_Latn"``).  This processor's ``create_decoder_prompt``
    converts it to the corresponding special-token ID via
    ``tokenizer.convert_tokens_to_ids``, so the decoder starts generating
    in the target language.

    **Encoder (source language):**
    The encoder text is tokenized by ``_call_hf_processor``.  NLLB's
    tokenizer prepends the source-language token automatically when
    ``tokenizer.src_lang`` is set.  The default is ``"eng_Latn"``.
    For any other source language, pass ``src_lang`` inside
    ``mm_processor_kwargs`` on the **encoder** prompt:

    .. code-block:: python

        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {"text": source_text},
                "mm_processor_kwargs": {"src_lang": "amh_Ethi"},
            },
            "decoder_prompt": "eng_Latn",
        }

    See ``make_nllb_prompt()`` for a convenience helper.
    """

    def create_decoder_prompt(
        self,
        prompt: str | list[int],
        mm_items,
    ) -> list[int]:
        """Convert target language code → single token ID list.

        NLLB generation must begin with the target language token.
        Using ``convert_tokens_to_ids`` is more reliable than
        ``tokenizer.encode(…, add_special_tokens=False)`` for special tokens.
        """
        if isinstance(prompt, str) and prompt:
            tokenizer = self.info.get_tokenizer()
            lang_id = tokenizer.convert_tokens_to_ids(prompt)
            # language codes are always valid special tokens; unk means wrong code
            if lang_id is not None and lang_id != tokenizer.unk_token_id:
                return [lang_id]
        if isinstance(prompt, (list, tuple)):
            return list(prompt)
        return [self.info.get_tokenizer().eos_token_id]

    def _call_hf_processor(
        self,
        prompt: str | list,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ):
        """Tokenize encoder text, honouring an optional ``src_lang`` kwarg.

        For non-English source languages, pass
        ``mm_processor_kwargs={"src_lang": "<FLORES-200 code>"}`` on the
        encoder prompt dict.  The source-language token is prepended
        manually (thread-safe — does not mutate the shared tokenizer).
        """
        import torch as _torch
        from transformers.feature_extraction_utils import BatchFeature

        tokenizer = self.info.get_tokenizer()
        result: dict = {}

        # --- encoder text --------------------------------------------------
        has_encoder_data = mm_data is not None and "texts" in mm_data
        if has_encoder_data:
            encoder_texts = mm_data["texts"]
            encoder_text = encoder_texts[0] if encoder_texts else ""

            src_lang: str = mm_kwargs.get("src_lang", tokenizer.src_lang)
            src_lang_id: int = tokenizer.convert_tokens_to_ids(src_lang)

            # Tokenize without specials, then manually wrap as NLLB expects:
            #   [src_lang_token]  [text tokens…]  [EOS]
            text_ids = tokenizer(
                encoder_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]  # shape (1, seq_len)

            eos_id = tokenizer.eos_token_id
            prefix = _torch.tensor([[src_lang_id]])
            suffix = _torch.tensor([[eos_id]])
            result["encoder_input_ids"] = _torch.cat(
                [prefix, text_ids, suffix], dim=1
            )

        # --- decoder placeholder  ------------------------------------------
        # In vLLM >=0.18 the rendering pipeline may pass already-tokenized
        # token IDs (a list of ints) instead of a string.  Pass through.
        if (
            isinstance(prompt, (list, tuple))
            and len(prompt) > 0
            and isinstance(prompt[0], int)
        ):
            result["input_ids"] = _torch.tensor([prompt])
        else:
            tokenized = tokenizer(
                prompt if prompt else "",
                add_special_tokens=False,
                return_tensors="pt",
                **tok_kwargs,
            )
            result["input_ids"] = tokenized["input_ids"]

        return BatchFeature(result)


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def make_nllb_prompt(
    source_text: str,
    src_lang: str,
    tgt_lang: str,
) -> dict:
    """Build a vLLM encoder-decoder prompt dict for NLLB translation.

    Args:
        source_text: Text to translate.
        src_lang:    FLORES-200 source language code (e.g. ``"eng_Latn"``).
        tgt_lang:    FLORES-200 target language code (e.g. ``"fra_Latn"``).

    Returns:
        A prompt dict ready to pass to ``LLM.generate()``.

    Example::

        from vllm import LLM, SamplingParams
        from vllm_bart_plugin.nllb import make_nllb_prompt

        llm = LLM("facebook/nllb-200-distilled-600M", ...)
        prompt = make_nllb_prompt(
            "The United Nations was founded in 1945.",
            src_lang="eng_Latn",
            tgt_lang="fra_Latn",
        )
        out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=60))
        print(out[0].outputs[0].text)
    """
    return {
        "encoder_prompt": {
            "prompt": "",
            "multi_modal_data": {"text": source_text},
            "mm_processor_kwargs": {"src_lang": src_lang},
        },
        "decoder_prompt": tgt_lang,
    }


# ---------------------------------------------------------------------------
# Top-level model class registered with vLLM
# ---------------------------------------------------------------------------

@MULTIMODAL_REGISTRY.register_processor(
    M2M100MultiModalProcessor,
    info=M2M100ProcessingInfo,
    dummy_inputs=M2M100DummyInputsBuilder,
)
class M2M100ForConditionalGeneration(nn.Module, SupportsQuant, SupportsMultiModal):
    """vLLM model for M2M-100 and NLLB distilled models.

    Registered HuggingFace architecture string:
      ``M2M100ForConditionalGeneration``

    Used by:
      facebook/nllb-200-distilled-600M
      facebook/nllb-200-distilled-1.3B
      facebook/nllb-200-3.3B
    """

    # M2M100 HF checkpoints already use the full model.encoder.* / model.decoder.*
    # prefix structure, so no key remapping is needed at the top level.
    # Key remapping for q/k/v → qkv_proj is handled in M2M100Model.load_weights.
    hf_to_vllm_mapper = None

    # embed_positions.weights is a non-persistent buffer — skip missing warning.
    keys_to_ignore_on_load_missing = [
        "model.encoder.embed_positions.weights",
        "model.decoder.embed_positions.weights",
    ]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: M2M100Config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        self.config = config

        self.model = M2M100Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        # lm_head.weight is tied to the shared embedding
        self.lm_head = ParallelLMHead(config.vocab_size, config.d_model, bias=False)
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size, config.vocab_size
        )

    def get_language_model(self) -> nn.Module:
        return self.model.decoder

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.decoder.embed_tokens(input_ids)

    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:
        encoder_input_ids_list = self._parse_and_validate_encoder_input(**kwargs)

        if not encoder_input_ids_list:
            raise ValueError(
                "encoder_input_ids_list is empty. "
                "Ensure multimodal data is being passed correctly."
            )

        encoder_outputs: list[torch.Tensor] = []
        for encoder_input_ids in encoder_input_ids_list:
            encoder_positions = torch.arange(
                encoder_input_ids.size(-1),
                dtype=torch.long,
                device=encoder_input_ids.device,
            )
            encoder_output = self.model.encoder(
                input_ids=encoder_input_ids.squeeze(0),
                positions=encoder_positions,
            )
            encoder_outputs.append(encoder_output)

        return encoder_outputs

    def _parse_and_validate_encoder_input(
        self, **kwargs: object
    ) -> list[torch.Tensor]:
        encoder_input_ids = kwargs.get(
            "encoder_input_ids", kwargs.get("input_ids")
        )
        if encoder_input_ids is None:
            return []
        if not isinstance(encoder_input_ids, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of encoder_input_ids. Got: {type(encoder_input_ids)}"
            )
        if isinstance(encoder_input_ids, list):
            return list(encoder_input_ids)
        # Tensor path: unbind along batch dim
        return encoder_input_ids.unsqueeze(1).unbind(dim=0)

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
            encoder_outputs = torch.cat(encoder_outputs, dim=0)
        return self.model(input_ids, positions, inputs_embeds, encoder_outputs)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # No final_logits_bias unlike BART
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights_list = list(weights)

        # Collect the shared embedding weight (appears under multiple names;
        # all point to the same tensor in the HF checkpoint).
        shared_embedding_weight: torch.Tensor | None = None
        filtered: list[tuple[str, torch.Tensor]] = []

        for name, loaded_weight in weights_list:
            if name in (
                "model.shared.weight",
                "model.encoder.embed_tokens.weight",
                "model.decoder.embed_tokens.weight",
                "lm_head.weight",
            ):
                if shared_embedding_weight is None:
                    shared_embedding_weight = loaded_weight
                # All four names are tied — only capture once
                continue
            filtered.append((name, loaded_weight))

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["cls.", "pooler."]),
        )
        loaded_params = loader.load_weights(filtered)

        # Tie shared embedding: lm_head ↔ encoder.embed_tokens ↔ decoder.embed_tokens
        if shared_embedding_weight is not None:
            weight_loader = getattr(
                self.lm_head.weight, "weight_loader", default_weight_loader
            )
            weight_loader(self.lm_head.weight, shared_embedding_weight)

            self.model.encoder.embed_tokens.weight = self.lm_head.weight
            self.model.decoder.embed_tokens.weight = self.lm_head.weight
            loaded_params.update(
                {
                    "model.shared.weight",
                    "model.encoder.embed_tokens.weight",
                    "model.decoder.embed_tokens.weight",
                    "lm_head.weight",
                }
            )

        # Declare ignored buffers so vLLM doesn't warn about missing keys
        for key in self.keys_to_ignore_on_load_missing:
            loaded_params.add(key)

        return loaded_params
