# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, TypedDict

import torch
from torch import nn
from transformers import (
    BatchFeature,
    Florence2Config,
    Florence2Processor,
)
from transformers.models.florence2.modeling_florence2 import (
    Florence2MultiModalProjector,
    Florence2VisionBackbone,
)
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

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
        self.encoder = BartEncoder(config,
                                   cache_config=cache_config,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.encoder")
        self.decoder = BartDecoder(config,
                                   cache_config=cache_config,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.decoder")

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
        self.model = Florence2LanguageModel(vllm_config=vllm_config,
                                            prefix=f"{prefix}.model")
        embed_scale = math.sqrt(
            config.d_model) if config.scale_embedding else 1.0

        self.vocab_size = config.vocab_size
        self.lm_head = BartParallelLMHead(self.vocab_size,
                                          config.d_model,
                                          embed_scale=embed_scale)

        self.logits_processor = LogitsProcessor(self.vocab_size,
                                                config.vocab_size)
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
        return self.model(input_ids,
                          positions,
                          inputs_embeds=inputs_embeds,
                          encoder_outputs=encoder_outputs)

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

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
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
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
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
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Florence2ProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> Florence2Config:
        hf_config = self.ctx.get_hf_config(Florence2Config)
        hf_config.decoder_start_token_id = hf_config.text_config.decoder_start_token_id
        return hf_config

    def get_hf_processor(self, **kwargs: object) -> Florence2Processor:
        return self.ctx.get_hf_processor(Florence2Processor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_num_image_tokens(self) -> int:
        processor_config = self.ctx.get_hf_image_processor_config()
        return processor_config["image_seq_length"]


class Florence2DummyInputsBuilder(
        BaseDummyInputsBuilder[Florence2ProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width = target_height = self.info.get_hf_config().vision_config.projection_dim

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class Florence2MultiModalProcessor(
        EncDecMultiModalProcessor[Florence2ProcessingInfo]):
    skip_decoder_start_token: bool = True

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return bool(mm_items.get_all_counts().get("image", 0))

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
    ) -> str | list[int]:
        if isinstance(prompt, list):
            tokenizer = self.info.get_tokenizer()
            prompt = tokenizer.decode(prompt, skip_special_tokens=True)
        return prompt

    def create_decoder_prompt(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
    ) -> str | list[int]:
        text_config = self.info.get_hf_config().text_config
        decoder_prompt = [text_config.decoder_start_token_id]
        forced_bos = getattr(text_config, "forced_bos_token_id", None)
        if forced_bos is not None:
            decoder_prompt.append(forced_bos)
        return decoder_prompt

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            processed_outputs = super()._call_hf_processor(
                prompt, mm_data, mm_kwargs, tok_kwargs)
        else:
            hf_processor = self.info.get_hf_processor()
            tokenizer = hf_processor.tokenizer
            prompt = hf_processor._construct_prompts([prompt])[0]
            processed_outputs = tokenizer(prompt,
                                          add_special_tokens=True,
                                          return_tensors="pt")

        image_token_id = self.info.get_hf_config().image_token_id
        input_ids = processed_outputs["input_ids"]
        encoder_input_ids = input_ids[input_ids != image_token_id]
        processed_outputs["encoder_input_ids"] = encoder_input_ids.unsqueeze(0)
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
        hf_config = self.info.get_hf_config()
        num_image_tokens = self.info.get_num_image_tokens()
        image_placeholder = [hf_config.image_token_id] * num_image_tokens

        insertion: list[int]
        image_items = out_mm_kwargs.get("image", [])
        if image_items:
            item_data = image_items[0].get_data()
            enc_ids = item_data.get("encoder_input_ids")
            if enc_ids is not None:
                insertion = image_placeholder + enc_ids.tolist()
            else:
                # Cache hit: encoder_input_ids not available; fall back to the
                # image placeholders only.
                insertion = image_placeholder
        else:
            insertion = image_placeholder

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
    dummy_inputs=Florence2DummyInputsBuilder)
class Florence2ForConditionalGeneration(nn.Module, SupportsMultiModal):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

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
        self.vision_config = config.vision_config
        self.processor_config = processor_config
        assert config.vision_config.model_type == "florence_vision", (
            f"only Florence Vision is supported for now. "
            f"Received model type: {config.vision_config.model_type}"
        )

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = Florence2VisionBackbone(config.vision_config)
            self.multi_modal_projector = Florence2MultiModalProjector(config)

        with self._mark_language_model(vllm_config):
            self.language_model = Florence2LanguageForConditionalGeneration(
                vllm_config=vllm_config.with_hf_config(config.text_config),
                prefix=f"{prefix}.language_model",
            )
        self.pad_token_id = config.text_config.pad_token_id
        self._pad_id = self.pad_token_id

    def _validate_pixel_values(
        self, data: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor | list[torch.Tensor]:

        size = self.processor_config["size"]
        h, w = size["height"], size["width"]
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = tuple(*map(str, expected_dims))
                raise ValueError(
                    "The expected shape of pixel values per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(self, **kwargs: object):
        pixel_values: list[list[torch.Tensor]] | list[torch.Tensor] | torch.Tensor | None = kwargs.pop(
            "pixel_values", None)
        image_embeds: list[list[torch.Tensor]] | list[torch.Tensor] | torch.Tensor | None = kwargs.pop(
                                         "image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None and image_embeds is not None:
            raise ValueError(
                "Both pixel values and image embeds are provided.")

        if pixel_values is not None:
            return Florence2ImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(pixel_values),
            )

        if image_embeds is not None:
            raise NotImplementedError

        raise AssertionError("This line should be unreachable.")

    def _parse_and_validate_encoder_input(self, **kwargs: object) -> list[torch.Tensor]:
        encoder_input_ids = kwargs.get("encoder_input_ids", kwargs.get("input_ids"))

        if encoder_input_ids is None:
            return []

        if not isinstance(encoder_input_ids, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of encoder input_ids. "
                f"Got type: {type(encoder_input_ids)}"
            )

        # Return as a list of tensors (one per item in the batch)
        if isinstance(encoder_input_ids, list):
            # Already a list - ensure each item is valid
            result = []
            for item in encoder_input_ids:
                if isinstance(item, torch.Tensor):
                    if item.dim() == 0:
                        item = item.unsqueeze(0)
                    result.append(item)
                else:
                    result.append(item)
            return result
        else:
            # [1xD]xN times
            return encoder_input_ids.unsqueeze(1).unbind(dim=0)

    def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(dtype)
        image_outputs = self.vision_tower(pixel_values)
        image_features = self.multi_modal_projector(image_outputs.last_hidden_state)
        return image_features

    def _process_image_input(
            self, image_input: Florence2ImagePixelInputs) -> torch.Tensor:
        assert image_input["type"] == "pixel_values"
        pixel_values = image_input["data"]
        return self._encode_image(pixel_values)

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
                "encoder_input_ids_list is empty - this should not happen. "
                "Check that multimodal data is being passed correctly."
            )

        # Process each encoder input separately and return a list of outputs
        # NOTE (NickLucche): Basic encoder batching optimization: BART input sequences
        # can have different lengths. Due to computational load of encoder being very
        # low here, we batch all sequences to run a single forward by max_seq padding.
        lengths = [t.numel() for t in encoder_input_ids_list]
        max_len = max(lengths) if lengths else 0
        assert max_len > 0, "Empty encoder_input_ids encountered."
        same_len = all(l == max_len for l in lengths)
        if len(encoder_input_ids_list) == 1:
            batch_encoder_input_ids = encoder_input_ids_list[0]
        elif same_len:
            # [1xD]xN =>NxD
            batch_encoder_input_ids = torch.cat(encoder_input_ids_list, dim=0)
        else:
            batch_encoder_input_ids = torch.full(
                (len(encoder_input_ids_list), max_len),
                fill_value=self._pad_id,
                dtype=encoder_input_ids_list[0].dtype,
                device=encoder_input_ids_list[0].device,
            )
            for i, t in enumerate(encoder_input_ids_list):
                batch_encoder_input_ids[i, : t.numel()] = t.squeeze()
        # Create (B, T) positions: 0..T-1 for each item.
        # batch_encoder_positions = torch.arange(
        #     max_len,
        #     dtype=torch.long,
        #     device=batch_encoder_input_ids.device,
        # ).unsqueeze(0).expand(batch_encoder_input_ids.size(0), -1)

        inputs_embeds = self.language_model.model.encoder.embed_tokens(batch_encoder_input_ids)
        inputs_embeds = torch.cat([vision_embeddings, inputs_embeds], dim=-2)
        batch_encoder_positions = torch.arange(
            inputs_embeds.size(1),
            dtype=torch.long,
            device=inputs_embeds.device,
        ).unsqueeze(0).expand(inputs_embeds.size(0), -1)

        # Run encoder once on the batch.
        batch_encoder_output = self.language_model.model.encoder(
            input_ids=batch_encoder_input_ids,
            positions=batch_encoder_positions,
            inputs_embeds=inputs_embeds,
        )
        # Split back into list[(T, H)] to match expected downstream format.
        # If we had to pad, slice back to the original lengths per item.
        encoder_outputs: list[torch.Tensor] = batch_encoder_output.unbind(dim=0)
        if not same_len:
            encoder_outputs = [
                out[:l] for out, l in zip(encoder_outputs, lengths)
            ]
        return encoder_outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_outputs: list[torch.Tensor] | None = None,
        # num_encoder_outputs: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            input_ids
                torch.Tensor of *decoder* input token ids.
            positions
                torch.Tensor of *decoder* position indices.
            encoder_input_ids
                torch.Tensor of *encoder* input token ids.
            encoder_positions
                torch.Tensor of *encoder* position indices
        Returns:
            Output torch.Tensor
        """
        # EncoderDecoderModelState passes an empty list on decode steps.
        enc_states = (
            torch.cat(encoder_outputs, dim=0) if encoder_outputs else None
        )

        hidden_states = self.language_model(input_ids,
                                            positions,
                                            encoder_outputs=enc_states,
                                            inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
