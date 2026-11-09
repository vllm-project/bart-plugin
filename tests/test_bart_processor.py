"""Tests for BART multimodal processor helpers."""

from types import SimpleNamespace

import pytest
import torch
from vllm.exceptions import VLLMValidationError

from vllm_bart_plugin.bart import _HF_PROCESSOR_TAKES_PROMPT


def test_text_data_parser_handles_empty_inputs():
    from vllm_bart_plugin.bart import TextDataParser

    parser = TextDataParser()

    assert parser._parse_text_data("").data == [""]
    assert parser._parse_text_data([]).data == [""]


def test_create_encoder_prompt_uses_placeholder_for_decoder_tokens():
    from vllm_bart_plugin.bart import BartMultiModalProcessor

    processor = BartMultiModalProcessor.__new__(BartMultiModalProcessor)

    assert processor.create_encoder_prompt([0, 1, 2], {"texts": ["encoder text"]}) == [0]


def _fake_processor(max_position_embeddings: int = 1024):
    from vllm_bart_plugin.bart import BartMultiModalProcessor

    class FakeTokenizer:
        def __call__(self, text, return_tensors="pt", **kwargs):
            if text == "encoder text":
                return {"input_ids": torch.tensor([[11, 12, 13]])}
            return {"input_ids": torch.tensor([[21, 22]])}

    class FakeInfo:
        def get_tokenizer(self):
            return FakeTokenizer()

        def get_hf_config(self):
            return SimpleNamespace(max_position_embeddings=max_position_embeddings)

    processor = BartMultiModalProcessor.__new__(BartMultiModalProcessor)
    processor.info = FakeInfo()
    return processor


@pytest.mark.skipif(
    not _HF_PROCESSOR_TAKES_PROMPT, reason="vLLM < 0.31 processor interface"
)
def test_call_hf_processor_accepts_pretokenized_decoder_prompt():
    processor = _fake_processor()

    out = processor._call_hf_processor(
        [7, 8, 9],
        {"texts": ["encoder text"]},
        {},
        {},
    )

    assert torch.equal(out["encoder_input_ids"], torch.tensor([[11, 12, 13]]))
    assert torch.equal(out["input_ids"], torch.tensor([[7, 8, 9]]))


@pytest.mark.skipif(
    _HF_PROCESSOR_TAKES_PROMPT, reason="vLLM 0.31+ processor interface"
)
def test_call_hf_processor_tokenizes_only_encoder_text():
    processor = _fake_processor()

    out = processor._call_hf_processor({"texts": ["encoder text"]}, {})

    assert torch.equal(out["encoder_input_ids"], torch.tensor([[11, 12, 13]]))
    assert "input_ids" not in out


def test_encoder_input_longer_than_position_table_is_rejected():
    """Over-long encoder input must fail the request, not hit a device-side
    assert in the positional embedding that kills the engine. It must be a
    VLLMValidationError so the server returns a clean 400 and keeps the
    connection open."""
    processor = _fake_processor(max_position_embeddings=2)

    with pytest.raises(
        VLLMValidationError, match="exceeds the model's maximum encoder length"
    ):
        processor._encoder_input_ids({"texts": ["encoder text"]})
