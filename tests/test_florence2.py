"""Tests for the Florence-2 multimodal model plugin."""

import os

import pytest
import torch
from transformers import Florence2Config

MODEL_NAME = "florence-community/Florence-2-base-ft"


def _small_vision_config():
    """Tiny 1-stage Florence2 config for fast CPU tests."""
    cfg = Florence2Config()
    vc = cfg.vision_config
    vc.embed_dim = [64]
    vc.depths = [1]
    vc.num_heads = [4]
    vc.num_groups = [4]
    vc.patch_size = [7]
    vc.patch_stride = [4]
    vc.patch_padding = [3]
    vc.patch_prenorm = [False]
    vc.drop_path_rate = 0.0
    return cfg, vc


# ---------------------------------------------------------------------------
# Unit tests — vision architecture (CPU, no weights)
# ---------------------------------------------------------------------------


class TestFlorenceVisionDropPath:
    def test_eval_is_identity(self):
        from vllm_bart_plugin.florence2 import Florence2VisionDropPath

        m = Florence2VisionDropPath(drop_prob=0.9).eval()
        x = torch.randn(2, 16)
        assert torch.equal(m(x), x)

    def test_training_drops_samples(self):
        from vllm_bart_plugin.florence2 import Florence2VisionDropPath

        torch.manual_seed(0)
        m = Florence2VisionDropPath(drop_prob=0.5).train()
        out = m(torch.ones(64, 16))
        assert not torch.all(out == 1)


class TestFlorenceVisionConvEmbed:
    @pytest.mark.parametrize("pre_norm", [True, False])
    def test_output_channels(self, pre_norm):
        from vllm_bart_plugin.florence2 import Florence2VisionConvEmbed

        m = Florence2VisionConvEmbed(
            patch_size=7,
            in_channels=3,
            embed_dim=64,
            stride=4,
            padding=3,
            pre_norm=pre_norm,
        )
        out = m(torch.randn(1, 3, 64, 64))
        assert out.shape[1] == 64


class TestFlorenceVisionWindowAttention:
    def test_exact_window(self):
        from vllm_bart_plugin.florence2 import Florence2VisionWindowAttention

        m = Florence2VisionWindowAttention(dim=32, num_heads=4, window_size=4)
        assert m(torch.randn(1, 4, 4, 32)).shape == (1, 16, 32)

    def test_input_requires_padding(self):
        from vllm_bart_plugin.florence2 import Florence2VisionWindowAttention

        m = Florence2VisionWindowAttention(dim=32, num_heads=4, window_size=4)
        # 6 is not divisible by 4; output should still be (B, 6*6, C)
        assert m(torch.randn(1, 6, 6, 32)).shape == (1, 36, 32)


class TestFlorenceVisionBackbone:
    def test_output_shape(self):
        from vllm_bart_plugin.florence2 import Florence2VisionBackbone

        _, vc = _small_vision_config()
        out = Florence2VisionBackbone(vc)(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, vc.embed_dim[-1], 16, 16)


class TestFlorenceVisionPositionalEmbeddingCosine1D:
    def test_output_shape_and_no_batch_dim(self):
        from vllm_bart_plugin.florence2 import (
            Florence2VisionPositionalEmbeddingCosine1D,
        )

        m = Florence2VisionPositionalEmbeddingCosine1D(embed_dim=64, max_seq_len=100)
        assert m(torch.randn(2, 5, 64)).shape == (5, 64)

    def test_raises_if_exceeds_max(self):
        from vllm_bart_plugin.florence2 import (
            Florence2VisionPositionalEmbeddingCosine1D,
        )

        m = Florence2VisionPositionalEmbeddingCosine1D(embed_dim=64, max_seq_len=10)
        with pytest.raises(AssertionError):
            m(torch.randn(1, 20, 64))


class TestFlorenceMultiModalProjector:
    def test_output_shape(self):
        from vllm_bart_plugin.florence2 import Florence2MultiModalProjector

        cfg, vc = _small_vision_config()
        vc.projection_dim = 128
        m = Florence2MultiModalProjector(cfg)
        out = m(torch.randn(2, vc.embed_dim[-1], 12, 12))
        # (B, 1 spatial-avg token + H*W tokens, proj_dim)
        assert out.shape == (2, 1 + 12 * 12, vc.projection_dim)


# ---------------------------------------------------------------------------
# Integration tests — full model inference (GPU required)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def florence2_llm():
    from vllm import LLM

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    return LLM(
        model=MODEL_NAME,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        mm_processor_cache_gb=0,
    )


@pytest.fixture(scope="module")
def stop_sign_image():
    from vllm.assets.image import ImageAsset

    return ImageAsset("stop_sign").pil_image


@pytest.fixture(scope="module")
def sampling_params():
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        max_tokens=20,
        repetition_penalty=1.5,
        skip_special_tokens=False,
    )


@pytest.mark.slow
class TestFlorenceInference:
    def test_caption(self, florence2_llm, stop_sign_image, sampling_params):
        outputs = florence2_llm.generate(
            [
                {
                    "prompt": "<DETAILED_CAPTION>",
                    "multi_modal_data": {"image": stop_sign_image},
                }
            ],
            sampling_params=sampling_params,
        )
        assert len(outputs[0].outputs[0].text) > 0

    def test_object_detection_has_loc_tokens(
        self, florence2_llm, stop_sign_image, sampling_params
    ):
        outputs = florence2_llm.generate(
            [
                {
                    "encoder_prompt": {
                        "prompt": "<OD>",
                        "multi_modal_data": {"image": stop_sign_image},
                    },
                    "decoder_prompt": "",
                }
            ],
            sampling_params=sampling_params,
        )
        assert "<loc_" in outputs[0].outputs[0].text

    def test_batch_inference(self, florence2_llm, stop_sign_image, sampling_params):
        prompts = [
            {"prompt": "<CAPTION>", "multi_modal_data": {"image": stop_sign_image}},
            {
                "prompt": "<DETAILED_CAPTION>",
                "multi_modal_data": {"image": stop_sign_image},
            },
        ]
        outputs = florence2_llm.generate(prompts, sampling_params=sampling_params)
        assert all(len(o.outputs[0].text) > 0 for o in outputs)

    def test_encoder_length_within_limit(self, stop_sign_image):
        """Processor output must not exceed BART max_position_embeddings."""
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        out = processor(
            text="<DETAILED_CAPTION>", images=stop_sign_image, return_tensors="pt"
        )
        assert out["input_ids"].shape[1] <= 1024
