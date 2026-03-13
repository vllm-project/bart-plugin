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


def _run_task(llm, processor, image, task_prompt, text_input=None, max_tokens=100):
    """Helper: run one Florence-2 task and return the post-processed result."""
    from vllm import SamplingParams

    prompt = task_prompt if text_input is None else task_prompt + text_input
    params = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, skip_special_tokens=False
    )
    outputs = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": image}}],
        sampling_params=params,
    )
    raw = outputs[0].outputs[0].text
    return processor.post_process_generation(
        raw, task=task_prompt, image_size=image.size
    )


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
def florence2_processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def stop_sign_image():
    from vllm.assets.image import ImageAsset

    return ImageAsset("stop_sign").pil_image.convert("RGB")


@pytest.mark.slow
class TestFlorenceInference:
    # ------------------------------------------------------------------
    # Caption tasks — check for semantically meaningful keywords
    # ------------------------------------------------------------------

    def test_caption(self, florence2_llm, florence2_processor, stop_sign_image):
        result = _run_task(
            florence2_llm,
            florence2_processor,
            stop_sign_image,
            "<CAPTION>",
            max_tokens=30,
        )
        text = result["<CAPTION>"].lower()
        assert (
            "car" in text or "stop" in text
        ), f"<CAPTION> output missing expected content: {text!r}"

    def test_detailed_caption(
        self, florence2_llm, florence2_processor, stop_sign_image
    ):
        result = _run_task(
            florence2_llm,
            florence2_processor,
            stop_sign_image,
            "<DETAILED_CAPTION>",
            max_tokens=80,
        )
        text = result["<DETAILED_CAPTION>"].lower()
        # Must mention the car and give some background detail — guards against the
        # KV-cache encoder_seq_lens regression that previously produced garbled output.
        assert "car" in text, f"<DETAILED_CAPTION> missing 'car': {text!r}"
        assert len(text.split()) >= 10, f"<DETAILED_CAPTION> too short: {text!r}"

    def test_more_detailed_caption(
        self, florence2_llm, florence2_processor, stop_sign_image
    ):
        result = _run_task(
            florence2_llm,
            florence2_processor,
            stop_sign_image,
            "<MORE_DETAILED_CAPTION>",
            max_tokens=100,
        )
        text = result["<MORE_DETAILED_CAPTION>"].lower()
        assert (
            "stop sign" in text or "sign" in text
        ), f"<MORE_DETAILED_CAPTION> missing 'stop sign': {text!r}"
        assert len(text.split()) >= 10, f"<MORE_DETAILED_CAPTION> too short: {text!r}"

    # ------------------------------------------------------------------
    # Structured-output tasks — check schema and key labels
    # ------------------------------------------------------------------

    def test_object_detection(
        self, florence2_llm, florence2_processor, stop_sign_image
    ):
        result = _run_task(
            florence2_llm, florence2_processor, stop_sign_image, "<OD>", max_tokens=300
        )
        od = result["<OD>"]
        assert "bboxes" in od and "labels" in od
        assert len(od["bboxes"]) == len(od["labels"]) > 0
        # Each bbox must be a 4-element list with non-negative coords
        for bbox in od["bboxes"]:
            assert len(bbox) == 4 and all(c >= 0 for c in bbox)
        labels = od["labels"]
        assert (
            "stop sign" in labels
        ), f"Expected 'stop sign' in OD labels, got: {labels}"
        assert (
            "car" in labels or "building" in labels
        ), f"Expected common objects in OD labels, got: {labels}"

    def test_dense_region_caption(
        self, florence2_llm, florence2_processor, stop_sign_image
    ):
        result = _run_task(
            florence2_llm,
            florence2_processor,
            stop_sign_image,
            "<DENSE_REGION_CAPTION>",
            max_tokens=250,
        )
        drc = result["<DENSE_REGION_CAPTION>"]
        assert "bboxes" in drc and "labels" in drc
        assert len(drc["bboxes"]) == len(drc["labels"]) > 0
        assert (
            "stop sign" in drc["labels"]
        ), f"Expected 'stop sign' in dense captions, got: {drc['labels']}"

    def test_region_proposal(self, florence2_llm, florence2_processor, stop_sign_image):
        result = _run_task(
            florence2_llm,
            florence2_processor,
            stop_sign_image,
            "<REGION_PROPOSAL>",
            max_tokens=100,
        )
        rp = result["<REGION_PROPOSAL>"]
        assert "bboxes" in rp and "labels" in rp
        assert len(rp["bboxes"]) > 0
        # Region proposal labels are always empty strings
        assert all(label == "" for label in rp["labels"])

    def test_ocr_with_region(self, florence2_llm, florence2_processor, stop_sign_image):
        result = _run_task(
            florence2_llm,
            florence2_processor,
            stop_sign_image,
            "<OCR_WITH_REGION>",
            max_tokens=250,
        )
        ocr = result["<OCR_WITH_REGION>"]
        assert "quad_boxes" in ocr and "labels" in ocr
        assert len(ocr["quad_boxes"]) == len(ocr["labels"]) > 0
        # Each quad box must be 8 coords
        for quad in ocr["quad_boxes"]:
            assert len(quad) == 8
        # "STOP" is the most prominent text in the image
        joined = " ".join(ocr["labels"])
        assert (
            "STOP" in joined
        ), f"Expected 'STOP' in OCR_WITH_REGION labels, got: {joined!r}"

    def test_caption_to_phrase_grounding(
        self, florence2_llm, florence2_processor, stop_sign_image
    ):
        result = _run_task(
            florence2_llm,
            florence2_processor,
            stop_sign_image,
            "<CAPTION_TO_PHRASE_GROUNDING>",
            text_input="A stop sign on a street corner.",
            max_tokens=80,
        )
        cpg = result["<CAPTION_TO_PHRASE_GROUNDING>"]
        assert "bboxes" in cpg and "labels" in cpg
        assert len(cpg["bboxes"]) > 0
        assert any(
            "stop sign" in lbl.lower() for lbl in cpg["labels"]
        ), f"Expected 'stop sign' grounded, got labels: {cpg['labels']}"

    # ------------------------------------------------------------------
    # Batch tests
    # ------------------------------------------------------------------

    def test_batch_inference(self, florence2_llm, florence2_processor, stop_sign_image):
        """Multiple prompts in one batch must all produce non-empty output."""
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=0.0, max_tokens=30, skip_special_tokens=False
        )
        prompts = [
            {"prompt": "<CAPTION>", "multi_modal_data": {"image": stop_sign_image}},
            {
                "prompt": "<DETAILED_CAPTION>",
                "multi_modal_data": {"image": stop_sign_image},
            },
        ]
        outputs = florence2_llm.generate(prompts, sampling_params=params)
        assert all(len(o.outputs[0].text) > 0 for o in outputs)
