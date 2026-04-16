"""Pytest configuration and fixtures for BART plugin tests."""

import pytest
import torch


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture
def vllm_config_ctx():
    """Context manager that sets a minimal vLLM config.

    Required for tests that instantiate vLLM attention layers directly
    (Attention, MMEncoderAttention, CrossAttention all call
    get_current_vllm_config() during __init__).
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        yield vllm_config


@pytest.fixture(scope="session")
def device():
    """Get the device to use for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def test_prompts():
    """Sample prompts for testing."""
    return [
        {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "The quick brown fox jumps over the lazy dog.",
                },
            },
            "decoder_prompt": "<s>",
        },
        {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "Machine learning is a subset of artificial intelligence.",
                },
            },
            "decoder_prompt": "<s>",
        },
    ]


@pytest.fixture
def skip_if_no_gpu(cuda_available):
    """Skip test if GPU is not available."""
    if not cuda_available:
        pytest.skip("GPU not available")
