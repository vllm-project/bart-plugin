"""Pytest configuration and fixtures for BART plugin tests."""

import pytest
import torch


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def device():
    """Get the device to use for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def small_model_name():
    """Small BART model for quick tests."""
    return "facebook/bart-base"


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
