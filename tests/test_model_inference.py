"""Tests for BART model inference."""

import pytest
from vllm import LLM, SamplingParams


class TestModelInference:
    """Test BART model inference capabilities."""

    @pytest.fixture(scope="class")
    def llm(self, small_model_name):
        """Create LLM instance for tests."""
        return LLM(
            model=small_model_name,
            tensor_parallel_size=1,
            enforce_eager=True,
            max_model_len=512,
            max_num_seqs=4,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=0.3,
            dtype="float16",
        )

    @pytest.mark.slow
    def test_single_generation(self, llm):
        """Test single prompt generation."""
        params = SamplingParams(temperature=0.0, max_tokens=10)

        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "The president of the United States is",
                },
            },
            "decoder_prompt": "<s>",
        }

        outputs = llm.generate([prompt], sampling_params=params)

        assert len(outputs) == 1
        assert len(outputs[0].outputs) > 0
        assert len(outputs[0].outputs[0].text) > 0

    @pytest.mark.slow
    def test_batch_generation(self, llm, test_prompts):
        """Test batch generation with multiple prompts."""
        params = SamplingParams(temperature=0.0, max_tokens=10)

        outputs = llm.generate(test_prompts, sampling_params=params)

        assert len(outputs) == len(test_prompts)
        for output in outputs:
            assert len(output.outputs) > 0
            assert len(output.outputs[0].text) > 0

    @pytest.mark.slow
    def test_different_sampling_params(self, llm):
        """Test generation with different sampling parameters."""
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "Machine learning is",
                },
            },
            "decoder_prompt": "<s>",
        }

        # Greedy decoding
        greedy_params = SamplingParams(temperature=0.0, max_tokens=10)
        greedy_outputs = llm.generate([prompt], sampling_params=greedy_params)
        assert len(greedy_outputs[0].outputs[0].text) > 0

        # Sampling with temperature
        sample_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=10)
        sample_outputs = llm.generate([prompt], sampling_params=sample_params)
        assert len(sample_outputs[0].outputs[0].text) > 0

        # Nucleus sampling
        nucleus_params = SamplingParams(temperature=0.7, top_p=0.5, max_tokens=10)
        nucleus_outputs = llm.generate([prompt], sampling_params=nucleus_params)
        assert len(nucleus_outputs[0].outputs[0].text) > 0

    @pytest.mark.slow
    def test_max_tokens_respected(self, llm):
        """Test that max_tokens parameter is respected."""
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "The quick brown fox jumps over the lazy dog.",
                },
            },
            "decoder_prompt": "<s>",
        }

        max_tokens = 5
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

        outputs = llm.generate([prompt], sampling_params=params)

        # Count tokens in output (approximate)
        output_text = outputs[0].outputs[0].text
        # The output might be slightly less due to EOS token
        # but should not exceed max_tokens significantly
        assert len(output_text.split()) <= max_tokens + 2  # Allow small margin

    @pytest.mark.slow
    def test_encoder_decoder_prompt_format(self, llm):
        """Test that encoder-decoder prompt format works correctly."""
        # Test with explicit encoder and decoder prompts
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "Donald Trump is the president of",
                },
            },
            "decoder_prompt": "<s>the United",
        }

        params = SamplingParams(temperature=0.0, max_tokens=5)
        outputs = llm.generate([prompt], sampling_params=params)

        assert len(outputs) == 1
        output_text = outputs[0].outputs[0].text
        assert len(output_text) > 0

    @pytest.mark.slow
    def test_empty_encoder_text(self, llm):
        """Test generation with minimal encoder text."""
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "<s>",  # Minimal text
                },
            },
            "decoder_prompt": "<s>Hello",
        }

        params = SamplingParams(temperature=0.0, max_tokens=5)
        outputs = llm.generate([prompt], sampling_params=params)

        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].text) > 0

    @pytest.mark.slow
    def test_deterministic_output(self, llm):
        """Test that temperature=0 produces deterministic outputs."""
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "The capital of France is",
                },
            },
            "decoder_prompt": "<s>",
        }

        params = SamplingParams(temperature=0.0, max_tokens=10)

        # Generate twice with same params
        outputs1 = llm.generate([prompt], sampling_params=params)
        outputs2 = llm.generate([prompt], sampling_params=params)

        # Should produce same output
        text1 = outputs1[0].outputs[0].text
        text2 = outputs2[0].outputs[0].text

        assert text1 == text2, "Deterministic generation should produce same output"

    @pytest.mark.slow
    def test_output_metadata(self, llm):
        """Test that output contains expected metadata."""
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "Testing metadata",
                },
            },
            "decoder_prompt": "<s>",
        }

        params = SamplingParams(temperature=0.0, max_tokens=5)
        outputs = llm.generate([prompt], sampling_params=params)

        output = outputs[0]

        # Check output structure
        assert hasattr(output, 'outputs')
        assert len(output.outputs) > 0
        assert hasattr(output.outputs[0], 'text')

        # Check that we can access token IDs if available
        if hasattr(output.outputs[0], 'token_ids'):
            assert len(output.outputs[0].token_ids) > 0


class TestModelEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture(scope="class")
    def llm(self, small_model_name):
        """Create LLM instance for tests."""
        return LLM(
            model=small_model_name,
            tensor_parallel_size=1,
            enforce_eager=True,
            max_model_len=512,
            gpu_memory_utilization=0.3,
            dtype="float16",
        )

    @pytest.mark.slow
    def test_long_encoder_input(self, llm):
        """Test with longer encoder input."""
        long_text = " ".join(["word"] * 100)  # 100 words

        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": long_text,
                },
            },
            "decoder_prompt": "<s>",
        }

        params = SamplingParams(temperature=0.0, max_tokens=10)

        try:
            outputs = llm.generate([prompt], sampling_params=params)
            assert len(outputs) == 1
        except Exception as e:
            pytest.fail(f"Failed with long input: {e}")

    @pytest.mark.slow
    def test_multiple_decoder_starts(self, llm):
        """Test with different decoder prompt starts."""
        encoder_text = "The president of the United States is"

        decoder_starts = ["<s>", "<s>Donald", "<s>Joe"]

        for decoder_start in decoder_starts:
            prompt = {
                "encoder_prompt": {
                    "prompt": "",
                    "multi_modal_data": {
                        "text": encoder_text,
                    },
                },
                "decoder_prompt": decoder_start,
            }

            params = SamplingParams(temperature=0.0, max_tokens=5)
            outputs = llm.generate([prompt], sampling_params=params)

            assert len(outputs) == 1
            assert len(outputs[0].outputs[0].text) > 0
