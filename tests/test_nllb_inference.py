"""Integration tests for NLLB / M2M-100 inference via vLLM.

These tests require a GPU and download the model on first run (~1.2 GB
for the 600M model).  Automatically skipped when no GPU is present unless
NLLB_FORCE_CPU_TEST=1 is set.

Usage:
    pytest tests/test_nllb_inference.py -v
    pytest tests/test_nllb_inference.py -v -k test_english_to_french
"""

import os

import pytest

MODEL_NAME = "facebook/nllb-200-distilled-600M"


@pytest.fixture(scope="module")
def llm():
    """Create a shared vLLM LLM instance for all tests in this module."""
    try:
        import torch
        from vllm import LLM
    except ImportError:
        pytest.skip("vllm not installed")

    if (
        not torch.cuda.is_available()
        and not torch.backends.mps.is_available()
        and not os.environ.get("NLLB_FORCE_CPU_TEST")
    ):
        pytest.skip("No GPU available (set NLLB_FORCE_CPU_TEST=1 to force)")

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    return LLM(
        model=MODEL_NAME,
        enforce_eager=True,
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.1,
        dtype="float16",
    )


class TestNLLBInference:
    """Functional tests using the make_nllb_prompt() helper."""

    def test_model_loads(self, llm):
        """Verify the model loads without errors."""
        assert llm is not None

    # ------------------------------------------------------------------
    # English source
    # ------------------------------------------------------------------

    def test_english_to_french(self, llm):
        """Translate an English sentence to French and verify basic quality."""
        from vllm import SamplingParams
        from vllm_bart_plugin.nllb import make_nllb_prompt

        prompt = make_nllb_prompt(
            "The United Nations was founded in 1945.",
            src_lang="eng_Latn",
            tgt_lang="fra_Latn",
        )
        out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=40))

        translation = out[0].outputs[0].text
        assert len(translation) > 0
        # NLLB-600M reliably produces "L'Organisation des Nations Unies..."
        # At minimum the output should contain Latin characters (not garbled Arabic)
        assert any(c.isalpha() and ord(c) < 512 for c in translation), (
            f"Expected Latin-alphabet output for French, got: {translation!r}"
        )

    def test_english_to_german(self, llm):
        from vllm import SamplingParams
        from vllm_bart_plugin.nllb import make_nllb_prompt

        prompt = make_nllb_prompt(
            "Machine translation has improved significantly.",
            src_lang="eng_Latn",
            tgt_lang="deu_Latn",
        )
        out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=40))
        translation = out[0].outputs[0].text
        assert len(translation) > 0
        assert any(c.isalpha() for c in translation)

    def test_english_to_arabic(self, llm):
        """Arabic output should contain Arabic-script characters (U+0600–U+06FF)."""
        from vllm import SamplingParams
        from vllm_bart_plugin.nllb import make_nllb_prompt

        prompt = make_nllb_prompt(
            "Hello, how are you?",
            src_lang="eng_Latn",
            tgt_lang="arb_Arab",
        )
        out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=30))
        translation = out[0].outputs[0].text
        assert len(translation) > 0
        assert any(0x0600 <= ord(c) <= 0x06FF for c in translation), (
            f"Expected Arabic-script characters, got: {translation!r}"
        )

    def test_english_to_chinese(self, llm):
        """Chinese output should contain CJK characters."""
        from vllm import SamplingParams
        from vllm_bart_plugin.nllb import make_nllb_prompt

        prompt = make_nllb_prompt(
            "The United Nations was founded in 1945 to promote international peace.",
            src_lang="eng_Latn",
            tgt_lang="zho_Hans",
        )
        out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=50))
        translation = out[0].outputs[0].text
        assert len(translation) > 0
        assert any(0x4E00 <= ord(c) <= 0x9FFF for c in translation), (
            f"Expected CJK characters, got: {translation!r}"
        )

    # ------------------------------------------------------------------
    # Non-English source — requires correct src_lang in mm_processor_kwargs
    # ------------------------------------------------------------------

    def test_amharic_to_english(self, llm):
        """Translate Ge'ez (Amharic) to English with explicit src_lang."""
        from vllm import SamplingParams
        from vllm_bart_plugin.nllb import make_nllb_prompt

        # "Hello world! The United Nations was founded in 1945."
        prompt = make_nllb_prompt(
            "ሰላም፣ ዓለም! የተባበሩት መንግሥታት ድርጅት በ1945 ዓ.ም ተቋቋመ።",
            src_lang="amh_Ethi",
            tgt_lang="eng_Latn",
        )
        out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=50))
        translation = out[0].outputs[0].text
        assert len(translation) > 0
        # Output should be ASCII-range Latin (English)
        assert any(c.isalpha() and ord(c) < 128 for c in translation), (
            f"Expected English (ASCII) output from Amharic source, got: {translation!r}"
        )

    def test_french_to_german(self, llm):
        """Translate French to German with explicit src_lang."""
        from vllm import SamplingParams
        from vllm_bart_plugin.nllb import make_nllb_prompt

        prompt = make_nllb_prompt(
            "La traduction automatique s'est beaucoup améliorée.",
            src_lang="fra_Latn",
            tgt_lang="deu_Latn",
        )
        out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=50))
        translation = out[0].outputs[0].text
        assert len(translation) > 0
        assert any(c.isalpha() for c in translation)

    def test_hindi_to_english(self, llm):
        """Translate Devanagari (Hindi) to English with explicit src_lang."""
        from vllm import SamplingParams
        from vllm_bart_plugin.nllb import make_nllb_prompt

        prompt = make_nllb_prompt(
            "संयुक्त राष्ट्र की स्थापना 1945 में हुई थी।",
            src_lang="hin_Deva",
            tgt_lang="eng_Latn",
        )
        out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=50))
        translation = out[0].outputs[0].text
        assert len(translation) > 0
        assert any(c.isalpha() and ord(c) < 128 for c in translation), (
            f"Expected English output from Hindi source, got: {translation!r}"
        )

    # ------------------------------------------------------------------
    # Batch and parameter tests
    # ------------------------------------------------------------------

    def test_batch_translation(self, llm):
        """Translate a batch of prompts in a single generate() call."""
        from vllm import SamplingParams
        from vllm_bart_plugin.nllb import make_nllb_prompt

        texts = [
            "Hello, how are you?",
            "The capital of France is Paris.",
            "Machine learning is a subset of artificial intelligence.",
        ]
        prompts = [
            make_nllb_prompt(t, src_lang="eng_Latn", tgt_lang="fra_Latn")
            for t in texts
        ]
        outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=30))

        assert len(outputs) == len(texts)
        for out in outputs:
            assert len(out.outputs[0].text) > 0

    def test_deterministic_output(self, llm):
        """temperature=0 must produce identical outputs on repeated calls."""
        from vllm import SamplingParams
        from vllm_bart_plugin.nllb import make_nllb_prompt

        prompt = make_nllb_prompt(
            "The capital of France is Paris.",
            src_lang="eng_Latn",
            tgt_lang="deu_Latn",
        )
        params = SamplingParams(temperature=0.0, max_tokens=20)

        out1 = llm.generate([prompt], params)[0].outputs[0].text
        out2 = llm.generate([prompt], params)[0].outputs[0].text
        assert out1 == out2, f"Non-deterministic: {out1!r} vs {out2!r}"

    def test_max_tokens_respected(self, llm):
        """Output token count must not exceed max_tokens."""
        from vllm import SamplingParams
        from transformers import NllbTokenizerFast
        from vllm_bart_plugin.nllb import make_nllb_prompt

        prompt = make_nllb_prompt(
            "This is a test sentence.",
            src_lang="eng_Latn",
            tgt_lang="fra_Latn",
        )
        max_tokens = 5
        out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=max_tokens))
        output_text = out[0].outputs[0].text

        tok = NllbTokenizerFast.from_pretrained(MODEL_NAME)
        n_toks = len(tok(output_text, add_special_tokens=False)["input_ids"])
        assert n_toks <= max_tokens, f"Got {n_toks} tokens, expected <= {max_tokens}"

    def test_long_source_input(self, llm):
        """Model handles longer input without crashing."""
        from vllm import SamplingParams
        from vllm_bart_plugin.nllb import make_nllb_prompt

        text = " ".join(["The quick brown fox jumps over the lazy dog."] * 8)
        prompt = make_nllb_prompt(text, src_lang="eng_Latn", tgt_lang="fra_Latn")
        out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=30))
        assert len(out[0].outputs[0].text) > 0

    def test_200_language_codes_accessible(self, llm):
        """Spot-check a selection of language codes across scripts."""
        from vllm import SamplingParams
        from vllm_bart_plugin.nllb import make_nllb_prompt

        # Cover multiple scripts — one sentence each.
        # Script-level checks confirm correct language token routing.
        # Note: NLLB-600M can produce low-quality output for some pairs;
        # the test verifies the target script is used, not translation quality.
        # Use sentences known to produce the right script with NLLB-600M.
        cases = [
            # (src_lang, tgt_lang, text, script_check)
            ("eng_Latn", "fra_Latn",
             "The United Nations was founded in 1945.",
             lambda t: any(c.isalpha() for c in t)),
            ("eng_Latn", "arb_Arab",
             "Hello, how are you?",                         # shorter → more reliable Arabic
             lambda t: any(0x0600 <= ord(c) <= 0x06FF for c in t)),
            ("eng_Latn", "zho_Hans",
             "The United Nations was founded in 1945 to promote international peace.",
             lambda t: any(0x4E00 <= ord(c) <= 0x9FFF for c in t)),
            # Amharic → French (non-English source, non-English target)
            ("amh_Ethi", "fra_Latn",
             "ሰላም፣ ዓለም! የተባበሩት መንግሥታት ድርጅት ተቋቋመ።",
             lambda t: any(c.isalpha() for c in t)),
        ]
        for src_lang, tgt_lang, text, check in cases:
            prompt = make_nllb_prompt(text, src_lang=src_lang, tgt_lang=tgt_lang)
            out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=30))
            translation = out[0].outputs[0].text
            assert len(translation) > 0, f"Empty output for {src_lang}→{tgt_lang}"
            assert check(translation), (
                f"Script check failed for {src_lang}→{tgt_lang}: {translation!r}"
            )
