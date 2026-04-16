"""Example: NLLB translation with vLLM via the bart-plugin.

Supported models (all use model_type=m2m_100):
  facebook/nllb-200-distilled-600M   (~1.2 GB)
  facebook/nllb-200-distilled-1.3B   (~2.6 GB)
  facebook/nllb-200-3.3B             (~6.6 GB)

Language codes follow the FLORES-200 format: <language>_<script>
  English   → eng_Latn
  French    → fra_Latn
  German    → deu_Latn
  Arabic    → arb_Arab
  Chinese   → zho_Hans
  Amharic   → amh_Ethi
  Hindi     → hin_Deva
  (200+ languages supported)

Run:
    python example_nllb_usage.py

Required:
    pip install vllm-bart-plugin
"""

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
from vllm_bart_plugin.nllb import make_nllb_prompt

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# ---------------------------------------------------------------------------
# Demo 1: English → multiple target languages
# ---------------------------------------------------------------------------

ENGLISH_TEXTS = [
    "The United Nations was founded in 1945.",
    "Machine translation has improved significantly in recent years.",
    "Hello, how are you doing today?",
]

TARGET_LANGS = [
    ("French",  "fra_Latn"),
    ("German",  "deu_Latn"),
    ("Spanish", "spa_Latn"),
    ("Arabic",  "arb_Arab"),
    ("Chinese", "zho_Hans"),
]

# ---------------------------------------------------------------------------
# Demo 2: Non-English source → English
# ---------------------------------------------------------------------------

NON_ENGLISH_TEXTS = [
    # Amharic (Ge'ez script)
    ("amh_Ethi", "eng_Latn", "ሰላም፣ ዓለም! የተባበሩት መንግሥታት ድርጅት በ1945 ዓ.ም ተቋቋመ።"),
    # French → German
    ("fra_Latn", "deu_Latn", "La traduction automatique s'est beaucoup améliorée."),
    # Hindi → English
    ("hin_Deva", "eng_Latn", "संयुक्त राष्ट्र की स्थापना 1945 में हुई थी।"),
]


def main():
    llm = LLM(
        model=MODEL_NAME,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.15,
        dtype="float16",
    )
    params = SamplingParams(temperature=0.0, max_tokens=60)

    # --- Demo 1: English source -------------------------------------------
    print("=" * 60)
    print("Demo 1: English → multiple languages")
    print("=" * 60)

    for tgt_name, tgt_lang in TARGET_LANGS:
        prompts = [
            make_nllb_prompt(text, src_lang="eng_Latn", tgt_lang=tgt_lang)
            for text in ENGLISH_TEXTS
        ]
        outputs = llm.generate(prompts, sampling_params=params)
        print(f"\n→ {tgt_name} ({tgt_lang})")
        for text, out in zip(ENGLISH_TEXTS, outputs):
            print(f"  [EN] {text}")
            print(f"  [{tgt_lang[:3].upper()}] {out.outputs[0].text}")

    # --- Demo 2: Non-English sources --------------------------------------
    print("\n" + "=" * 60)
    print("Demo 2: Non-English sources")
    print("=" * 60)

    for src_lang, tgt_lang, text in NON_ENGLISH_TEXTS:
        prompt = make_nllb_prompt(text, src_lang=src_lang, tgt_lang=tgt_lang)
        out = llm.generate([prompt], sampling_params=params)[0]
        print(f"\n[{src_lang}] {text}")
        print(f"[{tgt_lang}] {out.outputs[0].text}")


if __name__ == "__main__":
    main()
