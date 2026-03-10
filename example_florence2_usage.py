#!/usr/bin/env python3
"""
Florence-2 example usage of the vLLM BART plugin.

This script demonstrates how to use Florence-2 models with vLLM
after installing the BART plugin.
"""

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset


def main():
    """Run Florence-2 model examples."""
    model_name = "florence-community/Florence-2-large-ft"

    llm = LLM(
        model=model_name,
        mm_processor_cache_gb=0,
        enforce_eager=True,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=20,
        # skip_special_tokens=False is needed to present
        # grounding tokens like <loc_0><loc_1>
        skip_special_tokens=False,
    )
    outputs = llm.generate(
        [
            {  # NOTE implicit prompt with task token
                "prompt": "<DETAILED_CAPTION>",
                "multi_modal_data": {"image": ImageAsset("stop_sign").pil_image},
            },
            # Not supported without changes to vllm core
            # {  # Test explicit encoder/decoder prompt
            #     "encoder_prompt": {
            #         "prompt": "The president of the United States is",
            #     },
            #     "decoder_prompt": "<s>Donald",
            # },
            {  # NOTE Explicit encoder/decoder prompt
                "encoder_prompt": {
                    "prompt": "<OD>",
                    "multi_modal_data": {"image": ImageAsset("stop_sign").pil_image},
                },
                "decoder_prompt": "",
            },
        ],
        sampling_params=params,
    )
    for o in outputs:
        generated_text = o.outputs[0].text
        print("output:", generated_text)


if __name__ == "__main__":
    main()
