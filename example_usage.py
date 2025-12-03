#!/usr/bin/env python3
"""
Example usage of the vLLM BART plugin.

This script demonstrates how to use BART models with vLLM
after installing the BART plugin.
"""

from vllm import LLM, SamplingParams


def main():
    """Run BART model examples."""
    model_name = "facebook/bart-large-cnn"

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        enforce_eager=False,
        # max_model_len=1024,
        max_num_seqs=4,
        max_num_batched_tokens=11024,
        gpu_memory_utilization=0.5,
        dtype="float16",
    )

    params = SamplingParams(temperature=0.0, max_tokens=20)
    outputs = llm.generate(
        [
            # Not supported
            # {
            #     "prompt": "The president of the United States is",
            # },
            # Not supported without changes to vllm core
            # {  # Test explicit encoder/decoder prompt
            #     "encoder_prompt": {
            #         "prompt": "The president of the United States is",
            #     },
            #     "decoder_prompt": "<s>Donald",
            # },
            {  # NOTE Explicit encoder/decoder prompt. Use <s> to start decoder prompt
                "encoder_prompt": {
                    "prompt": "",
                    # NOTE This format is needed st we don't have to add custom encoder-only prompt
                    # logic in preprocess.py (vllm core) to convert encoder_token_ids to mm text item
                    "multi_modal_data": {
                        "text": "The president of the United States is",
                    },
                },
                "decoder_prompt": "<s>Donald",
            },
            {
                "encoder_prompt": {
                    "prompt": "",
                # NOTE output is really sensible to the BOS token which should always be present in decoder prompt!
                    "multi_modal_data": {
                        "text": "<s>",
                    },
                },
                "decoder_prompt": "<s>Ronald McDonald is",
            },
        ],
        sampling_params=params,
    )
    for o in outputs:
        generated_text = o.outputs[0].text
        print("output:", generated_text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
