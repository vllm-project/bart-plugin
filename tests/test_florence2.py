"""Tests for the Florence-2 multimodal model plugin."""

import pytest

MODEL_NAME = "florence-community/Florence-2-base-ft"


@pytest.fixture()
def stop_sign_image():
    from vllm.assets.image import ImageAsset

    return ImageAsset("stop_sign").pil_image.convert("RGB")


@pytest.mark.slow
class TestFlorence2GreedyDecoding:
    @pytest.mark.parametrize("prompt", ["<DETAILED_CAPTION>", "<OD>"])
    def test_greedy_decoding_matches(self, stop_sign_image, prompt):
        from vllm import LLM, SamplingParams

        florence2_llm = LLM(
            model=MODEL_NAME,
            enforce_eager=True,
            gpu_memory_utilization=0.5,
            mm_processor_cache_gb=0,
        )
        params = SamplingParams(
            temperature=0.0, max_tokens=256, skip_special_tokens=False, seed=0
        )
        output = florence2_llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": stop_sign_image}}],
            sampling_params=params,
        )[0].outputs[0]

        EXPECTED_TOKEN_IDS = {
            "<DETAILED_CAPTION>": [
                0, 1121, 42, 2274, 52, 64, 192, 10, 512, 15, 5, 921, 4, 96, 5, 3618, 89,
                32, 3413, 6, 3980, 6, 6904, 6, 4666, 6, 5569, 1033, 6, 20588, 6, 82, 8, 103,
                97, 8720, 4, 2,
            ],
            "<OD>": [
                0, 11557, 50269, 50269, 51267, 50949, 50269, 50269, 50579, 50942, 51032,
                50269, 51267, 50934, 5901, 50797, 50814, 51187, 51030, 4539, 14118, 50803,
                50869, 50809, 50882, 462, 3914, 50354, 50269, 50377, 50329, 50452, 50269,
                50476, 50298, 5970, 50792, 50812, 50821, 50914, 50742, 50804, 50758, 50868,
                50636, 50798, 50650, 50859, 50653, 50805, 50664, 50839, 8287, 1203, 50436,
                50402, 50631, 50692, 50765, 50759, 50778, 50777, 4328, 1671, 6870, 73, 7424,
                50651, 50876, 50699, 50985, 13630, 51074, 50943, 51159, 51028, 50837, 50947,
                50915, 51030, 2,
            ],
        }

        EXPECTED_TEXT = {
            "<DETAILED_CAPTION>": (
                "<s>In this image we can see a car on the road. In the background "
                "there are buildings, trees, boards, lights, railings, poles, people "
                "and some other objects."
            ),
            "<OD>": (
                "<s>building<loc_0><loc_0><loc_998><loc_680><loc_0><loc_0><loc_310>"
                "<loc_673><loc_763><loc_0><loc_998><loc_665>car<loc_528><loc_545>"
                "<loc_918><loc_761>handbag<loc_534><loc_600><loc_540><loc_613>lamp"
                "<loc_85><loc_0><loc_108><loc_60><loc_183><loc_0><loc_207><loc_29>"
                "person<loc_523><loc_543><loc_552><loc_645><loc_473><loc_535>"
                "<loc_489><loc_599><loc_367><loc_529><loc_381><loc_590><loc_384>"
                "<loc_536><loc_395><loc_570>stop sign<loc_167><loc_133><loc_362>"
                "<loc_423><loc_496><loc_490><loc_509><loc_508>trash bin/can<loc_382>"
                "<loc_607><loc_430><loc_716>wheel<loc_805><loc_674><loc_890><loc_759>"
                "<loc_568><loc_678><loc_646><loc_761>"
            ),
        }

        assert output.token_ids == EXPECTED_TOKEN_IDS[prompt]
        assert output.text == EXPECTED_TEXT[prompt]
