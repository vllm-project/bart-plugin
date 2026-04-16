"""vLLM BART / NLLB / M2M-100 model plugin.

This plugin registers BART, Florence-2, and NLLB/M2M-100 models with
vLLM's ModelRegistry, allowing them to be used with vLLM's inference engine.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.model_executor.models.registry import ModelRegistry

__version__ = "0.1.0"


def register_bart_model() -> None:
    """Register BART, Florence-2, and NLLB/M2M-100 models with vLLM's ModelRegistry.

    This function is called automatically when the plugin is loaded
    through vLLM's plugin discovery mechanism.
    """
    try:
        from vllm.logger import init_logger
        from vllm.model_executor.models.registry import ModelRegistry

        logger = init_logger(__name__)

        ModelRegistry.register_model(
            "BartForConditionalGeneration",
            "vllm_bart_plugin.bart:BartForConditionalGeneration",
        )
        ModelRegistry.register_model(
            "Florence2ForConditionalGeneration",
            "vllm_bart_plugin.florence2:Florence2ForConditionalGeneration",
        )
        # M2M100ForConditionalGeneration covers all NLLB distilled models:
        #   facebook/nllb-200-distilled-600M
        #   facebook/nllb-200-distilled-1.3B
        #   facebook/nllb-200-3.3B
        ModelRegistry.register_model(
            "M2M100ForConditionalGeneration",
            "vllm_bart_plugin.nllb:M2M100ForConditionalGeneration",
        )

        logger.info("Successfully registered BART, Florence-2, and NLLB/M2M-100 models with vLLM")

    except Exception as e:
        logger.error(f"Failed to register models: {e}")
        raise


__all__ = [
    "register_bart_model",
    "__version__",
]
