# vLLM BART Model Plugin

This plugin adds support for BART (Bidirectional and Auto-Regressive Transformers) models to vLLM's inference engine.

## Overview

BART is an encoder-decoder model that is particularly effective for sequence-to-sequence tasks like summarization, translation, and text generation. This plugin integrates the BART model implementation with vLLM's plugin system, allowing you to use BART models with vLLM's optimized inference engine.

## Installation

### Prerequisites

This plugin requires [uv](https://docs.astral.sh/uv/) for package management. If you don't have it installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### From Source

1. Clone this repository:
```bash
git clone <repository-url>
cd bart-plugin
```

2. Install the plugin in development mode:
```bash
uv pip install -e .
```

Or install directly:
```bash
uv pip install .
```

### Verify Installation

After installation, vLLM should automatically discover and load the BART plugin. You can verify by checking the vLLM logs when initializing a model.

```bash
python verify_plugin.py
```

## Usage

### Basic Usage

Run `python example_usage.py` or

```python
import vllm_bart_plugin
from vllm import LLM, SamplingParams
model_name = "facebook/bart-large-cnn"

llm = LLM(
    model=model_name,
    max_model_len=1024,
    gpu_memory_utilization=0.5,
    dtype="float16",
)
params = SamplingParams(temperature=0.0, max_tokens=20)
outputs = llm.generate(
    [
        {  
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "The president of the United States is",
                },
            },
            "decoder_prompt": "<s>Donald",
        },
        {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "<s>",
                },
            },
            "decoder_prompt": "<s>The capital of France is",
        },
    ],
    sampling_params=params,
)
for o in outputs:
    generated_text = o.outputs[0].text
    print("output:", generated_text)
```

## Plugin Architecture

This plugin follows vLLM's plugin system architecture:

1. **Entry Point**: Registered via setuptools entry_points in `setup.py`
2. **Registration Function**: `register_bart_model()` in `__init__.py` registers the model with vLLM's ModelRegistry
3. **Model Implementation**: The BART model class in `bart.py` implements vLLM's model interfaces

### Plugin Discovery

vLLM automatically discovers plugins using Python's entry point mechanism. The plugin is registered under the `vllm.plugins` group and is loaded when vLLM initializes.

## Model Features

The BART implementation includes:

- **Encoder-Decoder Architecture**: Full support for BART's encoder-decoder structure
- **Cross-Attention**: Proper implementation of cross-attention between encoder and decoder
- **Multi-Modal Support**: Integration with vLLM's multimodal processing pipeline
- **Quantization Support**: Compatible with vLLM's quantization features
- **Tensor Parallelism**: Support for distributed inference

## Supported Models

This plugin should work with any BART-based model from HuggingFace, including:

- `facebook/bart-large-cnn`
- `facebook/bart-large`
- `facebook/bart-large-cnn`
- Other BART variants and fine-tuned models

## TODO
 - [ ] Support `MBartForConditionalGeneration`

## Environment Variables

You can control plugin loading with the `VLLM_PLUGINS` environment variable:

```bash
# Load all plugins (default)
export VLLM_PLUGINS=all

# Load only specific plugins
export VLLM_PLUGINS=bart

# Disable all plugins
export VLLM_PLUGINS=none
```

### `VLLM_BART_ENCODER_MAX_SEQ_PADDING`

Enable a small optimization that **batches encoder forwards** by padding all encoder inputs in a batch to the maximum encoder sequence length, running the encoder once, then slicing outputs back to per-item lengths.

- **Default**: disabled
- **Enable**:

```bash
export VLLM_BART_ENCODER_MAX_SEQ_PADDING=1
```

Notes:
- Requires `pad_token_id` to be set in the HF config. If it is missing, the plugin will log a warning and keep the optimization disabled.


## Development

### Project Structure

```
bart-plugin/
├── vllm_bart_plugin/
│   ├── __init__.py          # Plugin registration
│   └── bart.py              # BART model implementation
├── setup.py                 # Package configuration and entry points
├── README.md                # This file
└── LICENSE                  # License file
```

### Running Tests

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run fast tests
pytest -m "not slow"

# Run all tests
pytest

# Run with coverage
pytest --cov=vllm_bart_plugin --cov-report=html
```

See [tests/README.md](tests/README.md) for detailed testing documentation.

### Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

Quick checklist:
1. Install pre-commit hooks: `pre-commit install`
2. Code follows style guidelines (enforced by pre-commit)
3. All tests pass: `pytest`
4. Documentation is updated for new features

Pre-commit hooks will automatically:
- Format code with ruff
- Check types with mypy
- Scan for security issues
- Validate documentation style

## Troubleshooting

### Plugin Not Loading

If the plugin isn't being discovered:

1. Verify installation: `uv pip list | grep vllm-bart-plugin`
2. Check entry points: `python -c "from importlib.metadata import entry_points; print(list(entry_points(group='vllm.plugins')))"`
3. Enable verbose logging: Set `VLLM_LOGGING_LEVEL=DEBUG`
4. Run the verification script: `python verify_plugin.py`

### Model Not Found

If vLLM doesn't recognize the BART model:

1. Ensure the plugin loaded successfully (check logs) you should see
```bash
[2025-12-19 14:32:11] INFO __init__.py:33: Successfully registered BART model with vLLM
```
2. Verify the model architecture name matches: `BartForConditionalGeneration`
3. Try explicitly setting `trust_remote_code=False`

### Import Errors

Make sure all dependencies are installed:

```bash
uv pip install vllm torch transformers
```