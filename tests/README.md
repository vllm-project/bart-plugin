# BART Plugin Tests

This directory contains the test suite for the vLLM BART plugin.

## Test Structure

```
tests/
├── __init__.py                      # Test package marker
├── conftest.py                      # Pytest configuration and fixtures
├── test_plugin_registration.py     # Plugin registration tests
├── test_model_initialization.py    # Model initialization tests
└── test_model_inference.py         # Model inference tests
```

## Running Tests

### Install Test Dependencies

```bash
uv pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
# Plugin registration tests (fast)
pytest tests/test_plugin_registration.py

# Model initialization tests (slow)
pytest tests/test_model_initialization.py

# Model inference tests (slow)
pytest tests/test_model_inference.py
```

### Run Tests by Marker

```bash
# Skip slow tests
pytest -m "not slow"

# Run only slow tests
pytest -m slow

# Run only GPU tests
pytest -m gpu
```

### Run Specific Tests

```bash
# Run specific test class
pytest tests/test_plugin_registration.py::TestPluginRegistration

# Run specific test method
pytest tests/test_plugin_registration.py::TestPluginRegistration::test_plugin_installed
```

### Verbose Output

```bash
# Show more details
pytest -v

# Show print statements
pytest -s

# Show all output
pytest -vv -s
```

## Test Categories

### Fast Tests (Plugin Registration)

These tests run quickly and verify:
- Plugin package installation
- Entry point registration
- Model class imports
- Interface implementations

Run with:
```bash
pytest tests/test_plugin_registration.py
```

### Slow Tests (Model Initialization & Inference)

These tests download models and perform inference:
- Model loading and initialization
- Single and batch generation
- Different sampling strategies
- Edge cases

Run with:
```bash
pytest -m slow
```

## Test Fixtures

Available fixtures (defined in `conftest.py`):

- `cuda_available`: Check if CUDA is available
- `device`: Get device for tests ('cuda' or 'cpu')
- `small_model_name`: Small model for testing (`facebook/bart-base`)
- `test_prompts`: Sample prompts for inference tests
- `skip_if_no_gpu`: Skip test if GPU not available

## Writing New Tests

### Basic Test Structure

```python
import pytest

class TestMyFeature:
    """Test my new feature."""

    def test_something(self):
        """Test that something works."""
        assert True

    @pytest.mark.slow
    def test_slow_operation(self, llm):
        """Test that requires model loading."""
        # This will be skipped with -m "not slow"
        result = llm.generate(...)
        assert result is not None
```

### Using Fixtures

```python
def test_with_model(self, small_model_name):
    """Test using the small_model_name fixture."""
    from vllm import LLM
    llm = LLM(model=small_model_name)
    # ...

def test_with_prompts(self, test_prompts):
    """Test using the test_prompts fixture."""
    assert len(test_prompts) > 0
    # ...
```

### Marking Tests

```python
import pytest

@pytest.mark.slow
def test_slow_feature():
    """This test takes a long time."""
    pass

@pytest.mark.gpu
def test_gpu_feature(skip_if_no_gpu):
    """This test requires a GPU."""
    pass
```

## Continuous Integration

For CI/CD pipelines:

```bash
# Fast tests only (for quick feedback)
pytest -m "not slow" --tb=short

# All tests (for complete validation)
pytest --tb=short --maxfail=3
```

## Coverage

To generate coverage reports:

```bash
# Install coverage tool
uv pip install pytest-cov

# Run with coverage
pytest --cov=vllm_bart_plugin --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html
```

## Troubleshooting

### Model Download Issues

Tests will download `facebook/bart-base` on first run. If you encounter issues:

```bash
# Pre-download the model
python -c "from transformers import BartForConditionalGeneration; BartForConditionalGeneration.from_pretrained('facebook/bart-base')"
```

### CUDA/GPU Issues

If GPU tests fail:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Skip GPU tests
pytest -m "not gpu"
```

### Memory Issues

If you encounter OOM errors:

```bash
# Reduce GPU memory utilization in tests
# Edit conftest.py and reduce gpu_memory_utilization value

# Or run fewer tests at once
pytest --maxfail=1  # Stop after first failure
```

### Import Errors

If imports fail:

```bash
# Ensure plugin is installed
uv pip install -e .

# Verify installation
python verify_plugin.py
```

## Test Performance

Approximate test runtimes:

- `test_plugin_registration.py`: ~2 seconds
- `test_model_initialization.py`: ~30-60 seconds (model download + loading)
- `test_model_inference.py`: ~60-120 seconds (inference tests)

Total runtime: ~2-3 minutes for first run, ~1-2 minutes for subsequent runs (cached models).

## Best Practices

1. **Fast tests first**: Run fast tests before slow tests
2. **Use markers**: Mark slow tests with `@pytest.mark.slow`
3. **Fixtures for setup**: Use fixtures for common setup code
4. **Descriptive names**: Use clear test names that describe what's being tested
5. **One concept per test**: Each test should verify one specific behavior
6. **Assertions**: Include clear assertion messages for failures

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [vLLM Testing Guide](https://docs.vllm.ai/en/latest/contributing/)
- [BART Model Tests](https://github.com/huggingface/transformers/tree/main/tests/models/bart)
