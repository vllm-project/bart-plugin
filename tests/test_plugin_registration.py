"""Tests for plugin registration and discovery."""

import pytest
from importlib.metadata import entry_points


class TestPluginRegistration:
    """Test plugin registration with vLLM."""

    def test_plugin_installed(self):
        """Test that the plugin package is installed."""
        try:
            import vllm_bart_plugin
            assert vllm_bart_plugin.__version__ is not None
        except ImportError:
            pytest.fail("vllm_bart_plugin package not installed")

    def test_entry_point_registered(self):
        """Test that the plugin entry point is registered."""
        # Get all vllm.plugins entry points
        vllm_plugins = list(entry_points(group='vllm.plugins'))

        # Check if bart plugin is registered
        plugin_names = [ep.name for ep in vllm_plugins]
        assert 'bart' in plugin_names, f"bart plugin not found in {plugin_names}"

    def test_entry_point_loads(self):
        """Test that the entry point can be loaded."""
        vllm_plugins = entry_points(group='vllm.plugins')

        bart_plugin = None
        for ep in vllm_plugins:
            if ep.name == 'bart':
                bart_plugin = ep
                break

        assert bart_plugin is not None, "bart plugin entry point not found"

        # Load the entry point
        register_fn = bart_plugin.load()
        assert callable(register_fn), "Entry point did not load a callable"

    def test_registration_function_exists(self):
        """Test that register_bart_model function exists."""
        from vllm_bart_plugin import register_bart_model
        assert callable(register_bart_model)

    def test_model_class_importable(self):
        """Test that the BART model class can be imported."""
        from vllm_bart_plugin.bart import BartForConditionalGeneration
        assert BartForConditionalGeneration is not None

    def test_model_registration(self):
        """Test that the model can be registered with ModelRegistry."""
        from vllm_bart_plugin import register_bart_model

        # Register the model
        try:
            register_bart_model()
        except Exception as e:
            pytest.fail(f"Model registration failed: {e}")

        # Verify it's in the registry
        from vllm.model_executor.models.registry import ModelRegistry

        try:
            model_cls = ModelRegistry._get_model("BartForConditionalGeneration")
            assert model_cls is not None
        except Exception as e:
            pytest.fail(f"Model not found in registry: {e}")

    def test_model_interfaces(self):
        """Test that BART model implements required interfaces."""
        from vllm_bart_plugin.bart import BartForConditionalGeneration
        from vllm.model_executor.models.interfaces import (
            SupportsMultiModal,
        )
        from vllm.model_executor.models.interfaces_base import SupportsQuant

        # Check if the class implements the interfaces
        # Note: This checks if the class has the interface in its MRO
        assert issubclass(BartForConditionalGeneration, SupportsMultiModal)
        assert issubclass(BartForConditionalGeneration, SupportsQuant)
