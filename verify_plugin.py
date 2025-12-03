#!/usr/bin/env python3
"""
Verification script for vLLM BART plugin installation.

This script checks that the plugin is properly installed and registered
with vLLM's plugin system.
"""

import sys


def check_plugin_installed():
    """Check if the plugin package is installed."""
    print("Checking plugin installation...")
    try:
        import vllm_bart_plugin
        print(f"✓ Plugin package found (version {vllm_bart_plugin.__version__})")
        return True
    except ImportError as e:
        print(f"✗ Plugin package not found: {e}")
        print("  Run: pip install -e .")
        return False


def check_entry_points():
    """Check if the plugin is registered as an entry point."""
    print("\nChecking entry point registration...")
    try:
        from importlib.metadata import entry_points

        # Get all vllm.plugins entry points
        vllm_plugins = entry_points(group='vllm.plugins')

        # Look for bart plugin
        bart_plugin = None
        for ep in vllm_plugins:
            if ep.name == 'bart':
                bart_plugin = ep
                break

        if bart_plugin:
            print(f"✓ Plugin registered as entry point: {bart_plugin.value}")
            return True
        else:
            print("✗ Plugin entry point not found")
            print(f"  Available plugins: {[ep.name for ep in vllm_plugins]}")
            return False

    except Exception as e:
        print(f"✗ Error checking entry points: {e}")
        return False


def check_registration_function():
    """Check if the registration function can be loaded."""
    print("\nChecking registration function...")
    try:
        from vllm_bart_plugin import register_bart_model
        print("✓ Registration function loaded successfully")
        return True
    except ImportError as e:
        print(f"✗ Cannot load registration function: {e}")
        return False


def check_model_class():
    """Check if the BART model class can be imported."""
    print("\nChecking BART model class...")
    try:
        from vllm_bart_plugin.bart import BartForConditionalGeneration
        print("✓ BartForConditionalGeneration class imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Cannot import BART model class: {e}")
        return False


def check_vllm_dependencies():
    """Check if vLLM and its dependencies are available."""
    print("\nChecking vLLM dependencies...")

    dependencies = {
        'vllm': 'vLLM',
        'torch': 'PyTorch',
        'transformers': 'Transformers',
    }

    all_ok = True
    for module_name, display_name in dependencies.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name} {version}")
        except ImportError:
            print(f"✗ {display_name} not found")
            all_ok = False

    return all_ok


def test_model_registration():
    """Test that the model can be registered with ModelRegistry."""
    print("\nTesting model registration...")
    try:
        # Import the registration function
        from vllm_bart_plugin import register_bart_model

        # Try to register the model
        register_bart_model()

        # Check if it's in the registry
        from vllm.model_executor.models.registry import ModelRegistry

        # Try to get the model class
        model_cls = ModelRegistry._get_model("BartForConditionalGeneration")
        print(f"✓ Model registered successfully: {model_cls}")
        return True

    except Exception as e:
        print(f"✗ Model registration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_cuda_availability():
    """Check CUDA availability (optional)."""
    print("\nChecking CUDA availability (optional)...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available (version {torch.version.cuda})")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA not available (CPU-only mode)")
        return True
    except Exception as e:
        print(f"⚠ Cannot check CUDA: {e}")
        return True  # Non-critical


def main():
    """Run all verification checks."""
    print("=" * 80)
    print("vLLM BART Plugin - Verification Script")
    print("=" * 80)

    checks = [
        ("Plugin Installation", check_plugin_installed),
        ("Entry Point Registration", check_entry_points),
        ("Registration Function", check_registration_function),
        ("Model Class Import", check_model_class),
        ("vLLM Dependencies", check_vllm_dependencies),
        ("Model Registration", test_model_registration),
        ("CUDA Availability", check_cuda_availability),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("Verification Summary")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("\n" + "=" * 80)
    print(f"Results: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All checks passed! The plugin is ready to use.")
        print("\nNext steps:")
        print("  1. Run example_usage.py to test the plugin")
        print("  2. Read README.md for usage instructions")
        print("  3. Check INSTALL.md for troubleshooting tips")
        return 0
    else:
        print("\n⚠ Some checks failed. Please review the errors above.")
        print("\nTroubleshooting:")
        print("  1. Ensure the plugin is installed: pip install -e .")
        print("  2. Check that vLLM is installed: pip install vllm")
        print("  3. See INSTALL.md for detailed troubleshooting")
        return 1


if __name__ == "__main__":
    sys.exit(main())
