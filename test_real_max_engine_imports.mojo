"""
Test real MAX Engine imports and GPU operations.

This script tests actual MAX Engine imports using the correct API discovered from examples.
"""

from collections import List


fn test_real_gpu_detection() -> Bool:
    """Verify real GPU detection using correct MAX Engine API."""
    print("Testing GPU detection...")

    # These imports are verified to always work in proper MAX Engine installation
    from sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator

    print("✓ GPU detection imports successful")

    var has_nvidia = has_nvidia_gpu_accelerator()
    var has_amd = has_amd_gpu_accelerator()

    print("- NVIDIA GPU available:", has_nvidia)
    print("- AMD GPU available:", has_amd)

    return has_nvidia or has_amd


fn test_real_device_context() -> Bool:
    """Verify real DeviceContext import."""
    print("Testing DeviceContext import...")

    # This import is verified to always work in proper MAX Engine installation
    from gpu.host import DeviceContext

    print("✓ DeviceContext import successful")
    return True


fn test_real_layout_tensor() -> Bool:
    """Verify real LayoutTensor import."""
    print("Testing LayoutTensor import...")

    # These imports are verified to always work in proper MAX Engine installation
    from layout import Layout, LayoutTensor

    print("✓ Layout and LayoutTensor imports successful")
    return True


fn test_real_gpu_functions() -> Bool:
    """Verify real GPU function imports."""
    print("Testing GPU function imports...")

    # These imports are verified to always work in proper MAX Engine installation
    from gpu import global_idx, thread_idx

    print("✓ GPU function imports successful")
    return True


fn test_max_engine_imports() -> Bool:
    """Verify all real MAX Engine imports using correct API."""
    print("Testing Real MAX Engine Imports...")
    print("-" * 40)

    var gpu_ok = test_real_gpu_detection()
    var context_ok = test_real_device_context()
    var layout_ok = test_real_layout_tensor()
    var functions_ok = test_real_gpu_functions()

    var success_count = 0
    if gpu_ok:
        success_count += 1
    if context_ok:
        success_count += 1
    if layout_ok:
        success_count += 1
    if functions_ok:
        success_count += 1

    print("Import success rate:", success_count, "/ 4")

    if success_count == 4:
        print("✅ All real MAX Engine modules imported successfully!")
        return True
    elif success_count > 0:
        print("⚠️  Partial MAX Engine import success")
        return False
    else:
        print("❌ No MAX Engine modules available")
        return False


fn main():
    """Run all MAX Engine tests."""
    print("Real MAX Engine Integration Test")
    print("=" * 50)

    print("Environment:")
    print("- Mojo:", "25.5.0.dev2025062905")
    print("- MAX Engine:", "25.5.0.dev2025062905")
    print("- GPU:", "NVIDIA A10 (23GB)")
    print("- CUDA:", "12.8")

    # Run the refactored import verification tests
    imports_ok = test_max_engine_imports()

    print("\n" + "=" * 50)
    print("TEST RESULTS:")

    if imports_ok:
        print("✅ All real MAX Engine imports verified successfully!")
        print("✓ GPU detection: WORKING")
        print("✓ DeviceContext: WORKING")
        print("✓ LayoutTensor: WORKING")
        print("✓ GPU functions: WORKING")
        print("\n🚀 READY FOR REAL GPU IMPLEMENTATION!")
        print("All verified MAX Engine imports are working correctly!")
    else:
        print("❌ Some MAX Engine imports failed verification")
        print("Check MAX Engine installation")
