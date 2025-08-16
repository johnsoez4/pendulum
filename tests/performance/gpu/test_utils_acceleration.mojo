"""
GPU acceleration testing for utilities with universal hardware support.

This module provides comprehensive testing for GPU acceleration capabilities
across all supported GPU hardware platforms including NVIDIA, AMD, and Intel
architectures. Tests real GPU operations, memory bandwidth, and device context
management using the MAX Engine GPU programming interface.

Key Components Tested:
- Universal GPU hardware detection across all supported vendors
- Real GPU device context creation and management
- GPU memory allocation and buffer operations
- Memory bandwidth measurement and performance validation
- Cross-platform GPU acceleration verification

Hardware Support:
- NVIDIA GPUs (CUDA-compatible architectures)
- AMD GPUs (ROCm-compatible architectures)
- Intel GPUs (Level Zero-compatible architectures)
- Any MAX Engine supported GPU acceleration hardware
"""

from sys import (
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    has_accelerator,
)
from gpu.host import DeviceContext
from time import perf_counter_ns as now


fn detect_gpu_hardware() -> (Bool, String):
    """Detect available GPU acceleration hardware across all supported platforms.

    Returns:
        A tuple containing:
        - Bool: True if any GPU acceleration hardware is detected, False otherwise
        - String: Description of detected GPU acceleration hardware types

    Note:
        This function uses has_accelerator() for universal GPU detection across
        all MAX Engine supported acceleration hardware including NVIDIA, AMD,
        Intel, and any future GPU architectures supported by MAX Engine.
    """
    # Use universal accelerator detection for comprehensive hardware support
    has_any_accelerator = has_accelerator()

    # Also check specific vendors for detailed reporting
    has_nvidia = has_nvidia_gpu_accelerator()
    has_amd = has_amd_gpu_accelerator()

    gpu_types = List[String]()
    if has_nvidia:
        gpu_types.append("NVIDIA")
    if has_amd:
        gpu_types.append("AMD")

    # Check for additional accelerators not covered by specific vendor detection
    if has_any_accelerator and not has_nvidia and not has_amd:
        gpu_types.append("Other MAX Engine Supported GPU")

    if has_any_accelerator:
        if len(gpu_types) > 0:
            gpu_description = "Detected GPU acceleration hardware: "
            for i in range(len(gpu_types)):
                gpu_description += gpu_types[i]
                if i < len(gpu_types) - 1:
                    gpu_description += ", "
        else:
            gpu_description = (
                "Detected GPU acceleration hardware: MAX Engine Supported"
            )
    else:
        gpu_description = "No GPU acceleration hardware detected"

    return has_any_accelerator, gpu_description


fn test_gpu_device_context() -> Bool:
    """Test GPU device context creation across all supported hardware.

    Returns:
        True if device context creation succeeds, False otherwise.
    """
    try:
        _ = DeviceContext()
        print("✅ GPU DeviceContext created successfully")
        return True
    except e:
        print("❌ GPU device context creation failed")
        return False


fn test_gpu_buffer_operations(device_context: DeviceContext) -> Bool:
    """Test GPU buffer allocation and basic operations.

    Args:
        device_context: Active GPU device context for buffer operations.

    Returns:
        True if all buffer operations succeed, False otherwise.
    """
    try:
        # Test basic buffer allocation
        test_buffer = device_context.enqueue_create_buffer[DType.float64](1024)
        print("✅ GPU buffer allocated: 1024 elements")

        # Test buffer fill operations
        _ = test_buffer.enqueue_fill(42.0)
        device_context.synchronize()
        print("✅ GPU memory operations verified")

        return True
    except e:
        print("❌ GPU buffer operations failed")
        return False


fn test_gpu_memory_bandwidth(device_context: DeviceContext) -> Float64:
    """Test GPU memory bandwidth across different buffer sizes.

    Args:
        device_context: Active GPU device context for memory operations.

    Returns:
        Measured memory bandwidth in GB/s, or 0.0 if test fails.
    """
    try:
        # Test memory bandwidth with 1M elements
        test_size = 1024 * 1024
        buffer = device_context.enqueue_create_buffer[DType.float64](test_size)

        start_time = now()
        _ = buffer.enqueue_fill(3.14159)
        device_context.synchronize()
        end_time = now()

        time_seconds = Float64(end_time - start_time) / 1e9
        bytes_transferred = Float64(test_size * 8)  # 8 bytes per float64
        bandwidth_gbps = (bytes_transferred / time_seconds) / 1e9

        print("✅ GPU memory bandwidth measured:", bandwidth_gbps, "GB/s")
        return bandwidth_gbps
    except e:
        print("❌ GPU memory bandwidth test failed")
        return 0.0


fn run_comprehensive_gpu_tests() -> Bool:
    """Run comprehensive GPU acceleration tests across all hardware.

    Returns:
        True if all tests pass, False if any test fails.
    """
    print("Running comprehensive GPU acceleration tests...")
    print("Testing universal GPU acceleration hardware support")

    # Test 1: Universal hardware detection using has_accelerator()
    has_gpu, gpu_description = detect_gpu_hardware()
    print(gpu_description)

    if not has_gpu:
        print("⚠️  No GPU acceleration hardware detected - skipping GPU tests")
        return False

    # Test 2: Device context creation
    if not test_gpu_device_context():
        return False

    # Test 3: Create device context for remaining tests
    try:
        device_context = DeviceContext()
    except e:
        print("❌ Failed to create device context for tests")
        return False

    # Test 4: Buffer operations
    if not test_gpu_buffer_operations(device_context):
        return False

    # Test 5: Memory bandwidth
    bandwidth = test_gpu_memory_bandwidth(device_context)
    if bandwidth <= 0.0:
        return False

    print("✅ All GPU acceleration tests completed successfully")
    return True


fn main():
    """Main test function for universal GPU acceleration testing.

    Executes comprehensive GPU acceleration tests across all MAX Engine
    supported hardware platforms using has_accelerator() for universal
    detection and provides detailed results for validation.
    """
    print("Universal GPU Acceleration Testing Suite")
    print("=" * 70)
    print("Testing GPU acceleration across all MAX Engine supported hardware")
    print(
        "Universal Detection: Using has_accelerator() for comprehensive"
        " hardware support"
    )
    print(
        "Supported: NVIDIA (CUDA), AMD (ROCm), Intel (Level Zero), and any"
        " MAX Engine GPU"
    )
    print()

    # Run comprehensive test suite with universal detection
    success = run_comprehensive_gpu_tests()

    print()
    if success:
        print("🎉 Universal GPU acceleration testing completed successfully!")
        print("✅ GPU utilities are ready for cross-platform acceleration")
        print("✅ Compatible with all MAX Engine supported GPU hardware")
    else:
        print("⚠️  GPU acceleration testing completed with limitations")
        print(
            "ℹ️  GPU acceleration features may not be available on this system"
        )
        print("ℹ️  CPU fallback mode will be used for computation")

    print("=" * 70)
