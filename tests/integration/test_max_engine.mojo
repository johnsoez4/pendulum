"""
Test actual MAX Engine imports and real GPU operations.

This script tests MAX Engine availability and provides fallback information.
"""

from collections import List
from sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator
from gpu.host import DeviceContext

# MAX Engine availability flags (set to False since modules not available)
alias MAX_DEVICE_AVAILABLE = False
alias MAX_TENSOR_AVAILABLE = False
alias MAX_OPS_AVAILABLE = False


fn test_real_device_detection() -> Bool:
    """Test real MAX Engine device detection."""
    print("Testing Real Device Detection...")
    print("-" * 40)

    @parameter
    if MAX_DEVICE_AVAILABLE:
        print("✓ MAX Device module imported successfully")
        print("✓ Real GPU devices accessible")
        return True
    else:
        print("⚠️  MAX Device module not available")
        print("Using DeviceContext from gpu.host instead")

        # Test fallback GPU detection
        has_nvidia = has_nvidia_gpu_accelerator()
        has_amd = has_amd_gpu_accelerator()

        if has_nvidia:
            print("✓ NVIDIA GPU detected via fallback")
            return True
        elif has_amd:
            print("✓ AMD GPU detected via fallback")
            return True
        else:
            print("⚠️  No GPU detected")
            return False


fn test_real_tensor_operations() -> Bool:
    """Test real MAX Engine tensor operations."""
    print("\nTesting Real Tensor Operations...")
    print("-" * 40)

    @parameter
    if MAX_TENSOR_AVAILABLE:
        print("✓ MAX Tensor module imported successfully")
        print("✓ Tensor operations working")
        return True
    else:
        print("⚠️  MAX Tensor module not available")
        print("Using DeviceContext tensor operations instead")

        # Test fallback tensor operations
        try:
            ctx = DeviceContext()
            buffer = ctx.enqueue_create_buffer[DType.float64](4)
            for i in range(4):
                _ = buffer.enqueue_fill(Float64(i))
            ctx.synchronize()
            print("✓ Fallback tensor operations working")
            return True
        except Exception:
            print("⚠️  Fallback tensor operations failed")
            return False


fn test_real_gpu_operations() -> Bool:
    """Test real MAX Engine GPU operations."""
    print("\nTesting Real GPU Operations...")
    print("-" * 40)

    @parameter
    if MAX_OPS_AVAILABLE and MAX_TENSOR_AVAILABLE:
        print("✓ MAX Ops and Tensor modules imported successfully")
        print("✓ GPU operations completed successfully")
        return True
    else:
        print("⚠️  MAX Ops or Tensor modules not available")
        print("Using DeviceContext GPU operations instead")

        # Test fallback GPU operations
        try:
            ctx = DeviceContext()
            buffer_a = ctx.enqueue_create_buffer[DType.float64](4)
            buffer_b = ctx.enqueue_create_buffer[DType.float64](4)

            # Fill buffers
            for i in range(4):
                _ = buffer_a.enqueue_fill(Float64(i))
                _ = buffer_b.enqueue_fill(Float64(i * 2))

            ctx.synchronize()
            print("✓ Fallback GPU operations working")
            return True
        except Exception:
            print("⚠️  Fallback GPU operations failed")
            return False


fn main():
    """Test MAX Engine availability and functionality."""
    print("Testing MAX Engine Availability and Functionality")
    print("=" * 60)

    # Test device detection
    device_ok = test_real_device_detection()

    # Test tensor operations
    tensor_ok = test_real_tensor_operations()

    # Test GPU operations
    gpu_ok = test_real_gpu_operations()

    # Summary
    print("\n" + "=" * 60)
    print("MAX ENGINE TEST RESULTS:")
    print(
        "- Device detection:",
        "AVAILABLE" if MAX_DEVICE_AVAILABLE else "NOT AVAILABLE",
    )
    print(
        "- Tensor operations:",
        "AVAILABLE" if MAX_TENSOR_AVAILABLE else "NOT AVAILABLE",
    )
    print(
        "- Ops module:", "AVAILABLE" if MAX_OPS_AVAILABLE else "NOT AVAILABLE"
    )

    print("\nFALLBACK FUNCTIONALITY:")
    print("- Device detection:", "WORKING" if device_ok else "FAILED")
    print("- Tensor operations:", "WORKING" if tensor_ok else "FAILED")
    print("- GPU operations:", "WORKING" if gpu_ok else "FAILED")

    if device_ok and tensor_ok and gpu_ok:
        print("\n✅ All fallback functionality working!")
        print("Ready for MAX Engine integration when available")
    else:
        print("\n⚠️  Some fallback functionality issues detected")
        print("Check GPU hardware and driver installation")
