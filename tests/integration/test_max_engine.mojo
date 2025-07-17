"""
Test actual MAX Engine imports and real GPU operations.

This script attempts to use real MAX Engine imports and perform actual GPU operations.
"""

from collections import List

# Try actual MAX Engine imports
try:
    from max.device import Device, get_device_count, get_device

    alias MAX_DEVICE_AVAILABLE = True
except ImportError:
    alias MAX_DEVICE_AVAILABLE = False

try:
    from max.tensor import Tensor, TensorSpec, DType

    alias MAX_TENSOR_AVAILABLE = True
except ImportError:
    alias MAX_TENSOR_AVAILABLE = False

try:
    from max.graph import ops

    alias MAX_OPS_AVAILABLE = True
except ImportError:
    alias MAX_OPS_AVAILABLE = False


fn test_real_device_detection():
    """Test real MAX Engine device detection."""
    print("Testing Real Device Detection...")
    print("-" * 40)

    @parameter
    if MAX_DEVICE_AVAILABLE:
        print("✓ MAX Device module imported successfully")

        # Try to get device count
        try:
            var device_count = get_device_count()
            print("Real GPU devices detected:", device_count)

            if device_count > 0:
                # Try to get device information
                device = get_device(0)
                print("✓ Successfully accessed GPU device 0")
                return True
            else:
                print("✗ No GPU devices found")
                return False
        except:
            print("✗ Failed to access GPU devices")
            return False
    else:
        print("✗ MAX Device module not available")
        return False


fn test_real_tensor_operations():
    """Test real MAX Engine tensor operations."""
    print("\nTesting Real Tensor Operations...")
    print("-" * 40)

    @parameter
    if MAX_TENSOR_AVAILABLE:
        print("✓ MAX Tensor module imported successfully")

        try:
            # Create tensor specification
            shape = List[Int]()
            shape.append(2)
            shape.append(2)

            # Create tensor spec
            spec = TensorSpec(DType.float64, shape)
            print("✓ TensorSpec created successfully")

            # Create tensor
            tensor = Tensor[DType.float64](spec)
            print("✓ Tensor created successfully")

            return True
        except:
            print("✗ Failed to create tensors")
            return False
    else:
        print("✗ MAX Tensor module not available")
        return False


fn test_real_gpu_operations():
    """Test real MAX Engine GPU operations."""
    print("\nTesting Real GPU Operations...")
    print("-" * 40)

    @parameter
    if MAX_OPS_AVAILABLE and MAX_TENSOR_AVAILABLE:
        print("✓ MAX Ops module imported successfully")

        try:
            # Create test tensors
            shape = List[Int]()
            shape.append(2)
            shape.append(2)

            spec = TensorSpec(DType.float64, shape)
            tensor_a = Tensor[DType.float64](spec)
            tensor_b = Tensor[DType.float64](spec)

            # Test addition
            var result_add = add(tensor_a, tensor_b)
            print("✓ Tensor addition successful")

            # Test multiplication
            var result_mul = multiply(tensor_a, tensor_b)
            print("✓ Tensor multiplication successful")

            return True
        except:
            print("✗ Failed to perform GPU operations")
            return False
    else:
        print("✗ MAX Ops or Tensor modules not available")
        return False


fn main():
    """Run real MAX Engine tests."""
    print("Real MAX Engine GPU Operations Test")
    print("=" * 50)

    print("Import Status:")
    print(
        "- Device module:",
        "AVAILABLE" if MAX_DEVICE_AVAILABLE else "NOT AVAILABLE",
    )
    print(
        "- Tensor module:",
        "AVAILABLE" if MAX_TENSOR_AVAILABLE else "NOT AVAILABLE",
    )
    print(
        "- Ops module:", "AVAILABLE" if MAX_OPS_AVAILABLE else "NOT AVAILABLE"
    )

    # Run tests
    device_ok = test_real_device_detection()
    tensor_ok = test_real_tensor_operations()
    ops_ok = test_real_gpu_operations()

    print("\n" + "=" * 50)
    print("REAL GPU TEST RESULTS:")

    if device_ok:
        print("✅ Real GPU device detection: SUCCESS")
    else:
        print("❌ Real GPU device detection: FAILED")

    if tensor_ok:
        print("✅ Real tensor operations: SUCCESS")
    else:
        print("❌ Real tensor operations: FAILED")

    if ops_ok:
        print("✅ Real GPU operations: SUCCESS")
    else:
        print("❌ Real GPU operations: FAILED")

    if device_ok and tensor_ok and ops_ok:
        print("\n🎉 ALL REAL GPU TESTS PASSED!")
        print("Ready to implement production GPU acceleration!")
    else:
        print("\n⚠️  Some real GPU tests failed")
        print("Check MAX Engine installation and GPU drivers")
