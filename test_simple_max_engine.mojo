"""
Simple test for MAX Engine imports and basic functionality.
"""

from collections import List

fn test_max_device_import() -> Bool:
    """Test MAX Engine device import."""
    print("Testing MAX Engine device import...")
    
    # Try to import max.device
    try:
        from max.device import Device, get_device_count
        print("✓ Successfully imported max.device")
        
        # Try to get device count
        var device_count = get_device_count()
        print("✓ Device count:", device_count)
        
        if device_count > 0:
            print("✓ GPU devices available:", device_count)
            return True
        else:
            print("✗ No GPU devices found")
            return False
            
    except ImportError:
        print("✗ Failed to import max.device")
        return False
    except:
        print("✗ Error accessing GPU devices")
        return False

fn test_max_tensor_import() -> Bool:
    """Test MAX Engine tensor import."""
    print("\nTesting MAX Engine tensor import...")
    
    try:
        from max.tensor import Tensor, TensorSpec, DType
        print("✓ Successfully imported max.tensor")
        
        # Try to create a simple tensor spec
        shape = List[Int]()
        shape.append(2)
        shape.append(2)
        
        spec = TensorSpec(DType.float64, shape)
        print("✓ TensorSpec created successfully")
        
        return True
        
    except ImportError:
        print("✗ Failed to import max.tensor")
        return False
    except:
        print("✗ Error creating tensor spec")
        return False

fn test_max_ops_import() -> Bool:
    """Test MAX Engine operations import."""
    print("\nTesting MAX Engine ops import...")
    
    try:
        from max.ops import add, multiply
        print("✓ Successfully imported max.ops")
        return True
        
    except ImportError:
        print("✗ Failed to import max.ops")
        return False
    except:
        print("✗ Error with max.ops")
        return False

fn main():
    """Run simple MAX Engine tests."""
    print("Simple MAX Engine Test")
    print("=" * 40)
    
    print("Environment:")
    print("- Mojo: 25.5.0.dev2025062905")
    print("- MAX Engine: 25.5.0.dev2025062905")
    print("- GPU: NVIDIA A10 (23GB)")
    
    # Test imports
    device_ok = test_max_device_import()
    tensor_ok = test_max_tensor_import()
    ops_ok = test_max_ops_import()
    
    print("\n" + "=" * 40)
    print("Results:")
    
    if device_ok:
        print("✅ Device import: SUCCESS")
    else:
        print("❌ Device import: FAILED")
        
    if tensor_ok:
        print("✅ Tensor import: SUCCESS")
    else:
        print("❌ Tensor import: FAILED")
        
    if ops_ok:
        print("✅ Ops import: SUCCESS")
    else:
        print("❌ Ops import: FAILED")
    
    if device_ok and tensor_ok and ops_ok:
        print("\n🎉 All MAX Engine imports successful!")
        print("Ready for real GPU implementation!")
    else:
        print("\n⚠️  Some imports failed")
        print("Will use fallback patterns")
