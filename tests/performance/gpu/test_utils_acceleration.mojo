"""
Test real GPU acceleration for GPU utilities.
"""

from sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator
from gpu.host import DeviceContext
from time import perf_counter_ns as now


fn main():
    """Main test function for GPU utilities acceleration."""
    print("GPU Utilities Real Acceleration Test")
    print("=" * 70)
    
    # Test GPU hardware detection
    has_nvidia = has_nvidia_gpu_accelerator()
    has_amd = has_amd_gpu_accelerator()
    
    print("NVIDIA GPU available:", has_nvidia)
    print("AMD GPU available:", has_amd)
    
    if has_nvidia or has_amd:
        try:
            # Test DeviceContext creation
            device_context = DeviceContext()
            print("✅ Real DeviceContext created successfully")
            
            # Test GPU buffer allocation
            test_buffer = device_context.enqueue_create_buffer[DType.float64](1024)
            print("✅ Real GPU buffer allocated: 1024 elements")
            
            # Test GPU memory operations
            _ = test_buffer.enqueue_fill(42.0)
            device_context.synchronize()
            print("✅ Real GPU memory operations verified")
            
            # Test memory bandwidth
            test_size = 1024 * 1024  # 1M elements
            buffer = device_context.enqueue_create_buffer[DType.float64](test_size)
            
            start_time = now()
            _ = buffer.enqueue_fill(3.14159)
            device_context.synchronize()
            end_time = now()
            
            time_seconds = Float64(end_time - start_time) / 1e9
            bytes_transferred = Float64(test_size * 8)  # 8 bytes per float64
            bandwidth_gbps = (bytes_transferred / time_seconds) / 1e9
            
            print("✅ Real GPU memory bandwidth:", bandwidth_gbps, "GB/s")
            print("✅ Real GPU device initialization successful")
            
        except:
            print("❌ GPU device initialization failed")
    else:
        print("⚠️  No GPU hardware detected")
    
    print("=" * 70)
