# Troubleshooting Guide - Real GPU Implementation

## Overview

This guide provides solutions for common issues encountered when using the real GPU implementation with MAX Engine DeviceContext on compatible GPU hardware.

## Quick Diagnostics

### System Health Check
Run this diagnostic script to identify common issues:

```bash
#!/bin/bash
echo "=== GPU Implementation Diagnostics ==="

# 1. Environment Check
echo "1. Environment Check:"
echo "   Mojo Version: $(mojo -v 2>/dev/null || echo 'NOT FOUND')"
echo "   MAX Version: $(max --version 2>/dev/null || echo 'NOT FOUND')"
echo "   Pixi Status: $(pixi info 2>/dev/null | grep -q 'pixi' && echo 'OK' || echo 'NOT ACTIVE')"

# 2. GPU Hardware Check
echo "2. GPU Hardware Check:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "   NVIDIA GPU: NOT DETECTED"
rocm-smi --showproductname 2>/dev/null || echo "   AMD GPU: NOT DETECTED"

# 3. GPU Toolkit Check
echo "3. GPU Toolkit Check:"
nvcc --version 2>/dev/null | grep "release" || echo "   CUDA: NOT FOUND"
hipcc --version 2>/dev/null | grep "HIP" || echo "   ROCm: NOT FOUND"

# 4. Mojo GPU Test
echo "4. Mojo GPU Test:"
mojo -c "from sys import has_nvidia_gpu_accelerator; print('   GPU Detected:', has_nvidia_gpu_accelerator())" 2>/dev/null || echo "   Mojo GPU Test: FAILED"

# 5. DeviceContext Test
echo "5. DeviceContext Test:"
mojo -c "from gpu.host import DeviceContext; print('   DeviceContext: OK')" 2>/dev/null || echo "   DeviceContext: FAILED"

echo "=== Diagnostics Complete ==="
```

Save as `diagnose_gpu.sh` and run: `bash diagnose_gpu.sh`

## Common Issues and Solutions

### Issue 1: GPU Not Detected

#### Symptoms
```bash
mojo -c "from sys import has_nvidia_gpu_accelerator; print(has_nvidia_gpu_accelerator())"
# Output: False
```

#### Diagnosis
```bash
# Check GPU hardware
nvidia-smi
# Should show GPU information

# Check driver version
nvidia-smi --query-gpu=driver_version --format=csv,noheader
# Should show driver version (recommended: 535.0+)
```

#### Solutions

**Solution 1: Install/Update GPU Drivers**
```bash
# For NVIDIA GPUs (Ubuntu/Debian)
sudo apt update
sudo apt install nvidia-driver-535

# For AMD GPUs (Ubuntu/Debian)
sudo apt install amdgpu-dkms rocm-dev

# Verify installation
nvidia-smi  # For NVIDIA
rocm-smi    # For AMD
```

**Solution 2: Verify GPU Toolkit Installation**
```bash
# For NVIDIA: Check CUDA version
nvcc --version

# For AMD: Check ROCm version
hipcc --version

# Install toolkit if missing (follow vendor-specific guides)
```

**Solution 3: Restart and Verify**
```bash
# Restart system
sudo reboot

# Verify after restart
nvidia-smi  # For NVIDIA
rocm-smi    # For AMD
mojo -c "from sys import has_nvidia_gpu_accelerator; print(has_nvidia_gpu_accelerator())"
```

### Issue 2: DeviceContext Initialization Fails

#### Symptoms
```mojo
from gpu.host import DeviceContext
var ctx = DeviceContext()  # Raises exception
```

#### Diagnosis
```bash
# Check MAX Engine installation
max --version

# Check GPU memory (NVIDIA)
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Check for conflicting processes (NVIDIA)
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

# For AMD GPUs, use rocm-smi equivalent commands
```

#### Solutions

**Solution 1: Reinstall MAX Engine**
```bash
# Reinstall MAX Engine
pixi remove max
pixi add max

# Verify installation
max --version
```

**Solution 2: Clear GPU Memory**
```bash
# Kill GPU processes if necessary
sudo nvidia-smi --gpu-reset

# Or restart NVIDIA services
sudo systemctl restart nvidia-persistenced
```

**Solution 3: Check GPU Memory**
```bash
# Ensure sufficient GPU memory (>2GB free recommended)
nvidia-smi

# If memory is full, restart system or kill GPU processes
```

### Issue 3: Buffer Allocation Fails

#### Symptoms
```mojo
var buffer = ctx.enqueue_create_buffer[DType.float64](large_size)
# Raises out-of-memory exception
```

#### Diagnosis
```bash
# Check available GPU memory
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits

# Calculate buffer size requirements
# Float64 = 8 bytes per element
# Buffer size in MB = (elements * 8) / (1024 * 1024)
```

#### Solutions

**Solution 1: Reduce Buffer Size**
```mojo
# Use smaller buffer sizes
var max_elements = 10000  # Adjust based on available memory
var safe_size = min(requested_size, max_elements)
var buffer = ctx.enqueue_create_buffer[DType.float64](safe_size)
```

**Solution 2: Batch Processing**
```mojo
# Process data in batches
var batch_size = 1000
var total_elements = 50000

for batch_start in range(0, total_elements, batch_size):
    var batch_end = min(batch_start + batch_size, total_elements)
    var current_batch_size = batch_end - batch_start
    
    var batch_buffer = ctx.enqueue_create_buffer[DType.float64](current_batch_size)
    # Process batch
    ctx.synchronize()
```

**Solution 3: Memory Management**
```mojo
# Explicit memory management (if needed)
fn process_with_memory_management(mut ctx: DeviceContext) raises:
    try:
        var buffer = ctx.enqueue_create_buffer[DType.float64](10000)
        # Use buffer
        ctx.synchronize()
        # Buffer automatically cleaned up when going out of scope
    except:
        print("Memory allocation failed - using smaller buffer")
        var small_buffer = ctx.enqueue_create_buffer[DType.float64](1000)
        ctx.synchronize()
```

### Issue 4: Poor GPU Performance

#### Symptoms
```mojo
var speedup = benchmark_gpu_vs_cpu()
print(speedup)  # Output: < 2.0x (expected: > 3.0x)
```

#### Diagnosis
```bash
# Check GPU utilization during operation
nvidia-smi dmon -s u -c 10

# Check GPU clocks
nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv,noheader

# Check thermal throttling
nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader
```

#### Solutions

**Solution 1: Optimize Buffer Sizes**
```mojo
# Use optimal buffer sizes (powers of 2 often work well)
var optimal_sizes = List[Int]()
optimal_sizes.append(1024)   # 1K elements
optimal_sizes.append(4096)   # 4K elements
optimal_sizes.append(16384)  # 16K elements

# Test different sizes to find optimal
for i in range(len(optimal_sizes)):
    var size = optimal_sizes[i]
    var speedup = benchmark_with_size(size)
    print("Size:", size, "Speedup:", speedup, "x")
```

**Solution 2: Proper Synchronization**
```mojo
# Ensure proper synchronization for accurate timing
fn benchmark_with_sync(mut ctx: DeviceContext) raises -> Float64:
    # CPU benchmark
    var cpu_start = Float64(now()) / 1_000_000_000.0
    # CPU operations here
    var cpu_end = Float64(now()) / 1_000_000_000.0
    var cpu_time = cpu_end - cpu_start
    
    # GPU benchmark with proper sync
    ctx.synchronize()  # Ensure GPU is idle before starting
    var gpu_start = Float64(now()) / 1_000_000_000.0
    
    # GPU operations here
    var buffer = ctx.enqueue_create_buffer[DType.float64](10000)
    for i in range(10000):
        _ = buffer.enqueue_fill(Float64(i) * 0.001)
    
    ctx.synchronize()  # Wait for completion before timing
    var gpu_end = Float64(now()) / 1_000_000_000.0
    var gpu_time = gpu_end - gpu_start
    
    return cpu_time / gpu_time
```

**Solution 3: Check System Load**
```bash
# Check CPU usage
top -p $(pgrep mojo)

# Check system memory
free -h

# Check disk I/O
iostat -x 1 5

# Reduce system load if necessary
```

### Issue 5: Compilation Errors

#### Symptoms
```bash
mojo run gpu_program.mojo
# Error: package 'gpu.host' does not contain 'DeviceContext'
```

#### Diagnosis
```bash
# Check Mojo version
mojo -v

# Check MAX Engine installation
pixi list | grep max

# Check import paths
mojo -c "import sys; print(sys.path)"
```

#### Solutions

**Solution 1: Update Mojo and MAX Engine**
```bash
# Update pixi environment
pixi update

# Verify versions
mojo -v      # Should be 25.5.0+
max --version # Should be 25.5.0+
```

**Solution 2: Fix Import Statements**
```mojo
# Correct imports for real GPU implementation
from sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator
from gpu.host import DeviceContext
from time import perf_counter_ns as now

# Avoid deprecated or simulation imports
# DON'T USE: from gpu import SimulatedDevice  # Old simulation code
```

**Solution 3: Clean Build**
```bash
# Clean Mojo cache
rm -rf ~/.modular/mojo_cache

# Rebuild project
mojo build gpu_program.mojo
```

### Issue 6: Runtime Crashes

#### Symptoms
```bash
mojo run gpu_program.mojo
# Segmentation fault or unexpected termination
```

#### Diagnosis
```bash
# Run with debug information
mojo run --debug gpu_program.mojo

# Check system logs
dmesg | tail -20

# Check GPU error logs
nvidia-smi --query-gpu=ecc.errors.corrected.total,ecc.errors.uncorrected.total --format=csv,noheader
```

#### Solutions

**Solution 1: Add Error Handling**
```mojo
fn safe_gpu_operation() raises:
    """Safe GPU operation with comprehensive error handling."""
    try:
        var ctx = DeviceContext()
        
        try:
            var buffer = ctx.enqueue_create_buffer[DType.float64](10000)
            
            for i in range(10000):
                _ = buffer.enqueue_fill(Float64(i) * 0.001)
            
            ctx.synchronize()
            print("✓ GPU operation successful")
            
        except:
            print("⚠️  GPU buffer operation failed")
            # CPU fallback here
            
    except:
        print("❌ DeviceContext initialization failed")
        # Handle initialization failure
```

**Solution 2: Validate Input Parameters**
```mojo
fn validate_and_allocate(mut ctx: DeviceContext, size: Int) raises -> Bool:
    """Validate parameters before GPU allocation."""
    # Check size limits
    if size <= 0:
        print("❌ Invalid buffer size:", size)
        return False
    
    if size > 1000000:  # 1M elements max
        print("⚠️  Buffer size too large:", size)
        return False
    
    # Check available memory (simplified check)
    try:
        var test_buffer = ctx.enqueue_create_buffer[DType.float64](min(size, 1000))
        ctx.synchronize()
        return True
    except:
        print("❌ Memory allocation test failed")
        return False
```

**Solution 3: Gradual Testing**
```mojo
fn test_gpu_incrementally() raises:
    """Test GPU functionality incrementally."""
    print("1. Testing DeviceContext...")
    var ctx = DeviceContext()
    print("   ✓ DeviceContext OK")
    
    print("2. Testing small buffer...")
    var small_buffer = ctx.enqueue_create_buffer[DType.float64](100)
    ctx.synchronize()
    print("   ✓ Small buffer OK")
    
    print("3. Testing medium buffer...")
    var medium_buffer = ctx.enqueue_create_buffer[DType.float64](1000)
    ctx.synchronize()
    print("   ✓ Medium buffer OK")
    
    print("4. Testing operations...")
    for i in range(1000):
        _ = medium_buffer.enqueue_fill(Float64(i) * 0.001)
    ctx.synchronize()
    print("   ✓ Operations OK")
    
    print("✅ All GPU tests passed")
```

## Performance Optimization Tips

### 1. Buffer Size Optimization
```mojo
# Test different buffer sizes to find optimal performance
var test_sizes = List[Int]()
test_sizes.append(512)
test_sizes.append(1024)
test_sizes.append(2048)
test_sizes.append(4096)
test_sizes.append(8192)

var best_size = 1024
var best_performance = 0.0

for i in range(len(test_sizes)):
    var size = test_sizes[i]
    var performance = benchmark_buffer_size(size)
    if performance > best_performance:
        best_performance = performance
        best_size = size

print("Optimal buffer size:", best_size, "elements")
```

### 2. Batch Processing
```mojo
# Process large datasets in optimal batches
fn process_large_dataset(mut ctx: DeviceContext, total_size: Int) raises:
    var optimal_batch_size = 4096  # Determined from testing
    
    for batch_start in range(0, total_size, optimal_batch_size):
        var batch_end = min(batch_start + optimal_batch_size, total_size)
        var batch_size = batch_end - batch_start
        
        var batch_buffer = ctx.enqueue_create_buffer[DType.float64](batch_size)
        
        # Process batch
        for i in range(batch_size):
            var value = Float64(batch_start + i) * 0.001
            _ = batch_buffer.enqueue_fill(value)
        
        ctx.synchronize()
        print("Processed batch:", batch_start, "to", batch_end)
```

### 3. Memory Usage Monitoring
```mojo
fn monitor_gpu_memory() raises:
    """Monitor GPU memory usage during operations."""
    # This would require additional GPU monitoring APIs
    # For now, use external monitoring
    print("Monitor GPU memory with: nvidia-smi dmon -s m")
```

## Getting Help

### 1. Check Documentation
- [Real GPU Implementation Guide](REAL_GPU_IMPLEMENTATION_GUIDE.md)
- [Migration Guide](MIGRATION_GUIDE.md)
- [API Documentation](API_DOCUMENTATION.md)

### 2. Verify Installation
```bash
# Complete installation verification
bash diagnose_gpu.sh

# Run migration verification
mojo run verify_migration.mojo
```

### 3. Community Resources
- Mojo Documentation: https://docs.modular.com/mojo/
- MAX Engine Documentation: https://docs.modular.com/max/
- GPU Programming Guide: https://docs.modular.com/mojo/manual/gpu/

### 4. Report Issues
When reporting issues, include:
- Output of `bash diagnose_gpu.sh`
- Mojo and MAX Engine versions
- GPU hardware information
- Complete error messages
- Minimal reproduction code

---

**🔧 Troubleshooting Complete**

This guide covers the most common issues with real GPU implementation. For additional help, refer to the documentation links above or run the diagnostic scripts provided.
