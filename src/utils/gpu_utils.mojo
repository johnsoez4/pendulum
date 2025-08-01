"""
GPU utilities for pendulum project.

This module provides GPU detection, capability testing, and device management
utilities for the pendulum AI control system. It handles automatic GPU detection
with graceful CPU fallback and provides configuration options for compute mode selection.
"""

from collections import List
from memory import UnsafePointer
from time import perf_counter_ns as now

# Real MAX Engine imports for GPU operations (VERIFIED WORKING)
from sys import (
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    has_accelerator,
)
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

# Note: These are the working MAX Engine imports for GPU acceleration
# Current implementation uses DeviceContext and GPU kernels for real GPU operations
# Future optimization could use: from max.graph import ops (verified available)

# GPU availability determined at runtime using real MAX Engine API
# All GPU availability checks now use has_accelerator() for dynamic detection


@fieldwise_init
struct GPUDeviceInfo:
    """
    Structure to hold information about a specific GPU device.
    """

    var is_valid: Bool
    var name: String
    var memory_total_mb: Int
    var memory_free_mb: Int
    var compute_capability: String


struct GPUDetectionResult(Copyable):
    """Result of GPU detection with detailed information."""

    var gpu_available: Bool
    var gpu_type: String
    var device_count: Int
    var recommended_mode: String

    fn __init__(out self):
        """Initialize with default values indicating no GPU."""
        self.gpu_available = False
        self.gpu_type = "none"
        self.device_count = 0
        self.recommended_mode = "cpu"


struct GPUCapabilities(Copyable):
    """
    Structure to hold GPU capability information.
    """

    var gpu_available: Bool
    var device_count: Int
    var device_name: String
    var memory_total: Int
    var memory_free: Int
    var compute_capability: String
    var max_engine_available: Bool

    fn __init__(out self):
        """Initialize with default values indicating no GPU."""
        self.gpu_available = False
        self.device_count = 0
        self.device_name = "None"
        self.memory_total = 0
        self.memory_free = 0
        self.compute_capability = "None"
        # MAX Engine availability will be determined at runtime
        self.max_engine_available = False


struct ComputeMode:
    """
    Enumeration for compute mode selection.
    """

    alias AUTO = 0  # Automatic GPU detection with CPU fallback
    alias GPU_ONLY = 1  # Force GPU-only mode (fail if GPU unavailable)
    alias CPU_ONLY = 2  # Force CPU-only mode (for benchmarking)
    alias HYBRID = 3  # Use both GPU and CPU for different operations


struct GPUManager:
    """
    GPU device manager for the pendulum project.

    Features:
    - Automatic GPU detection and capability assessment
    - Graceful fallback to CPU when GPU is unavailable
    - Configuration options for compute mode selection
    - Device management and memory allocation
    - Memory management and optimization
    """

    var capabilities: GPUCapabilities
    var compute_mode: Int
    var device_initialized: Bool
    var fallback_to_cpu: Bool

    fn __init__(out self, compute_mode: Int = ComputeMode.AUTO) raises:
        """
        Initialize GPU manager with specified compute mode.

        Args:
            compute_mode: Compute mode selection (AUTO, GPU_ONLY, CPU_ONLY, HYBRID).
        """
        self.capabilities = GPUCapabilities()
        self.compute_mode = compute_mode
        self.device_initialized = False
        self.fallback_to_cpu = False

        # Initialize GPU capabilities
        self._detect_gpu_capabilities()

        # Initialize device based on compute mode
        self._initialize_compute_device()

    fn _detect_gpu_capabilities(mut self) raises:
        """Detect and assess GPU capabilities using real MAX Engine API."""
        print("Detecting GPU capabilities...")

        # Use real MAX Engine GPU detection API
        if has_accelerator():
            # Check specific GPU types for detailed information
            has_nvidia = has_nvidia_gpu_accelerator()
            has_amd = has_amd_gpu_accelerator()

            if has_nvidia:
                print("✓ NVIDIA GPU detected and available for acceleration")
            elif has_amd:
                print("✓ AMD GPU detected and available for acceleration")
            else:
                print("✓ GPU accelerator detected and available")

            self.capabilities.gpu_available = True
            # Get detailed GPU information
            self.capabilities.max_engine_available = self._try_gpu_detection()
        else:
            print("⚠️  No GPU accelerator detected - using CPU fallback")
            self.capabilities.gpu_available = False
            self.capabilities.max_engine_available = False

    fn _try_gpu_detection(mut self) raises -> Bool:
        """
        Attempt to detect GPU devices using MAX engine.

        Returns:
            True if GPU is detected and usable, False otherwise
        """
        # Skip GPU detection if CPU_ONLY mode is requested
        if self.compute_mode == ComputeMode.CPU_ONLY:
            return False

        # Real MAX Engine GPU detection implementation
        # This implements the actual pattern for MAX Engine integration
        # Ready for real MAX Engine APIs when available

        print("Real MAX Engine: GPU device detection starting...")

        # Get information about the primary GPU
        gpu_info = self._query_gpu_device()
        if gpu_info.is_valid:
            self.capabilities.device_count = 1
            self.capabilities.device_name = gpu_info.name
            self.capabilities.memory_total = gpu_info.memory_total_mb
            self.capabilities.memory_free = gpu_info.memory_free_mb
            self.capabilities.compute_capability = gpu_info.compute_capability

            print("Real GPU detected:", gpu_info.name)
            print(
                "Memory:",
                gpu_info.memory_total_mb,
                "MB total,",
                gpu_info.memory_free_mb,
                "MB free",
            )
            print(
                "Compute capability:",
                gpu_info.compute_capability,
            )
            return True
        else:
            print("Real MAX Engine: Failed to get GPU device information")
            return False

    fn _query_gpu_device(self) -> GPUDeviceInfo:
        """
        Query GPU device information for the primary GPU.

        Returns:
            GPUDeviceInfo with device information
        """
        # GPU device query implementation
        # This function interfaces with GPU hardware detection

        # Attempt to detect GPU hardware (any accelerator)
        gpu_detected = has_accelerator()

        if gpu_detected:
            # Get GPU properties
            gpu_name = self._get_gpu_name()
            memory_info = self._get_gpu_memory_info()
            compute_cap = self._get_compute_capability()

            return GPUDeviceInfo(
                is_valid=True,
                name=gpu_name,
                memory_total_mb=memory_info[0],
                memory_free_mb=memory_info[1],
                compute_capability=compute_cap,
            )
        else:
            return GPUDeviceInfo(
                is_valid=False,
                name="No GPU Detected",
                memory_total_mb=0,
                memory_free_mb=0,
                compute_capability="0.0",
            )

    fn _get_gpu_name(self) -> String:
        """
        Get the actual GPU device name by reading from hardware.

        Returns:
            GPU device name string read from actual GPU hardware
        """
        try:
            if has_nvidia_gpu_accelerator():
                # Read actual NVIDIA GPU name from hardware
                var ctx = DeviceContext(0, api="cuda")
                var gpu_name = ctx.name()
                return gpu_name

            elif has_amd_gpu_accelerator():
                # Read actual AMD GPU name from hardware
                var ctx = DeviceContext(0, api="hip")
                var gpu_name = ctx.name()
                return gpu_name

            else:
                # Generic GPU - try default DeviceContext
                _ = DeviceContext()  # Test if DeviceContext works
                return "Generic GPU"

        except Exception:
            # If hardware reading fails, return fallback based on detection
            if has_nvidia_gpu_accelerator():
                return "NVIDIA GPU (Detection Failed)"
            elif has_amd_gpu_accelerator():
                return "AMD GPU (Detection Failed)"
            else:
                return "Unknown GPU"

    fn _get_gpu_memory_info(self) -> (Int, Int):
        """
        Get real GPU memory information by reading from hardware.

        Returns:
            Tuple of (total_memory_mb, free_memory_mb) read from actual GPU hardware.
        """
        try:
            if has_nvidia_gpu_accelerator():
                # Read actual NVIDIA GPU memory from hardware
                var ctx = DeviceContext(0, api="cuda")
                var memory_info = ctx.get_memory_info()
                var free_memory_bytes = memory_info[0]
                var total_memory_bytes = memory_info[1]

                # Convert bytes to MB
                var total_memory_mb = Int(
                    Float64(total_memory_bytes) / (1024.0 * 1024.0)
                )
                var free_memory_mb = Int(
                    Float64(free_memory_bytes) / (1024.0 * 1024.0)
                )

                print(
                    "✓ Real GPU memory detected:",
                    total_memory_mb,
                    "MB total,",
                    free_memory_mb,
                    "MB free",
                )
                return (total_memory_mb, free_memory_mb)

            elif has_amd_gpu_accelerator():
                # Read actual AMD GPU memory from hardware
                var ctx = DeviceContext(0, api="hip")
                var memory_info = ctx.get_memory_info()
                var free_memory_bytes = memory_info[0]
                var total_memory_bytes = memory_info[1]

                # Convert bytes to MB
                var total_memory_mb = Int(
                    Float64(total_memory_bytes) / (1024.0 * 1024.0)
                )
                var free_memory_mb = Int(
                    Float64(free_memory_bytes) / (1024.0 * 1024.0)
                )

                print(
                    "✓ Real GPU memory detected:",
                    total_memory_mb,
                    "MB total,",
                    free_memory_mb,
                    "MB free",
                )
                return (total_memory_mb, free_memory_mb)

            else:
                # Generic GPU - try default DeviceContext
                var ctx = DeviceContext()
                var memory_info = ctx.get_memory_info()
                var free_memory_bytes = memory_info[0]
                var total_memory_bytes = memory_info[1]

                # Convert bytes to MB
                var total_memory_mb = Int(
                    Float64(total_memory_bytes) / (1024.0 * 1024.0)
                )
                var free_memory_mb = Int(
                    Float64(free_memory_bytes) / (1024.0 * 1024.0)
                )

                return (total_memory_mb, free_memory_mb)

        except e:
            print("⚠️  GPU memory query failed:", String(e))
            print("  - Hardware memory information unavailable")
            print("  - GPU device may not support memory queries")

            # Return zero values to indicate memory query failure
            # Calling code should handle this appropriately
            return (0, 0)

    fn _get_compute_capability(self) -> String:
        """
        Get GPU compute capability by reading actual values from GPU hardware.

        Returns:
            Compute capability string read from actual GPU hardware
        """
        try:
            if has_nvidia_gpu_accelerator():
                # Read actual NVIDIA GPU compute capability from hardware
                var ctx = DeviceContext(0, api="cuda")
                var compute_capability = ctx.compute_capability()

                # Format compute capability as "major.minor" (e.g., "8.6")
                var major = compute_capability // 10
                var minor = compute_capability % 10
                return String(major) + "." + String(minor)

            elif has_amd_gpu_accelerator():
                # Read actual AMD GPU information from hardware
                var ctx = DeviceContext(0, api="hip")
                var gpu_name = ctx.name()

                # Extract architecture from actual GPU name
                if "MI300" in gpu_name or "MI355" in gpu_name:
                    return "CDNA3"  # Latest AMD data center architecture
                elif "MI250" in gpu_name or "MI210" in gpu_name:
                    return "CDNA2"  # Previous generation data center
                elif "RX 7" in gpu_name or "RDNA3" in gpu_name:
                    return "RDNA3"  # Consumer RDNA3 architecture
                elif "RX 6" in gpu_name or "RDNA2" in gpu_name:
                    return "RDNA2"  # Consumer RDNA2 architecture
                else:
                    # Return generic identifier for unknown AMD GPUs
                    return "AMD-" + gpu_name[:10]  # First 10 chars of GPU name

            else:
                # Generic GPU - try to get basic information
                _ = DeviceContext()  # Test if DeviceContext works
                return "Generic"

        except Exception:
            # If hardware reading fails, return fallback based on detection
            if has_nvidia_gpu_accelerator():
                return "Unknown-NVIDIA"  # Conservative NVIDIA fallback
            elif has_amd_gpu_accelerator():
                return "Unknown-AMD"  # Conservative AMD fallback
            else:
                return "Unknown"

    fn _initialize_compute_device(mut self) raises:
        """Initialize compute device based on detected capabilities and mode."""
        print("Initializing compute device...")

        if self.compute_mode == ComputeMode.CPU_ONLY:
            print("Compute mode CPU_ONLY - GPU acceleration disabled")
            self.fallback_to_cpu = True
            self.device_initialized = True
            return

        if (
            self.compute_mode == ComputeMode.GPU_ONLY
            and not self.capabilities.gpu_available
        ):
            print("ERROR: GPU_ONLY mode requested but no GPU available")
            self.device_initialized = False
            return

        if self.capabilities.gpu_available:
            if self._initialize_gpu_device():
                print("GPU device initialized successfully")
                self.device_initialized = True
                self.fallback_to_cpu = False
            else:
                print("GPU initialization failed - falling back to CPU")
                self.fallback_to_cpu = True
                self.device_initialized = True
        else:
            print("No GPU available - using CPU")
            self.fallback_to_cpu = True
            self.device_initialized = True

    fn _initialize_gpu_device(mut self) raises -> Bool:
        """
        Initialize GPU device for computation using real MAX Engine DeviceContext.

        Returns:
            True if GPU initialization successful, False otherwise
        """
        print("Real GPU Device: Initialization starting...")

        try:
            # Create real DeviceContext for GPU operations
            device_context = DeviceContext()
            print("✓ Real DeviceContext created successfully")

            # Test GPU memory allocation to verify device functionality
            test_buffer_size = 1024  # 1KB test allocation
            test_buffer = device_context.enqueue_create_buffer[DType.float64](
                test_buffer_size
            )
            print(
                "✓ Real GPU memory allocation test successful (",
                test_buffer_size,
                "elements)",
            )

            # Test GPU memory operations
            _ = test_buffer.enqueue_fill(42.0)
            device_context.synchronize()
            print("✓ Real GPU memory operations verified")

            # Update capabilities with real device information
            self.capabilities.max_engine_available = True
            self.capabilities.gpu_available = True

            print("✓ Real GPU device initialization completed")
            print("  - DeviceContext: Active and ready")
            print("  - GPU memory: Allocated and tested")
            print("  - Compute context: Prepared for operations")
            print("  - Device ready for real MAX Engine operations")

            return True

        except e:
            print("⚠️  Real GPU device initialization failed:", String(e))
            print("  - DeviceContext creation failed")
            print("  - GPU hardware unavailable, using CPU processing")

            # Update capabilities to reflect GPU unavailability
            self.capabilities.max_engine_available = False
            self.capabilities.gpu_available = False
            self.fallback_to_cpu = True

            return False

    fn is_gpu_available(self) -> Bool:
        """Check if GPU is available and initialized."""
        return self.device_initialized and not self.fallback_to_cpu

    fn should_use_gpu(self) -> Bool:
        """Determine if GPU should be used for computation."""
        return (
            self.is_gpu_available()
            and self.compute_mode != ComputeMode.CPU_ONLY
        )

    fn allocate_gpu_buffer(
        self, size: Int, dtype: DType = DType.float64
    ) raises -> Bool:
        """
        Allocate GPU buffer using real DeviceContext operations.

        Args:
            size: Number of elements to allocate.
            dtype: Data type for the buffer.

        Returns:
            True if allocation successful, False otherwise.
        """
        if not self.is_gpu_available():
            print("GPU not available for buffer allocation")
            return False

        try:
            device_context = DeviceContext()

            # Create buffer with appropriate data type
            if dtype == DType.float64:
                _ = device_context.enqueue_create_buffer[DType.float64](size)
                print(
                    "✓ Real GPU buffer allocated:", size, "elements of float64"
                )
            elif dtype == DType.float32:
                _ = device_context.enqueue_create_buffer[DType.float32](size)
                print(
                    "✓ Real GPU buffer allocated:", size, "elements of float32"
                )
            else:
                print("Unsupported data type for GPU buffer allocation")
                return False

            device_context.synchronize()
            return True

        except e:
            print(
                "⚠️  GPU buffer allocation failed, size:",
                size,
                "- Error:",
                String(e),
            )
            return False


fn detect_gpu_hardware(context: String = "general") -> GPUDetectionResult:
    """
    Centralized GPU detection with context-specific messaging.

    Args:
        context: Context for detection (e.g., "neural_network", "matrix_ops", "validation").

    Returns:
        GPUDetectionResult with detection details.
    """
    result = GPUDetectionResult()

    if has_accelerator():
        # Check specific GPU types for detailed information
        has_nvidia = has_nvidia_gpu_accelerator()
        has_amd = has_amd_gpu_accelerator()

        if has_nvidia:
            print("✓ NVIDIA GPU detected for", context)
            result.gpu_type = "nvidia"
        elif has_amd:
            print("✓ AMD GPU detected for", context)
            result.gpu_type = "amd"
        else:
            print("✓ GPU accelerator detected for", context)
            result.gpu_type = "unknown"

        result.gpu_available = True
        result.recommended_mode = "gpu"
        # For now, assume 1 device per GPU type detected
        # In a full implementation, this would enumerate all available devices
        result.device_count = 1
    else:
        print("⚠️  No GPU detected for", context, "- using CPU fallback")
        result.gpu_available = False
        result.gpu_type = "none"
        result.recommended_mode = "cpu"
        result.device_count = 0

    return result
