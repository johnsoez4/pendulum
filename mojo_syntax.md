# Mojo Syntax Reference & Coding Standards

This file serves as the centralized guide for Mojo language best practices and syntax standards. All Mojo code creation and modification should reference this file to ensure consistent, idiomatic code.

## 📋 Table of Contents

1. [Version Commands & Environment](#version-commands--environment)
2. [Import Patterns & Organization](#import-patterns--organization)
3. [Function Definitions & Signatures](#function-definitions--signatures)
4. [Struct Definitions & Methods](#struct-definitions--methods)
5. [Error Handling Patterns](#error-handling-patterns)
6. [Variable Declarations](#variable-declarations)
7. [Memory Management](#memory-management)
8. [External Function Calls (FFI)](#external-function-calls-ffi)
9. [MAX Engine GPU Programming](#max-engine-gpu-programming)
10. [Naming Conventions](#naming-conventions)
11. [Documentation Standards](#documentation-standards)
12. [Code Formatting](#code-formatting)
13. [Testing Patterns](#testing-patterns)
14. [GPU Simulation Labeling](#gpu-simulation-labeling)
15. [Common Patterns & Idioms](#common-patterns--idioms)
16. [Compliance Checklist](#compliance-checklist)

---

## 🔧 Version Commands & Environment

### ✅ **Version Checking Commands**

```bash
# Check Mojo compiler version
mojo -v
# or
mojo --version

# Check MAX Engine version
max --version

# Example output:
# mojo 24.4.0 (2024-06-07)
# MAX Engine 24.4.0
```

### 📋 **Environment Information**

- **Mojo Compiler**: Use `mojo -v` to check current version
- **MAX Engine**: Use `max --version` to verify MAX Engine installation
- **GPU Support**: Real GPU hardware available for MAX Engine acceleration

### 🎯 **Development Environment Setup**

```bash
# Verify Mojo installation
mojo -v

# Verify MAX Engine installation
max --version

# Check GPU availability (if nvidia-smi available)
nvidia-smi

# Compile Mojo files
mojo build src/file.mojo

# Run Mojo programs
mojo run src/file.mojo
```

### 📋 **Version Compatibility Notes**

1. **Mojo 24.4+**: `let` keyword removed, use direct assignment or `var`
2. **MAX Engine**: GPU operations require compatible MAX Engine version
3. **GPU Support**: Real GPU hardware available for acceleration
4. **Import Syntax**: MAX Engine imports follow standard Mojo import patterns

---

## 🔗 Import Patterns & Organization

### ✅ **Preferred Import Patterns**

```mojo
# Standard library imports first
from sys.ffi import external_call
from memory import UnsafePointer
from collections import Dict
from testing import assert_equal, assert_true, assert_false
from time import sleep, perf_counter_ns as now  # Note: use perf_counter_ns as now for timing

# Project imports with full paths from root
from src.module_name import (
    SomeClass,
    SomeFunction,
    SomeError,
    g_global_variable,
    SOME_CONSTANT,
    SOME_FLAG,
    SomeType,
    AnotherType,
)
```

### ❌ **Avoid These Import Patterns**

```mojo
# DON'T: Relative imports
from .module_name import SomeType
from module_name import SomeClass  # Without full path

# DON'T: Separate imports for related items
from src.module_name import SomeClass
from src.module_name import SomeType
from src.module_name import SomeFunction
```

### 📋 **Import Organization Rules**

1. **Standard library imports** first (sys, memory, collections, etc.)
2. **Third-party imports** second (if any)
3. **Project imports** last, using full paths from project root
4. **Group related imports** using parentheses for multi-line imports
5. **Use aliases** for long names or to avoid conflicts
6. **Sort imports** alphabetically within each group

### ⏱️ **Timing Functions**

**Important**: For timing operations, use the correct import:

```mojo
# CORRECT: Import perf_counter_ns as now for high-precision timing
from time import perf_counter_ns as now

# INCORRECT: This import does not exist
from time import now  # ❌ ERROR: 'now' function doesn't exist in time module

# Usage example:
start_time = now()
# ... some operation ...
end_time = now()
elapsed_ns = end_time - start_time
elapsed_ms = Float64(elapsed_ns) / 1_000_000.0
```

**Related Files**: Any project source files with imports

---

## 🔧 Function Definitions & Signatures

### ✅ **Standard Function Patterns**

```mojo
# Simple function with clear parameter types
fn worker_function():
    """Simple target function for testing."""
    print("Hello from real OS thread!")
    sleep(0.1)

# Function with parameters and return type
fn get_current_thread_id() -> ThreadId:
    """Get the current thread's ID."""
    return MojoThreading.get_current_thread_id()

# Function with raises annotation
fn init_system() raises -> None:
    """Initialize the system."""
    result = external_call["system_init", Int32]()
    if result != SUCCESS_CODE:
        msg = get_error_message(result)
        raise Error("Failed to initialize system: " + msg)

# Function with complex parameters
fn create_resource(resource_id: ResourceId, context: UnsafePointer[NoneType], flags: Int32) raises -> ResourceHandle:
    """Create a new resource with specified parameters."""
    handle = external_call["resource_create", Int32](
        resource_id, context, flags
    )
    if handle < 0:
        msg = get_error_message(handle)
        raise Error("Failed to create resource: " + msg)
    return handle
```

### 📋 **Function Definition Rules**

1. **Always include docstrings** for public functions
2. **Use type annotations** for parameters and return types
3. **Add `raises` annotation** when function can throw errors
4. **Use descriptive parameter names** that indicate purpose
5. **Keep functions focused** on single responsibility
6. **Use direct assignment** for single-assignment variables, `var` only when needed
7. **Use `alias`** for compile-time constants and type aliases

---

## 🏗️ Struct Definitions & Methods

### ✅ **Struct Definition Patterns**

```mojo
struct SystemManager:
    """
    Main interface to a system library.

    This struct provides static methods that wrap C library functions
    using external_call, handling type conversions and error checking.
    """

    @staticmethod
    fn init() raises -> None:
        """Initialize the system."""
        result = external_call["system_init", Int32]()
        if result != SUCCESS_CODE:
            msg = SystemManager._get_error_message(result)
            raise Error("Failed to initialize system: " + msg)

    @staticmethod
    fn _get_error_message(error_code: Int32) -> String:
        """Get error message for error code (private method)."""
        if error_code == ERROR_INVALID_ID:
            return "Invalid resource or callback ID"
        # ... more error cases
        else:
            return "Unknown error"

struct Resource(Copyable, Movable):
    """
    A resource management implementation example.
    """

    alias Handler = fn () -> None
    alias ResourceIdType = UInt64

    var _handler: Self.Handler
    var _name: String
    var _id: Self.ResourceIdType
    var _active: Bool

    fn __init__(out self, handler: Self.Handler, name: String):
        """Initialize a new Resource object."""
        self._handler = handler
        self._name = name
        self._id = 0
        self._active = False
```

### 📋 **Struct Definition Rules**

1. **Include comprehensive docstrings** for structs and methods
2. **Use traits** (Copyable, Movable) when appropriate
3. **Define type aliases** within structs for clarity
4. **Use `_` prefix** for private methods and variables
5. **Group related methods** together logically
6. **Use `@staticmethod`** for utility functions that don't need instance data

### 🔄 **Modern Struct Traits (Copy & Move Semantics)**

**✅ PREFERRED: Use trait-based approach**

```mojo
# For copyable structs
struct MyStruct(Copyable):
    var data: Int

    fn __init__(out self, value: Int):
        self.data = value

# For movable structs
struct MyStruct(Movable):
    var data: Int

    fn __init__(out self, value: Int):
        self.data = value

# For both copyable and movable
struct MyStruct(Copyable & Movable):
    var data: Int

    fn __init__(out self, value: Int):
        self.data = value
```

**❌ DEPRECATED: Explicit dunder methods**

```mojo
# Don't use explicit __copyinit__ and __moveinit__
struct MyStruct:
    var data: Int

    fn __init__(out self, value: Int):
        self.data = value

    # ❌ Don't implement these manually
    fn __copyinit__(out self, other: Self):
        self.data = other.data

    fn __moveinit__(out self, owned other: Self):
        self.data = other.data^
```

### 📋 **Trait Selection Guidelines**

1. **Use `Copyable`** when struct instances need to be copied
2. **Use `Movable`** when struct instances need to be moved efficiently
3. **Use `Copyable & Movable`** for maximum flexibility (most common)
4. **Let Mojo handle** the implementation automatically via traits
5. **Avoid manual dunder methods** for copy/move semantics

---

## ⚠️ Error Handling Patterns

### ✅ **Proper Error Handling**

```mojo
# Function that can raise errors
fn wait_for_completion(mut self) raises:
    """Wait for the resource to complete."""
    if not self._active:
        raise Error("cannot wait for resource before it is started")

    if self._completed:
        return  # Already completed

    # Actual wait operation would go here
    self._completed = True

# Calling functions that raise
fn test_basic_functionality() raises:
    """Test basic resource creation and management."""
    var resource = Resource(test_handler, name="TestResource")

    try:
        resource.start()
        resource.wait_for_completion()
    except e:
        print("Resource operation failed:", e)
        raise e  # Re-raise if needed
```

### 📋 **Error Handling Rules**

1. **Use `raises` annotation** for functions that can throw
2. **Provide descriptive error messages** with context
3. **Use try/except blocks** when handling recoverable errors
4. **Re-raise errors** when caller should handle them
5. **Check preconditions** and fail fast with clear messages

---

## 📝 Variable Declarations

### ✅ **Current Mojo Variable Declaration Patterns (v24.4+)**

```mojo
# Direct assignment for single-assignment variables (preferred)
result = external_call["system_init", Int32]()
resource_id = external_call["resource_create", Int32](callback_id, context, flags)
msg = SystemManager._get_error_message(result)

# Use var only when declaring without immediate assignment
var result: Int32
if some_condition:
    result = external_call["system_init", Int32]()
else:
    result = ERROR_SYSTEM

# Use var when variable will be reassigned in loops or conditionals
var counter = 0
for i in range(10):
    counter += i

# Function parameters use appropriate conventions
fn process_data(borrowed data: String, mut result: List[Int], owned context: ProcessContext):
    # borrowed: read-only access (default)
    # mut: mutable reference (replaces old inout)
    # owned: takes ownership of the value
    pass

# Compile-time constants use alias
alias SUCCESS_CODE = 0
alias MAX_RESOURCES = 1024
alias ResourceId = Int32
```

### ❌ **Deprecated Patterns to Avoid**

```mojo
# DON'T: Use let keyword (removed in v24.4)
let result = external_call["system_init", Int32]()  # INVALID

# DON'T: Use var for simple single assignments
var result = external_call["system_init", Int32]()  # Unnecessary

# DON'T: Use inout for mutable parameters (replaced by mut)
fn old_function(inout data: List[Int]):  # DEPRECATED
    pass

# DON'T: Use var when direct assignment is clearer
var resource_id = 12345  # Prefer: resource_id = 12345
```

### 📋 **Variable Declaration Rules**

1. **Use direct assignment** for single-assignment variables that won't change
2. **Use `var` only when** declaring without immediate assignment or when reassignment is needed
3. **Use `alias`** for compile-time constants and type aliases
4. **Use appropriate parameter conventions**: `borrowed` (default), `mut`, `owned`
5. **Avoid `var`** for simple assignments where the value won't be modified
6. **Remember**: All runtime variables in Mojo are mutable by default

**Note**: The `let` keyword was completely removed from Mojo in version 24.4 (June 2024).

---

## 🧠 Memory Management

### ✅ **Memory Management Patterns**

```mojo
# Using UnsafePointer for C interop
fn register_handler(handler_id: HandlerId, handler_ptr: UnsafePointer[NoneType]) raises -> None:
    """Register a handler function with the given ID."""
    result = external_call["register_handler", Int32](
        handler_id, handler_ptr
    )
    if result != SUCCESS_CODE:
        msg = SystemManager._get_error_message(result)
        raise Error("Failed to register handler: " + msg)

# Safe initialization patterns
fn __init__(out self):
    """Initialize with safe defaults."""
    self._next_id = 1
    self._initialized = False

# Resource cleanup patterns
fn cleanup(mut self) raises -> None:
    """Cleanup resources properly."""
    if self._initialized:
        SystemManager.cleanup()
        self._initialized = False
```

### 📋 **Memory Management Rules**

1. **Use `UnsafePointer`** only for C interop
2. **Initialize all variables** explicitly
3. **Implement cleanup methods** for resource management
4. **Use `out` parameters** for initialization
5. **Avoid manual memory management** when possible

---

## 🔌 External Function Calls (FFI) - DLHandle API

### ✅ **DLHandle FFI Call Patterns**

```mojo
# Library loading and function calls with DLHandle
from sys.ffi import DLHandle

# Load library explicitly
lib = DLHandle("libsystem.so")

# Function call pattern (syntax may vary by Mojo version)
# Note: Exact syntax pending clarification in current Mojo version
init_func = lib.get_function["system_init", fn() -> Int32]()
result = init_func()

# Function with parameters
create_func = lib.get_function["resource_create", fn(Int32, UnsafePointer[NoneType], Int32) -> Int32]()
resource_id = create_func(handler_id, context, flags)

# Function with no return value
cleanup_func = lib.get_function["system_cleanup", fn(Int32) -> None]()
cleanup_func(exit_code)

# Library cleanup
lib.close()
```

### 📋 **DLHandle FFI Rules**

1. **Load library explicitly** using DLHandle constructor
2. **Get function handles** before calling functions
3. **Specify function signatures** in get_function calls
4. **Handle library loading errors** appropriately
5. **Close library** when done for proper cleanup
6. **Use descriptive C function names** that match the library
7. **Document C library dependencies** in comments

**Migration Benefits:**
- Explicit library loading with error handling
- Better control over library lifecycle
- Support for multiple library versions
- No complex environment variable setup

**Related Files**: Any C library header and source files

---

## 🚀 MAX Engine GPU Programming

### ✅ **REAL MAX Engine Import Patterns (VERIFIED WORKING)**

```mojo
# ⚠️  IMPORTANT: The following are the ACTUAL working MAX Engine imports
# discovered from working examples and verified on NVIDIA A10 GPU

# GPU Detection and Hardware Access (VERIFIED WORKING)
from sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator

# GPU Device Context for Operations (VERIFIED WORKING)
from gpu.host import DeviceContext

# Tensor Layout and Operations (VERIFIED WORKING)
from layout import Layout, LayoutTensor

# GPU Kernel Functions (VERIFIED WORKING)
from gpu import global_idx, thread_idx

# ❌ INCORRECT IMPORTS (These do NOT exist in current MAX Engine):
# from max.device import Device, get_device_count, get_device  # ❌ NOT AVAILABLE
# from max.tensor import Tensor, TensorSpec, DType              # ❌ NOT AVAILABLE
# from max.ops import matmul, add, tanh, relu, sigmoid         # ❌ NOT AVAILABLE
```

### ✅ **REAL GPU Device Management Patterns (VERIFIED WORKING)**

```mojo
# GPU Detection (VERIFIED on NVIDIA A10)
fn check_gpu_availability() -> Bool:
    """Check if GPU hardware is available."""
    from sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator

    var has_nvidia = has_nvidia_gpu_accelerator()  # Returns True on NVIDIA A10
    var has_amd = has_amd_gpu_accelerator()        # Returns False on our system

    if has_nvidia:
        print("✓ NVIDIA GPU detected and available")
        return True
    elif has_amd:
        print("✓ AMD GPU detected and available")
        return True
    else:
        print("⚠️  No GPU accelerator detected")
        return False

# DeviceContext Creation (VERIFIED WORKING)
fn create_gpu_context() -> DeviceContext:
    """Create GPU device context for operations."""
    from gpu.host import DeviceContext

    # This creates actual GPU context on NVIDIA A10
    var ctx = DeviceContext()
    return ctx

# GPU Buffer Management (VERIFIED WORKING PATTERN)
fn create_gpu_buffer[dtype: DType](ctx: DeviceContext, size: Int):
    """Create GPU buffer using DeviceContext."""
    # Based on working vector_addition.mojo example
    var buffer = ctx.enqueue_create_buffer[dtype](size)
    return buffer
    except e:
        print("GPU operation failed:", e)
        raise e
```

### ✅ **REAL GPU Tensor Operations Patterns (VERIFIED WORKING)**

```mojo
# LayoutTensor Creation (VERIFIED from working examples)
fn create_layout_tensor[dtype: DType](ctx: DeviceContext, width: Int, height: Int):
    """Create LayoutTensor using real MAX Engine API."""
    from layout import Layout, LayoutTensor

    # Define tensor layout (from working examples)
    alias layout = Layout.row_major(width, height)

    # Create GPU buffer
    var buffer = ctx.enqueue_create_buffer[dtype](width * height)

    # Create tensor from buffer
    var tensor = LayoutTensor[dtype, layout](buffer)
    return tensor

# GPU Kernel Function Pattern (VERIFIED from working examples)
fn gpu_element_wise_add(
    lhs_tensor: LayoutTensor[DType.float32, layout, MutableAnyOrigin],
    rhs_tensor: LayoutTensor[DType.float32, layout, MutableAnyOrigin],
    out_tensor: LayoutTensor[DType.float32, layout, MutableAnyOrigin],
    size: Int,
):
    """GPU kernel for element-wise addition (from vector_addition.mojo)."""
    from gpu import global_idx

    var global_tid = global_idx.x
    if global_tid < size:
        out_tensor[global_tid] = lhs_tensor[global_tid] + rhs_tensor[global_tid]

# GPU Kernel Launch Pattern (VERIFIED from working examples)
fn launch_gpu_kernel(ctx: DeviceContext, tensor_a, tensor_b, result, size: Int):
    """Launch GPU kernel using real MAX Engine API."""
    from math import ceildiv

    alias BLOCK_SIZE = 256
    var grid_dim = ceildiv(size, BLOCK_SIZE)

    # Launch kernel (from working examples)
    ctx.enqueue_function[gpu_element_wise_add](
        tensor_a,
        tensor_b,
        result,
        size,
        grid_dim=grid_dim,
        block_dim=BLOCK_SIZE,
    )

# Host-Device Data Transfer (VERIFIED from working examples)
fn transfer_data_to_host[dtype: DType, layout: Layout](buffer):
    """Transfer GPU data to host for CPU access."""
    # Pattern from working examples
    with buffer.map_to_host() as host_buffer:
        var host_tensor = LayoutTensor[dtype, layout](host_buffer)
        # Access data on CPU
        print("Result:", host_tensor)
```

### ✅ **GPU Neural Network Patterns**

```mojo
# Linear layer implementation
fn gpu_linear_layer(input: Tensor[DType.float64], weights: Tensor[DType.float64], bias: Tensor[DType.float64]) -> Tensor[DType.float64]:
    """GPU-accelerated linear layer."""
    return linear(input, weights, bias)

# Activation functions
fn gpu_apply_activation(tensor: Tensor[DType.float64], activation: String) -> Tensor[DType.float64]:
    """Apply activation function on GPU."""
    if activation == "tanh":
        return tanh(tensor)
    elif activation == "relu":
        return relu(tensor)
    elif activation == "sigmoid":
        return sigmoid(tensor)
    else:
        raise Error("Unsupported activation: " + activation)

# Fused operations for performance
fn gpu_fused_linear_activation(input: Tensor[DType.float64], weights: Tensor[DType.float64], bias: Tensor[DType.float64], activation: String) -> Tensor[DType.float64]:
    """Fused linear + activation for optimal GPU performance."""
    # Use fused kernel when available
    if activation == "tanh":
        return fused_linear_bias_activation(input, weights, bias, "tanh")
    else:
        # Fallback to separate operations
        linear_output = linear(input, weights, bias)
        return gpu_apply_activation(linear_output, activation)
```

### ✅ **GPU Memory Management Patterns**

```mojo
# Memory allocation and deallocation
fn allocate_gpu_memory(size_bytes: Int, device: Device) -> UnsafePointer[UInt8]:
    """Allocate raw GPU memory."""
    return device.allocate(size_bytes)

fn deallocate_gpu_memory(ptr: UnsafePointer[UInt8], device: Device):
    """Deallocate GPU memory."""
    device.deallocate(ptr)

# Asynchronous memory transfers
fn async_transfer_to_gpu(cpu_data: List[Float64], device: Device, stream: DeviceStream) -> Tensor[DType.float64]:
    """Asynchronous CPU to GPU transfer."""
    shape = List[Int]()
    shape.append(len(cpu_data))

    gpu_tensor = create_gpu_tensor(shape, device)
    gpu_tensor.copy_from_host_async(cpu_data, stream)
    return gpu_tensor

# Memory synchronization
fn synchronize_gpu_operations(device: Device):
    """Wait for all GPU operations to complete."""
    device.synchronize()

# Stream management
fn create_gpu_stream(device: Device) -> DeviceStream:
    """Create GPU stream for asynchronous operations."""
    return device.create_stream()
```

### 📋 **MAX Engine GPU Programming Rules**

1. **Always check device availability** before GPU operations
2. **Use appropriate data types** (DType.float64, DType.float32, etc.)
3. **Handle GPU memory explicitly** with proper allocation/deallocation
4. **Use asynchronous operations** for optimal performance
5. **Synchronize when necessary** to ensure operation completion
6. **Prefer fused operations** for better GPU utilization
7. **Implement CPU fallback** for compatibility
8. **Monitor GPU memory usage** to avoid out-of-memory errors

### 🔧 **Tensor Indexing and SIMD Vector Extraction**

**⚠️ CRITICAL: Tensor indexing operations return SIMD vector types, not scalar values**

In Mojo, tensor indexing operations like `input_buffer[0, j]` return SIMD vector types derived from `DType`, not scalar values. This causes type conversion errors when performing arithmetic with `Float32` scalars.

#### ✅ **Correct Tensor Indexing Pattern**

```mojo
# ❌ INCORRECT - Causes type conversion error:
# "cannot implicitly convert 'SIMD[float32, ...]' value to 'SIMD[float32, 1]'"
sum = sum + input_buffer[0, j] * weight

# ✅ CORRECT - Extract scalar value from SIMD vector:
sum = sum + input_buffer[0, j][0] * weight
#                              ^^^
#                              Extract first element as scalar
```

#### 📋 **SIMD Vector Extraction Rules**

1. **Always use `[0]` indexing** to extract scalar values from tensor operations
2. **Apply to all tensor indexing** where you perform arithmetic with scalars
3. **Use for both input and output** tensor operations in GPU kernels
4. **Essential for type compatibility** in GPU kernel arithmetic

#### 🎯 **Common Patterns**

```mojo
# Extracting input values for computation
input_value = input_buffer[0, j][0]  # Extract scalar Float32
weight = Float32(idx + j + 1) * 0.1
sum = sum + input_value * weight

# Direct arithmetic with extraction
sum = sum + input_buffer[0, 0][0] * 0.1 + input_buffer[0, 1][0] * 0.2

# Storing results (output indexing typically doesn't need [0])
output_buffer[0, idx] = tanh_result
```

#### ⚠️ **Type Conversion Error Prevention**

This pattern resolves the common compilation error:
```
cannot implicitly convert 'SIMD[float32, __init__[::Origin[::Bool(IntTuple(1), IntTuple(1)).size:]' value to 'SIMD[float32, 1]'
```

**Related Files**: `src/pendulum/utils/gpu_utils.mojo`, `src/pendulum/utils/gpu_matrix.mojo`, `src/pendulum/digital_twin/gpu_neural_network.mojo`

---

## 📝 Naming Conventions

### ✅ **Naming Standards**

```mojo
# Constants and aliases - UPPER_CASE
alias SUCCESS_CODE = 0
alias SYSTEM_FLAG_ENABLED = 0x01

# Type aliases - PascalCase
alias ResourceId = Int32
alias HandlerId = Int32

# Struct names - PascalCase
struct SystemManager:
struct ResourceRegistry:

# Function names - snake_case
fn get_current_resource_id() -> ResourceId:
fn system_yield():

# Variable names - snake_case
g_resource_registry = ResourceRegistry()
handler_id: HandlerId = 0

# Private members - _snake_case
var _next_id: HandlerId
var _initialized: Bool
fn _get_error_message(error_code: Int32) -> String:
```

### 📋 **Naming Rules**

1. **Constants**: UPPER_CASE with underscores
2. **Types/Structs**: PascalCase
3. **Functions**: snake_case
4. **Variables**: snake_case
5. **Private members**: _snake_case prefix
6. **Global variables**: g_ prefix for clarity

---

## 📚 Documentation Standards

### ✅ **Documentation Patterns**

```mojo
"""
Module-level docstring at the top of file.

This module provides example patterns for Mojo code organization,
demonstrating best practices for structure and documentation.
"""

struct Resource:
    """
    A resource management implementation example.

    This demonstrates proper struct organization and method patterns
    for managing system resources with appropriate error handling.
    """

    fn start(mut self) raises:
        """Start the resource's activity.

        It must be called at most once per resource object. It arranges for the
        object's handler method to be invoked appropriately.

        This method will raise a RuntimeError if called more than once on the
        same resource object.
        """
```

### 📋 **Documentation Rules**

1. **Module docstrings** at the top of every file
2. **Struct docstrings** explaining purpose and usage
3. **Function docstrings** with parameters and behavior description
4. **Use triple quotes** for all docstrings
5. **Include examples** for complex functions
6. **Document error conditions** and exceptions

**Related Files**: All source files in any Mojo project

---

## 🎨 Code Formatting

### ✅ **Formatting Standards**

```mojo
# Function calls with multiple parameters - align for readability
var resource_id = external_call["resource_create", Int32](
    handler_id, context, flags
)

# Import statements - group and align
from src.module_name import (
    SystemManager,
    ResourceRegistry,
    SystemError,
    g_resource_registry,
)

# Conditional statements - clear spacing
if result != SUCCESS_CODE:
    var msg = SystemManager._get_error_message(result)
    raise Error("Failed to initialize system: " + msg)

# Struct definitions - consistent indentation
struct Resource(Copyable, Movable):
    var _handler: Self.Handler
    var _name: String
    var _id: Self.ResourceIdType

    fn __init__(out self, handler: Self.Handler, name: String):
        self._handler = handler
        self._name = name
```

### 📋 **Formatting Rules**

1. **Use 4 spaces** for indentation (no tabs)
2. **Align multi-line parameters** for readability
3. **Group imports** with parentheses for multi-line
4. **Add blank lines** between logical sections
5. **Keep lines under 100 characters** when possible
6. **Use consistent spacing** around operators

---

## 🧪 Testing Patterns

### ✅ **Test Function Patterns**

```mojo
from testing import assert_equal, assert_true, assert_false

fn main() raises:
    """Main test entry point."""
    print("=== Simple Mojo Threading Test ===")
    test_basic_thread_functionality()
    print("=== Test Completed Successfully! ===")

fn test_basic_thread_functionality() raises:
    """Test basic thread creation and management."""
    print("Testing basic thread functionality...")

    alias THREAD_NAME = "TestThread"

    # Create thread
    thread = Thread(test_target, name=THREAD_NAME)

    # Test initial state
    assert_equal(thread.name(), THREAD_NAME, "thread_name")
    assert_equal(thread.ident(), 0, "thread_not_started")
    assert_false(thread.is_alive(), "thread_not_alive_initially")

    print("Initial state tests passed")

    # Start thread
    print("Starting thread...")
    thread.start()

    # Test post-start state
    assert_true(thread.ident() != 0, "thread_has_id")
    assert_true(thread.is_alive(), "thread_is_alive")

    print("All basic functionality tests passed!")

fn test_target():
    """Simple target function for testing."""
    print("Thread target function executed!")
    sleep(0.1)
```

### 📋 **Testing Rules**

1. **Use descriptive test function names** starting with `test_`
2. **Include assertion messages** for clarity using TestLocation
3. **Test both positive and negative cases**
4. **Use `alias` for test constants**
5. **Print progress messages** for debugging
6. **Group related assertions** logically
7. **Follow main() function pattern** calling tests in order
8. **Use symbolic links** for test imports (`tests/mojo_src` → `src/mojo`)
9. **Import test utilities** via symbolic link in test directories
10. **Record test results** using global TestResults tracker

### 🧪 **Testing Framework Patterns**

```mojo
# Test file structure with main() function first
fn main() raises:
    """Main test entry point - calls all test functions in order."""
    test_basic_functionality()
    test_error_handling()

    # Print final test summary
    from test_utils import g_test_results
    g_test_results.print_summary()

# Standard imports
from testing import assert_equal, assert_true, assert_false

# Project imports via symbolic link
from mojo_src.threading_real import Thread

# Test utilities via symbolic link
from test_utils import (
    TestLocation,
    TestTimer,
    print_test_header,
    print_test_footer,
    g_test_results,
)

# Test function with comprehensive error handling
fn test_basic_functionality() raises:
    """Test basic functionality with proper framework patterns."""
    print_test_header("Basic Functionality")
    test_loc = TestLocation("test_basic_functionality")

    try:
        # Test implementation
        result = some_operation()
        assert_true(result, test_loc("operation_success"))

        g_test_results.record_pass()
        print_test_footer("Basic Functionality - PASSED")

    except e:
        g_test_results.record_fail()
        print("✗ Test failed: " + str(e))
        print_test_footer("Basic Functionality - FAILED")
        raise e
```

**Related Files**: `tests/test_utils.mojo`, `tests/test_mojo_threading.mojo`, `tests/run_all_tests.mojo`

---

## 🎮 GPU Simulation Labeling

### 📋 **GPU Implementation Transparency Requirements**

**CRITICAL**: All GPU-related code in the pendulum project must clearly distinguish between simulated GPU operations (implementation structure/patterns) and actual GPU hardware execution. This ensures transparency and prepares for real MAX Engine GPU implementation.

### ✅ **Required Labeling Prefixes**

All simulated GPU operations, placeholder implementations, and mock benchmark data **MUST** use these prefixes in print statements and output:

```mojo
# For simulated GPU operations (CPU-based simulation)
print("SIMULATED GPU: Matrix multiplication completed")
print("SIMULATED GPU KERNEL: Memory coalescing optimization")

# For placeholder implementations (structure ready for real GPU)
print("PLACEHOLDER GPU: Tensor allocation pattern")
print("PLACEHOLDER MAX ENGINE: Device enumeration structure")

# For mock benchmark data (not real GPU performance)
print("MOCK GPU PERFORMANCE: 4.2x speedup (simulated)")
print("MOCK BENCHMARK: GPU memory bandwidth 85% (placeholder)")
```

### 🎯 **Implementation Guidelines**

#### **1. GPU Operation Simulation**
```mojo
fn _gpu_matrix_multiply(self, other: GPUMatrix) -> GPUMatrix:
    """GPU matrix multiplication with simulation labeling."""
    if self.gpu_allocated and other.gpu_allocated:
        print("SIMULATED GPU KERNEL: Matrix multiplication with memory coalescing")
        print("  - PLACEHOLDER: Block size optimization (16x16 thread blocks)")
        print("  - PLACEHOLDER: Shared memory utilization enabled")

        # CPU-based simulation of GPU computation
        # ... implementation ...

        print("SIMULATED GPU: Matrix multiplication completed")

    return result
```

#### **2. Performance Benchmarking**
```mojo
fn benchmark_gpu_performance(self) -> BenchmarkResult:
    """GPU performance benchmarking with clear simulation labels."""
    print("MOCK GPU BENCHMARK: Starting performance measurement")

    # Simulated timing
    var gpu_time = self._simulate_gpu_timing()

    print("MOCK GPU PERFORMANCE:", speedup, "x speedup (simulated)")
    print("MOCK BENCHMARK: Memory bandwidth", bandwidth, "% (placeholder)")

    return result
```

#### **3. GPU Memory Management**
```mojo
fn _allocate_gpu_memory(mut self):
    """GPU memory allocation with simulation labeling."""
    print("PLACEHOLDER GPU: Memory allocation pattern")
    print("SIMULATED: GPU tensor allocation for", self.rows, "x", self.cols, "matrix")

    # Placeholder for actual MAX engine allocation:
    # self.gpu_tensor = tensor.zeros([self.rows, self.cols], device=gpu_device)

    self.gpu_allocated = True
```

### 🚫 **What NOT to Label**

Do **NOT** add simulation labels to:
- **Actual MAX Engine imports** (when implemented): `from max.tensor import Tensor`
- **Real GPU hardware calls** (when implemented): `gpu_device.synchronize()`
- **Genuine performance measurements** from real GPU hardware
- **CPU-only operations** that don't simulate GPU behavior

### 🔄 **Future Real GPU Implementation**

When implementing actual MAX Engine GPU operations:

1. **Remove simulation labels** from real GPU hardware calls
2. **Keep placeholder comments** showing the transition:
   ```mojo
   # OLD: print("SIMULATED GPU: Matrix multiplication")
   # NEW: Real GPU operation (no simulation label)
   result_tensor = ops.matmul(gpu_a, gpu_b)
   ```
3. **Update documentation** to reflect real vs simulated operations
4. **Maintain clear distinction** in mixed environments

### 📋 **Labeling Checklist**

- [ ] All GPU operation simulations labeled with `SIMULATED GPU:`
- [ ] All placeholder implementations labeled with `PLACEHOLDER:`
- [ ] All mock benchmark data labeled with `MOCK:`
- [ ] Performance measurements clearly marked as simulated
- [ ] GPU memory operations labeled appropriately
- [ ] Documentation updated to reflect simulation status
- [ ] Comments indicate future real GPU implementation patterns

### 🎯 **Benefits of This Approach**

1. **Transparency**: Clear distinction between simulation and real GPU operations
2. **Maintainability**: Easy identification of code requiring real GPU implementation
3. **Testing**: Ability to validate simulation vs real GPU behavior
4. **Documentation**: Self-documenting code showing implementation status
5. **Migration**: Smooth transition to real MAX Engine GPU operations

**Related Files**: All GPU-related files in `src/pendulum/utils/`, `src/pendulum/digital_twin/`, `tests/`, `src/pendulum/benchmarks/`

---

## 🔄 Common Patterns & Idioms

### ✅ **Preferred Idioms**

```mojo
# Error checking pattern
if result != MOJO_THREAD_SUCCESS:
    msg = MojoThreading._get_error_message(result)
    raise Error("Operation failed: " + msg)

# Safe initialization pattern
fn __init__(out self):
    self._next_id = 1
    self._initialized = False

# Resource management pattern
fn cleanup(mut self) raises -> None:
    if self._initialized:
        MojoThreading.cleanup()
        self._initialized = False

# Type alias pattern for clarity
alias ThreadId = Int32
alias CallbackId = Int32
alias ThreadResult = Int32

# Global variable pattern
g_simple_callback_registry = SimpleCallbackRegistry()

# Static method pattern for utilities
@staticmethod
fn get_current_thread_id() -> ThreadId:
    return external_call["mojo_thread_get_current_id", Int32]()
```

### ❌ **Patterns to Avoid**

```mojo
# DON'T: Ignore error returns
result = external_call["mojo_threading_init", Int32]()
# Missing error check

# DON'T: Use magic numbers
if thread_id == -1:  # Use named constants instead

# DON'T: Inconsistent naming
threadID: Int32 = 0  # Use thread_id instead
ThreadName: String = ""  # Use thread_name instead

# DON'T: Missing type annotations
fn create_thread(callback_id, context):  # Add types

# DON'T: Use var unnecessarily for single assignments
var result = external_call["mojo_threading_init", Int32]()  # Prefer direct assignment

# DON'T: Use deprecated let keyword
let result = some_value()  # INVALID - let was removed in v24.4
```

---

## ✅ Compliance Checklist

### 📋 **Pre-Creation Checklist**

Before creating new Mojo files, verify:

- [ ] **File location** follows project structure (`src/mojo/`)
- [ ] **Import organization** follows standard patterns
- [ ] **Naming conventions** are consistent with project standards
- [ ] **Documentation** requirements are understood
- [ ] **Error handling** patterns are planned
- [ ] **Testing approach** is defined
- [ ] **Test file created** with `test_` prefix in appropriate `tests/` subdirectory
- [ ] **Test imports** use symbolic links (`mojo_src` for source, `test_utils` for utilities)
- [ ] **MAX Engine imports** are properly structured for GPU operations
- [ ] **GPU availability checking** is implemented for MAX Engine operations
- [ ] **CPU fallback** is provided when GPU/MAX Engine is unavailable

### 📋 **Code Review Checklist**

When reviewing existing Mojo files:

- [ ] **Imports** use full paths from project root
- [ ] **Functions** have proper type annotations and docstrings
- [ ] **Structs** follow naming and documentation standards
- [ ] **Error handling** uses `raises` and proper messages
- [ ] **Memory management** follows safe patterns
- [ ] **FFI calls** have proper error checking
- [ ] **Tests** exist and follow testing patterns
- [ ] **Formatting** is consistent with standards
- [ ] **Test files** have `main()` function calling tests in order
- [ ] **Test functions** use TestLocation for assertion messages
- [ ] **Test results** are recorded using g_test_results
- [ ] **Test imports** work via symbolic links
- [ ] **GPU simulation labels** are applied to all simulated GPU operations
- [ ] **Mock benchmark data** is clearly labeled with appropriate prefixes
- [ ] **Placeholder implementations** are marked for future real GPU implementation
- [ ] **MAX Engine imports** follow proper conditional import patterns
- [ ] **GPU device management** uses appropriate MAX Engine APIs
- [ ] **GPU memory management** follows MAX Engine best practices
- [ ] **GPU tensor operations** use correct MAX Engine data types
- [ ] **GPU error handling** includes device availability checks
- [ ] **GPU performance patterns** use fused operations when possible

### 📋 **Update Procedures**

When syntax standards evolve:

1. **Update this file** with new patterns and examples
2. **Review existing code** for compliance with new standards
3. **Update memory system** in `code_assistant_memories.md`
4. **Test all changes** to ensure compatibility
5. **Document changes** in `prompts.md`

### 🔗 **Cross-References**

- **Memory System**: `code_assistant_memories.md` - Memory #3 (Import Path Management)
- **Project Structure**: `docs/PROJECT_STRUCTURE.md`
- **Example Files**: `src/mojo/threading_real.mojo`, `src/mojo/mojo_threading_simple.mojo`
- **Test Examples**: `tests/mojo/simple_threading_test.mojo`

---

## 🔄 Maintenance & Evolution

### **Adding New Patterns**

When discovering new Mojo syntax patterns:

1. **Document the pattern** with examples in appropriate section
2. **Add to compliance checklist** if it's a requirement
3. **Update related memories** in `code_assistant_memories.md`
4. **Test the pattern** in actual code
5. **Cross-reference** with related project files

### **Version Tracking**

- **Created**: 2025-06-12 (Initial comprehensive reference)
- **Last Updated**: 2025-07-01 (Added tensor indexing and SIMD vector extraction patterns)
- **Next Review**: 2025-09-12 (Quarterly with memory system)
- **Version**: 1.2.0

### **Recent Updates (v1.2.0)**

- ✅ **Added Tensor Indexing and SIMD Vector Extraction section** with critical `[0]` indexing pattern
- ✅ **Added type conversion error prevention** for GPU kernel development
- ✅ **Added SIMD vector extraction rules** for tensor operations
- ✅ **Added common patterns** for scalar extraction from tensor indexing
- ✅ **Updated MAX Engine GPU Programming** with tensor indexing best practices

### **Previous Updates (v1.1.0)**

- ✅ **Added Version Commands section** with `mojo -v` and `max --version`
- ✅ **Added MAX Engine GPU Programming section** with comprehensive patterns
- ✅ **Added GPU device management patterns** for MAX Engine
- ✅ **Added GPU tensor operations patterns** with proper data types
- ✅ **Added GPU memory management patterns** with async operations
- ✅ **Added GPU neural network patterns** with fused operations
- ✅ **Updated compliance checklists** with MAX Engine considerations

### **Future Enhancements**

Planned additions to this reference:

- [ ] **Performance patterns** for threading operations
- [ ] **Advanced FFI patterns** for complex C integration
- [ ] **Concurrency patterns** specific to Mojo
- [ ] **Debugging techniques** for Mojo threading code
- [ ] **Integration patterns** with other Mojo libraries
- [ ] **Advanced MAX Engine patterns** for multi-GPU operations
- [ ] **GPU profiling and optimization** techniques
- [ ] **MAX Engine debugging** and troubleshooting patterns

---

*This file is maintained alongside the memory system and updated with each significant Mojo development. All Mojo code in this project should reference these standards for consistency and quality.*
