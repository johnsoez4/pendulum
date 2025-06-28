# Mojo Syntax Reference & Coding Standards

This file serves as the centralized guide for Mojo language best practices and syntax standards. All Mojo code creation and modification should reference this file to ensure consistent, idiomatic code.

## 📋 Table of Contents

1. [Import Patterns & Organization](#import-patterns--organization)
2. [Function Definitions & Signatures](#function-definitions--signatures)
3. [Struct Definitions & Methods](#struct-definitions--methods)
4. [Error Handling Patterns](#error-handling-patterns)
5. [Variable Declarations](#variable-declarations)
6. [Memory Management](#memory-management)
7. [External Function Calls (FFI)](#external-function-calls-ffi)
8. [Naming Conventions](#naming-conventions)
8. [Documentation Standards](#documentation-standards)
9. [Code Formatting](#code-formatting)
10. [Testing Patterns](#testing-patterns)
11. [Common Patterns & Idioms](#common-patterns--idioms)
12. [Compliance Checklist](#compliance-checklist)

---

## 🔗 Import Patterns & Organization

### ✅ **Preferred Import Patterns**

```mojo
# Standard library imports first
from sys.ffi import external_call
from memory import UnsafePointer
from collections import Dict
from testing import assert_equal, assert_true, assert_false
from time import sleep, perf_counter_ns as now

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
- **Last Updated**: 2025-06-12
- **Next Review**: 2025-09-12 (Quarterly with memory system)
- **Version**: 1.0.0

### **Future Enhancements**

Planned additions to this reference:

- [ ] **Performance patterns** for threading operations
- [ ] **Advanced FFI patterns** for complex C integration
- [ ] **Concurrency patterns** specific to Mojo
- [ ] **Debugging techniques** for Mojo threading code
- [ ] **Integration patterns** with other Mojo libraries

---

*This file is maintained alongside the memory system and updated with each significant Mojo development. All Mojo code in this project should reference these standards for consistency and quality.*
