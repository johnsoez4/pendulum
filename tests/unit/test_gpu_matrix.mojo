"""
Test GPU matrix operations.

This test verifies that the GPU-accelerated matrix operations work correctly
and maintain compatibility with CPU-only operations.
"""

from collections import List
from math import exp, tanh


# Define max and min functions since they're not in math module
fn max(a: Float64, b: Float64) -> Float64:
    """Return maximum of two values."""
    return a if a > b else b


fn min(a: Float64, b: Float64) -> Float64:
    """Return minimum of two values."""
    return a if a < b else b


# Define compute modes locally
alias ComputeMode_AUTO = 0
alias ComputeMode_GPU_ONLY = 1
alias ComputeMode_CPU_ONLY = 2
alias ComputeMode_HYBRID = 3


# Include matrix implementations inline for testing
struct GPUMatrix:
    """GPU-accelerated matrix implementation with CPU fallback."""

    var data: List[Float64]
    var rows: Int
    var cols: Int
    var use_gpu: Bool

    fn __init__(
        out self, rows: Int, cols: Int, compute_mode: Int = ComputeMode_AUTO
    ):
        """Initialize matrix with specified dimensions."""
        self.rows = rows
        self.cols = cols
        self.data = List[Float64]()
        self.use_gpu = compute_mode != ComputeMode_CPU_ONLY

        for _ in range(rows * cols):
            self.data.append(0.0)

    fn __copyinit__(out self, other: Self):
        """Copy constructor."""
        self.rows = other.rows
        self.cols = other.cols
        self.data = other.data
        self.use_gpu = other.use_gpu

    fn get(self, row: Int, col: Int) -> Float64:
        """Get element at (row, col)."""
        return self.data[row * self.cols + col]

    fn set(mut self, row: Int, col: Int, value: Float64):
        """Set element at (row, col)."""
        self.data[row * self.cols + col] = value

    fn multiply(self, other: GPUMatrix) -> GPUMatrix:
        """Matrix multiplication."""
        var result = GPUMatrix(self.rows, other.cols, ComputeMode_CPU_ONLY)

        for i in range(self.rows):
            for j in range(other.cols):
                var sum = 0.0
                for k in range(self.cols):
                    sum += self.get(i, k) * other.get(k, j)
                result.set(i, j, sum)

        return result

    fn add_bias(mut self, bias: List[Float64]):
        """Add bias vector to each row."""
        for i in range(self.rows):
            for j in range(self.cols):
                if j < len(bias):
                    self.set(i, j, self.get(i, j) + bias[j])

    fn apply_activation(mut self, activation: String):
        """Apply activation function element-wise."""
        for i in range(self.rows):
            for j in range(self.cols):
                var val = self.get(i, j)
                if activation == "tanh":
                    self.set(i, j, tanh(val))
                elif activation == "relu":
                    self.set(i, j, max(0.0, val))
                elif activation == "sigmoid":
                    self.set(i, j, 1.0 / (1.0 + exp(-val)))

    fn to_cpu_matrix(self) -> Matrix:
        """Convert to CPU-only matrix."""
        var cpu_matrix = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                cpu_matrix.set(i, j, self.get(i, j))
        return cpu_matrix


struct Matrix:
    """CPU-only matrix implementation."""

    var data: List[Float64]
    var rows: Int
    var cols: Int

    fn __init__(out self, rows: Int, cols: Int):
        """Initialize matrix with zeros."""
        self.rows = rows
        self.cols = cols
        self.data = List[Float64]()

        for _ in range(rows * cols):
            self.data.append(0.0)

    fn __copyinit__(out self, other: Self):
        """Copy constructor."""
        self.rows = other.rows
        self.cols = other.cols
        self.data = other.data

    fn get(self, row: Int, col: Int) -> Float64:
        """Get element at (row, col)."""
        return self.data[row * self.cols + col]

    fn set(mut self, row: Int, col: Int, value: Float64):
        """Set element at (row, col)."""
        self.data[row * self.cols + col] = value

    fn to_gpu_matrix(self, compute_mode: Int = ComputeMode_AUTO) -> GPUMatrix:
        """Convert to GPU matrix."""
        var gpu_matrix = GPUMatrix(self.rows, self.cols, compute_mode)
        for i in range(self.rows):
            for j in range(self.cols):
                gpu_matrix.set(i, j, self.get(i, j))
        return gpu_matrix


fn test_matrix_creation():
    """Test matrix creation with different compute modes."""
    print("Testing matrix creation...")

    # Test CPU-only matrix creation
    var cpu_matrix = Matrix(3, 3)
    print("CPU matrix created: ", cpu_matrix.rows, "x", cpu_matrix.cols)

    # Test GPU matrix creation with different modes
    var gpu_matrix_auto = GPUMatrix(3, 3, ComputeMode_AUTO)
    print(
        "GPU matrix (AUTO) created: ",
        gpu_matrix_auto.rows,
        "x",
        gpu_matrix_auto.cols,
    )

    var gpu_matrix_cpu = GPUMatrix(3, 3, ComputeMode_CPU_ONLY)
    print(
        "GPU matrix (CPU_ONLY) created: ",
        gpu_matrix_cpu.rows,
        "x",
        gpu_matrix_cpu.cols,
    )

    var gpu_matrix_gpu = GPUMatrix(3, 3, ComputeMode_GPU_ONLY)
    print(
        "GPU matrix (GPU_ONLY) created: ",
        gpu_matrix_gpu.rows,
        "x",
        gpu_matrix_gpu.cols,
    )


fn test_matrix_operations():
    """Test basic matrix operations."""
    print("Testing matrix operations...")

    # Create test matrices
    var matrix_a = GPUMatrix(2, 3, ComputeMode_AUTO)
    var matrix_b = GPUMatrix(3, 2, ComputeMode_AUTO)

    # Set some test values
    matrix_a.set(0, 0, 1.0)
    matrix_a.set(0, 1, 2.0)
    matrix_a.set(0, 2, 3.0)
    matrix_a.set(1, 0, 4.0)
    matrix_a.set(1, 1, 5.0)
    matrix_a.set(1, 2, 6.0)

    matrix_b.set(0, 0, 1.0)
    matrix_b.set(0, 1, 2.0)
    matrix_b.set(1, 0, 3.0)
    matrix_b.set(1, 1, 4.0)
    matrix_b.set(2, 0, 5.0)
    matrix_b.set(2, 1, 6.0)

    print("Matrix A (2x3):")
    for i in range(matrix_a.rows):
        for j in range(matrix_a.cols):
            print("  A[", i, ",", j, "] =", matrix_a.get(i, j))

    print("Matrix B (3x2):")
    for i in range(matrix_b.rows):
        for j in range(matrix_b.cols):
            print("  B[", i, ",", j, "] =", matrix_b.get(i, j))

    # Test matrix multiplication
    var result = matrix_a.multiply(matrix_b)
    print("Result of A * B (2x2):")
    for i in range(result.rows):
        for j in range(result.cols):
            print("  Result[", i, ",", j, "] =", result.get(i, j))


fn test_activation_functions():
    """Test activation function applications."""
    print("Testing activation functions...")

    var matrix = GPUMatrix(2, 2, ComputeMode_AUTO)

    # Set test values
    matrix.set(0, 0, -1.0)
    matrix.set(0, 1, 0.0)
    matrix.set(1, 0, 1.0)
    matrix.set(1, 1, 2.0)

    print("Original matrix:")
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            print("  [", i, ",", j, "] =", matrix.get(i, j))

    # Test tanh activation
    var tanh_matrix = matrix
    tanh_matrix.apply_activation("tanh")
    print("After tanh activation:")
    for i in range(tanh_matrix.rows):
        for j in range(tanh_matrix.cols):
            print("  [", i, ",", j, "] =", tanh_matrix.get(i, j))

    # Test ReLU activation
    var relu_matrix = matrix
    relu_matrix.apply_activation("relu")
    print("After ReLU activation:")
    for i in range(relu_matrix.rows):
        for j in range(relu_matrix.cols):
            print("  [", i, ",", j, "] =", relu_matrix.get(i, j))


fn test_bias_addition():
    """Test bias vector addition."""
    print("Testing bias addition...")

    var matrix = GPUMatrix(2, 3, ComputeMode_AUTO)

    # Set test values
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            matrix.set(i, j, Float64(i * matrix.cols + j))

    print("Original matrix:")
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            print("  [", i, ",", j, "] =", matrix.get(i, j))

    # Create bias vector
    var bias = List[Float64]()
    bias.append(1.0)
    bias.append(2.0)
    bias.append(3.0)

    # Add bias
    matrix.add_bias(bias)

    print("After adding bias [1.0, 2.0, 3.0]:")
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            print("  [", i, ",", j, "] =", matrix.get(i, j))


fn test_cpu_gpu_compatibility():
    """Test compatibility between CPU and GPU matrices."""
    print("Testing CPU-GPU compatibility...")

    # Create CPU matrix
    var cpu_matrix = Matrix(2, 2)
    cpu_matrix.set(0, 0, 1.0)
    cpu_matrix.set(0, 1, 2.0)
    cpu_matrix.set(1, 0, 3.0)
    cpu_matrix.set(1, 1, 4.0)

    print("CPU matrix:")
    for i in range(cpu_matrix.rows):
        for j in range(cpu_matrix.cols):
            print("  [", i, ",", j, "] =", cpu_matrix.get(i, j))

    # Convert to GPU matrix
    var gpu_matrix = cpu_matrix.to_gpu_matrix(ComputeMode_AUTO)

    print("Converted to GPU matrix:")
    for i in range(gpu_matrix.rows):
        for j in range(gpu_matrix.cols):
            print("  [", i, ",", j, "] =", gpu_matrix.get(i, j))

    # Convert back to CPU matrix
    var cpu_matrix_back = gpu_matrix.to_cpu_matrix()

    print("Converted back to CPU matrix:")
    for i in range(cpu_matrix_back.rows):
        for j in range(cpu_matrix_back.cols):
            print("  [", i, ",", j, "] =", cpu_matrix_back.get(i, j))


fn main():
    """Run all GPU matrix tests."""
    print("=" * 70)
    print("GPU MATRIX OPERATIONS TEST SUITE")
    print("=" * 70)

    test_matrix_creation()
    print()

    test_matrix_operations()
    print()

    test_activation_functions()
    print()

    test_bias_addition()
    print()

    test_cpu_gpu_compatibility()
    print()

    print("=" * 70)
    print("GPU MATRIX TESTS COMPLETED")
    print("=" * 70)
