"""
Neural network architecture for pendulum digital twin.

This module implements a physics-informed neural network for modeling
pendulum dynamics using Mojo with GPU acceleration capabilities using
MAX Engine DeviceContext API and automatic CPU fallback.
"""

from collections import List
from math import exp, tanh, sqrt
from random import random_float64
from memory import UnsafePointer
from sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator
from gpu.host import DeviceContext
from gpu import thread_idx, block_dim, block_idx
from layout import Layout, LayoutTensor

# Import project modules
from src.utils.physics import PendulumState, PendulumPhysics
from src.utils.physics_constraints import apply_physics_constraints

# Model configuration constants (from config/pendulum_config.mojo)
alias MODEL_INPUT_DIM = 4  # [la_pos, pend_vel, pend_pos, cmd_volts]
alias MODEL_OUTPUT_DIM = 3  # [next_la_pos, next_pend_vel, next_pend_pos]
alias MODEL_HIDDEN_LAYERS = 3  # number of hidden layers
alias MODEL_HIDDEN_SIZE = 128  # neurons per hidden layer
alias MODEL_LEARNING_RATE = 0.001  # training learning rate


# GPU kernel functions for matrix operations
fn gpu_matrix_multiply_kernel(
    output: UnsafePointer[
        Scalar[DType.float64]
    ],  # GPU memory managed by DeviceContext
    a: UnsafePointer[
        Scalar[DType.float64]
    ],  # GPU memory managed by DeviceContext
    b: UnsafePointer[
        Scalar[DType.float64]
    ],  # GPU memory managed by DeviceContext
    rows_a: Int,
    cols_a: Int,
    cols_b: Int,
):
    """
    GPU kernel for matrix multiplication with thread-level parallelism.

    Performs C = A * B where A is rows_a x cols_a and B is cols_a x cols_b.
    Each thread computes one element of the output matrix.

    Args:
        output: Output matrix buffer (rows_a x cols_b).
        a: First input matrix buffer (rows_a x cols_a).
        b: Second input matrix buffer (cols_a x cols_b).
        rows_a: Number of rows in matrix A.
        cols_a: Number of columns in A / rows in B.
        cols_b: Number of columns in matrix B.
    """
    row = thread_idx.y + block_idx.y * block_dim.y
    col = thread_idx.x + block_idx.x * block_dim.x

    if row < rows_a and col < cols_b:
        var sum = Scalar[DType.float64](0.0)
        for k in range(cols_a):
            sum += a[row * cols_a + k] * b[k * cols_b + col]
        output[row * cols_b + col] = sum


fn gpu_add_bias_kernel(
    matrix: UnsafePointer[
        Scalar[DType.float64]
    ],  # GPU memory managed by DeviceContext
    bias: UnsafePointer[
        Scalar[DType.float64]
    ],  # GPU memory managed by DeviceContext
    rows: Int,
    cols: Int,
):
    """
    GPU kernel for adding bias vector to each row of a matrix.

    Each thread processes one matrix element, adding the corresponding bias value.

    Args:
        matrix: Matrix buffer to modify in-place.
        bias: Bias vector buffer.
        rows: Number of rows in the matrix.
        cols: Number of columns in the matrix.
    """
    row = thread_idx.y + block_idx.y * block_dim.y
    col = thread_idx.x + block_idx.x * block_dim.x

    if row < rows and col < cols:
        matrix[row * cols + col] += bias[col]


fn gpu_apply_activation_kernel(
    matrix: UnsafePointer[
        Scalar[DType.float64]
    ],  # GPU memory managed by DeviceContext
    rows: Int,
    cols: Int,
    activation_type: Int,  # 0=tanh, 1=relu, 2=sigmoid, 3=linear
):
    """
    GPU kernel for applying activation functions element-wise.

    Each thread processes one matrix element with the specified activation function.

    Args:
        matrix: Matrix buffer to modify in-place.
        rows: Number of rows in the matrix.
        cols: Number of columns in the matrix.
        activation_type: Type of activation (0=tanh, 1=relu, 2=sigmoid, 3=linear).
    """
    row = thread_idx.y + block_idx.y * block_dim.y
    col = thread_idx.x + block_idx.x * block_dim.x

    if row < rows and col < cols:
        idx = row * cols + col
        val = matrix[idx]

        if activation_type == 0:  # tanh
            matrix[idx] = tanh(val)
        elif activation_type == 1:  # relu
            matrix[idx] = val if val > 0.0 else 0.0
        elif activation_type == 2:  # sigmoid
            matrix[idx] = 1.0 / (1.0 + exp(-val))
        # activation_type == 3 (linear) requires no modification


struct Matrix(Copyable, Movable):
    """
    GPU-accelerated matrix implementation for neural network operations.

    This struct provides high-performance matrix operations using MAX Engine
    DeviceContext API with automatic CPU fallback when GPU hardware is unavailable.
    Supports real GPU acceleration for matrix multiplication, bias addition, and
    activation functions while maintaining identical mathematical results.

    Features:
        - GPU-accelerated matrix operations with thread-level parallelism
        - Automatic CPU fallback for reliability and compatibility
        - Proper GPU memory management and synchronization
        - Identical results between GPU and CPU implementations
        - Performance optimization following MAX Engine best practices
    """

    var data: List[Float64]
    var rows: Int
    var cols: Int
    var gpu_available: Bool

    fn __init__(out self, rows: Int, cols: Int):
        """
        Initialize matrix with zeros and detect GPU availability.

        Args:
            rows: Number of rows in the matrix.
            cols: Number of columns in the matrix.
        """
        self.rows = rows
        self.cols = cols
        self.data = List[Float64]()

        # Detect GPU availability for hardware acceleration
        self.gpu_available = (
            has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
        )

        for _ in range(rows * cols):
            self.data.append(0.0)

    fn get(self, row: Int, col: Int) -> Float64:
        """
        Get element at (row, col).

        Args:
            row: Row index.
            col: Column index.

        Returns:
            Float64: Element value at the specified position.
        """
        return self.data[row * self.cols + col]

    fn set(mut self, row: Int, col: Int, value: Float64):
        """
        Set element at (row, col).

        Args:
            row: Row index.
            col: Column index.
            value: Value to set at the specified position.
        """
        self.data[row * self.cols + col] = value

    fn multiply(self, other: Matrix) -> Matrix:
        """
        GPU-accelerated matrix multiplication with automatic CPU fallback.

        Attempts GPU acceleration first using DeviceContext and GPU kernels,
        automatically falls back to CPU implementation if GPU operations fail.
        Ensures identical mathematical results between GPU and CPU paths.

        Args:
            other: Matrix to multiply with (self * other).

        Returns:
            Matrix: Result of matrix multiplication.
        """
        result = Matrix(self.rows, other.cols)

        # Try GPU acceleration first if available
        if self.gpu_available and other.gpu_available:
            try:
                if self._gpu_multiply(other, result):
                    return result
            except e:
                # GPU failed, fall back to CPU
                pass

        # CPU fallback implementation
        self._cpu_multiply(other, result)
        return result

    fn _gpu_multiply(self, other: Matrix, mut result: Matrix) raises -> Bool:
        """
        GPU matrix multiplication implementation using DeviceContext API.

        Args:
            other: Second matrix for multiplication.
            result: Output matrix to store results.

        Returns:
            Bool: True if GPU operation succeeded, False otherwise.

        Raises:
            Error: If GPU operations fail.
        """
        try:
            with DeviceContext() as ctx:
                # Create GPU buffers for matrices
                buffer_size_a = self.rows * self.cols
                buffer_size_b = other.rows * other.cols
                buffer_size_result = result.rows * result.cols

                # Allocate GPU memory
                buffer_a = ctx.enqueue_create_buffer[DType.float64](
                    buffer_size_a
                )
                buffer_b = ctx.enqueue_create_buffer[DType.float64](
                    buffer_size_b
                )
                buffer_result = ctx.enqueue_create_buffer[DType.float64](
                    buffer_size_result
                )

                # Transfer data to GPU
                with buffer_a.map_to_host() as a_host:
                    for i in range(buffer_size_a):
                        a_host[i] = self.data[i]

                with buffer_b.map_to_host() as b_host:
                    for i in range(buffer_size_b):
                        b_host[i] = other.data[i]

                # Configure GPU kernel launch parameters
                alias BLOCK_SIZE = 16
                blocks_x = (result.cols + BLOCK_SIZE - 1) // BLOCK_SIZE
                blocks_y = (result.rows + BLOCK_SIZE - 1) // BLOCK_SIZE

                # Launch GPU kernel
                ctx.enqueue_function[gpu_matrix_multiply_kernel](
                    buffer_result.unsafe_ptr(),
                    buffer_a.unsafe_ptr(),
                    buffer_b.unsafe_ptr(),
                    self.rows,
                    self.cols,
                    other.cols,
                    grid_dim=(blocks_x, blocks_y),
                    block_dim=(BLOCK_SIZE, BLOCK_SIZE),
                )

                # Synchronize GPU operations
                ctx.synchronize()

                # Transfer results back to CPU
                with buffer_result.map_to_host() as result_host:
                    for i in range(buffer_size_result):
                        result.data[i] = Float64(result_host[i])

                return True

        except _:
            return False

        return False  # Should not reach here, but required for all paths

    fn _cpu_multiply(self, other: Matrix, mut result: Matrix):
        """
        CPU matrix multiplication fallback implementation.

        Args:
            other: Second matrix for multiplication.
            result: Output matrix to store results.
        """
        for i in range(self.rows):
            for j in range(other.cols):
                var sum = 0.0  # Needs var - reassigned in loop
                for k in range(self.cols):
                    sum += self.get(i, k) * other.get(k, j)
                result.set(i, j, sum)

    fn add_bias(mut self, bias: List[Float64]):
        """
        GPU-accelerated bias addition with automatic CPU fallback.

        Attempts GPU acceleration first using DeviceContext and GPU kernels,
        automatically falls back to CPU implementation if GPU operations fail.
        Ensures identical mathematical results between GPU and CPU paths.

        Args:
            bias: Bias vector to add to each row of the matrix.
        """
        # Try GPU acceleration first if available
        if self.gpu_available:
            try:
                if self._gpu_add_bias(bias):
                    return
            except e:
                # GPU failed, fall back to CPU
                pass

        # CPU fallback implementation
        self._cpu_add_bias(bias)

    fn _gpu_add_bias(mut self, bias: List[Float64]) raises -> Bool:
        """
        GPU bias addition implementation using DeviceContext API.

        Args:
            bias: Bias vector to add to each row.

        Returns:
            Bool: True if GPU operation succeeded, False otherwise.

        Raises:
            Error: If GPU operations fail.
        """
        try:
            with DeviceContext() as ctx:
                # Create GPU buffers
                buffer_size_matrix = self.rows * self.cols
                buffer_size_bias = len(bias)

                # Allocate GPU memory
                buffer_matrix = ctx.enqueue_create_buffer[DType.float64](
                    buffer_size_matrix
                )
                buffer_bias = ctx.enqueue_create_buffer[DType.float64](
                    buffer_size_bias
                )

                # Transfer data to GPU
                with buffer_matrix.map_to_host() as matrix_host:
                    for i in range(buffer_size_matrix):
                        matrix_host[i] = self.data[i]

                with buffer_bias.map_to_host() as bias_host:
                    for i in range(buffer_size_bias):
                        bias_host[i] = bias[i]

                # Configure GPU kernel launch parameters
                alias BLOCK_SIZE = 16
                blocks_x = (self.cols + BLOCK_SIZE - 1) // BLOCK_SIZE
                blocks_y = (self.rows + BLOCK_SIZE - 1) // BLOCK_SIZE

                # Launch GPU kernel
                ctx.enqueue_function[gpu_add_bias_kernel](
                    buffer_matrix.unsafe_ptr(),
                    buffer_bias.unsafe_ptr(),
                    self.rows,
                    self.cols,
                    grid_dim=(blocks_x, blocks_y),
                    block_dim=(BLOCK_SIZE, BLOCK_SIZE),
                )

                # Synchronize GPU operations
                ctx.synchronize()

                # Transfer results back to CPU
                with buffer_matrix.map_to_host() as matrix_host:
                    for i in range(buffer_size_matrix):
                        self.data[i] = Float64(matrix_host[i])

                return True

        except _:
            return False

        return False  # Should not reach here, but required for all paths

    fn _cpu_add_bias(mut self, bias: List[Float64]):
        """
        CPU bias addition fallback implementation.

        Args:
            bias: Bias vector to add to each row.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                if j < len(bias):
                    self.set(i, j, self.get(i, j) + bias[j])

    fn apply_activation(mut self, activation: String):
        """
        GPU-accelerated activation function application with automatic CPU fallback.

        Attempts GPU acceleration first using DeviceContext and GPU kernels,
        automatically falls back to CPU implementation if GPU operations fail.
        Ensures identical mathematical results between GPU and CPU paths.

        Args:
            activation: Activation function name ("tanh", "relu", "sigmoid", or "linear").
        """
        # Try GPU acceleration first if available
        if self.gpu_available:
            try:
                if self._gpu_apply_activation(activation):
                    return
            except e:
                # GPU failed, fall back to CPU
                pass

        # CPU fallback implementation
        self._cpu_apply_activation(activation)

    fn _gpu_apply_activation(mut self, activation: String) raises -> Bool:
        """
        GPU activation function implementation using DeviceContext API.

        Args:
            activation: Activation function name.

        Returns:
            Bool: True if GPU operation succeeded, False otherwise.

        Raises:
            Error: If GPU operations fail.
        """
        try:
            # Map activation string to integer for GPU kernel
            var activation_type: Int
            if activation == "tanh":
                activation_type = 0
            elif activation == "relu":
                activation_type = 1
            elif activation == "sigmoid":
                activation_type = 2
            elif activation == "linear":
                activation_type = 3
            else:
                return False  # Unknown activation function

            with DeviceContext() as ctx:
                # Create GPU buffer
                buffer_size = self.rows * self.cols
                buffer_matrix = ctx.enqueue_create_buffer[DType.float64](
                    buffer_size
                )

                # Transfer data to GPU
                with buffer_matrix.map_to_host() as matrix_host:
                    for i in range(buffer_size):
                        matrix_host[i] = self.data[i]

                # Configure GPU kernel launch parameters
                alias BLOCK_SIZE = 16
                blocks_x = (self.cols + BLOCK_SIZE - 1) // BLOCK_SIZE
                blocks_y = (self.rows + BLOCK_SIZE - 1) // BLOCK_SIZE

                # Launch GPU kernel
                ctx.enqueue_function[gpu_apply_activation_kernel](
                    buffer_matrix.unsafe_ptr(),
                    self.rows,
                    self.cols,
                    activation_type,
                    grid_dim=(blocks_x, blocks_y),
                    block_dim=(BLOCK_SIZE, BLOCK_SIZE),
                )

                # Synchronize GPU operations
                ctx.synchronize()

                # Transfer results back to CPU
                with buffer_matrix.map_to_host() as matrix_host:
                    for i in range(buffer_size):
                        self.data[i] = Float64(matrix_host[i])

                return True

        except _:
            return False

        return False  # Should not reach here, but required for all paths

    fn _cpu_apply_activation(mut self, activation: String):
        """
        CPU activation function fallback implementation.

        Args:
            activation: Activation function name.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.get(i, j)
                if activation == "tanh":
                    self.set(i, j, tanh(val))
                elif activation == "relu":
                    self.set(i, j, max(0.0, val))
                elif activation == "sigmoid":
                    self.set(i, j, 1.0 / (1.0 + exp(-val)))
                # Linear activation (no change) for output layer


struct NeuralLayer(Copyable, Movable):
    """
    Single neural network layer with weights, biases, and activation.

    This struct represents a fully connected neural network layer with
    configurable activation functions and Xavier weight initialization.
    """

    var weights: Matrix
    var biases: List[Float64]
    var activation: String
    var input_size: Int
    var output_size: Int

    fn __init__(
        out self, input_size: Int, output_size: Int, activation: String = "tanh"
    ):
        """
        Initialize layer with random weights.

        Args:
            input_size: Number of input neurons.
            output_size: Number of output neurons.
            activation: Activation function name (default: "tanh").
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = Matrix(input_size, output_size)
        self.biases = List[Float64]()

        # Initialize biases to zero
        for _ in range(output_size):
            self.biases.append(0.0)

        # Initialize weights with Xavier initialization
        self._initialize_weights()

    fn _initialize_weights(mut self):
        """
        Initialize weights using Xavier initialization.

        Uses Xavier/Glorot initialization to set initial weight values
        based on the number of input and output neurons.
        """
        scale = sqrt(2.0 / Float64(self.input_size + self.output_size))

        for i in range(self.input_size):
            for j in range(self.output_size):
                # Simple pseudo-random initialization (replace with proper random when available)
                val = scale * (Float64((i * 7 + j * 13) % 1000) / 1000.0 - 0.5)
                self.weights.set(i, j, val)

    fn forward(self, input: Matrix) -> Matrix:
        """
        Forward pass through the layer.

        Args:
            input: Input matrix to process through the layer.

        Returns:
            Matrix: Output after applying weights, biases, and activation.
        """
        output = input.multiply(self.weights)
        output.add_bias(self.biases)
        output.apply_activation(self.activation)
        return output


struct PendulumNeuralNetwork:
    """
    Physics-informed neural network for pendulum digital twin.

    Architecture:
    - Input: [la_position, pend_velocity, pend_position, cmd_volts]
    - Hidden layers: 3 layers with 128 neurons each
    - Output: [next_la_position, next_pend_velocity, next_pend_position]
    - Physics constraints: Integrated into loss function and predictions
    """

    var layers: List[NeuralLayer]
    var physics_model: PendulumPhysics
    var input_means: List[Float64]
    var input_stds: List[Float64]
    var output_means: List[Float64]
    var output_stds: List[Float64]
    var trained: Bool

    fn __init__(out self):
        """
        Initialize neural network architecture.

        Sets up the complete neural network with physics model integration,
        normalization parameters, and multi-layer architecture.
        """
        self.layers = List[NeuralLayer]()
        self.physics_model = PendulumPhysics()
        self.input_means = List[Float64]()
        self.input_stds = List[Float64]()
        self.output_means = List[Float64]()
        self.output_stds = List[Float64]()
        self.trained = False

        # Build network architecture
        self._build_architecture()

        # Initialize normalization parameters
        self._initialize_normalization()

    fn _build_architecture(mut self):
        """
        Build the neural network architecture.

        Creates a multi-layer neural network with configurable hidden layers
        and appropriate activation functions for regression tasks.
        """
        # Input layer to first hidden layer
        layer1 = NeuralLayer(MODEL_INPUT_DIM, MODEL_HIDDEN_SIZE, "tanh")
        self.layers.append(layer1)

        # Hidden layers
        for _ in range(MODEL_HIDDEN_LAYERS - 1):
            hidden_layer = NeuralLayer(
                MODEL_HIDDEN_SIZE, MODEL_HIDDEN_SIZE, "tanh"
            )
            self.layers.append(hidden_layer)

        # Output layer (linear activation for regression)
        output_layer = NeuralLayer(
            MODEL_HIDDEN_SIZE, MODEL_OUTPUT_DIM, "linear"
        )
        self.layers.append(output_layer)

    fn _initialize_normalization(mut self):
        """
        Initialize normalization parameters with default values.

        Sets up input and output normalization statistics that will be
        updated during training for proper data scaling.
        """
        # Input normalization (will be updated during training)
        for _ in range(MODEL_INPUT_DIM):
            self.input_means.append(0.0)
            self.input_stds.append(1.0)

        # Output normalization
        for _ in range(MODEL_OUTPUT_DIM):
            self.output_means.append(0.0)
            self.output_stds.append(1.0)

    fn normalize_input(self, input: List[Float64]) -> List[Float64]:
        """
        Normalize input using stored statistics.

        Args:
            input: Raw input vector to normalize.

        Returns:
            List[Float64]: Normalized input vector with zero mean and unit variance.
        """
        normalized = List[Float64]()

        for i in range(len(input)):
            if i < len(self.input_means):
                val = (input[i] - self.input_means[i]) / self.input_stds[i]
                normalized.append(val)
            else:
                normalized.append(input[i])

        return normalized

    fn denormalize_output(self, output: List[Float64]) -> List[Float64]:
        """
        Denormalize output using stored statistics.

        Args:
            output: Normalized output vector from the network.

        Returns:
            List[Float64]: Denormalized output vector in original scale.
        """
        denormalized = List[Float64]()

        for i in range(len(output)):
            if i < len(self.output_means):
                val = output[i] * self.output_stds[i] + self.output_means[i]
                denormalized.append(val)
            else:
                denormalized.append(output[i])

        return denormalized

    fn forward(self, input: List[Float64]) -> List[Float64]:
        """
        Forward pass through the network.

        Args:
            input: [la_position, pend_velocity, pend_position, cmd_volts].

        Returns:
            [next_la_position, next_pend_velocity, next_pend_position].
        """
        # Normalize input
        normalized_input = self.normalize_input(input)

        # Convert to matrix format
        var current_output = Matrix(1, len(normalized_input))
        for i in range(len(normalized_input)):
            current_output.set(0, i, normalized_input[i])

        # Forward pass through all layers
        for i in range(len(self.layers)):
            current_output = self.layers[i].forward(current_output)

        # Extract output
        raw_output = List[Float64]()
        for i in range(MODEL_OUTPUT_DIM):
            raw_output.append(current_output.get(0, i))

        # Denormalize output
        final_output = self.denormalize_output(raw_output)

        # Apply physics constraints
        return apply_physics_constraints(input, final_output)

    fn predict_next_state(
        self, current_state: List[Float64], dt: Float64 = 0.04
    ) -> List[Float64]:
        """
        Predict next state given current state.

        Args:
            current_state: [la_position, pend_velocity, pend_position, cmd_volts].
            dt: Time step (seconds).

        Returns:
            Predicted next state.
        """
        return self.forward(current_state)

    fn compute_physics_loss(
        self, input: List[Float64], prediction: List[Float64]
    ) -> Float64:
        """
        Compute physics-informed loss component.

        Args:
            input: Input state.
            prediction: Network prediction.

        Returns:
            Physics loss value.
        """
        # Convert to physics state
        current_state = PendulumState.from_data_sample(
            input[0], input[1], input[2], input[3]
        )
        predicted_state = PendulumState.from_data_sample(
            prediction[0], prediction[1], prediction[2], input[3]
        )

        # Check energy conservation (approximate)
        current_energy = current_state.total_energy()
        predicted_energy = predicted_state.total_energy()
        energy_loss = abs(predicted_energy - current_energy) / max(
            current_energy, 1e-6
        )

        # Check constraint violations
        constraint_loss = 0.0
        if not self.physics_model.validate_physics_constraints(predicted_state):
            constraint_loss = 10.0  # High penalty for constraint violations

        return energy_loss + constraint_loss

    fn set_normalization_parameters(
        mut self,
        input_means: List[Float64],
        input_stds: List[Float64],
        output_means: List[Float64],
        output_stds: List[Float64],
    ):
        """
        Set normalization parameters from training data.

        Args:
            input_means: Mean values for input normalization.
            input_stds: Standard deviation values for input normalization.
            output_means: Mean values for output denormalization.
            output_stds: Standard deviation values for output denormalization.
        """
        self.input_means = input_means
        self.input_stds = input_stds
        self.output_means = output_means
        self.output_stds = output_stds


fn create_pendulum_network() -> PendulumNeuralNetwork:
    """
    Create a pendulum neural network with default architecture.

    Returns:
        Initialized PendulumNeuralNetwork.
    """
    return PendulumNeuralNetwork()
