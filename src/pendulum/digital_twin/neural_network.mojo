"""
Neural network architecture for pendulum digital twin.

This module implements a physics-informed neural network for modeling
pendulum dynamics using Mojo. When MAX engine is available, this will
be enhanced with GPU acceleration.
"""

from collections import List
from math import exp, tanh, max, min, sqrt
from memory import UnsafePointer

# Import project modules
from config.pendulum_config import (
    MODEL_INPUT_DIM,
    MODEL_OUTPUT_DIM,
    MODEL_HIDDEN_LAYERS,
    MODEL_HIDDEN_SIZE,
    MODEL_LEARNING_RATE,
)
from src.pendulum.utils.physics import PendulumState, PendulumPhysics

struct Matrix:
    """
    Simple matrix implementation for neural network operations.
    
    This will be replaced with MAX engine tensors when available.
    """
    
    var data: List[Float64]
    var rows: Int
    var cols: Int
    
    fn __init__(out self, rows: Int, cols: Int):
        """Initialize matrix with zeros."""
        self.rows = rows
        self.cols = cols
        self.data = List[Float64]()
        
        for i in range(rows * cols):
            self.data.append(0.0)
    
    fn get(self, row: Int, col: Int) -> Float64:
        """Get element at (row, col)."""
        return self.data[row * self.cols + col]
    
    fn set(mut self, row: Int, col: Int, value: Float64):
        """Set element at (row, col)."""
        self.data[row * self.cols + col] = value
    
    fn multiply(self, other: Matrix) -> Matrix:
        """Matrix multiplication."""
        var result = Matrix(self.rows, other.cols)
        
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
                val = self.get(i, j)
                if activation == "tanh":
                    self.set(i, j, tanh(val))
                elif activation == "relu":
                    self.set(i, j, max(0.0, val))
                elif activation == "sigmoid":
                    self.set(i, j, 1.0 / (1.0 + exp(-val)))
                # Linear activation (no change) for output layer

struct NeuralLayer:
    """
    Single neural network layer with weights, biases, and activation.
    """
    
    var weights: Matrix
    var biases: List[Float64]
    var activation: String
    var input_size: Int
    var output_size: Int
    
    fn __init__(out self, input_size: Int, output_size: Int, activation: String = "tanh"):
        """Initialize layer with random weights."""
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = Matrix(input_size, output_size)
        self.biases = List[Float64]()
        
        # Initialize biases to zero
        for i in range(output_size):
            self.biases.append(0.0)
        
        # Initialize weights with Xavier initialization
        self._initialize_weights()
    
    fn _initialize_weights(mut self):
        """Initialize weights using Xavier initialization."""
        var scale = sqrt(2.0 / Float64(self.input_size + self.output_size))
        
        for i in range(self.input_size):
            for j in range(self.output_size):
                # Simple pseudo-random initialization (replace with proper random when available)
                var val = scale * (Float64((i * 7 + j * 13) % 1000) / 1000.0 - 0.5)
                self.weights.set(i, j, val)
    
    fn forward(self, input: Matrix) -> Matrix:
        """Forward pass through the layer."""
        var output = input.multiply(self.weights)
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
        """Initialize neural network architecture."""
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
        """Build the neural network architecture."""
        # Input layer to first hidden layer
        var layer1 = NeuralLayer(MODEL_INPUT_DIM, MODEL_HIDDEN_SIZE, "tanh")
        self.layers.append(layer1)
        
        # Hidden layers
        for i in range(MODEL_HIDDEN_LAYERS - 1):
            var hidden_layer = NeuralLayer(MODEL_HIDDEN_SIZE, MODEL_HIDDEN_SIZE, "tanh")
            self.layers.append(hidden_layer)
        
        # Output layer (linear activation for regression)
        var output_layer = NeuralLayer(MODEL_HIDDEN_SIZE, MODEL_OUTPUT_DIM, "linear")
        self.layers.append(output_layer)
    
    fn _initialize_normalization(mut self):
        """Initialize normalization parameters with default values."""
        # Input normalization (will be updated during training)
        for i in range(MODEL_INPUT_DIM):
            self.input_means.append(0.0)
            self.input_stds.append(1.0)
        
        # Output normalization
        for i in range(MODEL_OUTPUT_DIM):
            self.output_means.append(0.0)
            self.output_stds.append(1.0)
    
    fn normalize_input(self, input: List[Float64]) -> List[Float64]:
        """Normalize input using stored statistics."""
        var normalized = List[Float64]()
        
        for i in range(len(input)):
            if i < len(self.input_means):
                val = (input[i] - self.input_means[i]) / self.input_stds[i]
                normalized.append(val)
            else:
                normalized.append(input[i])
        
        return normalized
    
    fn denormalize_output(self, output: List[Float64]) -> List[Float64]:
        """Denormalize output using stored statistics."""
        var denormalized = List[Float64]()
        
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
            input: [la_position, pend_velocity, pend_position, cmd_volts]
            
        Returns:
            [next_la_position, next_pend_velocity, next_pend_position]
        """
        # Normalize input
        var normalized_input = self.normalize_input(input)
        
        # Convert to matrix format
        var current_output = Matrix(1, len(normalized_input))
        for i in range(len(normalized_input)):
            current_output.set(0, i, normalized_input[i])
        
        # Forward pass through all layers
        for i in range(len(self.layers)):
            current_output = self.layers[i].forward(current_output)
        
        # Extract output
        var raw_output = List[Float64]()
        for i in range(MODEL_OUTPUT_DIM):
            raw_output.append(current_output.get(0, i))
        
        # Denormalize output
        var final_output = self.denormalize_output(raw_output)
        
        # Apply physics constraints
        return self._apply_physics_constraints(input, final_output)
    
    fn _apply_physics_constraints(self, input: List[Float64], prediction: List[Float64]) -> List[Float64]:
        """
        Apply physics constraints to network predictions.
        
        Args:
            input: Original input state
            prediction: Raw network prediction
            
        Returns:
            Physics-constrained prediction
        """
        var constrained = List[Float64]()
        
        # Extract current state
        current_la_pos = input[0]
        current_pend_vel = input[1]
        current_pend_pos = input[2]
        current_cmd_volts = input[3]
        
        # Extract predictions
        pred_la_pos = prediction[0]
        pred_pend_vel = prediction[1]
        pred_pend_pos = prediction[2]
        
        # Apply actuator position constraints
        constrained_la_pos = max(-4.0, min(4.0, pred_la_pos))
        constrained.append(constrained_la_pos)
        
        # Apply velocity constraints
        constrained_pend_vel = max(-1000.0, min(1000.0, pred_pend_vel))
        constrained.append(constrained_pend_vel)
        
        # Apply angle continuity (no sudden jumps)
        angle_diff = pred_pend_pos - current_pend_pos
        if abs(angle_diff) > 180.0:
            # Handle angle wrapping
            if angle_diff > 180.0:
                pred_pend_pos -= 360.0
            elif angle_diff < -180.0:
                pred_pend_pos += 360.0
        
        constrained.append(pred_pend_pos)
        
        return constrained
    
    fn predict_next_state(self, current_state: List[Float64], dt: Float64 = 0.04) -> List[Float64]:
        """
        Predict next state given current state.
        
        Args:
            current_state: [la_position, pend_velocity, pend_position, cmd_volts]
            dt: Time step (seconds)
            
        Returns:
            Predicted next state
        """
        return self.forward(current_state)
    
    fn compute_physics_loss(self, input: List[Float64], prediction: List[Float64]) -> Float64:
        """
        Compute physics-informed loss component.
        
        Args:
            input: Input state
            prediction: Network prediction
            
        Returns:
            Physics loss value
        """
        # Convert to physics state
        current_state = PendulumState.from_data_sample(input[0], input[1], input[2], input[3])
        predicted_state = PendulumState.from_data_sample(prediction[0], prediction[1], prediction[2], input[3])
        
        # Check energy conservation (approximate)
        current_energy = current_state.total_energy()
        predicted_energy = predicted_state.total_energy()
        energy_loss = abs(predicted_energy - current_energy) / max(current_energy, 1e-6)
        
        # Check constraint violations
        constraint_loss = 0.0
        if not self.physics_model.validate_physics_constraints(predicted_state):
            constraint_loss = 10.0  # High penalty for constraint violations
        
        return energy_loss + constraint_loss
    
    fn set_normalization_parameters(mut self, input_means: List[Float64], input_stds: List[Float64],
                                   output_means: List[Float64], output_stds: List[Float64]):
        """Set normalization parameters from training data."""
        self.input_means = input_means
        self.input_stds = input_stds
        self.output_means = output_means
        self.output_stds = output_stds

@staticmethod
fn create_pendulum_network() -> PendulumNeuralNetwork:
    """
    Create a pendulum neural network with default architecture.
    
    Returns:
        Initialized PendulumNeuralNetwork
    """
    return PendulumNeuralNetwork()
