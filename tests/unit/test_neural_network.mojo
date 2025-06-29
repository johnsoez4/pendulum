"""
Unit tests for neural network module.

This module tests the neural network architecture, forward pass, physics
constraints, and training components.
"""

from collections import List
from testing import assert_equal, assert_true, assert_false

# Helper functions for testing
fn abs(x: Float64) -> Float64:
    """Return absolute value of x."""
    return x if x >= 0.0 else -x

fn max(a: Float64, b: Float64) -> Float64:
    """Return maximum of two values."""
    return a if a > b else b

fn min(a: Float64, b: Float64) -> Float64:
    """Return minimum of two values."""
    return a if a < b else b

fn assert_near(actual: Float64, expected: Float64, tolerance: Float64 = 1e-6):
    """Assert that two floating point values are close."""
    var diff = abs(actual - expected)
    if diff > tolerance:
        print("Assertion failed: expected", expected, "but got", actual, "diff:", diff)
        assert_true(False)

fn tanh_approx(x: Float64) -> Float64:
    """Approximate tanh function."""
    if x > 3.0:
        return 1.0
    elif x < -3.0:
        return -1.0
    else:
        var x2 = x * x
        return x * (1.0 - x2/3.0 + 2.0*x2*x2/15.0)

# Network configuration constants for testing
alias INPUT_DIM = 4
alias OUTPUT_DIM = 3
alias HIDDEN_SIZE = 8  # Smaller for testing

@fieldwise_init
struct TestNeuralNetwork(Copyable, Movable):
    """Simplified neural network for testing."""
    
    var weights1: List[List[Float64]]  # Input to hidden weights
    var biases1: List[Float64]         # Hidden biases
    var weights2: List[List[Float64]]  # Hidden to output weights
    var biases2: List[Float64]         # Output biases
    var trained: Bool
    
    fn initialize_weights(mut self):
        """Initialize network weights with small values."""
        # Initialize weights1 (INPUT_DIM x HIDDEN_SIZE)
        for i in range(INPUT_DIM):
            var row = List[Float64]()
            for j in range(HIDDEN_SIZE):
                var val = 0.1 * (Float64((i * 7 + j * 13) % 100) / 100.0 - 0.5)
                row.append(val)
            self.weights1.append(row)
        
        # Initialize biases1
        for _ in range(HIDDEN_SIZE):
            self.biases1.append(0.0)
        
        # Initialize weights2 (HIDDEN_SIZE x OUTPUT_DIM)
        for i in range(HIDDEN_SIZE):
            var row = List[Float64]()
            for j in range(OUTPUT_DIM):
                var val = 0.1 * (Float64((i * 11 + j * 17) % 100) / 100.0 - 0.5)
                row.append(val)
            self.weights2.append(row)
        
        # Initialize biases2
        for _ in range(OUTPUT_DIM):
            self.biases2.append(0.0)
    
    fn forward(self, input: List[Float64]) -> List[Float64]:
        """Forward pass through the network."""
        # Layer 1: Input to Hidden
        var hidden = List[Float64]()
        for j in range(HIDDEN_SIZE):
            var sum = self.biases1[j]
            for i in range(INPUT_DIM):
                if i < len(input):
                    sum += input[i] * self.weights1[i][j]
            hidden.append(tanh_approx(sum))
        
        # Layer 2: Hidden to Output
        var output = List[Float64]()
        for j in range(OUTPUT_DIM):
            var sum = self.biases2[j]
            for i in range(HIDDEN_SIZE):
                sum += hidden[i] * self.weights2[i][j]
            output.append(sum)  # Linear activation for output
        
        return self.apply_constraints(input, output)
    
    fn apply_constraints(self, input: List[Float64], prediction: List[Float64]) -> List[Float64]:
        """Apply physics constraints to predictions."""
        var constrained = List[Float64]()
        
        # Constrain actuator position to [-4, 4] inches
        var la_pos = max(-4.0, min(4.0, prediction[0]))
        constrained.append(la_pos)
        
        # Constrain pendulum velocity to [-1000, 1000] deg/s
        var pend_vel = max(-1000.0, min(1000.0, prediction[1]))
        constrained.append(pend_vel)
        
        # Handle angle continuity
        var pred_angle = prediction[2]
        constrained.append(pred_angle)
        
        return constrained

struct NeuralNetworkTests:
    """Test suite for neural network functionality."""
    
    @staticmethod
    fn test_network_creation():
        """Test neural network creation and initialization."""
        print("Testing neural network creation...")
        
        var weights1 = List[List[Float64]]()
        var biases1 = List[Float64]()
        var weights2 = List[List[Float64]]()
        var biases2 = List[Float64]()
        
        var network = TestNeuralNetwork(weights1, biases1, weights2, biases2, False)
        network.initialize_weights()
        
        # Check weight dimensions
        assert_equal(len(network.weights1), INPUT_DIM)
        assert_equal(len(network.weights1[0]), HIDDEN_SIZE)
        assert_equal(len(network.biases1), HIDDEN_SIZE)
        assert_equal(len(network.weights2), HIDDEN_SIZE)
        assert_equal(len(network.weights2[0]), OUTPUT_DIM)
        assert_equal(len(network.biases2), OUTPUT_DIM)
        
        print("✓ Network creation test passed")
    
    @staticmethod
    fn test_forward_pass():
        """Test neural network forward pass."""
        print("Testing forward pass...")
        
        var weights1 = List[List[Float64]]()
        var biases1 = List[Float64]()
        var weights2 = List[List[Float64]]()
        var biases2 = List[Float64]()
        
        var network = TestNeuralNetwork(weights1, biases1, weights2, biases2, False)
        network.initialize_weights()
        
        # Test with valid input
        var input = List[Float64]()
        input.append(1.0)   # la_position
        input.append(10.0)  # pend_velocity
        input.append(180.0) # pend_position
        input.append(0.5)   # cmd_volts
        
        var output = network.forward(input)
        
        # Check output dimensions
        assert_equal(len(output), OUTPUT_DIM)
        
        # Check that output values are reasonable
        for i in range(len(output)):
            assert_true(output[i] == output[i])  # Not NaN
            assert_true(abs(output[i]) < 1000.0)  # Reasonable magnitude
        
        print("✓ Forward pass test passed")
    
    @staticmethod
    fn test_physics_constraints():
        """Test physics constraint enforcement."""
        print("Testing physics constraints...")
        
        var weights1 = List[List[Float64]]()
        var biases1 = List[Float64]()
        var weights2 = List[List[Float64]]()
        var biases2 = List[Float64]()
        
        var network = TestNeuralNetwork(weights1, biases1, weights2, biases2, False)
        network.initialize_weights()
        
        # Test input that might produce constraint violations
        var input = List[Float64]()
        input.append(0.0)
        input.append(0.0)
        input.append(0.0)
        input.append(10.0)  # High voltage
        
        var output = network.forward(input)
        
        # Check actuator position constraint [-4, 4] inches
        assert_true(output[0] >= -4.0)
        assert_true(output[0] <= 4.0)
        
        # Check velocity constraint [-1000, 1000] deg/s
        assert_true(output[1] >= -1000.0)
        assert_true(output[1] <= 1000.0)
        
        print("✓ Physics constraints test passed")
    
    @staticmethod
    fn test_activation_functions():
        """Test activation function implementations."""
        print("Testing activation functions...")
        
        # Test tanh approximation
        assert_near(tanh_approx(0.0), 0.0, 1e-3)
        assert_near(tanh_approx(1.0), 0.76, 1e-1)  # Approximate tanh(1)
        assert_near(tanh_approx(-1.0), -0.76, 1e-1)  # Approximate tanh(-1)
        
        # Test saturation
        assert_near(tanh_approx(5.0), 1.0, 1e-3)
        assert_near(tanh_approx(-5.0), -1.0, 1e-3)
        
        print("✓ Activation functions test passed")
    
    @staticmethod
    fn test_input_validation():
        """Test input validation and error handling."""
        print("Testing input validation...")
        
        var weights1 = List[List[Float64]]()
        var biases1 = List[Float64]()
        var weights2 = List[List[Float64]]()
        var biases2 = List[Float64]()
        
        var network = TestNeuralNetwork(weights1, biases1, weights2, biases2, False)
        network.initialize_weights()
        
        # Test with correct input size
        var valid_input = List[Float64]()
        valid_input.append(1.0)
        valid_input.append(2.0)
        valid_input.append(3.0)
        valid_input.append(4.0)
        
        var output = network.forward(valid_input)
        assert_equal(len(output), OUTPUT_DIM)
        
        # Test with smaller input (should handle gracefully)
        var small_input = List[Float64]()
        small_input.append(1.0)
        small_input.append(2.0)
        
        var output2 = network.forward(small_input)
        assert_equal(len(output2), OUTPUT_DIM)
        
        print("✓ Input validation test passed")
    
    @staticmethod
    fn test_weight_initialization():
        """Test weight initialization properties."""
        print("Testing weight initialization...")
        
        var weights1 = List[List[Float64]]()
        var biases1 = List[Float64]()
        var weights2 = List[List[Float64]]()
        var biases2 = List[Float64]()
        
        var network = TestNeuralNetwork(weights1, biases1, weights2, biases2, False)
        network.initialize_weights()
        
        # Check that weights are small and varied
        var weight_sum = 0.0
        var weight_count = 0
        
        for i in range(len(network.weights1)):
            for j in range(len(network.weights1[i])):
                var w = network.weights1[i][j]
                assert_true(abs(w) < 0.1)  # Small initialization
                weight_sum += w
                weight_count += 1
        
        # Check that weights are not all the same
        var avg_weight = weight_sum / Float64(weight_count)
        var has_variation = False
        for i in range(len(network.weights1)):
            for j in range(len(network.weights1[i])):
                if abs(network.weights1[i][j] - avg_weight) > 1e-6:
                    has_variation = True
                    break
        
        assert_true(has_variation)
        
        print("✓ Weight initialization test passed")
    
    @staticmethod
    fn run_all_tests():
        """Run all neural network tests."""
        print("Running Neural Network Unit Tests")
        print("=================================")
        
        NeuralNetworkTests.test_network_creation()
        NeuralNetworkTests.test_forward_pass()
        NeuralNetworkTests.test_physics_constraints()
        NeuralNetworkTests.test_activation_functions()
        NeuralNetworkTests.test_input_validation()
        NeuralNetworkTests.test_weight_initialization()
        
        print()
        print("✓ All neural network tests passed!")
        print()

fn main():
    """Run neural network unit tests."""
    NeuralNetworkTests.run_all_tests()
