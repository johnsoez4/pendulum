"""
Performance benchmarks for the pendulum digital twin system.

This module tests real-time performance, throughput, and latency requirements
for the digital twin system including 25 Hz control loop capability.
"""

from collections import List
from time import now

# Helper functions for performance testing
fn abs(x: Float64) -> Float64:
    """Return absolute value of x."""
    return x if x >= 0.0 else -x

fn max(a: Float64, b: Float64) -> Float64:
    """Return maximum of two values."""
    return a if a > b else b

fn min(a: Float64, b: Float64) -> Float64:
    """Return minimum of two values."""
    return a if a < b else b

fn tanh_approx(x: Float64) -> Float64:
    """Approximate tanh function."""
    if x > 3.0:
        return 1.0
    elif x < -3.0:
        return -1.0
    else:
        var x2 = x * x
        return x * (1.0 - x2/3.0 + 2.0*x2*x2/15.0)

# Performance test configuration
alias BENCHMARK_ITERATIONS = 1000
alias TARGET_FREQUENCY_HZ = 25.0
alias TARGET_LATENCY_MS = 40.0  # 1/25 Hz = 40ms
alias INPUT_DIM = 4
alias OUTPUT_DIM = 3
alias HIDDEN_SIZE = 64

@fieldwise_init
struct BenchmarkNetwork(Copyable, Movable):
    """Neural network optimized for performance benchmarking."""
    
    var weights1: List[List[Float64]]
    var biases1: List[Float64]
    var weights2: List[List[Float64]]
    var biases2: List[Float64]
    var weights3: List[List[Float64]]
    var biases3: List[Float64]
    
    fn initialize_weights(mut self):
        """Initialize network weights for benchmarking."""
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
        
        # Initialize weights2 (HIDDEN_SIZE x HIDDEN_SIZE)
        for i in range(HIDDEN_SIZE):
            var row = List[Float64]()
            for j in range(HIDDEN_SIZE):
                var val = 0.1 * (Float64((i * 11 + j * 17) % 100) / 100.0 - 0.5)
                row.append(val)
            self.weights2.append(row)
        
        # Initialize biases2
        for _ in range(HIDDEN_SIZE):
            self.biases2.append(0.0)
        
        # Initialize weights3 (HIDDEN_SIZE x OUTPUT_DIM)
        for i in range(HIDDEN_SIZE):
            var row = List[Float64]()
            for j in range(OUTPUT_DIM):
                var val = 0.1 * (Float64((i * 19 + j * 23) % 100) / 100.0 - 0.5)
                row.append(val)
            self.weights3.append(row)
        
        # Initialize biases3
        for _ in range(OUTPUT_DIM):
            self.biases3.append(0.0)
    
    fn forward_optimized(self, input: List[Float64]) -> List[Float64]:
        """Optimized forward pass for performance benchmarking."""
        # Layer 1: Input to Hidden1
        var hidden1 = List[Float64]()
        for j in range(HIDDEN_SIZE):
            var sum = self.biases1[j]
            for i in range(INPUT_DIM):
                if i < len(input):
                    sum += input[i] * self.weights1[i][j]
            hidden1.append(tanh_approx(sum))
        
        # Layer 2: Hidden1 to Hidden2
        var hidden2 = List[Float64]()
        for j in range(HIDDEN_SIZE):
            var sum = self.biases2[j]
            for i in range(HIDDEN_SIZE):
                sum += hidden1[i] * self.weights2[i][j]
            hidden2.append(tanh_approx(sum))
        
        # Layer 3: Hidden2 to Output
        var output = List[Float64]()
        for j in range(OUTPUT_DIM):
            var sum = self.biases3[j]
            for i in range(HIDDEN_SIZE):
                sum += hidden2[i] * self.weights3[i][j]
            output.append(sum)
        
        # Apply constraints (optimized)
        return self.apply_constraints_fast(output)
    
    fn apply_constraints_fast(self, prediction: List[Float64]) -> List[Float64]:
        """Fast constraint application for performance."""
        var constrained = List[Float64]()
        
        # Actuator position constraint [-4, 4] inches
        constrained.append(max(-4.0, min(4.0, prediction[0])))
        
        # Velocity constraint [-1000, 1000] deg/s
        constrained.append(max(-1000.0, min(1000.0, prediction[1])))
        
        # Angle (no constraint for performance)
        constrained.append(prediction[2])
        
        return constrained

struct PerformanceTests:
    """Performance benchmark test suite."""
    
    @staticmethod
    fn test_inference_latency():
        """Test neural network inference latency."""
        print("Testing inference latency...")
        
        # Create and initialize network
        var weights1 = List[List[Float64]]()
        var biases1 = List[Float64]()
        var weights2 = List[List[Float64]]()
        var biases2 = List[Float64]()
        var weights3 = List[List[Float64]]()
        var biases3 = List[Float64]()
        
        var network = BenchmarkNetwork(weights1, biases1, weights2, biases2, weights3, biases3)
        network.initialize_weights()
        
        # Prepare test input
        var test_input = List[Float64]()
        test_input.append(2.0)    # la_position
        test_input.append(100.0)  # pend_velocity
        test_input.append(180.0)  # pend_position
        test_input.append(1.0)    # cmd_volts
        
        # Warm-up runs
        for _ in range(10):
            var _ = network.forward_optimized(test_input)
        
        # Benchmark inference latency
        var start_time = now()
        
        for i in range(BENCHMARK_ITERATIONS):
            var prediction = network.forward_optimized(test_input)
            # Ensure prediction is used to prevent optimization
            if len(prediction) != OUTPUT_DIM:
                print("Error: unexpected output size")
        
        var end_time = now()
        var total_time_ns = end_time - start_time
        var total_time_ms = Float64(total_time_ns) / 1_000_000.0
        var avg_latency_ms = total_time_ms / Float64(BENCHMARK_ITERATIONS)
        
        print("  Iterations:", BENCHMARK_ITERATIONS)
        print("  Total time:", total_time_ms, "ms")
        print("  Average latency:", avg_latency_ms, "ms")
        print("  Target latency:", TARGET_LATENCY_MS, "ms")
        
        # Check if meets real-time requirements
        if avg_latency_ms <= TARGET_LATENCY_MS:
            print("  ✓ Meets 25 Hz real-time requirement")
        else:
            print("  ⚠ Does not meet 25 Hz requirement")
        
        print("✓ Inference latency test completed")
    
    @staticmethod
    fn test_throughput():
        """Test system throughput (predictions per second)."""
        print("Testing system throughput...")
        
        # Create network
        var weights1 = List[List[Float64]]()
        var biases1 = List[Float64]()
        var weights2 = List[List[Float64]]()
        var biases2 = List[Float64]()
        var weights3 = List[List[Float64]]()
        var biases3 = List[Float64]()
        
        var network = BenchmarkNetwork(weights1, biases1, weights2, biases2, weights3, biases3)
        network.initialize_weights()
        
        # Prepare multiple test inputs
        var test_inputs = List[List[Float64]]()
        for i in range(100):
            var input = List[Float64]()
            input.append(Float64(i % 10) * 0.4 - 2.0)
            input.append(Float64(i % 20) * 10.0 - 100.0)
            input.append(Float64(i % 36) * 10.0)
            input.append(Float64(i % 5) * 0.4 - 1.0)
            test_inputs.append(input)
        
        # Benchmark throughput
        var start_time = now()
        var predictions_made = 0
        
        for iteration in range(BENCHMARK_ITERATIONS // 10):  # Fewer iterations for throughput test
            for i in range(len(test_inputs)):
                var prediction = network.forward_optimized(test_inputs[i])
                predictions_made += 1
        
        var end_time = now()
        var total_time_s = Float64(end_time - start_time) / 1_000_000_000.0
        var throughput = Float64(predictions_made) / total_time_s
        
        print("  Predictions made:", predictions_made)
        print("  Total time:", total_time_s, "seconds")
        print("  Throughput:", throughput, "predictions/second")
        print("  Target frequency:", TARGET_FREQUENCY_HZ, "Hz")
        
        # Check if meets throughput requirements
        if throughput >= TARGET_FREQUENCY_HZ:
            print("  ✓ Meets throughput requirement")
        else:
            print("  ⚠ Below target throughput")
        
        print("✓ Throughput test completed")
    
    @staticmethod
    fn test_real_time_simulation():
        """Test real-time control loop simulation."""
        print("Testing real-time control loop simulation...")
        
        # Create network
        var weights1 = List[List[Float64]]()
        var biases1 = List[Float64]()
        var weights2 = List[List[Float64]]()
        var biases2 = List[Float64]()
        var weights3 = List[List[Float64]]()
        var biases3 = List[Float64]()
        
        var network = BenchmarkNetwork(weights1, biases1, weights2, biases2, weights3, biases3)
        network.initialize_weights()
        
        # Simulate real-time control loop
        var control_frequency = 25.0  # 25 Hz
        var simulation_duration = 1.0  # 1 second
        var expected_cycles = Int(control_frequency * simulation_duration)
        
        var current_state = List[Float64]()
        current_state.append(0.0)    # Initial position
        current_state.append(0.0)    # Initial velocity
        current_state.append(180.0)  # Initial angle
        current_state.append(0.0)    # Initial command
        
        var start_time = now()
        var cycles_completed = 0
        var max_latency = 0.0
        var total_latency = 0.0
        
        for cycle in range(expected_cycles):
            var cycle_start = now()
            
            # Predict next state
            var prediction = network.forward_optimized(current_state)
            
            # Update state (simplified)
            current_state[0] = prediction[0]
            current_state[1] = prediction[1]
            current_state[2] = prediction[2]
            current_state[3] = 0.1 * Float64(cycle % 10)  # Varying command
            
            var cycle_end = now()
            var cycle_latency = Float64(cycle_end - cycle_start) / 1_000_000.0  # ms
            
            max_latency = max(max_latency, cycle_latency)
            total_latency += cycle_latency
            cycles_completed += 1
        
        var end_time = now()
        var total_simulation_time = Float64(end_time - start_time) / 1_000_000_000.0  # seconds
        var actual_frequency = Float64(cycles_completed) / total_simulation_time
        var avg_latency = total_latency / Float64(cycles_completed)
        
        print("  Expected cycles:", expected_cycles)
        print("  Completed cycles:", cycles_completed)
        print("  Simulation time:", total_simulation_time, "seconds")
        print("  Actual frequency:", actual_frequency, "Hz")
        print("  Target frequency:", control_frequency, "Hz")
        print("  Average latency:", avg_latency, "ms")
        print("  Maximum latency:", max_latency, "ms")
        print("  Target latency:", TARGET_LATENCY_MS, "ms")
        
        # Check real-time performance
        if actual_frequency >= control_frequency * 0.95:  # 95% of target
            print("  ✓ Meets real-time control requirements")
        else:
            print("  ⚠ Below real-time control requirements")
        
        if max_latency <= TARGET_LATENCY_MS:
            print("  ✓ Latency within acceptable bounds")
        else:
            print("  ⚠ Latency exceeds acceptable bounds")
        
        print("✓ Real-time simulation test completed")
    
    @staticmethod
    fn run_all_tests():
        """Run all performance tests."""
        print("Running Performance Benchmark Tests")
        print("===================================")
        print("Target: 25 Hz real-time control (40ms max latency)")
        print()
        
        PerformanceTests.test_inference_latency()
        print()
        PerformanceTests.test_throughput()
        print()
        PerformanceTests.test_real_time_simulation()
        
        print()
        print("✓ All performance tests completed!")
        print()

fn main():
    """Run performance benchmark tests."""
    PerformanceTests.run_all_tests()
