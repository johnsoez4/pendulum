"""
Performance benchmarks for the pendulum digital twin system.

This module tests real-time performance, throughput, and latency requirements
for the digital twin system including 25 Hz control loop capability using
official Mojo benchmark API with statistical analysis.

Key Components Tested:
- Neural network inference latency using official benchmark module
- System throughput measurement using proper Bench objects
- Real-time control loop simulation using BenchId and BenchConfig
- Performance validation against 25 Hz control requirements

Benchmark Methodology:
- Uses official Mojo benchmark module (Bench, Bencher, BenchId, BenchConfig)
- Statistical analysis through bencher.iter[] for accurate measurement
- Dead code elimination prevention following benchmark best practices
- Professional timing patterns following mojo_syntax.md guidelines
"""

from collections import List
from benchmark import Bench, Bencher, BenchId, BenchConfig, Unit


# Helper functions for performance testing


fn tanh_approx(x: Float64) -> Float64:
    """Approximate tanh function."""
    if x > 3.0:
        return 1.0
    elif x < -3.0:
        return -1.0
    else:
        x2 = x * x
        return x * (1.0 - x2 / 3.0 + 2.0 * x2 * x2 / 15.0)


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
            row = List[Float64]()
            for j in range(HIDDEN_SIZE):
                val = 0.1 * (Float64((i * 7 + j * 13) % 100) / 100.0 - 0.5)
                row.append(val)
            self.weights1.append(row)

        # Initialize biases1
        for _ in range(HIDDEN_SIZE):
            self.biases1.append(0.0)

        # Initialize weights2 (HIDDEN_SIZE x HIDDEN_SIZE)
        for i in range(HIDDEN_SIZE):
            row = List[Float64]()
            for j in range(HIDDEN_SIZE):
                val = 0.1 * (Float64((i * 11 + j * 17) % 100) / 100.0 - 0.5)
                row.append(val)
            self.weights2.append(row)

        # Initialize biases2
        for _ in range(HIDDEN_SIZE):
            self.biases2.append(0.0)

        # Initialize weights3 (HIDDEN_SIZE x OUTPUT_DIM)
        for i in range(HIDDEN_SIZE):
            row = List[Float64]()
            for j in range(OUTPUT_DIM):
                val = 0.1 * (Float64((i * 19 + j * 23) % 100) / 100.0 - 0.5)
                row.append(val)
            self.weights3.append(row)

        # Initialize biases3
        for _ in range(OUTPUT_DIM):
            self.biases3.append(0.0)

    fn forward_optimized(self, input: List[Float64]) -> List[Float64]:
        """Optimized forward pass for performance benchmarking."""
        # Layer 1: Input to Hidden1
        hidden1 = List[Float64]()
        for j in range(HIDDEN_SIZE):
            sum = self.biases1[j]
            for i in range(INPUT_DIM):
                if i < len(input):
                    sum += input[i] * self.weights1[i][j]
            hidden1.append(tanh_approx(sum))

        # Layer 2: Hidden1 to Hidden2
        hidden2 = List[Float64]()
        for j in range(HIDDEN_SIZE):
            sum = self.biases2[j]
            for i in range(HIDDEN_SIZE):
                sum += hidden1[i] * self.weights2[i][j]
            hidden2.append(tanh_approx(sum))

        # Layer 3: Hidden2 to Output
        output = List[Float64]()
        for j in range(OUTPUT_DIM):
            sum = self.biases3[j]
            for i in range(HIDDEN_SIZE):
                sum += hidden2[i] * self.weights3[i][j]
            output.append(sum)

        # Apply constraints (optimized)
        return self.apply_constraints_fast(output)

    fn apply_constraints_fast(self, prediction: List[Float64]) -> List[Float64]:
        """Fast constraint application for performance."""
        constrained = List[Float64]()

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
    fn test_inference_latency() raises:
        """Test neural network inference latency using official Mojo benchmark API.
        """
        print("Testing inference latency with official Mojo benchmark API...")

        # Create and initialize network with proper initialization
        network = BenchmarkNetwork(
            List[List[Float64]](),  # weights1
            List[Float64](),  # biases1
            List[List[Float64]](),  # weights2
            List[Float64](),  # biases2
            List[List[Float64]](),  # weights3
            List[Float64](),  # biases3
        )
        network.initialize_weights()

        # Prepare test input
        test_input = List[Float64]()
        test_input.append(2.0)  # la_position
        test_input.append(100.0)  # pend_velocity
        test_input.append(180.0)  # pend_position
        test_input.append(1.0)  # cmd_volts

        # Create official Mojo benchmark with proper configuration
        bench = Bench(BenchConfig())

        # Define benchmark function using official patterns
        @parameter
        @always_inline
        fn benchmark_inference_latency(mut bencher: Bencher) raises:
            """Benchmark neural network inference latency using official Mojo benchmark API.

            Args:
                bencher: Mojo benchmark iteration manager for statistical analysis.

            Raises:
                Error: If neural network forward pass fails.
            """

            @parameter
            @always_inline
            fn run_single_inference():
                """Execute single neural network inference with dead code elimination prevention.
                """
                prediction = network.forward_optimized(test_input)
                # Prevent dead code elimination (benchmark best practice)
                if len(prediction) == 0:
                    print("Unexpected empty prediction")

            bencher.iter[run_single_inference]()

        # Execute benchmark using official API
        bench.bench_function[benchmark_inference_latency](
            BenchId("neural_network", "inference_latency")
        )

        # Extract results using official benchmark API
        var avg_latency_ms: Float64 = 0.0
        for info in bench.info_vec:
            if info.name == "neural_network/inference_latency":
                avg_latency_ms = info.result.mean("ms")
                break

        print("  Official Mojo benchmark API results:")
        print("  Average latency:", avg_latency_ms, "ms")
        print("  Target latency:", TARGET_LATENCY_MS, "ms")

        # Check if meets real-time requirements
        if avg_latency_ms <= TARGET_LATENCY_MS:
            print("  ✓ Meets 25 Hz real-time requirement")
        else:
            print("  ⚠ Does not meet 25 Hz requirement")

        print(
            "✓ Inference latency test completed using official Mojo"
            " benchmark API"
        )

    @staticmethod
    fn test_throughput() raises:
        """Test system throughput using official Mojo benchmark API."""
        print("Testing system throughput with official Mojo benchmark API...")

        # Create network with proper initialization
        network = BenchmarkNetwork(
            List[List[Float64]](),  # weights1
            List[Float64](),  # biases1
            List[List[Float64]](),  # weights2
            List[Float64](),  # biases2
            List[List[Float64]](),  # weights3
            List[Float64](),  # biases3
        )
        network.initialize_weights()

        # Prepare multiple test inputs
        test_inputs = List[List[Float64]]()
        for i in range(100):
            input = List[Float64]()
            input.append(Float64(i % 10) * 0.4 - 2.0)
            input.append(Float64(i % 20) * 10.0 - 100.0)
            input.append(Float64(i % 36) * 10.0)
            input.append(Float64(i % 5) * 0.4 - 1.0)
            test_inputs.append(input)

        # Create official Mojo benchmark
        bench = Bench(BenchConfig())

        # Define benchmark function using official patterns
        @parameter
        @always_inline
        fn benchmark_throughput(mut bencher: Bencher) raises:
            """Benchmark system throughput using batch processing with official Mojo benchmark API.

            Args:
                bencher: Mojo benchmark iteration manager for statistical analysis.

            Raises:
                Error: If neural network forward pass fails during batch processing.
            """

            @parameter
            @always_inline
            fn run_batch_processing():
                """Execute batch neural network processing with dead code elimination prevention.
                """
                for i in range(len(test_inputs)):
                    prediction = network.forward_optimized(test_inputs[i])
                    # Prevent dead code elimination (benchmark best practice)
                    if len(prediction) == 0:
                        print("Unexpected empty prediction")

            bencher.iter[run_batch_processing]()

        # Execute benchmark using official API
        bench.bench_function[benchmark_throughput](
            BenchId("neural_network", "throughput")
        )

        # Extract results using official benchmark API
        var batch_time_s: Float64 = 0.0
        for info in bench.info_vec:
            if info.name == "neural_network/throughput":
                batch_time_s = info.result.mean("s")
                break

        # Calculate throughput from benchmark results
        throughput = Float64(len(test_inputs)) / batch_time_s

        print("  Official Mojo benchmark API results:")
        print("  Batch processing time:", batch_time_s, "seconds")
        print("  Throughput:", throughput, "predictions/second")
        print("  Target frequency:", TARGET_FREQUENCY_HZ, "Hz")

        # Check if meets throughput requirements
        if throughput >= TARGET_FREQUENCY_HZ:
            print("  ✓ Meets throughput requirement")
        else:
            print("  ⚠ Below target throughput")

        print("✓ Throughput test completed using official Mojo benchmark API")

    @staticmethod
    fn test_real_time_simulation() raises:
        """Test real-time control loop simulation using official Mojo benchmark API.
        """
        print(
            "Testing real-time control loop simulation with official Mojo"
            " benchmark API..."
        )

        # Create network with proper initialization
        network = BenchmarkNetwork(
            List[List[Float64]](),  # weights1
            List[Float64](),  # biases1
            List[List[Float64]](),  # weights2
            List[Float64](),  # biases2
            List[List[Float64]](),  # weights3
            List[Float64](),  # biases3
        )
        network.initialize_weights()

        # Simulation parameters (unchanged)
        control_frequency = 25.0
        simulation_duration = 1.0
        expected_cycles = Int(control_frequency * simulation_duration)

        # Initialize state (unchanged)
        current_state = List[Float64]()
        current_state.append(0.0)  # Initial position
        current_state.append(0.0)  # Initial velocity
        current_state.append(180.0)  # Initial angle
        current_state.append(0.0)  # Initial command

        # Create official Mojo benchmark
        bench = Bench(BenchConfig())

        # Define benchmark function for individual cycle timing
        @parameter
        @always_inline
        fn benchmark_simulation_cycle(mut bencher: Bencher) raises:
            """Benchmark individual simulation cycle latency using official Mojo benchmark API.

            Args:
                bencher: Mojo benchmark iteration manager for statistical analysis.

            Raises:
                Error: If neural network forward pass fails during simulation cycle.
            """

            @parameter
            @always_inline
            fn run_simulation_cycle():
                """Execute single simulation cycle with state updates and dead code elimination prevention.
                """
                prediction = network.forward_optimized(current_state)
                # Update state (simplified)
                current_state[0] = prediction[0]
                current_state[1] = prediction[1]
                current_state[2] = prediction[2]
                # Prevent dead code elimination (benchmark best practice)
                if len(prediction) == 0:
                    print("Unexpected empty prediction")

            bencher.iter[run_simulation_cycle]()

        # Execute cycle benchmark
        bench.bench_function[benchmark_simulation_cycle](
            BenchId("simulation", "cycle_latency")
        )

        # Define benchmark function for full simulation timing
        @parameter
        @always_inline
        fn benchmark_full_simulation(mut bencher: Bencher) raises:
            """Benchmark complete simulation timing using official Mojo benchmark API.

            Args:
                bencher: Mojo benchmark iteration manager for statistical analysis.

            Raises:
                Error: If neural network forward pass fails during full simulation.
            """

            @parameter
            @always_inline
            fn run_full_simulation():
                """Execute complete simulation with multiple cycles and dead code elimination prevention.
                """
                for cycle in range(expected_cycles):
                    prediction = network.forward_optimized(current_state)
                    current_state[0] = prediction[0]
                    current_state[1] = prediction[1]
                    current_state[2] = prediction[2]
                    current_state[3] = 0.1 * Float64(cycle % 10)
                    if len(prediction) == 0:
                        print("Unexpected empty prediction")

            bencher.iter[run_full_simulation]()

        # Execute full simulation benchmark
        bench.bench_function[benchmark_full_simulation](
            BenchId("simulation", "full_simulation")
        )

        # Extract results using official benchmark API
        var avg_cycle_latency_ms: Float64 = 0.0
        var simulation_time_s: Float64 = 0.0

        for info in bench.info_vec:
            if info.name == "simulation/cycle_latency":
                avg_cycle_latency_ms = info.result.mean("ms")
            elif info.name == "simulation/full_simulation":
                simulation_time_s = info.result.mean("s")

        # Calculate frequency from benchmark results
        actual_frequency = Float64(expected_cycles) / simulation_time_s

        print("  Official Mojo benchmark API results:")
        print("  Expected cycles:", expected_cycles)
        print("  Simulation time:", simulation_time_s, "seconds")
        print("  Actual frequency:", actual_frequency, "Hz")
        print("  Target frequency:", control_frequency, "Hz")
        print("  Average cycle latency:", avg_cycle_latency_ms, "ms")
        print("  Target latency:", TARGET_LATENCY_MS, "ms")

        # Check real-time performance
        if actual_frequency >= 25.0 * 0.95:  # 95% of target
            print("  ✓ Meets real-time control requirements")
        else:
            print("  ⚠ Below real-time control requirements")

        if avg_cycle_latency_ms <= TARGET_LATENCY_MS:
            print("  ✓ Latency within acceptable bounds")
        else:
            print("  ⚠ Latency exceeds acceptable bounds")

        print(
            "✓ Real-time simulation test completed using official Mojo"
            " benchmark API"
        )

    @staticmethod
    fn run_all_tests() raises:
        """Run all performance tests using official Mojo benchmark API."""
        print(
            "Running Performance Benchmark Tests with Official Mojo"
            " Benchmark API"
        )
        print(
            "===================================================================="
        )
        print("Target: 25 Hz real-time control (40ms max latency)")
        print(
            "Methodology: Official Mojo benchmark module (Bench, Bencher,"
            " BenchId, BenchConfig)"
        )
        print()

        PerformanceTests.test_inference_latency()
        print()
        PerformanceTests.test_throughput()
        print()
        PerformanceTests.test_real_time_simulation()

        print()
        print(
            "✓ All performance tests completed using official Mojo benchmark"
            " API!"
        )
        print(
            "✓ Statistical analysis through bencher.iter[] applied throughout"
        )
        print()


fn main() raises:
    """Run performance benchmark tests."""
    PerformanceTests.run_all_tests()
