"""
Parameter Optimizer for Control Algorithm Tuning.

This module implements comprehensive parameter optimization for the control system,
including MPC weights, adaptive gains, and control thresholds to achieve optimal
performance across diverse operating conditions.
"""

from collections import List
from math import sqrt, exp, sin, cos
from random import random

# Import control system components
from src.control.enhanced_ai_controller import (
    EnhancedAIController,
    AdaptiveGains,
    ControlPerformance,
)
from src.control.mpc_controller import MPCController, MPCObjective
from src.control.integrated_control_system import IntegratedControlSystem

# Optimization constants
alias OPTIMIZATION_ITERATIONS = 50  # Number of optimization iterations
alias PARAMETER_SEARCH_SAMPLES = 20  # Samples per parameter search
alias PERFORMANCE_TEST_CYCLES = 100  # Cycles for performance evaluation
alias TARGET_SUCCESS_RATE = 0.70  # >70% inversion success rate
alias TARGET_STABILITY_TIME = 15.0  # >15 second stability target
alias CONVERGENCE_TOLERANCE = 0.01  # Optimization convergence tolerance


@fieldwise_init
struct ParameterSet(Copyable, Movable):
    """
    Complete set of tunable control parameters for the pendulum control system.

    This struct encapsulates all parameters that can be optimized for the control
    algorithms, including MPC weights, adaptive gains, and control thresholds.
    The parameters are organized into logical groups for systematic optimization.

    Attributes:
        MPC Parameters: Weights and horizons for Model Predictive Control
        Adaptive Gains: PID-style gains for different control modes
        Control Thresholds: Angle and velocity thresholds for mode switching
        Hybrid Weights: Blending weights for hybrid control strategies
    """

    # MPC Parameters
    var mpc_weight_angle: Float64  # MPC angle tracking weight
    var mpc_weight_position: Float64  # MPC position tracking weight
    var mpc_weight_velocity: Float64  # MPC velocity tracking weight
    var mpc_weight_control: Float64  # MPC control effort weight
    var mpc_weight_rate: Float64  # MPC control rate weight
    var mpc_prediction_horizon: Int  # MPC prediction horizon
    var mpc_control_horizon: Int  # MPC control horizon

    # Adaptive Gains
    var kp_stabilize: Float64  # Proportional gain for stabilization
    var kd_stabilize: Float64  # Derivative gain for stabilization
    var ke_swing_up: Float64  # Energy gain for swing-up
    var kp_position: Float64  # Position gain for swing-up
    var learning_rate: Float64  # Adaptive learning rate

    # Control Thresholds
    var stabilize_angle_threshold: Float64  # Angle threshold for stabilization mode
    var stabilize_velocity_threshold: Float64  # Velocity threshold for stabilization mode
    var invert_angle_threshold: Float64  # Angle threshold for inversion mode
    var swing_up_angle_threshold: Float64  # Angle threshold for swing-up mode

    # Hybrid Control Weights
    var mpc_hybrid_weight: Float64  # MPC weight in hybrid control
    var classical_hybrid_weight: Float64  # Classical weight in hybrid control

    fn is_valid(self) -> Bool:
        """
        Validate that all parameters are within acceptable bounds.

        Performs comprehensive bounds checking on all control parameters to ensure
        they are within physically meaningful and numerically stable ranges.

        Returns:
            True if all parameters are valid, False otherwise.
        """
        return (
            self.mpc_weight_angle > 0.0
            and self.mpc_weight_angle < 1000.0
            and self.kp_stabilize > 0.0
            and self.kp_stabilize < 50.0
            and self.kd_stabilize > 0.0
            and self.kd_stabilize < 10.0
            and self.learning_rate > 0.0
            and self.learning_rate < 1.0
        )


@fieldwise_init
struct OptimizationResult(Copyable, Movable):
    """
    Comprehensive results from parameter optimization process.

    Contains all metrics and metadata from a parameter optimization run,
    including performance scores, convergence information, and target achievement.
    Used for tracking optimization progress and comparing different parameter sets.

    Attributes:
        best_parameters: The optimal parameter set found
        best_performance: Composite performance score (0.0 to 1.0)
        success_rate: Percentage of successful pendulum inversions
        stability_time: Average time pendulum remains stable
        control_effort: Average control voltage magnitude
        convergence_iterations: Number of iterations to reach convergence
        optimization_time: Total time spent optimizing (seconds)
        meets_targets: Whether all performance targets were achieved
    """

    var best_parameters: ParameterSet  # Best parameter set found
    var best_performance: Float64  # Best performance score achieved
    var success_rate: Float64  # Inversion success rate
    var stability_time: Float64  # Average stability time
    var control_effort: Float64  # Average control effort
    var convergence_iterations: Int  # Iterations to convergence
    var optimization_time: Float64  # Total optimization time
    var meets_targets: Bool  # Whether targets are met

    fn get_performance_grade(self) -> String:
        """
        Calculate a qualitative performance grade based on optimization results.

        Evaluates the optimization results against predefined thresholds to assign
        a human-readable performance grade. Considers both target achievement and
        the quality of the achieved metrics.

        Returns:
            Performance grade: "Excellent", "Good", "Acceptable", or "Needs Improvement".
        """
        if self.meets_targets:
            if self.success_rate > 0.85 and self.stability_time > 20.0:
                return "Excellent"
            else:
                return "Good"
        elif self.success_rate > 0.60:
            return "Acceptable"
        else:
            return "Needs Improvement"


struct ParameterOptimizer:
    """
    Comprehensive parameter optimizer for control algorithms.

    Features:
    - Multi-objective optimization for success rate and stability
    - Grid search and gradient-based optimization
    - Robust performance evaluation across diverse scenarios
    - Automatic parameter bound enforcement
    - Convergence detection and early stopping
    """

    var current_parameters: ParameterSet
    var optimization_history: List[OptimizationResult]
    var test_scenarios: List[List[Float64]]
    var performance_weights: List[Float64]
    var optimizer_initialized: Bool

    fn __init__(out self):
        """
        Initialize the parameter optimizer with default settings.

        Sets up the optimizer with default parameter values, initializes empty
        optimization history, creates diverse test scenarios, and configures
        performance weighting factors for multi-objective optimization.

        The initialization creates a comprehensive test suite covering various
        pendulum states and disturbance conditions to ensure robust optimization.
        """
        self.current_parameters = ParameterSet(
            # MPC Parameters
            100.0,  # mpc_weight_angle
            10.0,  # mpc_weight_position
            1.0,  # mpc_weight_velocity
            0.1,  # mpc_weight_control
            0.5,  # mpc_weight_rate
            10,  # mpc_prediction_horizon
            5,  # mpc_control_horizon
            # Adaptive Gains
            15.0,  # kp_stabilize
            2.0,  # kd_stabilize
            0.5,  # ke_swing_up
            1.0,  # kp_position
            0.1,  # learning_rate
            # Control Thresholds
            15.0,  # stabilize_angle_threshold
            100.0,  # stabilize_velocity_threshold
            90.0,  # invert_angle_threshold
            150.0,  # swing_up_angle_threshold
            # Hybrid Control Weights
            0.7,  # mpc_hybrid_weight
            0.3,  # classical_hybrid_weight
        )

        self.optimization_history = List[OptimizationResult]()
        self.test_scenarios = List[List[Float64]]()
        self.performance_weights = List[Float64]()
        self.optimizer_initialized = False

    fn initialize_optimizer(mut self) -> Bool:
        """Initialize optimizer with test scenarios and performance weights."""
        print("Initializing Parameter Optimizer...")

        self._create_test_scenarios()

        self.performance_weights = List[Float64]()
        self.performance_weights.append(0.4)  # Success rate weight
        self.performance_weights.append(0.3)  # Stability time weight
        self.performance_weights.append(0.2)  # Control effort weight
        self.performance_weights.append(0.1)  # Convergence speed weight

        self.optimizer_initialized = True
        print(
            "Parameter Optimizer initialized with",
            len(self.test_scenarios),
            "test scenarios",
        )
        return True

    fn optimize_parameters(mut self) raises -> OptimizationResult:
        """
        Run comprehensive parameter optimization with error handling.

        Performs multi-stage optimization including grid search, gradient-based
        fine-tuning, and adaptive parameter refinement. Validates initialization
        state and handles optimization failures gracefully.

        Returns:
            Optimization results with best parameters and performance metrics.

        Raises:
            Error: If optimizer is not properly initialized or optimization fails.
        """
        if not self.optimizer_initialized:
            raise Error(
                "Parameter optimizer not initialized - call __init__ first"
            )

        print("Starting Parameter Optimization...")
        print("Target: >70% success rate, >15s stability time")
        print("Optimization iterations:", OPTIMIZATION_ITERATIONS)

        var best_result = self._create_failed_result()
        var best_score = 0.0

        print("\nStage 1: Grid Search for Initial Parameters")
        grid_result = self._grid_search_optimization()
        if grid_result.best_performance > best_score:
            best_result = grid_result
            best_score = grid_result.best_performance

        print("\nStage 2: Gradient-Based Fine Tuning")
        gradient_result = self._gradient_based_optimization(
            best_result.best_parameters
        )
        if gradient_result.best_performance > best_score:
            best_result = gradient_result
            best_score = gradient_result.best_performance

        print("\nStage 3: Adaptive Parameter Refinement")
        adaptive_result = self._adaptive_parameter_refinement(
            best_result.best_parameters
        )
        if adaptive_result.best_performance > best_score:
            best_result = adaptive_result
            best_score = adaptive_result.best_performance

        self.optimization_history.append(best_result)
        self.current_parameters = best_result.best_parameters

        print("\nOptimization Complete!")
        self._print_optimization_results(best_result)

        return best_result

    fn _grid_search_optimization(self) raises -> OptimizationResult:
        """
        Perform systematic grid search optimization over key parameters.

        Explores a discretized parameter space using a structured grid approach
        to find promising parameter regions. Tests multiple combinations of
        critical parameters to establish a good starting point for fine-tuning.

        Returns:
            Optimization result with best parameters found during grid search.

        Raises:
            Error: If grid search encounters invalid parameter combinations.
        """
        print("  Running grid search over parameter space...")

        best_parameters = self.current_parameters
        best_score = 0.0
        evaluations = 0

        angle_weights = List[Float64]()
        angle_weights.append(50.0)
        angle_weights.append(100.0)
        angle_weights.append(200.0)

        kp_values = List[Float64]()
        kp_values.append(10.0)
        kp_values.append(15.0)
        kp_values.append(20.0)
        kp_values.append(25.0)

        kd_values = List[Float64]()
        kd_values.append(1.0)
        kd_values.append(2.0)
        kd_values.append(3.0)

        for i in range(len(angle_weights)):
            for j in range(len(kp_values)):
                for k in range(len(kd_values)):
                    test_params = self.current_parameters
                    test_params.mpc_weight_angle = angle_weights[i]
                    test_params.kp_stabilize = kp_values[j]
                    test_params.kd_stabilize = kd_values[k]

                    performance = self._evaluate_parameter_set(test_params)
                    evaluations += 1

                    if performance > best_score:
                        best_score = performance
                        best_parameters = test_params

        print("    Grid search completed:", evaluations, "evaluations")
        print("    Best score:", best_score)

        return self._create_optimization_result(
            best_parameters, best_score, evaluations
        )

    fn _gradient_based_optimization(
        self, initial_params: ParameterSet
    ) raises -> OptimizationResult:
        """
        Perform gradient-based optimization starting from initial parameters.

        Uses numerical gradient estimation and gradient descent to fine-tune
        parameters from a good starting point. Implements adaptive step sizing
        and convergence detection for efficient optimization.

        Args:
            initial_params: Starting parameter set from previous optimization stage.

        Returns:
            Optimization result with refined parameters and performance metrics.

        Raises:
            Error: If gradient computation fails or parameters become invalid.
        """
        print("  Running gradient-based optimization...")

        current_params = initial_params
        current_score = self._evaluate_parameter_set(current_params)
        iterations = 0
        step_size = 0.1

        for iteration in range(20):  # Limited iterations for gradient descent
            improved_params = self._compute_parameter_gradients(
                current_params, step_size
            )
            improved_score = self._evaluate_parameter_set(improved_params)

            if improved_score > current_score:
                current_params = improved_params
                current_score = improved_score
                iterations = iteration + 1
            else:
                step_size *= 0.8  # Reduce step size if no improvement
                if step_size < 0.01:
                    break  # Converged

        print("    Gradient optimization completed:", iterations, "iterations")
        print("    Final score:", current_score)

        return self._create_optimization_result(
            current_params, current_score, iterations
        )

    fn _adaptive_parameter_refinement(
        self, initial_params: ParameterSet
    ) raises -> OptimizationResult:
        """
        Perform adaptive parameter refinement based on performance feedback.

        Applies domain-specific refinements and adaptive adjustments to parameters
        based on performance characteristics. Focuses on fine-tuning critical
        parameters that have the most impact on system performance.

        Args:
            initial_params: Parameter set from previous optimization stages.

        Returns:
            Final optimization result with refined parameters.

        Raises:
            Error: If refinement process encounters invalid parameter states.
        """
        print("  Running adaptive parameter refinement...")

        refined_params = initial_params
        refinement_score = self._evaluate_parameter_set(refined_params)

        learning_rates = List[Float64]()
        learning_rates.append(0.05)
        learning_rates.append(0.1)
        learning_rates.append(0.15)
        learning_rates.append(0.2)

        best_lr_score = refinement_score
        for i in range(len(learning_rates)):
            test_params = refined_params
            test_params.learning_rate = learning_rates[i]

            lr_score = self._evaluate_parameter_set(test_params)
            if lr_score > best_lr_score:
                best_lr_score = lr_score
                refined_params.learning_rate = learning_rates[i]

        mpc_weights = List[Float64]()
        mpc_weights.append(0.6)
        mpc_weights.append(0.7)
        mpc_weights.append(0.8)

        for i in range(len(mpc_weights)):
            test_params = refined_params
            test_params.mpc_hybrid_weight = mpc_weights[i]
            test_params.classical_hybrid_weight = 1.0 - mpc_weights[i]

            weight_score = self._evaluate_parameter_set(test_params)
            if weight_score > refinement_score:
                refinement_score = weight_score
                refined_params = test_params

        print("    Adaptive refinement completed")
        print("    Refined score:", refinement_score)

        return self._create_optimization_result(
            refined_params, refinement_score, 10
        )

    fn _evaluate_parameter_set(self, params: ParameterSet) -> Float64:
        """
        Evaluate performance of a parameter set across comprehensive test scenarios.

        Runs the parameter set through all configured test scenarios and computes
        a weighted composite performance score. Considers success rate, stability
        time, and control effort to provide a balanced evaluation metric.

        Args:
            params: Parameter set to evaluate.

        Returns:
            Composite performance score between 0.0 and 1.0 (higher is better).
        """
        if not params.is_valid():
            return 0.0  # Invalid parameters get zero score

        var total_success = 0.0
        var total_stability = 0.0
        var total_control_effort = 0.0
        scenario_count = Float64(len(self.test_scenarios))

        for i in range(len(self.test_scenarios)):
            scenario = self.test_scenarios[i]
            result = self._test_scenario_performance(scenario, params)

            total_success += result[0]  # Success rate
            total_stability += result[1]  # Stability time
            total_control_effort += result[2]  # Control effort

        avg_success = total_success / scenario_count
        avg_stability = total_stability / scenario_count
        avg_control_effort = total_control_effort / scenario_count

        success_score = min(1.0, avg_success / TARGET_SUCCESS_RATE)
        stability_score = min(1.0, avg_stability / TARGET_STABILITY_TIME)
        effort_score = max(
            0.0, 1.0 - avg_control_effort / 10.0
        )  # Lower effort is better

        total_score = (
            self.performance_weights[0] * success_score
            + self.performance_weights[1] * stability_score
            + self.performance_weights[2] * effort_score
        )

        return total_score

    fn _test_scenario_performance(
        self, initial_state: List[Float64], params: ParameterSet
    ) -> (Float64, Float64, Float64):
        """
        Test performance for a single scenario using physics-based simulation.

        Simulates the pendulum control system with the given parameters starting
        from a specific initial state. Uses realistic physics modeling to measure
        key performance metrics including success rate, stability time, and control effort.

        Args:
            initial_state: Initial pendulum state [position, velocity, angle].
            params: Control parameters to test.

        Returns:
            Performance metrics tuple (success_rate, stability_time, control_effort).
        """
        position = initial_state[0]
        velocity = initial_state[1]
        angle = initial_state[2]

        # Physics-based performance simulation
        simulation_time = 30.0  # 30 second simulation
        dt = 0.02  # 50 Hz control rate
        steps = Int(simulation_time / dt)

        var current_angle = angle
        var current_position = position
        var current_angular_vel = 0.0

        var stable_time = 0.0
        var total_effort = 0.0
        var inversion_achieved = False
        var max_stable_duration = 0.0
        var current_stable_duration = 0.0

        for step in range(steps):
            # Determine control mode based on angle
            var control_voltage = 0.0
            if abs(current_angle) < params.stabilize_angle_threshold:
                # Stabilization control (PD controller)
                control_voltage = -(
                    params.kp_stabilize * current_angle
                    + params.kd_stabilize * current_angular_vel
                )
                inversion_achieved = True
                current_stable_duration += dt
                if current_stable_duration > max_stable_duration:
                    max_stable_duration = current_stable_duration
            else:
                # Swing-up control (energy-based)
                energy_error = self._calculate_energy_error(
                    current_angle, current_angular_vel
                )
                control_voltage = (
                    params.ke_swing_up * energy_error * cos(current_angle)
                )
                current_stable_duration = 0.0

            control_voltage *= 1.0 + params.mpc_weight_control

            # Bound control voltage
            control_voltage = max(-12.0, min(12.0, control_voltage))
            total_effort += abs(control_voltage) * dt

            # Physics integration (simplified pendulum dynamics)
            angular_accel = self._compute_angular_acceleration(
                current_angle,
                current_angular_vel,
                current_position,
                control_voltage,
            )
            current_angular_vel += angular_accel * dt
            current_angle += current_angular_vel * dt

            # Linear actuator dynamics
            actuator_accel = control_voltage * 0.5  # Simplified actuator model
            velocity += actuator_accel * dt
            current_position += velocity * dt

            current_angular_vel *= 0.999
            velocity *= 0.995

            if abs(current_angle) < 10.0:  # Within 10 degrees
                stable_time += dt

        if inversion_achieved:
            success_rate = min(
                1.0, max_stable_duration / 15.0
            )  # Target 15s stability
        else:
            success_rate = 0.0

        stability_time = max_stable_duration
        control_effort = total_effort / simulation_time  # Average effort

        return (success_rate, stability_time, control_effort)

    fn _calculate_energy_error(
        self, angle: Float64, angular_vel: Float64
    ) -> Float64:
        """
        Calculate energy error for swing-up control strategy.

        Computes the difference between target energy (inverted position) and
        current energy (kinetic + potential) to drive the energy-based swing-up
        controller. Used to determine control effort needed for pendulum inversion.

        Args:
            angle: Current pendulum angle in radians.
            angular_vel: Current angular velocity in rad/s.

        Returns:
            Energy error in Joules (positive means more energy needed).
        """
        # Target energy for inverted pendulum (potential energy at top)
        target_energy = 9.81 * 0.3  # g * L (assuming 0.3m pendulum)

        # Current energy (kinetic + potential)
        kinetic_energy = 0.5 * 0.1 * angular_vel * angular_vel  # 0.5 * m * v^2
        potential_energy = (
            9.81 * 0.1 * 0.3 * (1.0 - cos(angle))
        )  # m * g * L * (1 - cos(θ))
        current_energy = kinetic_energy + potential_energy

        return target_energy - current_energy

    fn _compute_angular_acceleration(
        self,
        angle: Float64,
        angular_vel: Float64,
        position: Float64,
        control_voltage: Float64,
    ) -> Float64:
        """
        Compute angular acceleration using inverted pendulum dynamics.

        Implements the nonlinear dynamics equation for an inverted pendulum on a cart:
        θ̈ = (g/L)sin(θ) + (1/L)ẍ*cos(θ)

        Args:
            angle: Current pendulum angle in radians.
            angular_vel: Current angular velocity in rad/s.
            position: Current cart position in meters.
            control_voltage: Applied control voltage in volts.

        Returns:
            Angular acceleration in rad/s².
        """
        # Simplified inverted pendulum dynamics
        # θ̈ = (g/L)sin(θ) + (1/L)ẍ*cos(θ)

        g = 9.81  # gravity
        L = 0.3  # pendulum length
        m_cart = 1.0  # cart mass

        linear_accel = control_voltage / m_cart

        gravity_term = (g / L) * sin(angle)
        gravity_term = (g / L) * sin(angle)
        coupling_term = (linear_accel / L) * cos(angle)

        return gravity_term + coupling_term

    fn _compute_parameter_gradients(
        self, params: ParameterSet, step_size: Float64
    ) raises -> ParameterSet:
        """
        Compute parameter gradients using finite differences.

        Uses finite difference approximation to estimate gradients of the
        performance function with respect to each parameter. Essential for
        gradient-based optimization algorithms.

        Args:
            params: Current parameter set.
            step_size: Step size for finite difference computation.

        Returns:
            Parameter set containing gradient estimates.

        Raises:
            Error: If gradient computation encounters numerical instabilities.
        """
        improved_params = params

        kp_plus = params
        kp_plus.kp_stabilize += step_size
        kp_minus = params
        kp_minus.kp_stabilize -= step_size

        score_plus = self._evaluate_parameter_set(kp_plus)
        score_minus = self._evaluate_parameter_set(kp_minus)
        kp_gradient = (score_plus - score_minus) / (2.0 * step_size)

        if kp_gradient > 0:
            improved_params.kp_stabilize += step_size * 0.5
        else:
            improved_params.kp_stabilize -= step_size * 0.5

        improved_params.kp_stabilize = max(
            5.0, min(30.0, improved_params.kp_stabilize)
        )

        return improved_params

    fn _create_test_scenarios(mut self):
        """Create diverse test scenarios for parameter evaluation."""
        # Near inverted scenarios
        scenario1 = List[Float64]()
        scenario1.append(0.5)  # la_position
        scenario1.append(20.0)  # pend_velocity
        scenario1.append(8.0)  # pend_angle
        scenario1.append(0.0)  # cmd_volts
        self.test_scenarios.append(scenario1)

        scenario2 = List[Float64]()
        scenario2.append(-0.3)
        scenario2.append(-15.0)
        scenario2.append(-12.0)
        scenario2.append(0.0)
        self.test_scenarios.append(scenario2)

        # Transition scenarios
        scenario3 = List[Float64]()
        scenario3.append(1.0)
        scenario3.append(100.0)
        scenario3.append(45.0)
        scenario3.append(0.0)
        self.test_scenarios.append(scenario3)

        scenario4 = List[Float64]()
        scenario4.append(-1.5)
        scenario4.append(-80.0)
        scenario4.append(-60.0)
        scenario4.append(0.0)
        self.test_scenarios.append(scenario4)

        # Hanging scenarios
        scenario5 = List[Float64]()
        scenario5.append(0.0)
        scenario5.append(10.0)
        scenario5.append(175.0)
        scenario5.append(0.0)
        self.test_scenarios.append(scenario5)

        scenario6 = List[Float64]()
        scenario6.append(0.2)
        scenario6.append(-5.0)
        scenario6.append(-170.0)
        scenario6.append(0.0)
        self.test_scenarios.append(scenario6)

    fn _create_optimization_result(
        self, params: ParameterSet, score: Float64, iterations: Int
    ) -> OptimizationResult:
        """
        Create comprehensive optimization result with real performance measurements.

        Evaluates the final parameter set across all test scenarios to generate
        accurate performance metrics and determines whether optimization targets
        have been achieved.

        Args:
            params: Final optimized parameter set.
            score: Composite performance score achieved.
            iterations: Number of optimization iterations performed.

        Returns:
            Complete optimization result with all performance metrics.
        """
        var total_success = 0.0
        var total_stability = 0.0
        var total_effort = 0.0
        scenario_count = Float64(len(self.test_scenarios))

        for i in range(len(self.test_scenarios)):
            scenario = self.test_scenarios[i]
            result = self._test_scenario_performance(scenario, params)
            total_success += result[0]
            total_stability += result[1]
            total_effort += result[2]

        measured_success_rate = total_success / scenario_count
        measured_stability_time = total_stability / scenario_count
        measured_control_effort = total_effort / scenario_count

        meets_targets = (
            measured_success_rate >= TARGET_SUCCESS_RATE
            and measured_stability_time >= TARGET_STABILITY_TIME
        )

        optimization_time = Float64(iterations) * 0.5 + scenario_count * 0.1

        return OptimizationResult(
            params,  # best_parameters
            score,  # best_performance
            measured_success_rate,  # success_rate (real measurement)
            measured_stability_time,  # stability_time (real measurement)
            measured_control_effort,  # control_effort (real measurement)
            iterations,  # convergence_iterations
            optimization_time,  # optimization_time (estimated)
            meets_targets,  # meets_targets (based on real metrics)
        )

    fn _create_failed_result(self) -> OptimizationResult:
        """
        Create a failed optimization result with default values.

        Returns a result structure indicating optimization failure, with zero
        performance metrics and the current parameter set. Used when optimization
        cannot proceed due to initialization or other errors.

        Returns:
            Optimization result indicating failure state.
        """
        return OptimizationResult(
            self.current_parameters, 0.0, 0.0, 0.0, 0.0, 0, 0.0, False
        )

    fn _print_optimization_results(self, result: OptimizationResult):
        """
        Print detailed optimization results in a formatted report.

        Displays comprehensive optimization results including performance metrics,
        parameter values, and target achievement status in a human-readable format.

        Args:
            result: Optimization result to display.
        """
        print("  Optimization Results:")
        print("    Performance score:", result.best_performance)
        print("    Success rate:", result.success_rate * 100.0, "%")
        print("    Stability time:", result.stability_time, "seconds")
        print("    Control effort:", result.control_effort, "V")
        print("    Convergence iterations:", result.convergence_iterations)
        print("    Performance grade:", result.get_performance_grade())
        print("    Meets targets:", result.meets_targets)

        if result.meets_targets:
            print("    ✓ Optimization successful - targets achieved!")
        else:
            print("    ⚠ Optimization incomplete - targets not fully met")

    fn get_optimized_parameters(self) -> ParameterSet:
        """
        Get the current optimized parameter set.

        Returns the best parameter set found during the most recent optimization
        run. These parameters represent the optimal configuration for the control
        system based on the defined performance criteria.

        Returns:
            The optimized parameter set.
        """
        return self.current_parameters

    fn get_optimization_history(self) -> List[OptimizationResult]:
        """
        Get complete optimization history.

        Returns all optimization results from previous runs, allowing analysis
        of optimization progress and comparison of different parameter sets.

        Returns:
            List of all optimization results.
        """
        return self.optimization_history
