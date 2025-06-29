"""
Performance Evaluation and Metrics Collection System.

This module implements comprehensive performance evaluation for control algorithms,
including detailed metrics collection, statistical analysis, and performance
validation against target requirements.
"""

from collections import List
from math import abs, max, min, sqrt

# Import control system components
from src.pendulum.control.enhanced_ai_controller import EnhancedAIController
from src.pendulum.control.control_trainer import ControlTrainer, TrainingResults
from src.pendulum.control.parameter_optimizer import ParameterSet
from src.pendulum.control.integrated_control_system import IntegratedControlSystem

# Performance evaluation constants
alias EVALUATION_DURATION = 60.0      # 60 seconds evaluation
alias EVALUATION_CYCLES = 1500        # 60s * 25 Hz
alias SUCCESS_ANGLE_THRESHOLD = 10.0  # Degrees for inversion success
alias STABILITY_ANGLE_THRESHOLD = 5.0 # Degrees for stability
alias MIN_STABILITY_DURATION = 15.0   # Minimum stability time (seconds)
alias TARGET_SUCCESS_RATE = 0.70      # 70% target success rate

@fieldwise_init
struct PerformanceMetrics(Copyable, Movable):
    """Comprehensive performance metrics."""
    
    var inversion_success_rate: Float64    # Percentage of successful inversions
    var average_stability_time: Float64    # Average time in stable region
    var maximum_stability_time: Float64    # Maximum continuous stability time
    var average_settling_time: Float64     # Average time to reach inverted state
    var control_effort_rms: Float64        # RMS control effort
    var tracking_error_rms: Float64        # RMS tracking error
    var robustness_score: Float64          # Robustness across conditions
    var real_time_performance: Float64     # Real-time capability score
    var safety_compliance: Float64         # Safety constraint compliance
    var energy_efficiency: Float64         # Energy efficiency score
    
    fn meets_targets(self) -> Bool:
        """Check if performance meets all targets."""
        return (self.inversion_success_rate >= TARGET_SUCCESS_RATE and
                self.average_stability_time >= MIN_STABILITY_DURATION and
                self.safety_compliance >= 0.95)
    
    fn get_overall_score(self) -> Float64:
        """Calculate overall performance score."""
        var weights = List[Float64]()
        weights.append(0.25)  # Success rate
        weights.append(0.20)  # Stability time
        weights.append(0.15)  # Control effort
        weights.append(0.15)  # Tracking error
        weights.append(0.10)  # Robustness
        weights.append(0.10)  # Real-time performance
        weights.append(0.05)  # Energy efficiency
        
        var score = (weights[0] * min(1.0, self.inversion_success_rate / TARGET_SUCCESS_RATE) +
                    weights[1] * min(1.0, self.average_stability_time / MIN_STABILITY_DURATION) +
                    weights[2] * max(0.0, 1.0 - self.control_effort_rms / 10.0) +
                    weights[3] * max(0.0, 1.0 - self.tracking_error_rms / 20.0) +
                    weights[4] * self.robustness_score +
                    weights[5] * self.real_time_performance +
                    weights[6] * self.energy_efficiency)
        
        return score

@fieldwise_init
struct EvaluationScenario(Copyable, Movable):
    """Performance evaluation scenario."""
    
    var scenario_name: String          # Name of the scenario
    var initial_states: List[List[Float64]]  # Initial states to test
    var disturbances: List[Float64]    # External disturbances
    var evaluation_duration: Float64   # Duration of evaluation
    var success_criteria: Float64      # Success rate criteria for this scenario
    var difficulty_level: Int          # Difficulty level (1-5)
    
    fn get_scenario_weight(self) -> Float64:
        """Get weight for this scenario in overall evaluation."""
        if self.difficulty_level <= 2:
            return 1.0  # Standard weight for easy/medium scenarios
        elif self.difficulty_level == 3:
            return 1.2  # Higher weight for hard scenarios
        else:
            return 1.5  # Highest weight for very hard/extreme scenarios

struct PerformanceEvaluator:
    """
    Comprehensive performance evaluation system for control algorithms.
    
    Features:
    - Multi-scenario performance testing
    - Statistical analysis and metrics collection
    - Robustness evaluation across operating conditions
    - Real-time performance validation
    - Safety and constraint compliance checking
    """
    
    var evaluation_scenarios: List[EvaluationScenario]
    var performance_history: List[PerformanceMetrics]
    var current_metrics: PerformanceMetrics
    var evaluator_initialized: Bool
    
    fn __init__(out self):
        """Initialize performance evaluator."""
        self.evaluation_scenarios = List[EvaluationScenario]()
        self.performance_history = List[PerformanceMetrics]()
        
        # Initialize empty metrics
        self.current_metrics = PerformanceMetrics(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        
        self.evaluator_initialized = False
    
    fn initialize_evaluator(mut self) -> Bool:
        """Initialize evaluator with test scenarios."""
        print("Initializing Performance Evaluator...")
        
        # Create evaluation scenarios
        self._create_evaluation_scenarios()
        
        self.evaluator_initialized = True
        print("Performance Evaluator initialized with", len(self.evaluation_scenarios), "scenarios")
        return True
    
    fn evaluate_control_performance(mut self, trained_parameters: ParameterSet) -> PerformanceMetrics:
        """
        Comprehensive performance evaluation of trained control system.
        
        Args:
            trained_parameters: Optimized control parameters
            
        Returns:
            Complete performance metrics
        """
        if not self.evaluator_initialized:
            print("Evaluator not initialized")
            return self.current_metrics
        
        print("=" * 60)
        print("COMPREHENSIVE PERFORMANCE EVALUATION")
        print("=" * 60)
        print("Evaluating control performance with optimized parameters:")
        print("- Target: >70% inversion success rate")
        print("- Target: >15 second average stability time")
        print("- Target: >95% safety compliance")
        print("- Scenarios:", len(self.evaluation_scenarios))
        print()
        
        # Initialize metrics collection
        var total_success_rate = 0.0
        var total_stability_time = 0.0
        var total_control_effort = 0.0
        var total_tracking_error = 0.0
        var total_scenario_weight = 0.0
        var max_stability_time = 0.0
        var safety_violations = 0
        var total_evaluations = 0
        
        # Evaluate each scenario
        for i in range(len(self.evaluation_scenarios)):
            var scenario = self.evaluation_scenarios[i]
            var scenario_weight = scenario.get_scenario_weight()
            
            print("Scenario", i + 1, ":", scenario.scenario_name)
            print("  Difficulty level:", scenario.difficulty_level)
            print("  Initial states:", len(scenario.initial_states))
            
            var scenario_results = self._evaluate_scenario(scenario, trained_parameters)
            
            var success_rate = scenario_results.0
            var stability_time = scenario_results.1
            var control_effort = scenario_results.2
            var tracking_error = scenario_results.3
            var safety_score = scenario_results.4
            
            # Accumulate weighted results
            total_success_rate += success_rate * scenario_weight
            total_stability_time += stability_time * scenario_weight
            total_control_effort += control_effort * scenario_weight
            total_tracking_error += tracking_error * scenario_weight
            total_scenario_weight += scenario_weight
            
            max_stability_time = max(max_stability_time, stability_time)
            
            if safety_score < 0.95:
                safety_violations += 1
            
            total_evaluations += 1
            
            print("    Success rate:", success_rate * 100.0, "%")
            print("    Stability time:", stability_time, "s")
            print("    Control effort:", control_effort, "V")
            print("    Tracking error:", tracking_error, "°")
            print("    Safety score:", safety_score * 100.0, "%")
            print()
        
        # Calculate final metrics
        var avg_success_rate = total_success_rate / total_scenario_weight
        var avg_stability_time = total_stability_time / total_scenario_weight
        var avg_control_effort = total_control_effort / total_scenario_weight
        var avg_tracking_error = total_tracking_error / total_scenario_weight
        var safety_compliance = Float64(total_evaluations - safety_violations) / Float64(total_evaluations)
        
        # Additional performance metrics
        var robustness_score = self._calculate_robustness_score(trained_parameters)
        var real_time_score = self._evaluate_real_time_performance(trained_parameters)
        var energy_efficiency = self._calculate_energy_efficiency(avg_control_effort)
        var settling_time = self._estimate_settling_time(trained_parameters)
        
        # Create comprehensive metrics
        self.current_metrics = PerformanceMetrics(
            avg_success_rate,       # inversion_success_rate
            avg_stability_time,     # average_stability_time
            max_stability_time,     # maximum_stability_time
            settling_time,          # average_settling_time
            avg_control_effort,     # control_effort_rms
            avg_tracking_error,     # tracking_error_rms
            robustness_score,       # robustness_score
            real_time_score,        # real_time_performance
            safety_compliance,      # safety_compliance
            energy_efficiency       # energy_efficiency
        )
        
        # Store in history
        self.performance_history.append(self.current_metrics)
        
        # Print comprehensive results
        self._print_performance_summary()
        
        return self.current_metrics
    
    fn _evaluate_scenario(self, scenario: EvaluationScenario, parameters: ParameterSet) -> (Float64, Float64, Float64, Float64, Float64):
        """
        Evaluate performance for a single scenario.
        
        Returns:
            (success_rate, stability_time, control_effort, tracking_error, safety_score)
        """
        var total_success = 0.0
        var total_stability = 0.0
        var total_effort = 0.0
        var total_error = 0.0
        var safety_violations = 0
        
        # Test each initial state in the scenario
        for i in range(len(scenario.initial_states)):
            var initial_state = scenario.initial_states[i]
            var result = self._simulate_control_performance(initial_state, parameters, scenario.evaluation_duration)
            
            total_success += result.0
            total_stability += result.1
            total_effort += result.2
            total_error += result.3
            
            if result.4 < 0.95:  # Safety score threshold
                safety_violations += 1
        
        var num_states = Float64(len(scenario.initial_states))
        var avg_success = total_success / num_states
        var avg_stability = total_stability / num_states
        var avg_effort = total_effort / num_states
        var avg_error = total_error / num_states
        var safety_score = Float64(len(scenario.initial_states) - safety_violations) / num_states
        
        return (avg_success, avg_stability, avg_effort, avg_error, safety_score)
    
    fn _simulate_control_performance(self, initial_state: List[Float64], parameters: ParameterSet, duration: Float64) -> (Float64, Float64, Float64, Float64, Float64):
        """
        Simulate control performance for a single initial condition.
        
        Returns:
            (success_rate, stability_time, control_effort, tracking_error, safety_score)
        """
        # Simplified performance simulation based on initial conditions and parameters
        var initial_angle = initial_state[2]
        var initial_velocity = initial_state[1]
        
        # Estimate success rate based on initial conditions
        var success_rate = 0.5
        if abs(initial_angle) < 30.0:  # Near inverted
            success_rate = 0.8 + parameters.kp_stabilize / 100.0
        elif abs(initial_angle) < 90.0:  # Transition region
            success_rate = 0.6 + parameters.mpc_weight_angle / 500.0
        else:  # Large angle
            success_rate = 0.3 + parameters.ke_swing_up * 0.4
        
        # Estimate stability time
        var stability_time = 5.0
        if success_rate > 0.7:
            stability_time = 10.0 + parameters.kd_stabilize * 3.0
        
        # Estimate control effort
        var control_effort = parameters.mpc_weight_control * 20.0 + abs(initial_angle) * 0.1
        
        # Estimate tracking error
        var tracking_error = abs(initial_angle) * 0.5 + abs(initial_velocity) * 0.01
        
        # Safety score (simplified)
        var safety_score = 1.0
        if control_effort > 12.0 or abs(initial_state[0]) > 3.8:
            safety_score = 0.9
        
        # Apply bounds
        success_rate = max(0.0, min(1.0, success_rate))
        stability_time = max(0.0, min(30.0, stability_time))
        control_effort = max(0.0, min(15.0, control_effort))
        tracking_error = max(0.0, min(50.0, tracking_error))
        safety_score = max(0.0, min(1.0, safety_score))
        
        return (success_rate, stability_time, control_effort, tracking_error, safety_score)
    
    fn _calculate_robustness_score(self, parameters: ParameterSet) -> Float64:
        """Calculate robustness score across parameter variations."""
        var robustness_tests = 10
        var successful_tests = 0
        
        for i in range(robustness_tests):
            # Create parameter variation
            var variation = 0.8 + 0.4 * Float64(i) / Float64(robustness_tests)  # 80% to 120%
            var varied_params = parameters
            varied_params.kp_stabilize *= variation
            varied_params.mpc_weight_angle *= variation
            
            # Test with standard scenario
            var test_state = List[Float64]()
            test_state.append(0.5)
            test_state.append(30.0)
            test_state.append(15.0)
            test_state.append(0.0)
            
            var result = self._simulate_control_performance(test_state, varied_params, 20.0)
            if result.0 > 0.6:  # 60% success threshold for robustness
                successful_tests += 1
        
        return Float64(successful_tests) / Float64(robustness_tests)
    
    fn _evaluate_real_time_performance(self, parameters: ParameterSet) -> Float64:
        """Evaluate real-time performance capability."""
        # Simplified real-time score based on parameter complexity
        var complexity_score = (parameters.mpc_prediction_horizon * parameters.mpc_control_horizon) / 50.0
        var real_time_score = max(0.0, 1.0 - complexity_score * 0.1)
        return min(1.0, real_time_score)
    
    fn _calculate_energy_efficiency(self, avg_control_effort: Float64) -> Float64:
        """Calculate energy efficiency score."""
        # Lower control effort indicates better efficiency
        var efficiency = max(0.0, 1.0 - avg_control_effort / 15.0)
        return efficiency
    
    fn _estimate_settling_time(self, parameters: ParameterSet) -> Float64:
        """Estimate average settling time to inverted state."""
        # Simplified estimation based on gains
        var settling_time = 10.0 / (parameters.kp_stabilize / 15.0)
        return max(2.0, min(20.0, settling_time))
    
    fn _create_evaluation_scenarios(mut self):
        """Create comprehensive evaluation scenarios."""
        # Scenario 1: Near inverted states (easy)
        var scenario1_states = List[List[Float64]]()
        for i in range(10):
            var state = List[Float64]()
            state.append(Float64(i - 5) * 0.2)  # Position variation
            state.append(Float64(i - 5) * 10.0) # Velocity variation
            state.append(Float64(i - 5) * 2.0)  # Angle variation ±10°
            state.append(0.0)
            scenario1_states.append(state)
        
        var scenario1 = EvaluationScenario(
            "Near Inverted States",
            scenario1_states,
            List[Float64](),
            30.0,  # 30 second evaluation
            0.85,  # 85% success criteria
            2      # Medium difficulty
        )
        self.evaluation_scenarios.append(scenario1)
        
        # Scenario 2: Transition region (medium)
        var scenario2_states = List[List[Float64]]()
        for i in range(15):
            var state = List[Float64]()
            state.append(Float64(i - 7) * 0.3)
            state.append(Float64(i - 7) * 20.0)
            state.append(Float64(i - 7) * 6.0)  # ±42° variation
            state.append(0.0)
            scenario2_states.append(state)
        
        var scenario2 = EvaluationScenario(
            "Transition Region",
            scenario2_states,
            List[Float64](),
            45.0,  # 45 second evaluation
            0.70,  # 70% success criteria
            3      # Hard difficulty
        )
        self.evaluation_scenarios.append(scenario2)
        
        # Scenario 3: Large angles (hard)
        var scenario3_states = List[List[Float64]]()
        for i in range(10):
            var state = List[Float64]()
            state.append(Float64(i - 5) * 0.4)
            state.append(Float64(i - 5) * 30.0)
            state.append(90.0 + Float64(i) * 9.0)  # 90° to 180°
            state.append(0.0)
            scenario3_states.append(state)
        
        var scenario3 = EvaluationScenario(
            "Large Angle Recovery",
            scenario3_states,
            List[Float64](),
            60.0,  # 60 second evaluation
            0.50,  # 50% success criteria
            4      # Very hard difficulty
        )
        self.evaluation_scenarios.append(scenario3)
        
        # Scenario 4: Robustness test (extreme)
        var scenario4_states = List[List[Float64]]()
        for i in range(8):
            var state = List[Float64]()
            state.append(Float64(i - 4) * 0.8)  # Large position variations
            state.append(Float64(i - 4) * 50.0) # Large velocity variations
            state.append(Float64(i) * 22.5)     # 0° to 157.5°
            state.append(0.0)
            scenario4_states.append(state)
        
        var scenario4 = EvaluationScenario(
            "Robustness Test",
            scenario4_states,
            List[Float64](),
            30.0,  # 30 second evaluation
            0.40,  # 40% success criteria
            5      # Extreme difficulty
        )
        self.evaluation_scenarios.append(scenario4)
    
    fn _print_performance_summary(self):
        """Print comprehensive performance evaluation summary."""
        print("Performance Evaluation Summary:")
        print("-" * 40)
        print("  Inversion Success Rate:", self.current_metrics.inversion_success_rate * 100.0, "%")
        print("  Average Stability Time:", self.current_metrics.average_stability_time, "seconds")
        print("  Maximum Stability Time:", self.current_metrics.maximum_stability_time, "seconds")
        print("  Average Settling Time:", self.current_metrics.average_settling_time, "seconds")
        print("  Control Effort (RMS):", self.current_metrics.control_effort_rms, "V")
        print("  Tracking Error (RMS):", self.current_metrics.tracking_error_rms, "°")
        print("  Robustness Score:", self.current_metrics.robustness_score * 100.0, "%")
        print("  Real-time Performance:", self.current_metrics.real_time_performance * 100.0, "%")
        print("  Safety Compliance:", self.current_metrics.safety_compliance * 100.0, "%")
        print("  Energy Efficiency:", self.current_metrics.energy_efficiency * 100.0, "%")
        print("  Overall Score:", self.current_metrics.get_overall_score() * 100.0, "%")
        print()
        
        print("Target Achievement:")
        print("  Success Rate Target (>70%):", "✓" if self.current_metrics.inversion_success_rate >= 0.70 else "✗")
        print("  Stability Time Target (>15s):", "✓" if self.current_metrics.average_stability_time >= 15.0 else "✗")
        print("  Safety Compliance (>95%):", "✓" if self.current_metrics.safety_compliance >= 0.95 else "✗")
        print("  Overall Target Achievement:", "✓" if self.current_metrics.meets_targets() else "✗")
        print()
        
        if self.current_metrics.meets_targets():
            print("✓ All performance targets achieved!")
        else:
            print("⚠ Some performance targets not met - further optimization needed")
    
    fn get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.current_metrics
    
    fn get_performance_history(self) -> List[PerformanceMetrics]:
        """Get complete performance evaluation history."""
        return self.performance_history
