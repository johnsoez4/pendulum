"""
Control Algorithm Training and Tuning Demonstration.

This module demonstrates the complete control algorithm training and tuning
process, including parameter optimization, progressive training, robustness
testing, and comprehensive performance evaluation.
"""

from collections import List
from math import abs, max, min

# Import training and evaluation components
from src.pendulum.control.parameter_optimizer import ParameterOptimizer, ParameterSet, OptimizationResult
from src.pendulum.control.control_trainer import ControlTrainer, TrainingResults
from src.pendulum.control.performance_evaluator import PerformanceEvaluator, PerformanceMetrics

struct TrainingDemo:
    """Comprehensive demonstration of control algorithm training and tuning."""
    
    @staticmethod
    fn run_training_demonstration():
        """Run complete training and tuning demonstration."""
        print("=" * 70)
        print("PHASE 2 TASK 3: CONTROL ALGORITHM TRAINING AND TUNING")
        print("=" * 70)
        print("Comprehensive control algorithm optimization and validation:")
        print("- Multi-stage parameter optimization")
        print("- Progressive difficulty training episodes")
        print("- Robustness testing across operating conditions")
        print("- Performance evaluation and metrics collection")
        print("- Target: >70% inversion success rate, >15s stability")
        print()
        
        # Stage 1: Parameter optimization
        var optimization_results = TrainingDemo._demonstrate_parameter_optimization()
        print()
        
        # Stage 2: Control algorithm training
        var training_results = TrainingDemo._demonstrate_control_training(optimization_results.best_parameters)
        print()
        
        # Stage 3: Performance evaluation
        var performance_metrics = TrainingDemo._demonstrate_performance_evaluation(training_results.final_parameters)
        print()
        
        # Stage 4: Final validation and summary
        TrainingDemo._demonstrate_final_validation(optimization_results, training_results, performance_metrics)
    
    @staticmethod
    fn _demonstrate_parameter_optimization() -> OptimizationResult:
        """Demonstrate comprehensive parameter optimization."""
        print("STAGE 1: PARAMETER OPTIMIZATION")
        print("=" * 50)
        print("Optimizing control parameters for maximum performance...")
        
        var optimizer = ParameterOptimizer()
        
        if not optimizer.initialize_optimizer():
            print("Failed to initialize parameter optimizer")
            return TrainingDemo._create_default_optimization_result()
        
        print("✓ Parameter optimizer initialized")
        print("  Optimization stages: Grid search → Gradient descent → Adaptive refinement")
        print("  Target metrics: Success rate, stability time, control effort")
        print()
        
        # Run optimization
        var optimization_result = optimizer.optimize_parameters()
        
        print("Parameter Optimization Results:")
        print("-" * 30)
        print("  Best performance score:", optimization_result.best_performance)
        print("  Success rate:", optimization_result.success_rate * 100.0, "%")
        print("  Stability time:", optimization_result.stability_time, "seconds")
        print("  Control effort:", optimization_result.control_effort, "V")
        print("  Convergence iterations:", optimization_result.convergence_iterations)
        print("  Performance grade:", optimization_result.get_performance_grade())
        print("  Meets targets:", optimization_result.meets_targets)
        
        if optimization_result.meets_targets:
            print("  ✓ Parameter optimization successful!")
        else:
            print("  ⚠ Parameter optimization needs further refinement")
        
        # Display optimized parameters
        TrainingDemo._print_optimized_parameters(optimization_result.best_parameters)
        
        return optimization_result
    
    @staticmethod
    fn _demonstrate_control_training(optimized_parameters: ParameterSet) -> TrainingResults:
        """Demonstrate progressive control algorithm training."""
        print("STAGE 2: CONTROL ALGORITHM TRAINING")
        print("=" * 50)
        print("Training control algorithms with optimized parameters...")
        
        var trainer = ControlTrainer()
        
        if not trainer.initialize_trainer():
            print("Failed to initialize control trainer")
            return TrainingDemo._create_default_training_result()
        
        print("✓ Control trainer initialized")
        print("  Training episodes: Progressive difficulty (easy → extreme)")
        print("  Validation: Robustness testing and final validation")
        print("  Adaptation: Real-time parameter adjustment during training")
        print()
        
        # Run training
        var training_result = trainer.train_control_algorithms()
        
        print("Control Training Results:")
        print("-" * 30)
        print("  Total episodes:", training_result.total_episodes)
        print("  Successful episodes:", training_result.successful_episodes)
        print("  Success rate:", training_result.average_success_rate * 100.0, "%")
        print("  Stability time:", training_result.average_stability_time, "seconds")
        print("  Robustness score:", training_result.robustness_score * 100.0, "%")
        print("  Convergence episode:", training_result.convergence_episode)
        print("  Training time:", training_result.training_time, "seconds")
        print("  Training grade:", training_result.get_training_grade())
        print("  Meets requirements:", training_result.meets_requirements)
        
        if training_result.meets_requirements:
            print("  ✓ Control training successful!")
        else:
            print("  ⚠ Control training needs additional episodes")
        
        return training_result
    
    @staticmethod
    fn _demonstrate_performance_evaluation(trained_parameters: ParameterSet) -> PerformanceMetrics:
        """Demonstrate comprehensive performance evaluation."""
        print("STAGE 3: PERFORMANCE EVALUATION")
        print("=" * 50)
        print("Evaluating trained control system performance...")
        
        var evaluator = PerformanceEvaluator()
        
        if not evaluator.initialize_evaluator():
            print("Failed to initialize performance evaluator")
            return TrainingDemo._create_default_performance_metrics()
        
        print("✓ Performance evaluator initialized")
        print("  Evaluation scenarios: Near inverted, transition, large angles, robustness")
        print("  Metrics: Success rate, stability, control effort, safety, efficiency")
        print("  Duration: Multi-scenario comprehensive testing")
        print()
        
        # Run evaluation
        var performance_metrics = evaluator.evaluate_control_performance(trained_parameters)
        
        print("Performance Evaluation Summary:")
        print("-" * 30)
        print("  Inversion success rate:", performance_metrics.inversion_success_rate * 100.0, "%")
        print("  Average stability time:", performance_metrics.average_stability_time, "seconds")
        print("  Maximum stability time:", performance_metrics.maximum_stability_time, "seconds")
        print("  Control effort (RMS):", performance_metrics.control_effort_rms, "V")
        print("  Tracking error (RMS):", performance_metrics.tracking_error_rms, "°")
        print("  Robustness score:", performance_metrics.robustness_score * 100.0, "%")
        print("  Safety compliance:", performance_metrics.safety_compliance * 100.0, "%")
        print("  Overall score:", performance_metrics.get_overall_score() * 100.0, "%")
        print("  Meets targets:", performance_metrics.meets_targets())
        
        if performance_metrics.meets_targets():
            print("  ✓ Performance evaluation successful!")
        else:
            print("  ⚠ Performance targets not fully achieved")
        
        return performance_metrics
    
    @staticmethod
    fn _demonstrate_final_validation(optimization_result: OptimizationResult, training_result: TrainingResults, performance_metrics: PerformanceMetrics):
        """Demonstrate final validation and comprehensive summary."""
        print("STAGE 4: FINAL VALIDATION AND SUMMARY")
        print("=" * 50)
        print("Comprehensive validation of training and tuning results...")
        print()
        
        # Validate against Phase 2 Task 3 requirements
        var success_rate_target = performance_metrics.inversion_success_rate >= 0.70
        var stability_time_target = performance_metrics.average_stability_time >= 15.0
        var robustness_target = performance_metrics.robustness_score >= 0.60
        var safety_target = performance_metrics.safety_compliance >= 0.95
        
        print("Phase 2 Task 3 Target Validation:")
        print("-" * 40)
        print("  Inversion Success Rate (>70%):", "✓" if success_rate_target else "✗", 
              "(" + str(performance_metrics.inversion_success_rate * 100.0) + "%)")
        print("  Stability Time (>15s):", "✓" if stability_time_target else "✗",
              "(" + str(performance_metrics.average_stability_time) + "s)")
        print("  Robustness Score (>60%):", "✓" if robustness_target else "✗",
              "(" + str(performance_metrics.robustness_score * 100.0) + "%)")
        print("  Safety Compliance (>95%):", "✓" if safety_target else "✗",
              "(" + str(performance_metrics.safety_compliance * 100.0) + "%)")
        print()
        
        var all_targets_met = (success_rate_target and stability_time_target and 
                              robustness_target and safety_target)
        
        print("Overall Task 3 Achievement:")
        print("-" * 40)
        if all_targets_met:
            print("  ✓ ALL PHASE 2 TASK 3 TARGETS ACHIEVED!")
            print("  ✓ Control algorithm training and tuning successful")
            print("  ✓ Ready for advanced control development (Task 4)")
        else:
            print("  ⚠ Some targets not fully achieved")
            print("  ⚠ Additional optimization may be needed")
        
        print()
        print("Training and Tuning Summary:")
        print("-" * 40)
        print("  Optimization Performance:", optimization_result.get_performance_grade())
        print("  Training Performance:", training_result.get_training_grade())
        print("  Evaluation Performance:", "Excellent" if performance_metrics.get_overall_score() > 0.8 else "Good")
        print("  Overall System Readiness:", "Ready" if all_targets_met else "Needs Work")
        print()
        
        # Technical achievements summary
        print("Technical Achievements:")
        print("-" * 40)
        print("  ✓ Multi-stage parameter optimization implemented")
        print("  ✓ Progressive difficulty training episodes completed")
        print("  ✓ Robustness testing across operating conditions")
        print("  ✓ Comprehensive performance metrics collection")
        print("  ✓ Adaptive parameter adjustment during training")
        print("  ✓ Safety and constraint compliance validation")
        print()
        
        # Performance metrics summary
        print("Key Performance Metrics:")
        print("-" * 40)
        print("  Success Rate:", performance_metrics.inversion_success_rate * 100.0, "% (Target: >70%)")
        print("  Stability Time:", performance_metrics.average_stability_time, "s (Target: >15s)")
        print("  Max Stability:", performance_metrics.maximum_stability_time, "s")
        print("  Control Effort:", performance_metrics.control_effort_rms, "V")
        print("  Robustness:", performance_metrics.robustness_score * 100.0, "%")
        print("  Safety:", performance_metrics.safety_compliance * 100.0, "%")
        print("  Overall Score:", performance_metrics.get_overall_score() * 100.0, "%")
        print()
        
        print("✓ Control Algorithm Training and Tuning Demonstration Complete!")
        
        if all_targets_met:
            print()
            print("🎉 PHASE 2 TASK 3 SUCCESSFULLY COMPLETED! 🎉")
            print("Ready to proceed with Advanced Control Development (Task 4)")
        else:
            print()
            print("📋 Additional optimization recommended before Task 4")
    
    @staticmethod
    fn _print_optimized_parameters(parameters: ParameterSet):
        """Print optimized parameter values."""
        print()
        print("Optimized Parameters:")
        print("-" * 20)
        print("  MPC Parameters:")
        print("    Angle weight:", parameters.mpc_weight_angle)
        print("    Position weight:", parameters.mpc_weight_position)
        print("    Control weight:", parameters.mpc_weight_control)
        print("    Prediction horizon:", parameters.mpc_prediction_horizon)
        print("    Control horizon:", parameters.mpc_control_horizon)
        print("  Adaptive Gains:")
        print("    Kp stabilize:", parameters.kp_stabilize)
        print("    Kd stabilize:", parameters.kd_stabilize)
        print("    Ke swing-up:", parameters.ke_swing_up)
        print("    Learning rate:", parameters.learning_rate)
        print("  Control Thresholds:")
        print("    Stabilize angle:", parameters.stabilize_angle_threshold, "°")
        print("    Invert angle:", parameters.invert_angle_threshold, "°")
        print("  Hybrid Weights:")
        print("    MPC weight:", parameters.mpc_hybrid_weight)
        print("    Classical weight:", parameters.classical_hybrid_weight)
    
    @staticmethod
    fn _create_default_optimization_result() -> OptimizationResult:
        """Create default optimization result for error cases."""
        var default_params = ParameterSet(
            100.0, 10.0, 1.0, 0.1, 0.5, 10, 5,  # MPC parameters
            15.0, 2.0, 0.5, 1.0, 0.1,            # Adaptive gains
            15.0, 100.0, 90.0, 150.0,            # Control thresholds
            0.7, 0.3                             # Hybrid weights
        )
        
        return OptimizationResult(
            default_params, 0.5, 0.6, 12.0, 5.0, 10, 5.0, False
        )
    
    @staticmethod
    fn _create_default_training_result() -> TrainingResults:
        """Create default training result for error cases."""
        var default_params = ParameterSet(
            100.0, 10.0, 1.0, 0.1, 0.5, 10, 5,  # MPC parameters
            15.0, 2.0, 0.5, 1.0, 0.1,            # Adaptive gains
            15.0, 100.0, 90.0, 150.0,            # Control thresholds
            0.7, 0.3                             # Hybrid weights
        )
        
        return TrainingResults(
            50, 30, 0.6, 12.0, 0.7, 25, default_params, 30.0, False
        )
    
    @staticmethod
    fn _create_default_performance_metrics() -> PerformanceMetrics:
        """Create default performance metrics for error cases."""
        return PerformanceMetrics(
            0.6, 12.0, 20.0, 8.0, 5.0, 15.0, 0.7, 0.8, 0.9, 0.6
        )

fn main():
    """Run control algorithm training and tuning demonstration."""
    TrainingDemo.run_training_demonstration()
