"""
Control Algorithm Training and Validation System.

This module implements comprehensive training and validation for control algorithms,
including robustness testing, performance metrics collection, and systematic
evaluation across diverse operating conditions.
"""

from collections import List
from math import abs, max, min, sqrt, sin, cos
from random import random

# Import control system components
from src.pendulum.control.enhanced_ai_controller import EnhancedAIController
from src.pendulum.control.parameter_optimizer import ParameterOptimizer, ParameterSet, OptimizationResult
from src.pendulum.control.integrated_control_system import IntegratedControlSystem

# Training constants
alias TRAINING_EPISODES = 100         # Number of training episodes
alias VALIDATION_EPISODES = 50        # Number of validation episodes
alias EPISODE_DURATION = 25.0         # Episode duration in seconds (625 cycles at 25 Hz)
alias ROBUSTNESS_TEST_VARIATIONS = 20 # Number of robustness test variations
alias PERFORMANCE_EVALUATION_CYCLES = 1000  # Cycles for performance evaluation

@fieldwise_init
struct TrainingEpisode(Copyable, Movable):
    """Single training episode configuration."""
    
    var initial_state: List[Float64]   # Initial pendulum state
    var disturbances: List[Float64]    # External disturbances during episode
    var target_performance: Float64    # Target performance for this episode
    var episode_duration: Float64      # Duration of episode (seconds)
    var difficulty_level: String       # "easy", "medium", "hard", "extreme"
    
    fn get_difficulty_score(self) -> Float64:
        """Get numerical difficulty score."""
        if self.difficulty_level == "easy":
            return 1.0
        elif self.difficulty_level == "medium":
            return 2.0
        elif self.difficulty_level == "hard":
            return 3.0
        else:  # extreme
            return 4.0

@fieldwise_init
struct TrainingResults(Copyable, Movable):
    """Results from control algorithm training."""
    
    var total_episodes: Int            # Total training episodes
    var successful_episodes: Int       # Episodes meeting performance targets
    var average_success_rate: Float64  # Average inversion success rate
    var average_stability_time: Float64 # Average stability time
    var robustness_score: Float64      # Robustness across conditions
    var convergence_episode: Int       # Episode where convergence achieved
    var final_parameters: ParameterSet # Final optimized parameters
    var training_time: Float64         # Total training time
    var meets_requirements: Bool       # Whether training targets are met
    
    fn get_training_grade(self) -> String:
        """Get training performance grade."""
        if self.meets_requirements:
            if self.average_success_rate > 0.85 and self.robustness_score > 0.8:
                return "Excellent"
            else:
                return "Good"
        elif self.average_success_rate > 0.60:
            return "Acceptable"
        else:
            return "Needs Improvement"

struct ControlTrainer:
    """
    Comprehensive control algorithm training system.
    
    Features:
    - Progressive difficulty training episodes
    - Robustness testing across diverse conditions
    - Performance metrics collection and analysis
    - Adaptive parameter optimization during training
    - Systematic validation and testing
    """
    
    var parameter_optimizer: ParameterOptimizer
    var training_episodes: List[TrainingEpisode]
    var validation_episodes: List[TrainingEpisode]
    var training_history: List[TrainingResults]
    var current_parameters: ParameterSet
    var trainer_initialized: Bool
    
    fn __init__(out self):
        """Initialize control trainer."""
        self.parameter_optimizer = ParameterOptimizer()
        self.training_episodes = List[TrainingEpisode]()
        self.validation_episodes = List[TrainingEpisode]()
        self.training_history = List[TrainingResults]()
        
        # Initialize with default parameters
        var default_params = ParameterSet(
            100.0, 10.0, 1.0, 0.1, 0.5, 10, 5,  # MPC parameters
            15.0, 2.0, 0.5, 1.0, 0.1,            # Adaptive gains
            15.0, 100.0, 90.0, 150.0,            # Control thresholds
            0.7, 0.3                             # Hybrid weights
        )
        self.current_parameters = default_params
        self.trainer_initialized = False
    
    fn initialize_trainer(mut self) -> Bool:
        """Initialize trainer with episodes and optimization setup."""
        print("Initializing Control Algorithm Trainer...")
        
        # Initialize parameter optimizer
        if not self.parameter_optimizer.initialize_optimizer():
            print("Failed to initialize parameter optimizer")
            return False
        
        # Create training episodes
        self._create_training_episodes()
        
        # Create validation episodes
        self._create_validation_episodes()
        
        self.trainer_initialized = True
        print("Control Trainer initialized with", len(self.training_episodes), "training episodes")
        print("Validation episodes:", len(self.validation_episodes))
        return True
    
    fn train_control_algorithms(mut self) -> TrainingResults:
        """
        Run comprehensive control algorithm training.
        
        Returns:
            Training results with performance metrics and optimized parameters
        """
        if not self.trainer_initialized:
            print("Trainer not initialized")
            return self._create_failed_training_result()
        
        print("=" * 60)
        print("CONTROL ALGORITHM TRAINING AND TUNING")
        print("=" * 60)
        print("Training objectives:")
        print("- Achieve >70% inversion success rate")
        print("- Maintain >15 second stability time")
        print("- Demonstrate robustness across conditions")
        print("- Optimize control parameters")
        print()
        
        var training_start_time = 0.0  # Simplified timing
        
        # Phase 1: Initial parameter optimization
        print("Phase 1: Initial Parameter Optimization")
        print("-" * 40)
        var optimization_result = self.parameter_optimizer.optimize_parameters()
        self.current_parameters = optimization_result.best_parameters
        print("Initial optimization complete")
        print()
        
        # Phase 2: Progressive training episodes
        print("Phase 2: Progressive Training Episodes")
        print("-" * 40)
        var episode_results = self._run_training_episodes()
        print("Training episodes complete")
        print()
        
        # Phase 3: Robustness testing
        print("Phase 3: Robustness Testing")
        print("-" * 40)
        var robustness_score = self._test_robustness()
        print("Robustness testing complete")
        print()
        
        # Phase 4: Final validation
        print("Phase 4: Final Validation")
        print("-" * 40)
        var validation_results = self._run_validation_episodes()
        print("Validation complete")
        print()
        
        var training_end_time = training_start_time + 30.0  # Simplified timing
        var training_time = training_end_time - training_start_time
        
        # Compile final results
        var final_results = TrainingResults(
            len(self.training_episodes),        # total_episodes
            episode_results.0,                  # successful_episodes
            episode_results.1,                  # average_success_rate
            episode_results.2,                  # average_stability_time
            robustness_score,                   # robustness_score
            episode_results.3,                  # convergence_episode
            self.current_parameters,            # final_parameters
            training_time,                      # training_time
            episode_results.1 > 0.70 and episode_results.2 > 15.0  # meets_requirements
        )
        
        # Store training results
        self.training_history.append(final_results)
        
        print("Training Summary:")
        self._print_training_results(final_results)
        
        return final_results
    
    fn _run_training_episodes(mut self) -> (Int, Float64, Float64, Int):
        """
        Run progressive training episodes.
        
        Returns:
            (successful_episodes, avg_success_rate, avg_stability_time, convergence_episode)
        """
        var successful_episodes = 0
        var total_success_rate = 0.0
        var total_stability_time = 0.0
        var convergence_episode = -1
        
        print("  Running", len(self.training_episodes), "training episodes...")
        
        for i in range(len(self.training_episodes)):
            var episode = self.training_episodes[i]
            
            print("    Episode", i + 1, "- Difficulty:", episode.difficulty_level)
            
            # Run episode with current parameters
            var episode_result = self._run_single_episode(episode)
            
            var success_rate = episode_result.0
            var stability_time = episode_result.1
            var control_effort = episode_result.2
            
            total_success_rate += success_rate
            total_stability_time += stability_time
            
            # Check if episode meets target performance
            if success_rate > episode.target_performance:
                successful_episodes += 1
                
                # Check for convergence
                if convergence_episode == -1 and success_rate > 0.70:
                    convergence_episode = i + 1
            
            print("      Success rate:", success_rate * 100.0, "%")
            print("      Stability time:", stability_time, "s")
            print("      Control effort:", control_effort, "V")
            
            # Adaptive parameter adjustment during training
            if i % 10 == 9:  # Every 10 episodes
                self._adaptive_parameter_update(success_rate, stability_time)
        
        var avg_success_rate = total_success_rate / Float64(len(self.training_episodes))
        var avg_stability_time = total_stability_time / Float64(len(self.training_episodes))
        
        print("  Training episodes summary:")
        print("    Successful episodes:", successful_episodes, "/", len(self.training_episodes))
        print("    Average success rate:", avg_success_rate * 100.0, "%")
        print("    Average stability time:", avg_stability_time, "seconds")
        print("    Convergence episode:", convergence_episode)
        
        return (successful_episodes, avg_success_rate, avg_stability_time, convergence_episode)
    
    fn _test_robustness(mut self) -> Float64:
        """Test control robustness across parameter variations."""
        print("  Testing robustness across", ROBUSTNESS_TEST_VARIATIONS, "parameter variations...")
        
        var robustness_tests = 0
        var successful_tests = 0
        
        # Test with parameter variations
        for i in range(ROBUSTNESS_TEST_VARIATIONS):
            # Create parameter variation (±10% variation)
            var variation_factor = 0.9 + 0.2 * (Float64(i) / Float64(ROBUSTNESS_TEST_VARIATIONS))
            var varied_params = self.current_parameters
            varied_params.kp_stabilize *= variation_factor
            varied_params.mpc_weight_angle *= variation_factor
            
            # Test with varied parameters
            var test_episode = self.training_episodes[i % len(self.training_episodes)]
            var test_result = self._run_single_episode_with_params(test_episode, varied_params)
            
            robustness_tests += 1
            if test_result.0 > 0.60:  # 60% success rate threshold for robustness
                successful_tests += 1
        
        var robustness_score = Float64(successful_tests) / Float64(robustness_tests)
        
        print("  Robustness test results:")
        print("    Successful tests:", successful_tests, "/", robustness_tests)
        print("    Robustness score:", robustness_score * 100.0, "%")
        
        return robustness_score
    
    fn _run_validation_episodes(mut self) -> (Float64, Float64):
        """
        Run final validation episodes.
        
        Returns:
            (validation_success_rate, validation_stability_time)
        """
        print("  Running", len(self.validation_episodes), "validation episodes...")
        
        var total_success = 0.0
        var total_stability = 0.0
        
        for i in range(len(self.validation_episodes)):
            var episode = self.validation_episodes[i]
            var result = self._run_single_episode(episode)
            
            total_success += result.0
            total_stability += result.1
        
        var avg_success = total_success / Float64(len(self.validation_episodes))
        var avg_stability = total_stability / Float64(len(self.validation_episodes))
        
        print("  Validation results:")
        print("    Average success rate:", avg_success * 100.0, "%")
        print("    Average stability time:", avg_stability, "seconds")
        
        return (avg_success, avg_stability)
    
    fn _run_single_episode(self, episode: TrainingEpisode) -> (Float64, Float64, Float64):
        """
        Run a single training episode.
        
        Returns:
            (success_rate, stability_time, control_effort)
        """
        return self._run_single_episode_with_params(episode, self.current_parameters)
    
    fn _run_single_episode_with_params(self, episode: TrainingEpisode, params: ParameterSet) -> (Float64, Float64, Float64):
        """
        Run a single episode with specific parameters.
        
        Returns:
            (success_rate, stability_time, control_effort)
        """
        # Simplified episode simulation
        var initial_angle = episode.initial_state[2]
        var difficulty = episode.get_difficulty_score()
        
        # Estimate performance based on initial conditions and parameters
        var base_success = 0.5
        var base_stability = 10.0
        var base_effort = 5.0
        
        # Adjust based on initial angle
        if abs(initial_angle) < 30.0:  # Near inverted
            base_success = 0.8
            base_stability = 15.0
        elif abs(initial_angle) > 150.0:  # Hanging
            base_success = 0.4
            base_stability = 5.0
        
        # Adjust based on parameters
        var param_factor = (params.kp_stabilize / 15.0 + params.mpc_weight_angle / 100.0) / 2.0
        var success_rate = min(1.0, base_success * param_factor)
        var stability_time = base_stability * param_factor
        var control_effort = base_effort / param_factor
        
        # Adjust based on difficulty
        success_rate /= difficulty
        stability_time /= sqrt(difficulty)
        control_effort *= difficulty
        
        # Apply bounds
        success_rate = max(0.0, min(1.0, success_rate))
        stability_time = max(0.0, min(30.0, stability_time))
        control_effort = max(0.0, min(15.0, control_effort))
        
        return (success_rate, stability_time, control_effort)
    
    fn _adaptive_parameter_update(mut self, success_rate: Float64, stability_time: Float64):
        """Adaptively update parameters based on recent performance."""
        if success_rate < 0.60:  # Poor performance, increase gains
            self.current_parameters.kp_stabilize *= 1.1
            self.current_parameters.mpc_weight_angle *= 1.1
        elif success_rate > 0.85:  # Excellent performance, can reduce for smoothness
            self.current_parameters.kp_stabilize *= 0.95
            self.current_parameters.mpc_weight_angle *= 0.95
        
        # Apply bounds
        self.current_parameters.kp_stabilize = max(5.0, min(30.0, self.current_parameters.kp_stabilize))
        self.current_parameters.mpc_weight_angle = max(50.0, min(200.0, self.current_parameters.mpc_weight_angle))
    
    fn _create_training_episodes(mut self):
        """Create progressive training episodes with increasing difficulty."""
        # Easy episodes (near inverted)
        for i in range(20):
            var angle = Float64(i - 10) * 2.0  # -20 to +18 degrees
            var episode = self._create_episode(angle, 20.0, "easy", 0.70)
            self.training_episodes.append(episode)
        
        # Medium episodes (transition region)
        for i in range(30):
            var angle = Float64(i - 15) * 4.0  # -60 to +56 degrees
            var episode = self._create_episode(angle, 50.0, "medium", 0.60)
            self.training_episodes.append(episode)
        
        # Hard episodes (large angles)
        for i in range(30):
            var angle = 90.0 + Float64(i) * 3.0  # 90 to 177 degrees
            var episode = self._create_episode(angle, 100.0, "hard", 0.50)
            self.training_episodes.append(episode)
        
        # Extreme episodes (challenging conditions)
        for i in range(20):
            var angle = 160.0 + Float64(i) * 1.0  # 160 to 179 degrees
            var episode = self._create_episode(angle, 200.0, "extreme", 0.40)
            self.training_episodes.append(episode)
    
    fn _create_validation_episodes(mut self):
        """Create validation episodes for final testing."""
        # Mixed difficulty validation episodes
        for i in range(VALIDATION_EPISODES):
            var angle = Float64(i - 25) * 7.0  # Wide range of angles
            var velocity = Float64(i % 10 - 5) * 20.0  # -100 to +80 deg/s
            var episode = self._create_episode(angle, velocity, "medium", 0.65)
            self.validation_episodes.append(episode)
    
    fn _create_episode(self, angle: Float64, velocity: Float64, difficulty: String, target: Float64) -> TrainingEpisode:
        """Create a single training episode."""
        var initial_state = List[Float64]()
        initial_state.append(0.0)      # la_position (centered)
        initial_state.append(velocity) # pend_velocity
        initial_state.append(angle)    # pend_angle
        initial_state.append(0.0)      # cmd_volts
        
        var disturbances = List[Float64]()
        disturbances.append(0.0)  # No disturbances for now
        
        return TrainingEpisode(
            initial_state,
            disturbances,
            target,
            EPISODE_DURATION,
            difficulty
        )
    
    fn _print_training_results(self, results: TrainingResults):
        """Print comprehensive training results."""
        print("  Final Training Results:")
        print("    Total episodes:", results.total_episodes)
        print("    Successful episodes:", results.successful_episodes)
        print("    Success rate:", results.average_success_rate * 100.0, "%")
        print("    Stability time:", results.average_stability_time, "seconds")
        print("    Robustness score:", results.robustness_score * 100.0, "%")
        print("    Convergence episode:", results.convergence_episode)
        print("    Training time:", results.training_time, "seconds")
        print("    Training grade:", results.get_training_grade())
        print("    Meets requirements:", results.meets_requirements)
        
        if results.meets_requirements:
            print("    ✓ Training successful - all targets achieved!")
        else:
            print("    ⚠ Training incomplete - some targets not met")
    
    fn _create_failed_training_result(self) -> TrainingResults:
        """Create failed training result."""
        return TrainingResults(
            0, 0, 0.0, 0.0, 0.0, -1, self.current_parameters, 0.0, False
        )
    
    fn get_trained_parameters(self) -> ParameterSet:
        """Get the final trained parameters."""
        return self.current_parameters
    
    fn get_training_history(self) -> List[TrainingResults]:
        """Get complete training history."""
        return self.training_history
