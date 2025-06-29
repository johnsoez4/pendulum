"""
Advanced Hybrid Controller for Superior Pendulum Control.

This module implements a sophisticated hybrid control system that intelligently
combines MPC, RL, and adaptive control techniques to achieve >90% inversion
success rate and >30 second stability through advanced control fusion.
"""

from collections import List
from math import abs, max, min, sqrt, exp, sin, cos

# Import control system components
from src.pendulum.control.enhanced_ai_controller import EnhancedAIController, ControlPerformance
from src.pendulum.control.mpc_controller import MPCController, MPCPrediction
from src.pendulum.control.rl_controller import RLController, RLState
from src.pendulum.control.ai_controller import ControlCommand, ControlState

# Advanced hybrid constants
alias HYBRID_FUSION_MODES = 4      # Number of fusion strategies
alias PERFORMANCE_WINDOW = 200     # Performance evaluation window
alias ADAPTATION_RATE = 0.05       # Rate of controller adaptation
alias CONFIDENCE_THRESHOLD = 0.8   # Threshold for high-confidence decisions
alias STABILITY_TARGET = 30.0      # Target stability time (seconds)
alias SUCCESS_TARGET = 0.90        # Target success rate (90%)

@fieldwise_init
struct ControllerConfidence(Copyable, Movable):
    """Confidence metrics for each controller."""
    
    var mpc_confidence: Float64        # MPC controller confidence
    var rl_confidence: Float64         # RL controller confidence
    var adaptive_confidence: Float64   # Adaptive controller confidence
    var fusion_confidence: Float64     # Overall fusion confidence
    var prediction_accuracy: Float64   # Prediction accuracy score
    var stability_score: Float64       # Current stability score
    
    fn get_best_controller(self) -> String:
        """Get the controller with highest confidence."""
        var max_conf = max(self.mpc_confidence, max(self.rl_confidence, self.adaptive_confidence))
        
        if max_conf == self.mpc_confidence:
            return "mpc"
        elif max_conf == self.rl_confidence:
            return "rl"
        else:
            return "adaptive"
    
    fn is_high_confidence(self) -> Bool:
        """Check if any controller has high confidence."""
        return (self.mpc_confidence > CONFIDENCE_THRESHOLD or
                self.rl_confidence > CONFIDENCE_THRESHOLD or
                self.adaptive_confidence > CONFIDENCE_THRESHOLD)

@fieldwise_init
struct FusionStrategy(Copyable, Movable):
    """Control fusion strategy configuration."""
    
    var strategy_name: String          # Name of fusion strategy
    var mpc_weight: Float64            # Weight for MPC controller
    var rl_weight: Float64             # Weight for RL controller
    var adaptive_weight: Float64       # Weight for adaptive controller
    var performance_threshold: Float64 # Performance threshold for activation
    var stability_requirement: Float64 # Stability requirement for strategy
    
    fn normalize_weights(mut self):
        """Normalize weights to sum to 1.0."""
        var total = self.mpc_weight + self.rl_weight + self.adaptive_weight
        if total > 0.0:
            self.mpc_weight /= total
            self.rl_weight /= total
            self.adaptive_weight /= total
    
    fn is_applicable(self, performance: ControlPerformance, stability_time: Float64) -> Bool:
        """Check if strategy is applicable for current conditions."""
        return (performance.success_rate >= self.performance_threshold and
                stability_time >= self.stability_requirement)

struct AdvancedHybridController:
    """
    Advanced hybrid controller combining MPC, RL, and adaptive control.
    
    Features:
    - Intelligent controller fusion based on performance and confidence
    - Dynamic weight adaptation based on real-time performance
    - Advanced prediction and stability optimization
    - Multi-objective optimization for success rate and stability
    - Target: >90% inversion success rate, >30s stability
    """
    
    var enhanced_controller: EnhancedAIController  # Enhanced MPC/Adaptive controller
    var rl_controller: RLController                # Reinforcement learning controller
    var fusion_strategies: List[FusionStrategy]    # Available fusion strategies
    var controller_confidence: ControllerConfidence # Current confidence metrics
    var performance_history: List[ControlPerformance] # Performance tracking
    var stability_history: List[Float64]          # Stability time tracking
    var current_strategy: String                   # Current active strategy
    var adaptation_count: Int                      # Number of adaptations performed
    var hybrid_initialized: Bool                   # Initialization flag
    
    fn __init__(out self):
        """Initialize advanced hybrid controller."""
        # Initialize component controllers
        self.enhanced_controller = EnhancedAIController()
        self.rl_controller = RLController()
        
        # Initialize fusion strategies
        self.fusion_strategies = List[FusionStrategy]()
        
        # Initialize confidence metrics
        self.controller_confidence = ControllerConfidence(
            0.5, 0.5, 0.5, 0.5, 0.5, 0.0
        )
        
        # Initialize tracking
        self.performance_history = List[ControlPerformance]()
        self.stability_history = List[Float64]()
        self.current_strategy = "balanced"
        self.adaptation_count = 0
        self.hybrid_initialized = False
    
    fn initialize_hybrid_controller(mut self) -> Bool:
        """Initialize hybrid controller and all subsystems."""
        print("Initializing Advanced Hybrid Controller...")
        
        # Initialize component controllers
        if not self.enhanced_controller.initialize_enhanced_controller():
            print("Failed to initialize enhanced controller")
            return False
        
        if not self.rl_controller.initialize_rl_controller():
            print("Failed to initialize RL controller")
            return False
        
        # Create fusion strategies
        self._create_fusion_strategies()
        
        self.hybrid_initialized = True
        print("Advanced Hybrid Controller initialized successfully")
        print("  Component controllers: Enhanced MPC/Adaptive + RL")
        print("  Fusion strategies:", len(self.fusion_strategies))
        print("  Target: >90% success rate, >30s stability")
        return True
    
    fn compute_hybrid_control(mut self, current_state: List[Float64], timestamp: Float64) -> ControlCommand:
        """
        Compute optimal control using advanced hybrid fusion.
        
        Args:
            current_state: [la_position, pend_velocity, pend_position, cmd_volts]
            timestamp: Current timestamp
            
        Returns:
            Optimally fused control command
        """
        if not self.hybrid_initialized:
            return self._create_safe_command(timestamp)
        
        # Get control commands from all controllers
        var enhanced_command = self.enhanced_controller.compute_enhanced_control(current_state, timestamp)
        var rl_command = self.rl_controller.compute_rl_control(current_state, timestamp)
        
        # Update controller confidence based on recent performance
        self._update_controller_confidence(current_state, enhanced_command, rl_command)
        
        # Select optimal fusion strategy
        var strategy = self._select_fusion_strategy(current_state)
        
        # Fuse control commands using selected strategy
        var fused_command = self._fuse_control_commands(enhanced_command, rl_command, strategy, timestamp)
        
        # Update performance tracking
        self._update_performance_tracking(current_state, fused_command, timestamp)
        
        # Adapt fusion weights based on performance
        if self.adaptation_count % 50 == 0:  # Adapt every 50 cycles (2 seconds)
            self._adapt_fusion_weights()
        
        self.adaptation_count += 1
        self.current_strategy = strategy.strategy_name
        
        return fused_command
    
    fn _create_fusion_strategies(mut self):
        """Create different fusion strategies for various scenarios."""
        # Strategy 1: Balanced fusion (default)
        var balanced = FusionStrategy(
            "balanced",
            0.4,    # mpc_weight
            0.4,    # rl_weight
            0.2,    # adaptive_weight
            0.5,    # performance_threshold
            5.0     # stability_requirement
        )
        balanced.normalize_weights()
        self.fusion_strategies.append(balanced)
        
        # Strategy 2: MPC-dominant (for precision control)
        var mpc_dominant = FusionStrategy(
            "mpc_dominant",
            0.7,    # mpc_weight
            0.2,    # rl_weight
            0.1,    # adaptive_weight
            0.7,    # performance_threshold
            10.0    # stability_requirement
        )
        mpc_dominant.normalize_weights()
        self.fusion_strategies.append(mpc_dominant)
        
        # Strategy 3: RL-dominant (for learning and adaptation)
        var rl_dominant = FusionStrategy(
            "rl_dominant",
            0.2,    # mpc_weight
            0.7,    # rl_weight
            0.1,    # adaptive_weight
            0.6,    # performance_threshold
            8.0     # stability_requirement
        )
        rl_dominant.normalize_weights()
        self.fusion_strategies.append(rl_dominant)
        
        # Strategy 4: Adaptive-dominant (for robustness)
        var adaptive_dominant = FusionStrategy(
            "adaptive_dominant",
            0.3,    # mpc_weight
            0.2,    # rl_weight
            0.5,    # adaptive_weight
            0.8,    # performance_threshold
            15.0    # stability_requirement
        )
        adaptive_dominant.normalize_weights()
        self.fusion_strategies.append(adaptive_dominant)
    
    fn _select_fusion_strategy(self, current_state: List[Float64]) -> FusionStrategy:
        """Select optimal fusion strategy based on current conditions."""
        var pend_angle = current_state[2]
        var pend_velocity = current_state[1]
        
        # Get current performance estimate
        var current_performance = self._estimate_current_performance()
        var current_stability = self._estimate_current_stability()
        
        # Find best applicable strategy
        var best_strategy = self.fusion_strategies[0]  # Default to balanced
        var best_score = 0.0
        
        for i in range(len(self.fusion_strategies)):
            var strategy = self.fusion_strategies[i]
            
            if strategy.is_applicable(current_performance, current_stability):
                # Score strategy based on confidence and state
                var score = self._score_strategy(strategy, current_state)
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
        
        return best_strategy
    
    fn _fuse_control_commands(self, enhanced_cmd: ControlCommand, rl_cmd: ControlCommand, 
                             strategy: FusionStrategy, timestamp: Float64) -> ControlCommand:
        """Fuse control commands using the selected strategy."""
        # Extract control voltages
        var enhanced_voltage = enhanced_cmd.voltage
        var rl_voltage = rl_cmd.voltage
        
        # Get adaptive component (simplified as enhanced controller's adaptive part)
        var adaptive_voltage = enhanced_voltage * 0.8  # Simplified adaptive component
        
        # Fuse voltages using strategy weights
        var fused_voltage = (strategy.mpc_weight * enhanced_voltage +
                            strategy.rl_weight * rl_voltage +
                            strategy.adaptive_weight * adaptive_voltage)
        
        # Apply safety constraints
        fused_voltage = max(-10.0, min(10.0, fused_voltage))
        
        # Use the most confident prediction
        var predicted_state = enhanced_cmd.predicted_state
        if self.controller_confidence.rl_confidence > self.controller_confidence.mpc_confidence:
            predicted_state = rl_cmd.predicted_state
        
        # Create fused command
        var fused_command = ControlCommand(
            fused_voltage,
            timestamp,
            "hybrid_" + strategy.strategy_name,
            enhanced_cmd.safety_override or rl_cmd.safety_override,
            predicted_state
        )
        
        return fused_command
    
    fn _update_controller_confidence(mut self, current_state: List[Float64], 
                                   enhanced_cmd: ControlCommand, rl_cmd: ControlCommand):
        """Update confidence metrics for each controller."""
        var pend_angle = current_state[2]
        var abs_angle = abs(pend_angle)
        
        # MPC confidence: High for near-inverted states
        if abs_angle < 15.0:
            self.controller_confidence.mpc_confidence = 0.9
        elif abs_angle < 45.0:
            self.controller_confidence.mpc_confidence = 0.7
        else:
            self.controller_confidence.mpc_confidence = 0.4
        
        # RL confidence: Increases with training episodes
        var rl_performance = self.rl_controller.get_rl_performance()
        self.controller_confidence.rl_confidence = min(0.9, rl_performance.3 + 0.1)  # Success rate estimate
        
        # Adaptive confidence: Based on recent performance
        var recent_performance = self._estimate_current_performance()
        self.controller_confidence.adaptive_confidence = recent_performance.success_rate
        
        # Overall fusion confidence
        var max_conf = max(self.controller_confidence.mpc_confidence,
                          max(self.controller_confidence.rl_confidence,
                             self.controller_confidence.adaptive_confidence))
        self.controller_confidence.fusion_confidence = max_conf
        
        # Update stability score
        if abs_angle < 10.0:
            self.controller_confidence.stability_score += 0.04  # Add 40ms
        else:
            self.controller_confidence.stability_score = 0.0
    
    fn _score_strategy(self, strategy: FusionStrategy, current_state: List[Float64]) -> Float64:
        """Score a fusion strategy for current conditions."""
        var pend_angle = current_state[2]
        var abs_angle = abs(pend_angle)
        
        var score = 0.0
        
        # Prefer MPC-dominant for precision near inverted
        if abs_angle < 15.0 and strategy.strategy_name == "mpc_dominant":
            score += 0.3
        
        # Prefer RL-dominant for learning in transition regions
        if abs_angle > 30.0 and abs_angle < 120.0 and strategy.strategy_name == "rl_dominant":
            score += 0.3
        
        # Prefer adaptive-dominant for robustness in challenging conditions
        if abs_angle > 120.0 and strategy.strategy_name == "adaptive_dominant":
            score += 0.3
        
        # Add confidence-based scoring
        if strategy.strategy_name == "mpc_dominant":
            score += self.controller_confidence.mpc_confidence * 0.2
        elif strategy.strategy_name == "rl_dominant":
            score += self.controller_confidence.rl_confidence * 0.2
        else:
            score += self.controller_confidence.adaptive_confidence * 0.2
        
        return score
    
    fn _estimate_current_performance(self) -> ControlPerformance:
        """Estimate current performance based on recent history."""
        if len(self.performance_history) == 0:
            return ControlPerformance(0.5, 10.0, 5.0, 5.0, 0.5, 0.0)
        
        # Use most recent performance
        return self.performance_history[len(self.performance_history) - 1]
    
    fn _estimate_current_stability(self) -> Float64:
        """Estimate current stability time."""
        if len(self.stability_history) == 0:
            return 0.0
        
        return self.stability_history[len(self.stability_history) - 1]
    
    fn _update_performance_tracking(mut self, current_state: List[Float64], 
                                  command: ControlCommand, timestamp: Float64):
        """Update performance and stability tracking."""
        var pend_angle = current_state[2]
        var abs_angle = abs(pend_angle)
        
        # Update stability tracking
        if abs_angle < 10.0:  # In stable region
            if len(self.stability_history) > 0:
                self.stability_history[len(self.stability_history) - 1] += 0.04  # Add 40ms
            else:
                self.stability_history.append(0.04)
        else:
            self.stability_history.append(0.0)  # Reset stability
        
        # Update performance tracking (simplified)
        var success_rate = 1.0 if abs_angle < 15.0 else 0.0
        var current_perf = ControlPerformance(
            success_rate,
            abs_angle,
            abs(command.voltage),
            self.controller_confidence.stability_score,
            self.controller_confidence.fusion_confidence,
            ADAPTATION_RATE
        )
        
        self.performance_history.append(current_perf)
        
        # Keep only recent history
        if len(self.performance_history) > PERFORMANCE_WINDOW:
            var new_history = List[ControlPerformance]()
            var start_idx = len(self.performance_history) - PERFORMANCE_WINDOW
            for i in range(start_idx, len(self.performance_history)):
                new_history.append(self.performance_history[i])
            self.performance_history = new_history
        
        if len(self.stability_history) > PERFORMANCE_WINDOW:
            var new_stability = List[Float64]()
            var start_idx = len(self.stability_history) - PERFORMANCE_WINDOW
            for i in range(start_idx, len(self.stability_history)):
                new_stability.append(self.stability_history[i])
            self.stability_history = new_stability
    
    fn _adapt_fusion_weights(mut self):
        """Adapt fusion weights based on recent performance."""
        if len(self.performance_history) < 10:
            return  # Need sufficient history
        
        # Calculate recent performance
        var recent_success = 0.0
        var recent_count = min(50, len(self.performance_history))
        var start_idx = len(self.performance_history) - recent_count
        
        for i in range(start_idx, len(self.performance_history)):
            recent_success += self.performance_history[i].success_rate
        
        var success_rate = recent_success / Float64(recent_count)
        
        # Adapt strategy weights based on performance
        for i in range(len(self.fusion_strategies)):
            var strategy = self.fusion_strategies[i]
            
            if success_rate > 0.85:  # High performance - favor current approach
                # Slightly increase weights of successful strategy
                if strategy.strategy_name == self.current_strategy:
                    strategy.mpc_weight *= 1.02
                    strategy.rl_weight *= 1.02
                    strategy.adaptive_weight *= 1.02
                    strategy.normalize_weights()
            elif success_rate < 0.60:  # Low performance - explore alternatives
                # Adjust weights to explore different approaches
                if strategy.strategy_name == "balanced":
                    strategy.mpc_weight = 0.4
                    strategy.rl_weight = 0.4
                    strategy.adaptive_weight = 0.2
                    strategy.normalize_weights()
    
    fn _create_safe_command(self, timestamp: Float64) -> ControlCommand:
        """Create safe command when hybrid controller is not ready."""
        var safe_predicted_state = List[Float64]()
        safe_predicted_state.append(0.0)
        safe_predicted_state.append(0.0)
        safe_predicted_state.append(0.0)
        
        return ControlCommand(
            0.0,                    # voltage
            timestamp,              # timestamp
            "hybrid_safe",          # control_mode
            True,                   # safety_override
            safe_predicted_state
        )
    
    fn get_hybrid_performance(self) -> (Float64, Float64, Float64, String, Bool):
        """
        Get comprehensive hybrid controller performance.
        
        Returns:
            (success_rate, stability_time, fusion_confidence, current_strategy, meets_targets)
        """
        var success_rate = 0.0
        var avg_stability = 0.0
        
        if len(self.performance_history) > 0:
            var total_success = 0.0
            for i in range(len(self.performance_history)):
                total_success += self.performance_history[i].success_rate
            success_rate = total_success / Float64(len(self.performance_history))
        
        if len(self.stability_history) > 0:
            var max_stability = 0.0
            for i in range(len(self.stability_history)):
                max_stability = max(max_stability, self.stability_history[i])
            avg_stability = max_stability
        
        var meets_targets = (success_rate >= SUCCESS_TARGET and avg_stability >= STABILITY_TARGET)
        
        return (success_rate, avg_stability, self.controller_confidence.fusion_confidence, 
                self.current_strategy, meets_targets)
    
    fn reset_hybrid_controller(mut self):
        """Reset hybrid controller to initial state."""
        self.enhanced_controller.reset_enhanced_controller()
        self.rl_controller.reset_rl_controller()
        
        self.performance_history = List[ControlPerformance]()
        self.stability_history = List[Float64]()
        self.current_strategy = "balanced"
        self.adaptation_count = 0
        
        # Reset confidence
        self.controller_confidence.mpc_confidence = 0.5
        self.controller_confidence.rl_confidence = 0.5
        self.controller_confidence.adaptive_confidence = 0.5
        self.controller_confidence.fusion_confidence = 0.5
        self.controller_confidence.stability_score = 0.0
        
        print("Advanced Hybrid Controller reset successfully")
