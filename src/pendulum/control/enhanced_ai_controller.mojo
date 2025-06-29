"""
Enhanced AI Controller with Advanced MPC Integration.

This module extends the basic AI controller with sophisticated Model Predictive
Control capabilities, adaptive control strategies, and enhanced performance
optimization for superior pendulum control.
"""

from collections import List
from math import abs, max, min, sqrt, exp, sin, cos

# Import project modules
from src.pendulum.utils.physics import PendulumState, PendulumPhysics
from src.pendulum.digital_twin.integrated_trainer import PendulumNeuralNetwork
from src.pendulum.control.ai_controller import AIController, ControlCommand, ControlState
from src.pendulum.control.mpc_controller import MPCController, MPCPrediction

# Enhanced control constants
alias ENHANCED_CONTROL_MODES = 5      # Number of control modes
alias ADAPTIVE_GAIN_ALPHA = 0.1      # Adaptive gain learning rate
alias PERFORMANCE_WINDOW = 100       # Performance evaluation window
alias MODE_SWITCH_THRESHOLD = 0.8    # Threshold for mode switching

@fieldwise_init
struct ControlPerformance(Copyable, Movable):
    """Control performance metrics."""
    
    var success_rate: Float64          # Recent success rate
    var average_error: Float64         # Average tracking error
    var control_effort: Float64        # Average control effort
    var stability_time: Float64        # Time in stable region
    var mode_effectiveness: Float64    # Effectiveness of current mode
    var adaptation_rate: Float64       # Rate of parameter adaptation
    
    fn is_performing_well(self) -> Bool:
        """Check if control performance is satisfactory."""
        return (self.success_rate > 0.7 and 
                self.average_error < 5.0 and
                self.mode_effectiveness > 0.6)

@fieldwise_init
struct AdaptiveGains(Copyable, Movable):
    """Adaptive control gains that adjust based on performance."""
    
    var kp_stabilize: Float64          # Proportional gain for stabilization
    var kd_stabilize: Float64          # Derivative gain for stabilization
    var ke_swing_up: Float64           # Energy gain for swing-up
    var kp_position: Float64           # Position gain for swing-up
    var mpc_weight_angle: Float64      # MPC angle weight
    var mpc_weight_control: Float64    # MPC control weight
    var learning_rate: Float64         # Adaptation learning rate
    
    fn adapt_gains(mut self, performance: ControlPerformance, error_signal: Float64):
        """Adapt gains based on performance feedback."""
        var adaptation = self.learning_rate * error_signal
        
        # Adapt stabilization gains
        if abs(error_signal) > 2.0:  # Large error, increase gains
            self.kp_stabilize += adaptation * 0.5
            self.kd_stabilize += adaptation * 0.1
        elif abs(error_signal) < 0.5:  # Small error, can reduce gains for smoothness
            self.kp_stabilize -= adaptation * 0.1
            self.kd_stabilize -= adaptation * 0.05
        
        # Apply bounds to gains
        self.kp_stabilize = max(5.0, min(30.0, self.kp_stabilize))
        self.kd_stabilize = max(0.5, min(5.0, self.kd_stabilize))
        self.ke_swing_up = max(0.1, min(2.0, self.ke_swing_up))
        self.kp_position = max(0.5, min(3.0, self.kp_position))

struct EnhancedAIController:
    """
    Enhanced AI Controller with advanced MPC and adaptive capabilities.
    
    Features:
    - Advanced Model Predictive Control with multi-step optimization
    - Adaptive control gains based on performance feedback
    - Intelligent mode switching based on system state and performance
    - Enhanced swing-up algorithms with energy optimization
    - Real-time performance monitoring and optimization
    - Robust error handling and graceful degradation
    """
    
    var base_controller: AIController
    var mpc_controller: MPCController
    var adaptive_gains: AdaptiveGains
    var performance_metrics: ControlPerformance
    var control_mode_history: List[String]
    var error_history: List[Float64]
    var performance_history: List[Float64]
    var enhanced_initialized: Bool
    
    fn __init__(out self):
        """Initialize enhanced AI controller."""
        # Initialize base components
        self.base_controller = AIController()
        self.mpc_controller = MPCController()
        
        # Initialize adaptive gains with default values
        self.adaptive_gains = AdaptiveGains(
            15.0,   # kp_stabilize
            2.0,    # kd_stabilize
            0.5,    # ke_swing_up
            1.0,    # kp_position
            100.0,  # mpc_weight_angle
            0.1,    # mpc_weight_control
            ADAPTIVE_GAIN_ALPHA  # learning_rate
        )
        
        # Initialize performance metrics
        self.performance_metrics = ControlPerformance(
            0.0,    # success_rate
            0.0,    # average_error
            0.0,    # control_effort
            0.0,    # stability_time
            0.0,    # mode_effectiveness
            0.0     # adaptation_rate
        )
        
        self.control_mode_history = List[String]()
        self.error_history = List[Float64]()
        self.performance_history = List[Float64]()
        self.enhanced_initialized = False
    
    fn initialize_enhanced_controller(mut self) -> Bool:
        """Initialize enhanced controller with all subsystems."""
        print("Initializing Enhanced AI Controller...")
        
        # Initialize base controller
        if not self.base_controller.initialize():
            print("Failed to initialize base AI controller")
            return False
        
        # Initialize MPC controller
        if not self.mpc_controller.initialize_mpc():
            print("Failed to initialize MPC controller")
            return False
        
        # Set initial MPC targets for inverted pendulum
        self.mpc_controller.set_mpc_target(0.0, 0.0, 0.0)  # Inverted, centered, stationary
        
        self.enhanced_initialized = True
        print("Enhanced AI Controller initialized successfully")
        return True
    
    fn compute_enhanced_control(mut self, current_state: List[Float64], timestamp: Float64) -> ControlCommand:
        """
        Compute optimal control using enhanced AI algorithms.
        
        Args:
            current_state: [la_position, pend_velocity, pend_position, cmd_volts]
            timestamp: Current timestamp
            
        Returns:
            Optimal control command with enhanced performance
        """
        if not self.enhanced_initialized:
            return self._create_safe_command(timestamp)
        
        # Analyze current state and determine optimal control strategy
        var control_mode = self._determine_enhanced_control_mode(current_state)
        var control_command: ControlCommand
        
        # Execute control based on selected mode
        if control_mode == "mpc_stabilize":
            control_command = self._mpc_stabilization_control(current_state, timestamp)
        elif control_mode == "mpc_invert":
            control_command = self._mpc_inversion_control(current_state, timestamp)
        elif control_mode == "adaptive_swing_up":
            control_command = self._adaptive_swing_up_control(current_state, timestamp)
        elif control_mode == "hybrid_control":
            control_command = self._hybrid_control(current_state, timestamp)
        else:  # fallback to base controller
            control_command = self.base_controller.compute_control(current_state, timestamp)
        
        # Update performance metrics and adapt gains
        self._update_enhanced_performance(current_state, control_command, timestamp)
        
        # Store control mode history
        self.control_mode_history.append(control_mode)
        
        return control_command
    
    fn _determine_enhanced_control_mode(self, current_state: List[Float64]) -> String:
        """Determine optimal control mode based on state and performance."""
        var pend_angle = current_state[2]
        var pend_velocity = current_state[1]
        var la_position = current_state[0]
        
        var abs_angle = abs(pend_angle)
        var abs_velocity = abs(pend_velocity)
        
        # Check if near inverted and should use MPC stabilization
        if abs_angle < 15.0 and abs_velocity < 100.0:
            if self.performance_metrics.is_performing_well():
                return "mpc_stabilize"
            else:
                return "hybrid_control"  # Use hybrid if MPC not performing well
        
        # Check if in transition region and should use MPC inversion
        elif abs_angle < 90.0 and abs_velocity < 300.0:
            return "mpc_invert"
        
        # Check if hanging and should use adaptive swing-up
        elif abs_angle > 150.0:
            return "adaptive_swing_up"
        
        # Default to hybrid control for intermediate states
        else:
            return "hybrid_control"
    
    fn _mpc_stabilization_control(mut self, current_state: List[Float64], timestamp: Float64) -> ControlCommand:
        """Advanced MPC control for stabilization around inverted state."""
        # Set MPC target to inverted state
        self.mpc_controller.set_mpc_target(0.0, 0.0, 0.0)
        
        # Compute MPC control
        var mpc_command = self.mpc_controller.compute_mpc_control(current_state, timestamp)
        
        # Enhance with adaptive gains
        var enhanced_voltage = mpc_command.voltage
        
        # Apply adaptive gain adjustment
        var angle_error = abs(current_state[2] - 0.0)  # Error from inverted
        if angle_error > 2.0:
            enhanced_voltage *= (1.0 + self.adaptive_gains.kp_stabilize / 15.0)  # Boost for large errors
        
        # Create enhanced command
        var enhanced_command = ControlCommand(
            enhanced_voltage,
            timestamp,
            "mpc_stabilize",
            mpc_command.safety_override,
            mpc_command.predicted_state
        )
        
        return enhanced_command
    
    fn _mpc_inversion_control(mut self, current_state: List[Float64], timestamp: Float64) -> ControlCommand:
        """MPC control for achieving inversion from arbitrary states."""
        # Set MPC target to inverted state with trajectory planning
        var target_angle = 0.0
        var target_position = 0.0
        var target_velocity = 0.0
        
        # Adjust target based on current state for smooth transition
        var current_angle = current_state[2]
        if abs(current_angle) > 30.0:
            # Intermediate target for large angle errors
            target_angle = current_angle * 0.5  # Move halfway toward inverted
        
        self.mpc_controller.set_mpc_target(target_angle, target_position, target_velocity)
        
        # Compute MPC control with enhanced prediction horizon
        var mpc_command = self.mpc_controller.compute_mpc_control(current_state, timestamp)
        
        # Enhance control with velocity feedforward
        var velocity_feedforward = current_state[1] * 0.01  # Small velocity compensation
        var enhanced_voltage = mpc_command.voltage + velocity_feedforward
        
        # Apply constraints
        enhanced_voltage = max(-10.0, min(10.0, enhanced_voltage))
        
        var enhanced_command = ControlCommand(
            enhanced_voltage,
            timestamp,
            "mpc_invert",
            mpc_command.safety_override,
            mpc_command.predicted_state
        )
        
        return enhanced_command
    
    fn _adaptive_swing_up_control(mut self, current_state: List[Float64], timestamp: Float64) -> ControlCommand:
        """Enhanced swing-up control with adaptive energy management."""
        var la_position = current_state[0]
        var pend_velocity = current_state[1]
        var pend_angle = current_state[2]
        
        # Enhanced energy-based control with adaptive gains
        var target_energy = 2.0  # Energy needed for inversion
        var current_energy = 0.5 * pend_velocity * pend_velocity * 0.001  # Simplified kinetic energy
        
        var energy_error = target_energy - current_energy
        var position_error = 0.0 - la_position  # Keep cart centered
        
        # Adaptive energy pumping with learned gains
        var energy_control = self.adaptive_gains.ke_swing_up * energy_error * sin(pend_angle * 3.14159 / 180.0)
        var position_control = self.adaptive_gains.kp_position * position_error
        
        var control_voltage = energy_control + position_control
        
        # Apply enhanced constraints
        control_voltage = max(-8.0, min(8.0, control_voltage))  # Slightly reduced for swing-up
        
        # Predict next state using base controller's digital twin
        var predicted_state = self.base_controller.digital_twin.forward(current_state)
        
        var command = ControlCommand(
            control_voltage,
            timestamp,
            "adaptive_swing_up",
            False,
            predicted_state
        )
        
        return command
    
    fn _hybrid_control(mut self, current_state: List[Float64], timestamp: Float64) -> ControlCommand:
        """Hybrid control combining MPC and classical control."""
        # Get both MPC and classical control commands
        var mpc_command = self.mpc_controller.compute_mpc_control(current_state, timestamp)
        var classical_command = self.base_controller.compute_control(current_state, timestamp)
        
        # Blend controls based on performance and state
        var mpc_weight = 0.7  # Default MPC weight
        var classical_weight = 0.3
        
        # Adjust weights based on performance
        if self.performance_metrics.mode_effectiveness > 0.8:
            mpc_weight = 0.8  # Increase MPC weight if performing well
            classical_weight = 0.2
        elif self.performance_metrics.mode_effectiveness < 0.4:
            mpc_weight = 0.4  # Reduce MPC weight if performing poorly
            classical_weight = 0.6
        
        # Blend control voltages
        var blended_voltage = mpc_weight * mpc_command.voltage + classical_weight * classical_command.voltage
        
        # Use MPC prediction as it's more sophisticated
        var hybrid_command = ControlCommand(
            blended_voltage,
            timestamp,
            "hybrid_control",
            mpc_command.safety_override or classical_command.safety_override,
            mpc_command.predicted_state
        )
        
        return hybrid_command
    
    fn _update_enhanced_performance(mut self, current_state: List[Float64], command: ControlCommand, timestamp: Float64):
        """Update enhanced performance metrics and adapt gains."""
        var pend_angle = current_state[2]
        var angle_error = abs(pend_angle)
        
        # Update error history
        self.error_history.append(angle_error)
        
        # Calculate recent performance
        if len(self.error_history) > PERFORMANCE_WINDOW:
            # Remove old entries
            var new_history = List[Float64]()
            var start_idx = len(self.error_history) - PERFORMANCE_WINDOW
            for i in range(start_idx, len(self.error_history)):
                new_history.append(self.error_history[i])
            self.error_history = new_history
        
        # Update performance metrics
        var recent_errors = self.error_history
        if len(recent_errors) > 0:
            var sum_error = 0.0
            var success_count = 0.0
            
            for i in range(len(recent_errors)):
                sum_error += recent_errors[i]
                if recent_errors[i] < 10.0:  # Success threshold
                    success_count += 1.0
            
            self.performance_metrics.average_error = sum_error / Float64(len(recent_errors))
            self.performance_metrics.success_rate = success_count / Float64(len(recent_errors))
            self.performance_metrics.control_effort = abs(command.voltage)
        
        # Adapt gains based on performance
        self.adaptive_gains.adapt_gains(self.performance_metrics, angle_error)
    
    fn _create_safe_command(self, timestamp: Float64) -> ControlCommand:
        """Create safe command when enhanced controller is not ready."""
        var safe_predicted_state = List[Float64]()
        safe_predicted_state.append(0.0)
        safe_predicted_state.append(0.0)
        safe_predicted_state.append(0.0)
        
        return ControlCommand(
            0.0,                        # voltage
            timestamp,                  # timestamp
            "enhanced_safe",            # control_mode
            True,                       # safety_override
            safe_predicted_state
        )
    
    fn get_enhanced_performance(self) -> (Float64, Float64, Float64, String):
        """
        Get enhanced controller performance metrics.
        
        Returns:
            (success_rate, average_error, control_effort, current_mode)
        """
        var current_mode = "unknown"
        if len(self.control_mode_history) > 0:
            current_mode = self.control_mode_history[len(self.control_mode_history) - 1]
        
        return (
            self.performance_metrics.success_rate,
            self.performance_metrics.average_error,
            self.performance_metrics.control_effort,
            current_mode
        )
    
    fn reset_enhanced_controller(mut self):
        """Reset enhanced controller to initial state."""
        self.base_controller.reset_controller()
        self.mpc_controller.reset_mpc()
        
        # Reset performance tracking
        self.control_mode_history = List[String]()
        self.error_history = List[Float64]()
        self.performance_history = List[Float64]()
        
        # Reset performance metrics
        self.performance_metrics.success_rate = 0.0
        self.performance_metrics.average_error = 0.0
        self.performance_metrics.control_effort = 0.0
        self.performance_metrics.stability_time = 0.0
        self.performance_metrics.mode_effectiveness = 0.0
        self.performance_metrics.adaptation_rate = 0.0
        
        print("Enhanced AI Controller reset successfully")
