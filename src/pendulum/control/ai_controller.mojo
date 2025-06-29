"""
AI Controller for Inverted Pendulum System.

This module implements the main AI control algorithm interface that integrates
with the digital twin for intelligent pendulum control. Supports both MPC and
RL-based control strategies.
"""

from collections import List
from math import abs, max, min, sqrt, exp, sin, cos

# Import project modules
from src.pendulum.utils.physics import PendulumState, PendulumPhysics
from src.pendulum.digital_twin.integrated_trainer import PendulumNeuralNetwork

# Control system constants
alias CONTROL_FREQUENCY = 25.0  # Hz
alias CONTROL_PERIOD = 1.0 / CONTROL_FREQUENCY  # 40ms
alias MAX_CONTROL_VOLTAGE = 10.0  # Volts
alias MIN_CONTROL_VOLTAGE = -10.0  # Volts
alias INVERTED_ANGLE_THRESHOLD = 10.0  # Degrees
alias STABILITY_THRESHOLD = 5.0  # Degrees for stability check

@fieldwise_init
struct ControlState(Copyable, Movable):
    """Current control system state and targets."""
    
    var target_angle: Float64      # Target pendulum angle (degrees)
    var target_position: Float64   # Target cart position (inches)
    var target_velocity: Float64   # Target angular velocity (deg/s)
    var control_mode: String       # "stabilize", "invert", "swing_up"
    var emergency_stop: Bool       # Emergency stop flag
    var last_control_time: Float64 # Last control update time
    
    fn is_inverted(self) -> Bool:
        """Check if target is inverted state."""
        return abs(self.target_angle) < INVERTED_ANGLE_THRESHOLD
    
    fn is_stable_inverted(self, current_angle: Float64) -> Bool:
        """Check if currently stable in inverted state."""
        return abs(current_angle) < STABILITY_THRESHOLD

@fieldwise_init
struct ControlCommand(Copyable, Movable):
    """Control command output."""
    
    var voltage: Float64           # Control voltage [-10, +10] V
    var timestamp: Float64         # Command timestamp
    var control_mode: String       # Active control mode
    var safety_override: Bool      # Safety system override active
    var predicted_state: List[Float64]  # Digital twin prediction
    
    fn is_valid(self) -> Bool:
        """Check if control command is valid."""
        return (self.voltage >= MIN_CONTROL_VOLTAGE and 
                self.voltage <= MAX_CONTROL_VOLTAGE and
                not self.safety_override)

struct AIController:
    """
    Main AI controller for inverted pendulum system.
    
    Integrates digital twin predictions with control algorithms to achieve
    and maintain inverted pendulum state. Supports multiple control strategies
    including MPC and reinforcement learning approaches.
    """
    
    var digital_twin: PendulumNeuralNetwork
    var physics_model: PendulumPhysics
    var control_state: ControlState
    var control_history: List[ControlCommand]
    var performance_metrics: List[Float64]
    var initialized: Bool
    
    fn __init__(out self):
        """Initialize AI controller with digital twin."""
        # Initialize components
        var weights1 = List[List[Float64]]()
        var biases1 = List[Float64]()
        var weights2 = List[List[Float64]]()
        var biases2 = List[Float64]()
        
        self.digital_twin = PendulumNeuralNetwork(weights1, biases1, weights2, biases2, True, 0.0, 0.0)
        self.physics_model = PendulumPhysics()
        
        # Initialize control state
        self.control_state = ControlState(
            0.0,        # target_angle (inverted)
            0.0,        # target_position (center)
            0.0,        # target_velocity (stationary)
            "stabilize", # control_mode
            False,      # emergency_stop
            0.0         # last_control_time
        )
        
        self.control_history = List[ControlCommand]()
        self.performance_metrics = List[Float64]()
        self.initialized = False
    
    fn initialize(mut self) -> Bool:
        """Initialize controller and validate digital twin."""
        # Initialize digital twin weights (simplified for demo)
        self.digital_twin.initialize_weights()
        
        # Validate digital twin functionality
        var test_input = List[Float64]()
        test_input.append(0.0)    # la_position
        test_input.append(0.0)    # pend_velocity
        test_input.append(180.0)  # pend_position (hanging)
        test_input.append(0.0)    # cmd_volts
        
        var prediction = self.digital_twin.forward(test_input)
        
        if len(prediction) == 3:
            self.initialized = True
            print("AI Controller initialized successfully")
            return True
        else:
            print("AI Controller initialization failed")
            return False
    
    fn compute_control(mut self, current_state: List[Float64], timestamp: Float64) -> ControlCommand:
        """
        Compute control command using AI algorithm.
        
        Args:
            current_state: [la_position, pend_velocity, pend_position, cmd_volts]
            timestamp: Current time in seconds
            
        Returns:
            Control command with voltage and metadata
        """
        if not self.initialized:
            return self._emergency_stop_command(timestamp)
        
        # Extract current state
        var la_position = current_state[0]
        var pend_velocity = current_state[1]
        var pend_angle = current_state[2]
        var last_voltage = current_state[3] if len(current_state) > 3 else 0.0
        
        # Determine control mode based on current state
        var control_mode = self._determine_control_mode(pend_angle)
        self.control_state.control_mode = control_mode
        
        # Compute control voltage based on mode
        var control_voltage = 0.0
        var predicted_state = List[Float64]()
        
        if control_mode == "stabilize":
            control_voltage = self._stabilize_control(current_state)
            predicted_state = self.digital_twin.forward(current_state)
        elif control_mode == "swing_up":
            control_voltage = self._swing_up_control(current_state)
            predicted_state = self.digital_twin.forward(current_state)
        else:  # invert mode
            control_voltage = self._invert_control(current_state)
            predicted_state = self.digital_twin.forward(current_state)
        
        # Apply safety constraints
        control_voltage = self._apply_safety_constraints(control_voltage, current_state)
        
        # Create control command
        var command = ControlCommand(
            control_voltage,
            timestamp,
            control_mode,
            False,  # safety_override
            predicted_state
        )
        
        # Update control history
        self.control_history.append(command)
        self.control_state.last_control_time = timestamp
        
        # Update performance metrics
        self._update_performance_metrics(current_state, command)
        
        return command
    
    fn _determine_control_mode(self, pend_angle: Float64) -> String:
        """Determine appropriate control mode based on pendulum angle."""
        var abs_angle = abs(pend_angle)
        
        if abs_angle < INVERTED_ANGLE_THRESHOLD:
            return "stabilize"  # Near inverted, stabilize
        elif abs_angle > 170.0:
            return "swing_up"   # Hanging down, swing up
        else:
            return "invert"     # In between, work toward inversion
    
    fn _stabilize_control(self, current_state: List[Float64]) -> Float64:
        """PD control for stabilizing inverted pendulum."""
        var pend_angle = current_state[2]
        var pend_velocity = current_state[1]
        
        # PD gains (tuned for stability)
        var kp = 15.0  # Proportional gain
        var kd = 2.0   # Derivative gain
        
        # Error from inverted position (0 degrees)
        var angle_error = 0.0 - pend_angle
        var velocity_error = 0.0 - pend_velocity
        
        # PD control law
        var control_voltage = kp * angle_error + kd * velocity_error
        
        return control_voltage
    
    fn _swing_up_control(self, current_state: List[Float64]) -> Float64:
        """Energy-based swing-up control."""
        var la_position = current_state[0]
        var pend_velocity = current_state[1]
        var pend_angle = current_state[2]
        
        # Energy-based control (simplified)
        var energy_gain = 0.5
        var position_gain = 1.0
        
        # Desired energy for inversion
        var target_energy = 2.0  # Approximate energy for inverted state
        var current_energy = 0.5 * pend_velocity * pend_velocity  # Simplified kinetic energy
        
        var energy_error = target_energy - current_energy
        var position_error = 0.0 - la_position  # Keep cart centered
        
        # Energy pumping control
        var control_voltage = energy_gain * energy_error * sin(pend_angle * 3.14159 / 180.0)
        control_voltage += position_gain * position_error
        
        return control_voltage
    
    fn _invert_control(self, current_state: List[Float64]) -> Float64:
        """Model predictive control for inversion."""
        # Use digital twin for prediction-based control
        var prediction = self.digital_twin.forward(current_state)
        
        var pend_angle = current_state[2]
        var pend_velocity = current_state[1]
        var predicted_angle = prediction[2] if len(prediction) > 2 else pend_angle
        
        # MPC-like control (simplified single-step)
        var angle_cost = abs(predicted_angle)  # Cost of being away from inverted
        var velocity_cost = abs(pend_velocity) * 0.1  # Penalize high velocities
        
        # Try different control actions and pick best
        var best_voltage = 0.0
        var best_cost = 1000.0
        
        for i in range(-10, 11):  # Test voltages from -10 to +10
            var test_voltage = Float64(i)
            var test_input = List[Float64]()
            test_input.append(current_state[0])
            test_input.append(current_state[1])
            test_input.append(current_state[2])
            test_input.append(test_voltage)
            
            var test_prediction = self.digital_twin.forward(test_input)
            var test_angle = test_prediction[2] if len(test_prediction) > 2 else pend_angle
            
            var cost = abs(test_angle) + abs(pend_velocity) * 0.1
            if cost < best_cost:
                best_cost = cost
                best_voltage = test_voltage
        
        return best_voltage
    
    fn _apply_safety_constraints(self, voltage: Float64, current_state: List[Float64]) -> Float64:
        """Apply safety constraints to control voltage."""
        # Clamp voltage to safe limits
        var safe_voltage = max(MIN_CONTROL_VOLTAGE, min(MAX_CONTROL_VOLTAGE, voltage))
        
        # Check for emergency conditions
        var la_position = current_state[0]
        var pend_velocity = current_state[1]
        
        # Emergency stop if actuator near limits
        if abs(la_position) > 3.8:  # Near 4-inch limit
            safe_voltage = 0.0
            self.control_state.emergency_stop = True
        
        # Emergency stop if velocity too high
        if abs(pend_velocity) > 900.0:  # Near 1000 deg/s limit
            safe_voltage *= 0.5  # Reduce control authority
        
        return safe_voltage
    
    fn _emergency_stop_command(self, timestamp: Float64) -> ControlCommand:
        """Generate emergency stop command."""
        var predicted_state = List[Float64]()
        predicted_state.append(0.0)
        predicted_state.append(0.0)
        predicted_state.append(0.0)
        
        return ControlCommand(
            0.0,           # voltage
            timestamp,     # timestamp
            "emergency",   # control_mode
            True,          # safety_override
            predicted_state
        )
    
    fn _update_performance_metrics(mut self, current_state: List[Float64], command: ControlCommand):
        """Update performance tracking metrics."""
        var pend_angle = current_state[2]
        
        # Track time in inverted state
        if abs(pend_angle) < INVERTED_ANGLE_THRESHOLD:
            self.performance_metrics.append(1.0)  # Success
        else:
            self.performance_metrics.append(0.0)  # Not inverted
    
    fn get_performance_summary(self) -> (Float64, Float64, Int):
        """
        Get performance summary.
        
        Returns:
            (inversion_success_rate, average_stability_time, total_commands)
        """
        if len(self.performance_metrics) == 0:
            return (0.0, 0.0, 0)
        
        var success_count = 0.0
        for i in range(len(self.performance_metrics)):
            success_count += self.performance_metrics[i]
        
        var success_rate = success_count / Float64(len(self.performance_metrics))
        var avg_stability = success_count * CONTROL_PERIOD  # Approximate stability time
        
        return (success_rate, avg_stability, len(self.control_history))
    
    fn reset_controller(mut self):
        """Reset controller state and history."""
        self.control_history = List[ControlCommand]()
        self.performance_metrics = List[Float64]()
        self.control_state.emergency_stop = False
        self.control_state.control_mode = "stabilize"
        print("AI Controller reset successfully")
