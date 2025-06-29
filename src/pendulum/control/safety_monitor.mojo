"""
Safety Monitor for Inverted Pendulum Control System.

This module implements comprehensive safety monitoring and constraint enforcement
for the pendulum control system, providing multiple layers of protection against
unsafe operating conditions.
"""

from collections import List
from math import abs, max, min, sqrt
from time import now

# Import project modules
from src.pendulum.utils.physics import PendulumState, PendulumPhysics
from src.pendulum.control.ai_controller import ControlCommand, ControlState

# Safety system constants
alias MAX_ACTUATOR_POSITION = 4.0      # inches
alias MAX_PENDULUM_VELOCITY = 1000.0   # deg/s
alias MAX_CONTROL_VOLTAGE = 10.0       # volts
alias MAX_ACCELERATION = 500.0         # deg/s^2
alias EMERGENCY_STOP_TIMEOUT = 5.0     # seconds
alias SAFETY_MARGIN_POSITION = 0.2     # inches
alias SAFETY_MARGIN_VELOCITY = 50.0    # deg/s

@fieldwise_init
struct SafetyViolation(Copyable, Movable):
    """Safety violation record."""
    
    var violation_type: String     # Type of violation
    var severity: String           # "warning", "critical", "emergency"
    var timestamp: Float64         # When violation occurred
    var current_value: Float64     # Value that caused violation
    var limit_value: Float64       # Safety limit that was exceeded
    var description: String        # Human-readable description
    
    fn is_critical(self) -> Bool:
        """Check if violation is critical or emergency level."""
        return self.severity == "critical" or self.severity == "emergency"

@fieldwise_init
struct SafetyStatus(Copyable, Movable):
    """Current safety system status."""
    
    var system_safe: Bool              # Overall system safety status
    var emergency_stop_active: Bool    # Emergency stop engaged
    var warnings_active: Int           # Number of active warnings
    var critical_violations: Int       # Number of critical violations
    var last_violation_time: Float64   # Time of last violation
    var safety_override_active: Bool   # Safety override in effect
    var monitoring_enabled: Bool       # Safety monitoring enabled
    
    fn has_active_violations(self) -> Bool:
        """Check if there are any active safety violations."""
        return self.warnings_active > 0 or self.critical_violations > 0

struct SafetyMonitor:
    """
    Comprehensive safety monitoring system for pendulum control.
    
    Provides multiple layers of safety protection including:
    - Real-time constraint monitoring
    - Predictive safety analysis
    - Emergency stop capabilities
    - Safety violation logging and analysis
    """
    
    var physics_model: PendulumPhysics
    var violation_history: List[SafetyViolation]
    var safety_status: SafetyStatus
    var last_state: List[Float64]
    var last_command: ControlCommand
    var monitoring_start_time: Float64
    var emergency_stop_time: Float64
    
    fn __init__(out self):
        """Initialize safety monitoring system."""
        self.physics_model = PendulumPhysics()
        self.violation_history = List[SafetyViolation]()
        
        # Initialize safety status
        self.safety_status = SafetyStatus(
            True,   # system_safe
            False,  # emergency_stop_active
            0,      # warnings_active
            0,      # critical_violations
            0.0,    # last_violation_time
            False,  # safety_override_active
            True    # monitoring_enabled
        )
        
        self.last_state = List[Float64]()
        var predicted_state = List[Float64]()
        predicted_state.append(0.0)
        predicted_state.append(0.0)
        predicted_state.append(0.0)
        
        self.last_command = ControlCommand(0.0, 0.0, "init", False, predicted_state)
        self.monitoring_start_time = 0.0
        self.emergency_stop_time = 0.0
    
    fn start_monitoring(mut self, timestamp: Float64):
        """Start safety monitoring system."""
        self.monitoring_start_time = timestamp
        self.safety_status.monitoring_enabled = True
        self.safety_status.system_safe = True
        self.safety_status.emergency_stop_active = False
        print("Safety monitoring started at time:", timestamp)
    
    fn check_safety(mut self, current_state: List[Float64], command: ControlCommand) -> SafetyStatus:
        """
        Comprehensive safety check of current state and control command.
        
        Args:
            current_state: [la_position, pend_velocity, pend_position, cmd_volts]
            command: Proposed control command
            
        Returns:
            Updated safety status
        """
        if not self.safety_status.monitoring_enabled:
            return self.safety_status
        
        # Reset violation counters
        self.safety_status.warnings_active = 0
        self.safety_status.critical_violations = 0
        
        # Check all safety constraints
        self._check_position_constraints(current_state, command.timestamp)
        self._check_velocity_constraints(current_state, command.timestamp)
        self._check_control_constraints(command)
        self._check_acceleration_constraints(current_state, command.timestamp)
        self._check_predictive_safety(current_state, command)
        
        # Update overall safety status
        self._update_safety_status(command.timestamp)
        
        # Store current state for next iteration
        self.last_state = current_state
        self.last_command = command
        
        return self.safety_status
    
    fn _check_position_constraints(mut self, current_state: List[Float64], timestamp: Float64):
        """Check actuator position constraints."""
        var la_position = current_state[0]
        var abs_position = abs(la_position)
        
        # Warning level check
        if abs_position > (MAX_ACTUATOR_POSITION - SAFETY_MARGIN_POSITION):
            var violation = SafetyViolation(
                "position_warning",
                "warning",
                timestamp,
                abs_position,
                MAX_ACTUATOR_POSITION - SAFETY_MARGIN_POSITION,
                "Actuator position approaching limit"
            )
            self.violation_history.append(violation)
            self.safety_status.warnings_active += 1
        
        # Critical level check
        if abs_position > MAX_ACTUATOR_POSITION:
            var violation = SafetyViolation(
                "position_critical",
                "critical",
                timestamp,
                abs_position,
                MAX_ACTUATOR_POSITION,
                "Actuator position limit exceeded"
            )
            self.violation_history.append(violation)
            self.safety_status.critical_violations += 1
    
    fn _check_velocity_constraints(mut self, current_state: List[Float64], timestamp: Float64):
        """Check pendulum velocity constraints."""
        var pend_velocity = current_state[1]
        var abs_velocity = abs(pend_velocity)
        
        # Warning level check
        if abs_velocity > (MAX_PENDULUM_VELOCITY - SAFETY_MARGIN_VELOCITY):
            var violation = SafetyViolation(
                "velocity_warning",
                "warning",
                timestamp,
                abs_velocity,
                MAX_PENDULUM_VELOCITY - SAFETY_MARGIN_VELOCITY,
                "Pendulum velocity approaching limit"
            )
            self.violation_history.append(violation)
            self.safety_status.warnings_active += 1
        
        # Critical level check
        if abs_velocity > MAX_PENDULUM_VELOCITY:
            var violation = SafetyViolation(
                "velocity_critical",
                "critical",
                timestamp,
                abs_velocity,
                MAX_PENDULUM_VELOCITY,
                "Pendulum velocity limit exceeded"
            )
            self.violation_history.append(violation)
            self.safety_status.critical_violations += 1
    
    fn _check_control_constraints(mut self, command: ControlCommand):
        """Check control command constraints."""
        var abs_voltage = abs(command.voltage)
        
        # Critical level check for control voltage
        if abs_voltage > MAX_CONTROL_VOLTAGE:
            var violation = SafetyViolation(
                "control_critical",
                "critical",
                command.timestamp,
                abs_voltage,
                MAX_CONTROL_VOLTAGE,
                "Control voltage limit exceeded"
            )
            self.violation_history.append(violation)
            self.safety_status.critical_violations += 1
    
    fn _check_acceleration_constraints(mut self, current_state: List[Float64], timestamp: Float64):
        """Check acceleration constraints using state history."""
        if len(self.last_state) == 0:
            return  # No previous state to compare
        
        var current_velocity = current_state[1]
        var last_velocity = self.last_state[1]
        var dt = timestamp - self.last_command.timestamp
        
        if dt > 0.0:
            var acceleration = abs((current_velocity - last_velocity) / dt)
            
            if acceleration > MAX_ACCELERATION:
                var violation = SafetyViolation(
                    "acceleration_warning",
                    "warning",
                    timestamp,
                    acceleration,
                    MAX_ACCELERATION,
                    "High acceleration detected"
                )
                self.violation_history.append(violation)
                self.safety_status.warnings_active += 1
    
    fn _check_predictive_safety(mut self, current_state: List[Float64], command: ControlCommand):
        """Check predictive safety using digital twin predictions."""
        if len(command.predicted_state) < 3:
            return  # No prediction available
        
        var predicted_la_pos = command.predicted_state[0]
        var predicted_pend_vel = command.predicted_state[1]
        
        # Check if prediction violates constraints
        if abs(predicted_la_pos) > (MAX_ACTUATOR_POSITION - SAFETY_MARGIN_POSITION):
            var violation = SafetyViolation(
                "predictive_position",
                "warning",
                command.timestamp,
                abs(predicted_la_pos),
                MAX_ACTUATOR_POSITION - SAFETY_MARGIN_POSITION,
                "Predicted position may violate constraints"
            )
            self.violation_history.append(violation)
            self.safety_status.warnings_active += 1
        
        if abs(predicted_pend_vel) > (MAX_PENDULUM_VELOCITY - SAFETY_MARGIN_VELOCITY):
            var violation = SafetyViolation(
                "predictive_velocity",
                "warning",
                command.timestamp,
                abs(predicted_pend_vel),
                MAX_PENDULUM_VELOCITY - SAFETY_MARGIN_VELOCITY,
                "Predicted velocity may violate constraints"
            )
            self.violation_history.append(violation)
            self.safety_status.warnings_active += 1
    
    fn _update_safety_status(mut self, timestamp: Float64):
        """Update overall safety status based on violations."""
        # Update violation timestamp
        if self.safety_status.warnings_active > 0 or self.safety_status.critical_violations > 0:
            self.safety_status.last_violation_time = timestamp
        
        # Determine if emergency stop is needed
        if self.safety_status.critical_violations > 0:
            self.safety_status.emergency_stop_active = True
            self.safety_status.safety_override_active = True
            self.emergency_stop_time = timestamp
            self.safety_status.system_safe = False
        
        # Check if emergency stop timeout has passed
        if self.safety_status.emergency_stop_active:
            if (timestamp - self.emergency_stop_time) > EMERGENCY_STOP_TIMEOUT:
                # Allow recovery if no recent violations
                if self.safety_status.critical_violations == 0:
                    self.safety_status.emergency_stop_active = False
                    self.safety_status.safety_override_active = False
                    self.safety_status.system_safe = True
        
        # System is safe if no critical violations and not in emergency stop
        if self.safety_status.critical_violations == 0 and not self.safety_status.emergency_stop_active:
            self.safety_status.system_safe = True
        else:
            self.safety_status.system_safe = False
    
    fn apply_safety_override(self, command: ControlCommand) -> ControlCommand:
        """Apply safety override to control command if necessary."""
        if not self.safety_status.safety_override_active:
            return command  # No override needed
        
        # Create safe command
        var safe_predicted_state = List[Float64]()
        safe_predicted_state.append(0.0)
        safe_predicted_state.append(0.0)
        safe_predicted_state.append(0.0)
        
        var safe_command = ControlCommand(
            0.0,                    # voltage (emergency stop)
            command.timestamp,      # timestamp
            "emergency_stop",       # control_mode
            True,                   # safety_override
            safe_predicted_state    # predicted_state
        )
        
        return safe_command
    
    fn get_safety_report(self) -> (Int, Int, Float64, Bool):
        """
        Get safety system report.
        
        Returns:
            (total_violations, critical_violations, uptime_percentage, system_safe)
        """
        var total_violations = len(self.violation_history)
        var critical_count = 0
        
        for i in range(len(self.violation_history)):
            if self.violation_history[i].is_critical():
                critical_count += 1
        
        var current_time = self.safety_status.last_violation_time
        if current_time == 0.0:
            current_time = self.monitoring_start_time + 1.0  # Avoid division by zero
        
        var total_time = current_time - self.monitoring_start_time
        var violation_time = Float64(critical_count) * 0.1  # Estimate violation duration
        var uptime_percentage = max(0.0, (total_time - violation_time) / total_time * 100.0)
        
        return (total_violations, critical_count, uptime_percentage, self.safety_status.system_safe)
    
    fn reset_safety_system(mut self):
        """Reset safety system to initial state."""
        self.violation_history = List[SafetyViolation]()
        self.safety_status.system_safe = True
        self.safety_status.emergency_stop_active = False
        self.safety_status.warnings_active = 0
        self.safety_status.critical_violations = 0
        self.safety_status.last_violation_time = 0.0
        self.safety_status.safety_override_active = False
        self.last_state = List[Float64]()
        print("Safety system reset successfully")
    
    fn enable_monitoring(mut self, enabled: Bool):
        """Enable or disable safety monitoring."""
        self.safety_status.monitoring_enabled = enabled
        if enabled:
            print("Safety monitoring enabled")
        else:
            print("Safety monitoring disabled")
    
    fn get_recent_violations(self, count: Int) -> List[SafetyViolation]:
        """Get the most recent safety violations."""
        var recent = List[SafetyViolation]()
        var start_idx = max(0, len(self.violation_history) - count)
        
        for i in range(start_idx, len(self.violation_history)):
            recent.append(self.violation_history[i])
        
        return recent
