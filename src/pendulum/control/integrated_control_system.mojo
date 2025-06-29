"""
Integrated Control System for Inverted Pendulum.

This module integrates all control system components including the AI controller,
safety monitor, state estimator, and digital twin to provide a complete
control solution for the inverted pendulum system.
"""

from collections import List
from time import now

# Import control system components
from src.pendulum.control.ai_controller import AIController, ControlCommand, ControlState
from src.pendulum.control.safety_monitor import SafetyMonitor, SafetyStatus, SafetyViolation
from src.pendulum.control.state_estimator import StateEstimator, FilteredState
from src.pendulum.digital_twin.integrated_trainer import PendulumNeuralNetwork

# System constants
alias CONTROL_LOOP_FREQUENCY = 25.0  # Hz
alias CONTROL_LOOP_PERIOD = 1.0 / CONTROL_LOOP_FREQUENCY  # 40ms

@fieldwise_init
struct SystemStatus(Copyable, Movable):
    """Overall system status."""
    
    var system_operational: Bool       # System is operational
    var control_active: Bool           # Control loop is active
    var safety_status: String          # Safety system status
    var performance_mode: String       # Current performance mode
    var uptime_seconds: Float64        # System uptime
    var total_control_cycles: Int      # Total control cycles executed
    var successful_inversions: Int     # Number of successful inversions
    var current_inversion_time: Float64 # Current inversion duration
    
    fn get_success_rate(self) -> Float64:
        """Calculate inversion success rate."""
        if self.total_control_cycles == 0:
            return 0.0
        return Float64(self.successful_inversions) / Float64(self.total_control_cycles) * 100.0

struct IntegratedControlSystem:
    """
    Complete integrated control system for inverted pendulum.
    
    Integrates:
    - AI Controller: Main control algorithm with MPC and learning capabilities
    - Safety Monitor: Multi-layer safety system with constraint enforcement
    - State Estimator: Advanced filtering and state estimation
    - Digital Twin: Physics-informed neural network for predictions
    
    Provides:
    - Real-time control loop at 25 Hz
    - Comprehensive safety monitoring
    - Performance tracking and optimization
    - Robust state estimation and filtering
    """
    
    var ai_controller: AIController
    var safety_monitor: SafetyMonitor
    var state_estimator: StateEstimator
    var system_status: SystemStatus
    var control_loop_active: Bool
    var system_start_time: Float64
    var last_control_time: Float64
    var performance_history: List[Float64]
    
    fn __init__(out self):
        """Initialize integrated control system."""
        # Initialize all subsystems
        self.ai_controller = AIController()
        self.safety_monitor = SafetyMonitor()
        self.state_estimator = StateEstimator()
        
        # Initialize system status
        self.system_status = SystemStatus(
            False,      # system_operational
            False,      # control_active
            "init",     # safety_status
            "startup",  # performance_mode
            0.0,        # uptime_seconds
            0,          # total_control_cycles
            0,          # successful_inversions
            0.0         # current_inversion_time
        )
        
        self.control_loop_active = False
        self.system_start_time = 0.0
        self.last_control_time = 0.0
        self.performance_history = List[Float64]()
    
    fn initialize_system(mut self, timestamp: Float64) -> Bool:
        """Initialize complete control system."""
        print("Initializing Integrated Control System...")
        
        # Initialize AI controller
        if not self.ai_controller.initialize():
            print("Failed to initialize AI controller")
            return False
        
        # Start safety monitoring
        self.safety_monitor.start_monitoring(timestamp)
        
        # System is now operational
        self.system_status.system_operational = True
        self.system_status.safety_status = "monitoring"
        self.system_status.performance_mode = "ready"
        self.system_start_time = timestamp
        
        print("Integrated Control System initialized successfully")
        return True
    
    fn start_control_loop(mut self, initial_state: List[Float64], timestamp: Float64) -> Bool:
        """Start the main control loop."""
        if not self.system_status.system_operational:
            print("Cannot start control loop - system not operational")
            return False
        
        # Initialize state estimator
        self.state_estimator.initialize_estimator(initial_state, timestamp)
        
        # Start control loop
        self.control_loop_active = True
        self.system_status.control_active = True
        self.system_status.performance_mode = "active"
        self.last_control_time = timestamp
        
        print("Control loop started at time:", timestamp)
        return True
    
    fn execute_control_cycle(mut self, raw_sensor_data: List[Float64], timestamp: Float64) -> ControlCommand:
        """
        Execute one complete control cycle.
        
        Args:
            raw_sensor_data: [la_position, pend_velocity, pend_position, cmd_volts]
            timestamp: Current timestamp
            
        Returns:
            Control command for actuator
        """
        if not self.control_loop_active:
            return self._create_safe_command(timestamp)
        
        # Update system uptime
        self.system_status.uptime_seconds = timestamp - self.system_start_time
        
        # Step 1: State Estimation
        var filtered_state = self.state_estimator.estimate_state(raw_sensor_data, timestamp)
        
        # Convert filtered state to control input format
        var control_input = List[Float64]()
        control_input.append(filtered_state.la_position)
        control_input.append(filtered_state.pend_velocity)
        control_input.append(filtered_state.pend_angle)
        control_input.append(raw_sensor_data[3] if len(raw_sensor_data) > 3 else 0.0)
        
        # Step 2: AI Control Computation
        var control_command = self.ai_controller.compute_control(control_input, timestamp)
        
        # Step 3: Safety Monitoring
        var safety_status = self.safety_monitor.check_safety(control_input, control_command)
        
        # Step 4: Apply Safety Override if Necessary
        var final_command = control_command
        if not safety_status.system_safe:
            final_command = self.safety_monitor.apply_safety_override(control_command)
            self.system_status.safety_status = "override_active"
        else:
            self.system_status.safety_status = "monitoring"
        
        # Step 5: Update Performance Metrics
        self._update_performance_metrics(filtered_state, final_command, timestamp)
        
        # Step 6: Update System Status
        self._update_system_status(filtered_state, timestamp)
        
        self.last_control_time = timestamp
        self.system_status.total_control_cycles += 1
        
        return final_command
    
    fn _create_safe_command(self, timestamp: Float64) -> ControlCommand:
        """Create safe default command when system is not active."""
        var safe_predicted_state = List[Float64]()
        safe_predicted_state.append(0.0)
        safe_predicted_state.append(0.0)
        safe_predicted_state.append(0.0)
        
        return ControlCommand(
            0.0,                # voltage
            timestamp,          # timestamp
            "system_inactive",  # control_mode
            True,               # safety_override
            safe_predicted_state
        )
    
    fn _update_performance_metrics(mut self, state: FilteredState, command: ControlCommand, timestamp: Float64):
        """Update system performance metrics."""
        # Check if pendulum is inverted
        var is_inverted = abs(state.pend_angle) < 10.0  # Within 10 degrees of inverted
        
        if is_inverted:
            self.system_status.current_inversion_time += (timestamp - self.last_control_time)
            
            # Count as successful inversion if maintained for >1 second
            if self.system_status.current_inversion_time > 1.0:
                self.system_status.successful_inversions += 1
        else:
            self.system_status.current_inversion_time = 0.0
        
        # Track performance history
        var performance_score = 0.0
        if is_inverted:
            performance_score = 1.0
        
        self.performance_history.append(performance_score)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:  # Last 40 seconds at 25 Hz
            var new_history = List[Float64]()
            var start_idx = len(self.performance_history) - 1000
            for i in range(start_idx, len(self.performance_history)):
                new_history.append(self.performance_history[i])
            self.performance_history = new_history
    
    fn _update_system_status(mut self, state: FilteredState, timestamp: Float64):
        """Update overall system status."""
        # Update performance mode based on current state
        if abs(state.pend_angle) < 5.0:
            self.system_status.performance_mode = "stabilizing"
        elif abs(state.pend_angle) < 30.0:
            self.system_status.performance_mode = "inverting"
        elif abs(state.pend_angle) > 150.0:
            self.system_status.performance_mode = "swinging_up"
        else:
            self.system_status.performance_mode = "transitioning"
    
    fn stop_control_loop(mut self):
        """Stop the control loop safely."""
        self.control_loop_active = False
        self.system_status.control_active = False
        self.system_status.performance_mode = "stopped"
        print("Control loop stopped")
    
    fn get_system_performance(self) -> (Float64, Float64, Float64, Int):
        """
        Get comprehensive system performance metrics.
        
        Returns:
            (success_rate, current_inversion_time, uptime, total_cycles)
        """
        var success_rate = self.system_status.get_success_rate()
        return (
            success_rate,
            self.system_status.current_inversion_time,
            self.system_status.uptime_seconds,
            self.system_status.total_control_cycles
        )
    
    fn get_subsystem_status(self) -> (String, String, String):
        """
        Get status of all subsystems.
        
        Returns:
            (ai_controller_status, safety_monitor_status, state_estimator_status)
        """
        var ai_status = "operational" if self.ai_controller.initialized else "not_initialized"
        var safety_status = self.system_status.safety_status
        var estimator_status = "operational"  # Simplified
        
        return (ai_status, safety_status, estimator_status)
    
    fn emergency_shutdown(mut self):
        """Emergency shutdown of entire system."""
        self.control_loop_active = False
        self.system_status.control_active = False
        self.system_status.system_operational = False
        self.system_status.safety_status = "emergency_shutdown"
        self.system_status.performance_mode = "shutdown"
        
        print("EMERGENCY SHUTDOWN ACTIVATED")
    
    fn reset_system(mut self):
        """Reset entire control system to initial state."""
        # Reset all subsystems
        self.ai_controller.reset_controller()
        self.safety_monitor.reset_safety_system()
        self.state_estimator.reset_estimator()
        
        # Reset system status
        self.system_status.system_operational = False
        self.system_status.control_active = False
        self.system_status.safety_status = "init"
        self.system_status.performance_mode = "startup"
        self.system_status.uptime_seconds = 0.0
        self.system_status.total_control_cycles = 0
        self.system_status.successful_inversions = 0
        self.system_status.current_inversion_time = 0.0
        
        self.control_loop_active = False
        self.system_start_time = 0.0
        self.last_control_time = 0.0
        self.performance_history = List[Float64]()
        
        print("Integrated Control System reset successfully")
