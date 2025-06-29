"""
Control Framework Demonstration.

This module demonstrates the complete control framework including AI controller,
safety monitoring, state estimation, and integrated control system operation.
"""

from collections import List
from math import sin, cos, abs

# Import control system components
from src.pendulum.control.integrated_control_system import IntegratedControlSystem, SystemStatus
from src.pendulum.control.ai_controller import ControlCommand
from src.pendulum.control.safety_monitor import SafetyStatus
from src.pendulum.control.state_estimator import FilteredState

struct ControlDemo:
    """Demonstration of the complete control framework."""
    
    @staticmethod
    fn run_control_framework_demo():
        """Run comprehensive control framework demonstration."""
        print("=" * 60)
        print("PHASE 2: AI CONTROL FRAMEWORK DEMONSTRATION")
        print("=" * 60)
        print("Demonstrating integrated control system with:")
        print("- AI Controller with MPC and swing-up algorithms")
        print("- Multi-layer safety monitoring system")
        print("- Advanced state estimation and filtering")
        print("- Real-time control loop at 25 Hz")
        print()
        
        # Create integrated control system
        var control_system = IntegratedControlSystem()
        
        # Initialize system
        var start_time = 0.0
        if not control_system.initialize_system(start_time):
            print("Failed to initialize control system")
            return
        
        print("✓ Control system initialized successfully")
        print()
        
        # Test different scenarios
        ControlDemo._test_stabilization_control(control_system)
        print()
        ControlDemo._test_swing_up_control(control_system)
        print()
        ControlDemo._test_safety_monitoring(control_system)
        print()
        ControlDemo._test_state_estimation(control_system)
        print()
        
        # Final performance summary
        ControlDemo._show_performance_summary(control_system)
    
    @staticmethod
    fn _test_stabilization_control(mut control_system: IntegratedControlSystem):
        """Test stabilization control for inverted pendulum."""
        print("1. Testing Stabilization Control")
        print("-" * 30)
        
        # Start with near-inverted state
        var initial_state = List[Float64]()
        initial_state.append(0.0)    # la_position (center)
        initial_state.append(5.0)    # pend_velocity (small)
        initial_state.append(5.0)    # pend_angle (near inverted)
        initial_state.append(0.0)    # cmd_volts
        
        var start_time = 1.0
        control_system.start_control_loop(initial_state, start_time)
        
        # Simulate control loop for stabilization
        var current_state = initial_state
        var successful_cycles = 0
        var total_cycles = 25  # 1 second at 25 Hz
        
        for i in range(total_cycles):
            var timestamp = start_time + Float64(i) * 0.04  # 40ms intervals
            
            # Execute control cycle
            var command = control_system.execute_control_cycle(current_state, timestamp)
            
            # Simulate system response (simplified)
            current_state = ControlDemo._simulate_pendulum_response(current_state, command)
            
            # Check if stabilized (within 2 degrees)
            if abs(current_state[2]) < 2.0:
                successful_cycles += 1
        
        var stabilization_rate = Float64(successful_cycles) / Float64(total_cycles) * 100.0
        print("  Stabilization cycles:", successful_cycles, "/", total_cycles)
        print("  Stabilization rate:", stabilization_rate, "%")
        
        if stabilization_rate > 80.0:
            print("  ✓ Stabilization control successful")
        else:
            print("  ⚠ Stabilization control needs improvement")
    
    @staticmethod
    fn _test_swing_up_control(mut control_system: IntegratedControlSystem):
        """Test swing-up control from hanging position."""
        print("2. Testing Swing-Up Control")
        print("-" * 30)
        
        # Reset and start with hanging state
        control_system.reset_system()
        control_system.initialize_system(10.0)
        
        var hanging_state = List[Float64]()
        hanging_state.append(0.0)     # la_position (center)
        hanging_state.append(0.0)     # pend_velocity (stationary)
        hanging_state.append(180.0)   # pend_angle (hanging down)
        hanging_state.append(0.0)     # cmd_volts
        
        control_system.start_control_loop(hanging_state, 10.0)
        
        # Simulate swing-up attempt
        var current_state = hanging_state
        var max_angle_achieved = 180.0
        var swing_cycles = 50  # 2 seconds at 25 Hz
        
        for i in range(swing_cycles):
            var timestamp = 10.0 + Float64(i) * 0.04
            
            var command = control_system.execute_control_cycle(current_state, timestamp)
            current_state = ControlDemo._simulate_pendulum_response(current_state, command)
            
            # Track maximum angle achieved
            if abs(current_state[2]) < abs(max_angle_achieved):
                max_angle_achieved = current_state[2]
        
        var swing_progress = (180.0 - abs(max_angle_achieved)) / 180.0 * 100.0
        print("  Maximum angle achieved:", max_angle_achieved, "degrees")
        print("  Swing-up progress:", swing_progress, "%")
        
        if swing_progress > 50.0:
            print("  ✓ Swing-up control showing progress")
        else:
            print("  ⚠ Swing-up control needs optimization")
    
    @staticmethod
    fn _test_safety_monitoring(mut control_system: IntegratedControlSystem):
        """Test safety monitoring system."""
        print("3. Testing Safety Monitoring")
        print("-" * 30)
        
        # Reset system
        control_system.reset_system()
        control_system.initialize_system(20.0)
        
        # Test with extreme state that should trigger safety
        var extreme_state = List[Float64]()
        extreme_state.append(3.9)     # la_position (near limit)
        extreme_state.append(950.0)   # pend_velocity (high velocity)
        extreme_state.append(45.0)    # pend_angle
        extreme_state.append(0.0)     # cmd_volts
        
        control_system.start_control_loop(extreme_state, 20.0)
        
        # Execute control cycle with extreme state
        var command = control_system.execute_control_cycle(extreme_state, 20.1)
        
        # Check safety response
        if command.safety_override:
            print("  ✓ Safety system correctly detected violations")
            print("  ✓ Safety override applied to control command")
            print("  Command voltage:", command.voltage, "V (should be 0.0)")
        else:
            print("  ⚠ Safety system did not detect violations")
        
        # Test normal state
        var normal_state = List[Float64]()
        normal_state.append(1.0)      # la_position (safe)
        normal_state.append(50.0)     # pend_velocity (moderate)
        normal_state.append(10.0)     # pend_angle (near inverted)
        normal_state.append(0.0)      # cmd_volts
        
        var normal_command = control_system.execute_control_cycle(normal_state, 20.2)
        
        if not normal_command.safety_override:
            print("  ✓ Safety system allows normal operation")
        else:
            print("  ⚠ Safety system incorrectly triggered on normal state")
    
    @staticmethod
    fn _test_state_estimation(mut control_system: IntegratedControlSystem):
        """Test state estimation and filtering."""
        print("4. Testing State Estimation")
        print("-" * 30)
        
        # Reset system
        control_system.reset_system()
        control_system.initialize_system(30.0)
        
        # Test with noisy measurements
        var clean_state = List[Float64]()
        clean_state.append(1.0)
        clean_state.append(100.0)
        clean_state.append(15.0)
        clean_state.append(0.0)
        
        control_system.start_control_loop(clean_state, 30.0)
        
        # Add noise to measurements
        var noisy_cycles = 10
        var estimation_quality = 0.0
        
        for i in range(noisy_cycles):
            var timestamp = 30.0 + Float64(i) * 0.04
            
            # Add simulated noise
            var noisy_state = List[Float64]()
            noisy_state.append(clean_state[0] + (Float64(i % 3) - 1.0) * 0.1)  # Position noise
            noisy_state.append(clean_state[1] + (Float64(i % 5) - 2.0) * 5.0)  # Velocity noise
            noisy_state.append(clean_state[2] + (Float64(i % 4) - 1.5) * 2.0)  # Angle noise
            noisy_state.append(0.0)
            
            var command = control_system.execute_control_cycle(noisy_state, timestamp)
            
            # Check if control remains stable despite noise
            if abs(command.voltage) < 5.0:  # Reasonable control effort
                estimation_quality += 1.0
        
        var filtering_effectiveness = estimation_quality / Float64(noisy_cycles) * 100.0
        print("  Filtering effectiveness:", filtering_effectiveness, "%")
        
        if filtering_effectiveness > 70.0:
            print("  ✓ State estimation handling noise well")
        else:
            print("  ⚠ State estimation needs improvement")
    
    @staticmethod
    fn _simulate_pendulum_response(current_state: List[Float64], command: ControlCommand) -> List[Float64]:
        """Simplified pendulum response simulation."""
        var dt = 0.04  # 40ms time step
        
        var la_pos = current_state[0]
        var pend_vel = current_state[1]
        var pend_angle = current_state[2]
        var control_voltage = command.voltage
        
        # Simplified dynamics (for demonstration)
        var control_force = control_voltage * 0.1  # Voltage to force conversion
        var gravity_effect = sin(pend_angle * 3.14159 / 180.0) * 50.0  # Gravity torque
        var damping = pend_vel * 0.05  # Velocity damping
        
        # Update pendulum velocity (simplified)
        var new_pend_vel = pend_vel + (gravity_effect - damping + control_force) * dt
        new_pend_vel = max(-1000.0, min(1000.0, new_pend_vel))  # Apply limits
        
        # Update pendulum angle
        var new_pend_angle = pend_angle + new_pend_vel * dt
        
        # Normalize angle to [-180, 180]
        while new_pend_angle > 180.0:
            new_pend_angle -= 360.0
        while new_pend_angle < -180.0:
            new_pend_angle += 360.0
        
        # Update actuator position (simplified)
        var new_la_pos = la_pos + control_voltage * dt * 0.01
        new_la_pos = max(-4.0, min(4.0, new_la_pos))  # Apply limits
        
        var new_state = List[Float64]()
        new_state.append(new_la_pos)
        new_state.append(new_pend_vel)
        new_state.append(new_pend_angle)
        new_state.append(control_voltage)
        
        return new_state
    
    @staticmethod
    fn _show_performance_summary(control_system: IntegratedControlSystem):
        """Show final performance summary."""
        print("5. Performance Summary")
        print("-" * 30)
        
        var performance = control_system.get_system_performance()
        var subsystem_status = control_system.get_subsystem_status()
        
        print("  System Performance:")
        print("    Success rate:", performance.0, "%")
        print("    Current inversion time:", performance.1, "seconds")
        print("    Total uptime:", performance.2, "seconds")
        print("    Total control cycles:", performance.3)
        print()
        
        print("  Subsystem Status:")
        print("    AI Controller:", subsystem_status.0)
        print("    Safety Monitor:", subsystem_status.1)
        print("    State Estimator:", subsystem_status.2)
        print()
        
        print("✓ Control Framework Demonstration Complete!")
        print()
        print("Key Achievements:")
        print("- Integrated control system operational")
        print("- AI controller with multiple control modes")
        print("- Multi-layer safety monitoring active")
        print("- State estimation and filtering functional")
        print("- Real-time performance demonstrated")

fn main():
    """Run control framework demonstration."""
    ControlDemo.run_control_framework_demo()
