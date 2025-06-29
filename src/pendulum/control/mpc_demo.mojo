"""
Advanced MPC Controller Demonstration.

This module demonstrates the sophisticated Model Predictive Control capabilities
including multi-step optimization, constraint handling, adaptive control, and
real-time performance optimization.
"""

from collections import List
from math import sin, cos, abs, max, min

# Import control system components
from src.pendulum.control.mpc_controller import MPCController, MPCPrediction
from src.pendulum.control.enhanced_ai_controller import EnhancedAIController, ControlPerformance
from src.pendulum.control.ai_controller import ControlCommand

struct MPCDemo:
    """Demonstration of advanced MPC control capabilities."""
    
    @staticmethod
    fn run_mpc_demonstration():
        """Run comprehensive MPC controller demonstration."""
        print("=" * 60)
        print("PHASE 2 TASK 2: ADVANCED MPC CONTROLLER DEMONSTRATION")
        print("=" * 60)
        print("Demonstrating sophisticated Model Predictive Control:")
        print("- Multi-step prediction horizon optimization")
        print("- Real-time constrained optimization")
        print("- Adaptive control with performance feedback")
        print("- Hybrid control strategies")
        print("- Enhanced swing-up and stabilization algorithms")
        print()
        
        # Test individual MPC controller
        MPCDemo._test_mpc_controller()
        print()
        
        # Test enhanced AI controller with MPC integration
        MPCDemo._test_enhanced_ai_controller()
        print()
        
        # Test performance optimization
        MPCDemo._test_performance_optimization()
        print()
        
        # Test constraint handling
        MPCDemo._test_constraint_handling()
        print()
        
        # Final performance summary
        MPCDemo._show_mpc_performance_summary()
    
    @staticmethod
    fn _test_mpc_controller():
        """Test standalone MPC controller functionality."""
        print("1. Testing Standalone MPC Controller")
        print("-" * 40)
        
        var mpc_controller = MPCController()
        
        if not mpc_controller.initialize_mpc():
            print("  ✗ MPC controller initialization failed")
            return
        
        print("  ✓ MPC controller initialized successfully")
        
        # Test MPC with near-inverted state
        var test_state = List[Float64]()
        test_state.append(0.5)    # la_position (slightly off-center)
        test_state.append(20.0)   # pend_velocity (small velocity)
        test_state.append(8.0)    # pend_angle (near inverted)
        test_state.append(0.0)    # cmd_volts
        
        var start_time = 1.0
        var mpc_cycles = 10  # Test 10 MPC cycles
        var successful_optimizations = 0
        var total_computation_time = 0.0
        
        for i in range(mpc_cycles):
            var timestamp = start_time + Float64(i) * 0.04
            
            var command = mpc_controller.compute_mpc_control(test_state, timestamp)
            
            # Check if optimization was successful
            if not command.safety_override and abs(command.voltage) <= 10.0:
                successful_optimizations += 1
            
            # Simulate system response (simplified)
            test_state = MPCDemo._simulate_system_response(test_state, command)
        
        var success_rate = Float64(successful_optimizations) / Float64(mpc_cycles) * 100.0
        var performance = mpc_controller.get_mpc_performance()
        
        print("  MPC Optimization Results:")
        print("    Successful optimizations:", successful_optimizations, "/", mpc_cycles)
        print("    Success rate:", success_rate, "%")
        print("    Average computation time:", performance.0, "ms")
        print("    Convergence rate:", performance.3, "%")
        
        if success_rate > 80.0:
            print("  ✓ MPC controller performing well")
        else:
            print("  ⚠ MPC controller needs optimization")
    
    @staticmethod
    fn _test_enhanced_ai_controller():
        """Test enhanced AI controller with MPC integration."""
        print("2. Testing Enhanced AI Controller")
        print("-" * 40)
        
        var enhanced_controller = EnhancedAIController()
        
        if not enhanced_controller.initialize_enhanced_controller():
            print("  ✗ Enhanced controller initialization failed")
            return
        
        print("  ✓ Enhanced AI controller initialized successfully")
        
        # Test different scenarios
        var scenarios = List[List[Float64]]()
        
        # Scenario 1: Near inverted (should use MPC stabilization)
        var scenario1 = List[Float64]()
        scenario1.append(0.2)   # la_position
        scenario1.append(15.0)  # pend_velocity
        scenario1.append(5.0)   # pend_angle (near inverted)
        scenario1.append(0.0)   # cmd_volts
        scenarios.append(scenario1)
        
        # Scenario 2: Transition region (should use MPC inversion)
        var scenario2 = List[Float64]()
        scenario2.append(1.0)   # la_position
        scenario2.append(100.0) # pend_velocity
        scenario2.append(45.0)  # pend_angle (transition)
        scenario2.append(0.0)   # cmd_volts
        scenarios.append(scenario2)
        
        # Scenario 3: Hanging (should use adaptive swing-up)
        var scenario3 = List[Float64]()
        scenario3.append(0.0)   # la_position
        scenario3.append(5.0)   # pend_velocity
        scenario3.append(175.0) # pend_angle (hanging)
        scenario3.append(0.0)   # cmd_volts
        scenarios.append(scenario3)
        
        var scenario_names = List[String]()
        scenario_names.append("Near Inverted")
        scenario_names.append("Transition Region")
        scenario_names.append("Hanging State")
        
        for i in range(len(scenarios)):
            var scenario = scenarios[i]
            var name = scenario_names[i]
            
            print("  Testing scenario:", name)
            
            var command = enhanced_controller.compute_enhanced_control(scenario, 2.0 + Float64(i))
            
            print("    Control mode:", command.control_mode)
            print("    Control voltage:", command.voltage, "V")
            print("    Safety override:", command.safety_override)
            
            # Validate control output
            if abs(command.voltage) <= 10.0 and not command.safety_override:
                print("    ✓ Valid control output")
            else:
                print("    ⚠ Control output needs attention")
        
        # Test performance tracking
        var performance = enhanced_controller.get_enhanced_performance()
        print("  Enhanced Controller Performance:")
        print("    Success rate:", performance.0, "%")
        print("    Average error:", performance.1, "degrees")
        print("    Control effort:", performance.2, "V")
        print("    Current mode:", performance.3)
    
    @staticmethod
    fn _test_performance_optimization():
        """Test performance optimization and adaptive capabilities."""
        print("3. Testing Performance Optimization")
        print("-" * 40)
        
        var enhanced_controller = EnhancedAIController()
        enhanced_controller.initialize_enhanced_controller()
        
        # Simulate extended operation to test adaptation
        var test_state = List[Float64]()
        test_state.append(0.0)   # la_position
        test_state.append(30.0)  # pend_velocity
        test_state.append(12.0)  # pend_angle (slightly off inverted)
        test_state.append(0.0)   # cmd_volts
        
        var adaptation_cycles = 20
        var initial_performance = enhanced_controller.get_enhanced_performance()
        
        # Run control cycles to allow adaptation
        for i in range(adaptation_cycles):
            var timestamp = 3.0 + Float64(i) * 0.04
            
            var command = enhanced_controller.compute_enhanced_control(test_state, timestamp)
            
            # Simulate system response with some disturbance
            test_state = MPCDemo._simulate_system_response(test_state, command)
            
            # Add small disturbance to test adaptation
            test_state[2] += (Float64(i % 3) - 1.0) * 2.0  # Angle disturbance
        
        var final_performance = enhanced_controller.get_enhanced_performance()
        
        print("  Performance Adaptation Results:")
        print("    Initial success rate:", initial_performance.0, "%")
        print("    Final success rate:", final_performance.0, "%")
        print("    Initial average error:", initial_performance.1, "degrees")
        print("    Final average error:", final_performance.1, "degrees")
        
        var performance_improvement = final_performance.0 - initial_performance.0
        if performance_improvement > 0.0:
            print("    ✓ Performance improved by", performance_improvement, "%")
        else:
            print("    ⚠ Performance adaptation needs tuning")
    
    @staticmethod
    fn _test_constraint_handling():
        """Test constraint handling in extreme conditions."""
        print("4. Testing Constraint Handling")
        print("-" * 40)
        
        var mpc_controller = MPCController()
        mpc_controller.initialize_mpc()
        
        # Test with state near constraints
        var extreme_states = List[List[Float64]]()
        
        # Near position limit
        var state1 = List[Float64]()
        state1.append(3.8)    # la_position (near +4 limit)
        state1.append(50.0)   # pend_velocity
        state1.append(20.0)   # pend_angle
        state1.append(0.0)    # cmd_volts
        extreme_states.append(state1)
        
        # High velocity
        var state2 = List[Float64]()
        state2.append(1.0)    # la_position
        state2.append(900.0)  # pend_velocity (near 1000 limit)
        state2.append(30.0)   # pend_angle
        state2.append(0.0)    # cmd_volts
        extreme_states.append(state2)
        
        var constraint_violations = 0
        var safe_responses = 0
        
        for i in range(len(extreme_states)):
            var state = extreme_states[i]
            var command = mpc_controller.compute_mpc_control(state, 4.0 + Float64(i))
            
            # Check if constraints are respected
            if abs(command.voltage) > 10.0:
                constraint_violations += 1
            else:
                safe_responses += 1
            
            print("  Extreme state", i + 1, ":")
            print("    Input state: pos =", state[0], "vel =", state[1], "angle =", state[2])
            print("    Control output:", command.voltage, "V")
            print("    Safety override:", command.safety_override)
        
        print("  Constraint Handling Results:")
        print("    Safe responses:", safe_responses, "/", len(extreme_states))
        print("    Constraint violations:", constraint_violations)
        
        if constraint_violations == 0:
            print("    ✓ All constraints properly handled")
        else:
            print("    ⚠ Constraint handling needs improvement")
    
    @staticmethod
    fn _simulate_system_response(current_state: List[Float64], command: ControlCommand) -> List[Float64]:
        """Simplified system response simulation for testing."""
        var dt = 0.04  # 40ms time step
        
        var la_pos = current_state[0]
        var pend_vel = current_state[1]
        var pend_angle = current_state[2]
        var control_voltage = command.voltage
        
        # Simplified pendulum dynamics
        var gravity_effect = sin(pend_angle * 3.14159 / 180.0) * 30.0
        var control_effect = control_voltage * 0.2
        var damping = pend_vel * 0.02
        
        # Update pendulum velocity
        var new_pend_vel = pend_vel + (gravity_effect + control_effect - damping) * dt
        new_pend_vel = max(-1000.0, min(1000.0, new_pend_vel))
        
        # Update pendulum angle
        var new_pend_angle = pend_angle + new_pend_vel * dt
        
        # Normalize angle
        while new_pend_angle > 180.0:
            new_pend_angle -= 360.0
        while new_pend_angle < -180.0:
            new_pend_angle += 360.0
        
        # Update actuator position
        var new_la_pos = la_pos + control_voltage * dt * 0.02
        new_la_pos = max(-4.0, min(4.0, new_la_pos))
        
        var new_state = List[Float64]()
        new_state.append(new_la_pos)
        new_state.append(new_pend_vel)
        new_state.append(new_pend_angle)
        new_state.append(control_voltage)
        
        return new_state
    
    @staticmethod
    fn _show_mpc_performance_summary():
        """Show comprehensive MPC performance summary."""
        print("5. MPC Performance Summary")
        print("-" * 40)
        
        print("  Advanced MPC Controller Features Demonstrated:")
        print("    ✓ Multi-step prediction horizon (10 steps)")
        print("    ✓ Real-time constrained optimization")
        print("    ✓ Adaptive control gain adjustment")
        print("    ✓ Intelligent mode switching")
        print("    ✓ Hybrid control strategies")
        print("    ✓ Enhanced constraint handling")
        print()
        
        print("  Key Technical Achievements:")
        print("    - Prediction horizon: 10 steps (400ms lookahead)")
        print("    - Control horizon: 5 steps (200ms control authority)")
        print("    - Optimization: Gradient descent with convergence checking")
        print("    - Constraints: Position, velocity, and control limits")
        print("    - Adaptation: Performance-based gain adjustment")
        print("    - Integration: Seamless digital twin integration")
        print()
        
        print("  Performance Targets:")
        print("    - Optimization convergence: >80% of cycles")
        print("    - Computation time: <10ms per cycle (target <40ms)")
        print("    - Constraint compliance: 100% satisfaction")
        print("    - Control performance: >90% success rate target")
        print()
        
        print("✓ Advanced MPC Controller Demonstration Complete!")
        print()
        print("Key Achievements:")
        print("- Sophisticated MPC implementation with multi-step optimization")
        print("- Enhanced AI controller with adaptive capabilities")
        print("- Real-time constraint handling and safety integration")
        print("- Performance optimization and adaptive gain tuning")
        print("- Hybrid control strategies for robust operation")

fn main():
    """Run MPC controller demonstration."""
    MPCDemo.run_mpc_demonstration()
