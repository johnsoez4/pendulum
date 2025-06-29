"""
Advanced Model Predictive Control (MPC) for Inverted Pendulum.

This module implements a sophisticated MPC controller that uses the digital twin
for multi-step predictions and real-time optimization to achieve optimal control
performance with constraint satisfaction.
"""

from collections import List
from math import abs, max, min, sqrt, exp, sin, cos

# Import project modules
from src.pendulum.utils.physics import PendulumState, PendulumPhysics
from src.pendulum.digital_twin.integrated_trainer import PendulumNeuralNetwork
from src.pendulum.control.ai_controller import ControlCommand, ControlState

# MPC configuration constants
alias MPC_PREDICTION_HORIZON = 10    # Number of prediction steps
alias MPC_CONTROL_HORIZON = 5        # Number of control steps
alias MPC_SAMPLE_TIME = 0.04         # 40ms sample time (25 Hz)
alias MPC_MAX_ITERATIONS = 20        # Maximum optimization iterations
alias MPC_CONVERGENCE_TOL = 1e-4     # Optimization convergence tolerance

# MPC cost function weights
alias WEIGHT_ANGLE_ERROR = 100.0     # Weight for angle tracking error
alias WEIGHT_POSITION_ERROR = 10.0   # Weight for position tracking error
alias WEIGHT_VELOCITY_ERROR = 1.0    # Weight for velocity error
alias WEIGHT_CONTROL_EFFORT = 0.1    # Weight for control effort
alias WEIGHT_CONTROL_RATE = 0.5      # Weight for control rate of change

@fieldwise_init
struct MPCPrediction(Copyable, Movable):
    """MPC prediction trajectory."""
    
    var predicted_states: List[List[Float64]]  # Predicted state trajectory
    var control_sequence: List[Float64]        # Optimal control sequence
    var cost_trajectory: List[Float64]         # Cost at each prediction step
    var constraint_violations: List[Bool]      # Constraint violation flags
    var optimization_converged: Bool           # Optimization convergence flag
    var computation_time: Float64              # Computation time (ms)
    
    fn is_valid(self) -> Bool:
        """Check if MPC prediction is valid."""
        return (self.optimization_converged and 
                len(self.predicted_states) == MPC_PREDICTION_HORIZON and
                len(self.control_sequence) == MPC_CONTROL_HORIZON)

@fieldwise_init
struct MPCObjective(Copyable, Movable):
    """MPC objective function configuration."""
    
    var target_angle: Float64          # Target pendulum angle (degrees)
    var target_position: Float64       # Target cart position (inches)
    var target_velocity: Float64       # Target angular velocity (deg/s)
    var weight_angle: Float64          # Weight for angle tracking
    var weight_position: Float64       # Weight for position tracking
    var weight_velocity: Float64       # Weight for velocity tracking
    var weight_control: Float64        # Weight for control effort
    var weight_rate: Float64           # Weight for control rate
    
    fn compute_stage_cost(self, state: List[Float64], control: Float64, prev_control: Float64) -> Float64:
        """Compute cost for a single MPC stage."""
        var angle_error = abs(state[2] - self.target_angle)
        var position_error = abs(state[0] - self.target_position)
        var velocity_error = abs(state[1] - self.target_velocity)
        var control_effort = control * control
        var control_rate = (control - prev_control) * (control - prev_control)
        
        return (self.weight_angle * angle_error * angle_error +
                self.weight_position * position_error * position_error +
                self.weight_velocity * velocity_error * velocity_error +
                self.weight_control * control_effort +
                self.weight_rate * control_rate)

struct MPCController:
    """
    Advanced Model Predictive Control for inverted pendulum.
    
    Features:
    - Multi-step prediction using digital twin
    - Real-time constrained optimization
    - Receding horizon control
    - Constraint handling for position, velocity, and control limits
    - Adaptive cost function weights
    - Real-time performance optimization
    """
    
    var digital_twin: PendulumNeuralNetwork
    var physics_model: PendulumPhysics
    var mpc_objective: MPCObjective
    var prediction_history: List[MPCPrediction]
    var control_history: List[Float64]
    var optimization_stats: List[Float64]
    var initialized: Bool
    
    fn __init__(out self):
        """Initialize MPC controller."""
        # Initialize digital twin
        var weights1 = List[List[Float64]]()
        var biases1 = List[Float64]()
        var weights2 = List[List[Float64]]()
        var biases2 = List[Float64]()
        
        self.digital_twin = PendulumNeuralNetwork(weights1, biases1, weights2, biases2, True, 0.0, 0.0)
        self.physics_model = PendulumPhysics()
        
        # Initialize MPC objective for inverted pendulum
        self.mpc_objective = MPCObjective(
            0.0,                    # target_angle (inverted)
            0.0,                    # target_position (center)
            0.0,                    # target_velocity (stationary)
            WEIGHT_ANGLE_ERROR,     # weight_angle
            WEIGHT_POSITION_ERROR,  # weight_position
            WEIGHT_VELOCITY_ERROR,  # weight_velocity
            WEIGHT_CONTROL_EFFORT,  # weight_control
            WEIGHT_CONTROL_RATE     # weight_rate
        )
        
        self.prediction_history = List[MPCPrediction]()
        self.control_history = List[Float64]()
        self.optimization_stats = List[Float64]()
        self.initialized = False
    
    fn initialize_mpc(mut self) -> Bool:
        """Initialize MPC controller and validate digital twin."""
        # Initialize digital twin
        self.digital_twin.initialize_weights()
        
        # Test digital twin functionality
        var test_state = List[Float64]()
        test_state.append(0.0)    # la_position
        test_state.append(0.0)    # pend_velocity
        test_state.append(180.0)  # pend_position (hanging)
        test_state.append(0.0)    # cmd_volts
        
        var prediction = self.digital_twin.forward(test_state)
        
        if len(prediction) == 3:
            self.initialized = True
            print("MPC Controller initialized successfully")
            return True
        else:
            print("MPC Controller initialization failed")
            return False
    
    fn compute_mpc_control(mut self, current_state: List[Float64], timestamp: Float64) -> ControlCommand:
        """
        Compute optimal control using Model Predictive Control.
        
        Args:
            current_state: [la_position, pend_velocity, pend_position, cmd_volts]
            timestamp: Current timestamp
            
        Returns:
            Optimal control command
        """
        if not self.initialized:
            return self._create_emergency_command(timestamp)
        
        var start_time = timestamp  # Simplified timing
        
        # Solve MPC optimization problem
        var mpc_prediction = self._solve_mpc_optimization(current_state)
        
        var end_time = timestamp + 0.001  # Simplified timing
        mpc_prediction.computation_time = (end_time - start_time) * 1000.0  # Convert to ms
        
        # Extract optimal control action
        var optimal_control = 0.0
        if mpc_prediction.is_valid() and len(mpc_prediction.control_sequence) > 0:
            optimal_control = mpc_prediction.control_sequence[0]
        
        # Apply safety constraints
        optimal_control = self._apply_mpc_constraints(optimal_control, current_state)
        
        # Create control command
        var predicted_state = List[Float64]()
        if len(mpc_prediction.predicted_states) > 1:
            predicted_state = mpc_prediction.predicted_states[1]  # Next state prediction
        else:
            predicted_state = current_state  # Fallback
        
        var command = ControlCommand(
            optimal_control,
            timestamp,
            "mpc",
            False,  # safety_override
            predicted_state
        )
        
        # Store prediction and control history
        self.prediction_history.append(mpc_prediction)
        self.control_history.append(optimal_control)
        
        # Update optimization statistics
        self._update_optimization_stats(mpc_prediction)
        
        return command
    
    fn _solve_mpc_optimization(self, current_state: List[Float64]) -> MPCPrediction:
        """Solve MPC optimization problem using gradient descent."""
        # Initialize control sequence
        var control_sequence = List[Float64]()
        for _ in range(MPC_CONTROL_HORIZON):
            control_sequence.append(0.0)  # Start with zero control
        
        var best_control_sequence = control_sequence
        var best_cost = 1e6
        var converged = False
        
        # Gradient descent optimization
        for iteration in range(MPC_MAX_ITERATIONS):
            # Evaluate current control sequence
            var prediction = self._predict_trajectory(current_state, control_sequence)
            var total_cost = self._evaluate_trajectory_cost(prediction.predicted_states, control_sequence)
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_control_sequence = control_sequence
            
            # Simple gradient descent update
            var improved_sequence = self._gradient_descent_step(current_state, control_sequence)
            
            # Check convergence
            var improvement = abs(total_cost - best_cost)
            if improvement < MPC_CONVERGENCE_TOL:
                converged = True
                break
            
            control_sequence = improved_sequence
        
        # Generate final prediction with best control sequence
        var final_prediction = self._predict_trajectory(current_state, best_control_sequence)
        final_prediction.control_sequence = best_control_sequence
        final_prediction.optimization_converged = converged
        
        return final_prediction
    
    fn _predict_trajectory(self, initial_state: List[Float64], control_sequence: List[Float64]) -> MPCPrediction:
        """Predict state trajectory using digital twin."""
        var predicted_states = List[List[Float64]]()
        var cost_trajectory = List[Float64]()
        var constraint_violations = List[Bool]()
        
        # Add initial state
        predicted_states.append(initial_state)
        
        var current_state = initial_state
        var prev_control = 0.0
        
        for i in range(MPC_PREDICTION_HORIZON):
            # Get control input for this step
            var control_input = 0.0
            if i < len(control_sequence):
                control_input = control_sequence[i]
            else:
                control_input = control_sequence[len(control_sequence) - 1]  # Hold last control
            
            # Create input for digital twin
            var twin_input = List[Float64]()
            twin_input.append(current_state[0])  # la_position
            twin_input.append(current_state[1])  # pend_velocity
            twin_input.append(current_state[2])  # pend_position
            twin_input.append(control_input)     # cmd_volts
            
            # Predict next state
            var next_state = self.digital_twin.forward(twin_input)
            
            # Add control input to state for consistency
            if len(next_state) == 3:
                next_state.append(control_input)
            
            predicted_states.append(next_state)
            
            # Compute stage cost
            var stage_cost = self.mpc_objective.compute_stage_cost(next_state, control_input, prev_control)
            cost_trajectory.append(stage_cost)
            
            # Check constraints
            var violates_constraints = self._check_state_constraints(next_state, control_input)
            constraint_violations.append(violates_constraints)
            
            # Update for next iteration
            current_state = next_state
            prev_control = control_input
        
        var prediction = MPCPrediction(
            predicted_states,
            List[Float64](),  # Will be filled by optimization
            cost_trajectory,
            constraint_violations,
            False,  # Will be set by optimization
            0.0     # Will be set by optimization
        )
        
        return prediction
    
    fn _evaluate_trajectory_cost(self, predicted_states: List[List[Float64]], control_sequence: List[Float64]) -> Float64:
        """Evaluate total cost of predicted trajectory."""
        var total_cost = 0.0
        var prev_control = 0.0
        
        for i in range(min(len(predicted_states) - 1, len(control_sequence))):
            var state = predicted_states[i + 1]  # Skip initial state
            var control = control_sequence[i]
            
            var stage_cost = self.mpc_objective.compute_stage_cost(state, control, prev_control)
            
            # Add constraint penalty
            if self._check_state_constraints(state, control):
                stage_cost += 1000.0  # High penalty for constraint violations
            
            total_cost += stage_cost
            prev_control = control
        
        return total_cost
    
    fn _gradient_descent_step(self, current_state: List[Float64], control_sequence: List[Float64]) -> List[Float64]:
        """Perform one gradient descent step for control optimization."""
        var improved_sequence = List[Float64]()
        var step_size = 0.1
        
        for i in range(len(control_sequence)):
            var current_control = control_sequence[i]
            
            # Compute finite difference gradient
            var perturbed_sequence_plus = control_sequence
            var perturbed_sequence_minus = control_sequence
            
            perturbed_sequence_plus[i] = current_control + step_size
            perturbed_sequence_minus[i] = current_control - step_size
            
            var cost_plus = self._evaluate_trajectory_cost(
                self._predict_trajectory(current_state, perturbed_sequence_plus).predicted_states,
                perturbed_sequence_plus
            )
            
            var cost_minus = self._evaluate_trajectory_cost(
                self._predict_trajectory(current_state, perturbed_sequence_minus).predicted_states,
                perturbed_sequence_minus
            )
            
            # Compute gradient
            var gradient = (cost_plus - cost_minus) / (2.0 * step_size)
            
            # Update control with gradient descent
            var new_control = current_control - 0.01 * gradient
            
            # Apply control constraints
            new_control = max(-10.0, min(10.0, new_control))
            
            improved_sequence.append(new_control)
        
        return improved_sequence
    
    fn _check_state_constraints(self, state: List[Float64], control: Float64) -> Bool:
        """Check if state and control violate constraints."""
        if len(state) < 3:
            return True  # Invalid state
        
        var la_position = state[0]
        var pend_velocity = state[1]
        
        # Check position constraints
        if abs(la_position) > 4.0:
            return True
        
        # Check velocity constraints
        if abs(pend_velocity) > 1000.0:
            return True
        
        # Check control constraints
        if abs(control) > 10.0:
            return True
        
        return False
    
    fn _apply_mpc_constraints(self, control: Float64, current_state: List[Float64]) -> Float64:
        """Apply final safety constraints to MPC control output."""
        # Clamp to voltage limits
        var constrained_control = max(-10.0, min(10.0, control))
        
        # Additional safety checks based on current state
        var la_position = current_state[0]
        var pend_velocity = current_state[1]
        
        # Reduce control authority near position limits
        if abs(la_position) > 3.5:
            constrained_control *= 0.5
        
        # Reduce control authority at high velocities
        if abs(pend_velocity) > 800.0:
            constrained_control *= 0.7
        
        return constrained_control
    
    fn _create_emergency_command(self, timestamp: Float64) -> ControlCommand:
        """Create emergency command when MPC fails."""
        var safe_predicted_state = List[Float64]()
        safe_predicted_state.append(0.0)
        safe_predicted_state.append(0.0)
        safe_predicted_state.append(0.0)
        
        return ControlCommand(
            0.0,                # voltage
            timestamp,          # timestamp
            "mpc_emergency",    # control_mode
            True,               # safety_override
            safe_predicted_state
        )
    
    fn _update_optimization_stats(mut self, prediction: MPCPrediction):
        """Update MPC optimization performance statistics."""
        self.optimization_stats.append(prediction.computation_time)
        
        # Keep only recent statistics
        if len(self.optimization_stats) > 100:
            var new_stats = List[Float64]()
            var start_idx = len(self.optimization_stats) - 100
            for i in range(start_idx, len(self.optimization_stats)):
                new_stats.append(self.optimization_stats[i])
            self.optimization_stats = new_stats
    
    fn get_mpc_performance(self) -> (Float64, Float64, Int, Float64):
        """
        Get MPC performance metrics.
        
        Returns:
            (avg_computation_time, max_computation_time, total_predictions, convergence_rate)
        """
        if len(self.optimization_stats) == 0:
            return (0.0, 0.0, 0, 0.0)
        
        var sum_time = 0.0
        var max_time = 0.0
        for i in range(len(self.optimization_stats)):
            sum_time += self.optimization_stats[i]
            max_time = max(max_time, self.optimization_stats[i])
        
        var avg_time = sum_time / Float64(len(self.optimization_stats))
        
        # Count converged predictions
        var converged_count = 0
        for i in range(len(self.prediction_history)):
            if self.prediction_history[i].optimization_converged:
                converged_count += 1
        
        var convergence_rate = Float64(converged_count) / Float64(len(self.prediction_history)) * 100.0
        
        return (avg_time, max_time, len(self.prediction_history), convergence_rate)
    
    fn set_mpc_target(mut self, target_angle: Float64, target_position: Float64, target_velocity: Float64):
        """Set MPC control targets."""
        self.mpc_objective.target_angle = target_angle
        self.mpc_objective.target_position = target_position
        self.mpc_objective.target_velocity = target_velocity
        print("MPC targets updated: angle =", target_angle, "position =", target_position, "velocity =", target_velocity)
    
    fn reset_mpc(mut self):
        """Reset MPC controller state."""
        self.prediction_history = List[MPCPrediction]()
        self.control_history = List[Float64]()
        self.optimization_stats = List[Float64]()
        print("MPC Controller reset successfully")
