"""
State Estimator for Inverted Pendulum Control System.

This module implements state estimation and filtering for the pendulum control
system, providing noise filtering, derivative estimation, and state prediction
capabilities for robust control.
"""

from collections import List
from math import abs, max, min, sqrt, exp

# Import project modules
from src.pendulum.utils.physics import PendulumState, PendulumPhysics

# State estimation constants
alias FILTER_ALPHA = 0.8           # Low-pass filter coefficient
alias DERIVATIVE_WINDOW = 5        # Window size for derivative estimation
alias OUTLIER_THRESHOLD = 3.0      # Standard deviations for outlier detection
alias MAX_STATE_CHANGE = 100.0     # Maximum allowed state change per step
alias ESTIMATION_FREQUENCY = 25.0  # Hz

@fieldwise_init
struct FilteredState(Copyable, Movable):
    """Filtered and estimated state information."""
    
    var la_position: Float64           # Filtered actuator position (inches)
    var la_velocity: Float64           # Estimated actuator velocity (inches/s)
    var pend_angle: Float64            # Filtered pendulum angle (degrees)
    var pend_velocity: Float64         # Filtered pendulum velocity (deg/s)
    var pend_acceleration: Float64     # Estimated pendulum acceleration (deg/s^2)
    var timestamp: Float64             # State timestamp
    var confidence: Float64            # Estimation confidence [0, 1]
    var outlier_detected: Bool         # Outlier detection flag
    
    fn is_valid(self) -> Bool:
        """Check if estimated state is valid."""
        return (self.confidence > 0.5 and 
                not self.outlier_detected and
                abs(self.la_position) <= 5.0 and
                abs(self.pend_velocity) <= 1200.0)

@fieldwise_init
struct StateHistory(Copyable, Movable):
    """Historical state data for estimation."""
    
    var positions: List[Float64]       # Position history
    var velocities: List[Float64]      # Velocity history
    var angles: List[Float64]          # Angle history
    var angular_velocities: List[Float64]  # Angular velocity history
    var timestamps: List[Float64]      # Timestamp history
    var max_history: Int               # Maximum history length
    
    fn add_state(mut self, la_pos: Float64, pend_vel: Float64, pend_angle: Float64, timestamp: Float64):
        """Add new state to history."""
        self.positions.append(la_pos)
        self.velocities.append(0.0)  # Will be estimated
        self.angles.append(pend_angle)
        self.angular_velocities.append(pend_vel)
        self.timestamps.append(timestamp)
        
        # Maintain maximum history length
        if len(self.positions) > self.max_history:
            # Remove oldest entries (simplified - in practice would use circular buffer)
            var new_positions = List[Float64]()
            var new_velocities = List[Float64]()
            var new_angles = List[Float64]()
            var new_angular_velocities = List[Float64]()
            var new_timestamps = List[Float64]()
            
            var start_idx = len(self.positions) - self.max_history
            for i in range(start_idx, len(self.positions)):
                new_positions.append(self.positions[i])
                new_velocities.append(self.velocities[i])
                new_angles.append(self.angles[i])
                new_angular_velocities.append(self.angular_velocities[i])
                new_timestamps.append(self.timestamps[i])
            
            self.positions = new_positions
            self.velocities = new_velocities
            self.angles = new_angles
            self.angular_velocities = new_angular_velocities
            self.timestamps = new_timestamps
    
    fn get_recent_states(self, count: Int) -> List[List[Float64]]:
        """Get recent states for analysis."""
        var recent = List[List[Float64]]()
        var start_idx = max(0, len(self.positions) - count)
        
        for i in range(start_idx, len(self.positions)):
            var state = List[Float64]()
            state.append(self.positions[i])
            state.append(self.angular_velocities[i])
            state.append(self.angles[i])
            state.append(self.timestamps[i])
            recent.append(state)
        
        return recent

struct StateEstimator:
    """
    Advanced state estimator for pendulum control system.
    
    Provides:
    - Noise filtering using low-pass filters
    - Derivative estimation using finite differences
    - Outlier detection and rejection
    - State prediction and validation
    - Confidence estimation
    """
    
    var physics_model: PendulumPhysics
    var state_history: StateHistory
    var filtered_state: FilteredState
    var last_raw_state: List[Float64]
    var filter_initialized: Bool
    var estimation_statistics: List[Float64]
    
    fn __init__(out self):
        """Initialize state estimator."""
        self.physics_model = PendulumPhysics()
        
        # Initialize state history
        var positions = List[Float64]()
        var velocities = List[Float64]()
        var angles = List[Float64]()
        var angular_velocities = List[Float64]()
        var timestamps = List[Float64]()
        
        self.state_history = StateHistory(
            positions, velocities, angles, angular_velocities, timestamps, 20
        )
        
        # Initialize filtered state
        self.filtered_state = FilteredState(
            0.0, 0.0, 180.0, 0.0, 0.0, 0.0, 1.0, False
        )
        
        self.last_raw_state = List[Float64]()
        self.filter_initialized = False
        self.estimation_statistics = List[Float64]()
    
    fn initialize_estimator(mut self, initial_state: List[Float64], timestamp: Float64):
        """Initialize estimator with first measurement."""
        if len(initial_state) >= 3:
            self.filtered_state.la_position = initial_state[0]
            self.filtered_state.pend_velocity = initial_state[1]
            self.filtered_state.pend_angle = initial_state[2]
            self.filtered_state.la_velocity = 0.0
            self.filtered_state.pend_acceleration = 0.0
            self.filtered_state.timestamp = timestamp
            self.filtered_state.confidence = 1.0
            self.filtered_state.outlier_detected = False
            
            self.last_raw_state = initial_state
            self.filter_initialized = True
            
            # Add to history
            self.state_history.add_state(
                initial_state[0], initial_state[1], initial_state[2], timestamp
            )
            
            print("State estimator initialized")
    
    fn estimate_state(mut self, raw_state: List[Float64], timestamp: Float64) -> FilteredState:
        """
        Estimate filtered state from raw measurements.
        
        Args:
            raw_state: [la_position, pend_velocity, pend_position, cmd_volts]
            timestamp: Current timestamp
            
        Returns:
            Filtered and estimated state
        """
        if not self.filter_initialized:
            self.initialize_estimator(raw_state, timestamp)
            return self.filtered_state
        
        # Extract raw measurements
        var raw_la_pos = raw_state[0]
        var raw_pend_vel = raw_state[1]
        var raw_pend_angle = raw_state[2]
        
        # Outlier detection
        var outlier_detected = self._detect_outliers(raw_state, timestamp)
        
        if outlier_detected:
            # Use prediction instead of measurement
            self.filtered_state = self._predict_state(timestamp)
            self.filtered_state.outlier_detected = True
            self.filtered_state.confidence *= 0.8  # Reduce confidence
        else:
            # Apply filtering
            self.filtered_state.la_position = self._apply_low_pass_filter(
                self.filtered_state.la_position, raw_la_pos, FILTER_ALPHA
            )
            
            self.filtered_state.pend_velocity = self._apply_low_pass_filter(
                self.filtered_state.pend_velocity, raw_pend_vel, FILTER_ALPHA
            )
            
            self.filtered_state.pend_angle = self._filter_angle(
                self.filtered_state.pend_angle, raw_pend_angle
            )
            
            # Estimate derivatives
            self.filtered_state.la_velocity = self._estimate_derivative(
                self.state_history.positions, self.state_history.timestamps
            )
            
            self.filtered_state.pend_acceleration = self._estimate_derivative(
                self.state_history.angular_velocities, self.state_history.timestamps
            )
            
            self.filtered_state.timestamp = timestamp
            self.filtered_state.outlier_detected = False
            self.filtered_state.confidence = min(1.0, self.filtered_state.confidence + 0.1)
        
        # Add to history
        self.state_history.add_state(raw_la_pos, raw_pend_vel, raw_pend_angle, timestamp)
        
        # Update statistics
        self._update_statistics()
        
        # Store raw state
        self.last_raw_state = raw_state
        
        return self.filtered_state
    
    fn _apply_low_pass_filter(self, filtered_value: Float64, raw_value: Float64, alpha: Float64) -> Float64:
        """Apply low-pass filter to reduce noise."""
        return alpha * filtered_value + (1.0 - alpha) * raw_value
    
    fn _filter_angle(self, filtered_angle: Float64, raw_angle: Float64) -> Float64:
        """Apply angle-specific filtering handling wraparound."""
        # Handle angle wraparound
        var angle_diff = raw_angle - filtered_angle
        
        if angle_diff > 180.0:
            angle_diff -= 360.0
        elif angle_diff < -180.0:
            angle_diff += 360.0
        
        var filtered_diff = self._apply_low_pass_filter(0.0, angle_diff, FILTER_ALPHA)
        var new_angle = filtered_angle + filtered_diff
        
        # Normalize to [-180, 180]
        while new_angle > 180.0:
            new_angle -= 360.0
        while new_angle < -180.0:
            new_angle += 360.0
        
        return new_angle
    
    fn _estimate_derivative(self, values: List[Float64], timestamps: List[Float64]) -> Float64:
        """Estimate derivative using finite differences."""
        var n = len(values)
        if n < 2:
            return 0.0
        
        # Use last two points for simple derivative
        var dt = timestamps[n-1] - timestamps[n-2]
        if dt <= 0.0:
            return 0.0
        
        var derivative = (values[n-1] - values[n-2]) / dt
        
        # Apply reasonable limits
        return max(-1000.0, min(1000.0, derivative))
    
    fn _detect_outliers(self, raw_state: List[Float64], timestamp: Float64) -> Bool:
        """Detect outliers in measurements."""
        if len(self.last_raw_state) == 0:
            return False
        
        var dt = timestamp - self.filtered_state.timestamp
        if dt <= 0.0:
            return True  # Invalid timestamp
        
        # Check for unreasonable changes
        var la_pos_change = abs(raw_state[0] - self.last_raw_state[0])
        var pend_vel_change = abs(raw_state[1] - self.last_raw_state[1])
        var pend_angle_change = abs(raw_state[2] - self.last_raw_state[2])
        
        # Maximum reasonable changes per time step
        var max_pos_change = 2.0 * dt  # 2 inches/second max
        var max_vel_change = 500.0 * dt  # 500 deg/s^2 max acceleration
        var max_angle_change = 100.0 * dt  # 100 deg/s max angular velocity
        
        if (la_pos_change > max_pos_change or 
            pend_vel_change > max_vel_change or
            pend_angle_change > max_angle_change):
            return True
        
        return False
    
    fn _predict_state(self, timestamp: Float64) -> FilteredState:
        """Predict state when measurement is unavailable."""
        var dt = timestamp - self.filtered_state.timestamp
        
        # Simple prediction using current velocity
        var predicted_la_pos = self.filtered_state.la_position + self.filtered_state.la_velocity * dt
        var predicted_pend_angle = self.filtered_state.pend_angle + self.filtered_state.pend_velocity * dt
        var predicted_pend_vel = self.filtered_state.pend_velocity + self.filtered_state.pend_acceleration * dt
        
        # Apply constraints
        predicted_la_pos = max(-4.5, min(4.5, predicted_la_pos))
        predicted_pend_vel = max(-1100.0, min(1100.0, predicted_pend_vel))
        
        var predicted_state = FilteredState(
            predicted_la_pos,
            self.filtered_state.la_velocity,
            predicted_pend_angle,
            predicted_pend_vel,
            self.filtered_state.pend_acceleration,
            timestamp,
            self.filtered_state.confidence * 0.9,  # Reduce confidence for prediction
            False
        )
        
        return predicted_state
    
    fn _update_statistics(mut self):
        """Update estimation performance statistics."""
        # Simple statistics tracking
        self.estimation_statistics.append(self.filtered_state.confidence)
        
        # Keep only recent statistics
        if len(self.estimation_statistics) > 100:
            var new_stats = List[Float64]()
            var start_idx = len(self.estimation_statistics) - 100
            for i in range(start_idx, len(self.estimation_statistics)):
                new_stats.append(self.estimation_statistics[i])
            self.estimation_statistics = new_stats
    
    fn get_estimation_quality(self) -> (Float64, Float64, Int):
        """
        Get estimation quality metrics.
        
        Returns:
            (average_confidence, current_confidence, outlier_count)
        """
        if len(self.estimation_statistics) == 0:
            return (0.0, 0.0, 0)
        
        var sum_confidence = 0.0
        for i in range(len(self.estimation_statistics)):
            sum_confidence += self.estimation_statistics[i]
        
        var avg_confidence = sum_confidence / Float64(len(self.estimation_statistics))
        
        # Count outliers in recent history
        var outlier_count = 0
        var recent_states = self.state_history.get_recent_states(10)
        # Simplified outlier counting
        
        return (avg_confidence, self.filtered_state.confidence, outlier_count)
    
    fn reset_estimator(mut self):
        """Reset state estimator to initial conditions."""
        self.state_history = StateHistory(
            List[Float64](), List[Float64](), List[Float64](), 
            List[Float64](), List[Float64](), 20
        )
        
        self.filtered_state = FilteredState(
            0.0, 0.0, 180.0, 0.0, 0.0, 0.0, 1.0, False
        )
        
        self.last_raw_state = List[Float64]()
        self.filter_initialized = False
        self.estimation_statistics = List[Float64]()
        
        print("State estimator reset successfully")
    
    fn get_state_prediction(self, prediction_time: Float64) -> FilteredState:
        """Get predicted state at future time."""
        return self._predict_state(self.filtered_state.timestamp + prediction_time)
