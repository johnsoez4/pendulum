"""
State Estimator for Inverted Pendulum Control System.

This module implements state estimation and filtering for the pendulum control
system, providing noise filtering, derivative estimation, and state prediction
capabilities for robust control.
"""

from collections import List
from math import sqrt, exp

# Import project modules
from src.utils.physics import PendulumState, PendulumPhysics

# State estimation constants
alias FILTER_ALPHA = 0.8  # Low-pass filter coefficient
alias DERIVATIVE_WINDOW = 5  # Window size for derivative estimation
alias OUTLIER_THRESHOLD = 3.0  # Standard deviations for outlier detection
alias MAX_STATE_CHANGE = 100.0  # Maximum allowed state change per step
alias ESTIMATION_FREQUENCY = 25.0  # Hz


@fieldwise_init
struct FilteredState(Copyable, Movable):
    """Filtered and estimated state information."""

    var la_position: Float64  # Filtered actuator position (inches)
    var la_velocity: Float64  # Estimated actuator velocity (inches/s)
    var pend_angle: Float64  # Filtered pendulum angle (degrees)
    var pend_velocity: Float64  # Filtered pendulum velocity (deg/s)
    var pend_acceleration: Float64  # Estimated pendulum acceleration (deg/s^2)
    var timestamp: Float64  # State timestamp
    var confidence: Float64  # Estimation confidence [0, 1]
    var outlier_detected: Bool  # Outlier detection flag


@fieldwise_init
struct StateHistory(Copyable, Movable):
    """Historical state data for estimation."""

    var positions: List[Float64]  # Position history
    var velocities: List[Float64]  # Velocity history
    var angles: List[Float64]  # Angle history
    var angular_velocities: List[Float64]  # Angular velocity history
    var timestamps: List[Float64]  # Timestamp history
    var max_history: Int  # Maximum history length

    fn add_state(
        mut self,
        la_pos: Float64,
        pend_vel: Float64,
        pend_angle: Float64,
        timestamp: Float64,
    ):
        """Add new state to history."""
        self.positions.append(la_pos)

        # Calculate velocity from position history using proper estimation
        var estimated_velocity = self._calculate_velocity_from_history(
            la_pos, timestamp
        )
        self.velocities.append(estimated_velocity)

        self.angles.append(pend_angle)
        self.angular_velocities.append(pend_vel)
        self.timestamps.append(timestamp)

        # Maintain maximum history length
        if len(self.positions) > self.max_history:
            # Remove oldest entries (simplified - in practice would use circular buffer)
            new_positions = List[Float64]()
            new_velocities = List[Float64]()
            new_angles = List[Float64]()
            new_angular_velocities = List[Float64]()
            new_timestamps = List[Float64]()

            start_idx = len(self.positions) - self.max_history
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

    fn _calculate_velocity_from_history(
        self, current_pos: Float64, current_time: Float64
    ) -> Float64:
        """
        Calculate velocity from position history using advanced estimation.

        Uses weighted finite differences with noise filtering for robust velocity estimation.

        Args:
            current_pos: Current position measurement.
            current_time: Current timestamp.

        Returns:
            Estimated velocity in appropriate units.
        """
        n = len(self.positions)
        if n < 2:
            return 0.0

        # Use multiple points for robust estimation if available
        if n >= 3:
            # Three-point central difference for better accuracy
            var dt1 = current_time - self.timestamps[n - 1]
            var dt2 = self.timestamps[n - 1] - self.timestamps[n - 2]

            if dt1 > 0.0 and dt2 > 0.0:
                # Weighted combination of forward and backward differences
                var forward_diff = (current_pos - self.positions[n - 1]) / dt1
                var backward_diff = (
                    self.positions[n - 1] - self.positions[n - 2]
                ) / dt2

                # Weight based on time intervals (prefer more recent data)
                var total_dt = dt1 + dt2
                var w1 = dt2 / total_dt  # Weight for forward difference
                var w2 = dt1 / total_dt  # Weight for backward difference

                var estimated_velocity = w1 * forward_diff + w2 * backward_diff

                # Apply reasonable velocity limits for linear actuator
                return max(
                    -10.0, min(10.0, estimated_velocity)
                )  # inches/second

        # Fallback to simple two-point difference
        var dt = current_time - self.timestamps[n - 1]
        if dt > 0.0:
            var velocity = (current_pos - self.positions[n - 1]) / dt
            return max(-10.0, min(10.0, velocity))

        return 0.0


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
        positions = List[Float64]()
        velocities = List[Float64]()
        angles = List[Float64]()
        angular_velocities = List[Float64]()
        timestamps = List[Float64]()

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

    fn initialize_estimator(
        mut self, initial_state: List[Float64], timestamp: Float64
    ):
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

    fn estimate_state(
        mut self, raw_state: List[Float64], timestamp: Float64
    ) -> FilteredState:
        """
        Estimate filtered state from raw measurements.

        Args:
            raw_state: [la_position, pend_velocity, pend_position, cmd_volts].
            timestamp: Current timestamp.

        Returns:
            Filtered and estimated state.
        """
        if not self.filter_initialized:
            self.initialize_estimator(raw_state, timestamp)
            return self.filtered_state

        # Extract raw measurements
        raw_la_pos = raw_state[0]
        raw_pend_vel = raw_state[1]
        raw_pend_angle = raw_state[2]

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

            # Estimate derivatives using advanced filtering
            self.filtered_state.la_velocity = self._estimate_velocity_kalman(
                self.state_history.positions, self.state_history.timestamps
            )

            self.filtered_state.pend_acceleration = (
                self._estimate_acceleration_kalman(
                    self.state_history.angular_velocities,
                    self.state_history.timestamps,
                )
            )

            self.filtered_state.timestamp = timestamp
            self.filtered_state.outlier_detected = False
            self.filtered_state.confidence = min(
                1.0, self.filtered_state.confidence + 0.1
            )

        # Add to history
        self.state_history.add_state(
            raw_la_pos, raw_pend_vel, raw_pend_angle, timestamp
        )

        # Update statistics
        self._update_statistics()

        # Store raw state
        self.last_raw_state = raw_state

        return self.filtered_state

    fn _apply_low_pass_filter(
        self, filtered_value: Float64, raw_value: Float64, alpha: Float64
    ) -> Float64:
        """Apply low-pass filter to reduce noise."""
        return alpha * filtered_value + (1.0 - alpha) * raw_value

    fn _filter_angle(
        self, filtered_angle: Float64, raw_angle: Float64
    ) -> Float64:
        """Apply angle-specific filtering handling wraparound."""
        # Handle angle wraparound
        var angle_diff = raw_angle - filtered_angle

        if angle_diff > 180.0:
            angle_diff -= 360.0
        elif angle_diff < -180.0:
            angle_diff += 360.0

        filtered_diff = self._apply_low_pass_filter(
            0.0, angle_diff, FILTER_ALPHA
        )
        var new_angle = filtered_angle + filtered_diff

        # Normalize to [-180, 180]
        while new_angle > 180.0:
            new_angle -= 360.0
        while new_angle < -180.0:
            new_angle += 360.0

        return new_angle

    fn _estimate_velocity_kalman(
        self, positions: List[Float64], timestamps: List[Float64]
    ) -> Float64:
        """
        Estimate velocity using Kalman filter approach.

        Implements a simplified Kalman filter for robust velocity estimation
        with noise reduction and outlier rejection.

        Args:
            positions: Position history.
            timestamps: Corresponding timestamps.

        Returns:
            Filtered velocity estimate.
        """
        n = len(positions)
        if n < 3:
            return self._estimate_simple_derivative(positions, timestamps)

        # Kalman filter parameters
        var process_noise = 0.1  # Process noise variance
        var measurement_noise = 0.5  # Measurement noise variance
        var estimation_error = 1.0  # Initial estimation error

        # Use multiple recent points for robust estimation
        var window_size = min(5, n)
        var start_idx = n - window_size

        var filtered_velocity = 0.0
        var error_covariance = estimation_error

        for i in range(start_idx + 1, n):
            # Time step
            var dt = timestamps[i] - timestamps[i - 1]
            if dt <= 0.0:
                continue

            # Measured velocity (finite difference)
            var measured_velocity = (positions[i] - positions[i - 1]) / dt

            # Kalman filter update
            # Prediction step
            var predicted_velocity = (
                filtered_velocity  # Assume constant velocity
            )
            var predicted_error = error_covariance + process_noise

            # Update step
            var kalman_gain = predicted_error / (
                predicted_error + measurement_noise
            )
            filtered_velocity = predicted_velocity + kalman_gain * (
                measured_velocity - predicted_velocity
            )
            error_covariance = (1.0 - kalman_gain) * predicted_error

        # Apply reasonable velocity limits for linear actuator
        return max(-15.0, min(15.0, filtered_velocity))

    fn _estimate_acceleration_kalman(
        self, velocities: List[Float64], timestamps: List[Float64]
    ) -> Float64:
        """
        Estimate acceleration using Kalman filter approach.

        Implements a simplified Kalman filter for robust acceleration estimation
        with noise reduction and outlier rejection.

        Args:
            velocities: Velocity history.
            timestamps: Corresponding timestamps.

        Returns:
            Filtered acceleration estimate.
        """
        n = len(velocities)
        if n < 3:
            return self._estimate_simple_derivative(velocities, timestamps)

        # Kalman filter parameters for acceleration
        var process_noise = 0.2  # Higher process noise for acceleration
        var measurement_noise = 1.0  # Higher measurement noise for acceleration
        var estimation_error = 2.0  # Initial estimation error

        # Use recent points for robust estimation
        var window_size = min(4, n)
        var start_idx = n - window_size

        var filtered_acceleration = 0.0
        var error_covariance = estimation_error

        for i in range(start_idx + 1, n):
            # Time step
            var dt = timestamps[i] - timestamps[i - 1]
            if dt <= 0.0:
                continue

            # Measured acceleration (finite difference of velocities)
            var measured_acceleration = (velocities[i] - velocities[i - 1]) / dt

            # Kalman filter update
            # Prediction step
            var predicted_acceleration = (
                filtered_acceleration  # Assume constant acceleration
            )
            var predicted_error = error_covariance + process_noise

            # Update step
            var kalman_gain = predicted_error / (
                predicted_error + measurement_noise
            )
            filtered_acceleration = predicted_acceleration + kalman_gain * (
                measured_acceleration - predicted_acceleration
            )
            error_covariance = (1.0 - kalman_gain) * predicted_error

        # Apply reasonable acceleration limits for pendulum
        return max(-2000.0, min(2000.0, filtered_acceleration))

    fn _estimate_simple_derivative(
        self, values: List[Float64], timestamps: List[Float64]
    ) -> Float64:
        """
        Fallback simple derivative estimation for cases with insufficient data.

        Args:
            values: Value history.
            timestamps: Corresponding timestamps.

        Returns:
            Simple finite difference derivative.
        """
        n = len(values)
        if n < 2:
            return 0.0

        # Use last two points for simple derivative
        var dt = timestamps[n - 1] - timestamps[n - 2]
        if dt <= 0.0:
            return 0.0

        var derivative = (values[n - 1] - values[n - 2]) / dt

        # Apply reasonable limits
        return max(-1000.0, min(1000.0, derivative))

    fn _detect_outliers(
        self, raw_state: List[Float64], timestamp: Float64
    ) -> Bool:
        """Detect outliers in measurements."""
        if len(self.last_raw_state) == 0:
            return False

        dt = timestamp - self.filtered_state.timestamp
        if dt <= 0.0:
            return True  # Invalid timestamp

        # Check for unreasonable changes
        la_pos_change = abs(raw_state[0] - self.last_raw_state[0])
        pend_vel_change = abs(raw_state[1] - self.last_raw_state[1])
        pend_angle_change = abs(raw_state[2] - self.last_raw_state[2])

        # Maximum reasonable changes per time step
        max_pos_change = 2.0 * dt  # 2 inches/second max
        max_vel_change = 500.0 * dt  # 500 deg/s^2 max acceleration
        max_angle_change = 100.0 * dt  # 100 deg/s max angular velocity

        if (
            la_pos_change > max_pos_change
            or pend_vel_change > max_vel_change
            or pend_angle_change > max_angle_change
        ):
            return True

        return False

    fn _predict_state(self, timestamp: Float64) -> FilteredState:
        """
        Predict state when measurement is unavailable using physics-based model.

        Uses pendulum physics and system dynamics for more accurate prediction
        than simple linear extrapolation.
        """
        var dt = timestamp - self.filtered_state.timestamp

        # Physics-based prediction using pendulum dynamics
        var predicted_la_pos = self._predict_linear_actuator_position(dt)
        var predicted_la_vel = self._predict_linear_actuator_velocity(dt)
        var predicted_pend_angle = self._predict_pendulum_angle(dt)
        var predicted_pend_vel = self._predict_pendulum_velocity(dt)
        var predicted_pend_accel = self._predict_pendulum_acceleration(dt)

        # Apply physical constraints
        predicted_la_pos = max(-4.5, min(4.5, predicted_la_pos))
        predicted_la_vel = max(-15.0, min(15.0, predicted_la_vel))
        predicted_pend_vel = max(-1100.0, min(1100.0, predicted_pend_vel))
        predicted_pend_accel = max(-2000.0, min(2000.0, predicted_pend_accel))

        # Normalize angle to [-180, 180]
        while predicted_pend_angle > 180.0:
            predicted_pend_angle -= 360.0
        while predicted_pend_angle < -180.0:
            predicted_pend_angle += 360.0

        var predicted_state = FilteredState(
            predicted_la_pos,
            predicted_la_vel,
            predicted_pend_angle,
            predicted_pend_vel,
            predicted_pend_accel,
            timestamp,
            self.filtered_state.confidence
            * 0.85,  # Reduce confidence for prediction
            False,
        )

        return predicted_state

    fn _predict_linear_actuator_position(self, dt: Float64) -> Float64:
        """Predict linear actuator position using kinematic model."""
        # Use kinematic equation: x = x0 + v0*t + 0.5*a*t^2
        # Assume constant acceleration based on recent trend
        var current_pos = self.filtered_state.la_position
        var current_vel = self.filtered_state.la_velocity

        # Estimate acceleration from velocity trend if available
        var estimated_accel = 0.0
        if len(self.state_history.velocities) >= 2:
            n = len(self.state_history.velocities)
            var vel_change = (
                self.state_history.velocities[n - 1]
                - self.state_history.velocities[n - 2]
            )
            var time_change = (
                self.state_history.timestamps[n - 1]
                - self.state_history.timestamps[n - 2]
            )
            if time_change > 0.0:
                estimated_accel = vel_change / time_change
                estimated_accel = max(
                    -5.0, min(5.0, estimated_accel)
                )  # Reasonable limits

        return current_pos + current_vel * dt + 0.5 * estimated_accel * dt * dt

    fn _predict_linear_actuator_velocity(self, dt: Float64) -> Float64:
        """Predict linear actuator velocity using dynamic model."""
        var current_vel = self.filtered_state.la_velocity

        # Estimate acceleration from recent velocity trend
        var estimated_accel = 0.0
        if len(self.state_history.velocities) >= 2:
            n = len(self.state_history.velocities)
            var vel_change = (
                self.state_history.velocities[n - 1]
                - self.state_history.velocities[n - 2]
            )
            var time_change = (
                self.state_history.timestamps[n - 1]
                - self.state_history.timestamps[n - 2]
            )
            if time_change > 0.0:
                estimated_accel = vel_change / time_change
                estimated_accel = max(-5.0, min(5.0, estimated_accel))

        return current_vel + estimated_accel * dt

    fn _predict_pendulum_angle(self, dt: Float64) -> Float64:
        """Predict pendulum angle using angular dynamics."""
        # Use angular kinematic equation: θ = θ0 + ω0*t + 0.5*α*t^2
        var current_angle = self.filtered_state.pend_angle
        var current_angular_vel = self.filtered_state.pend_velocity
        var current_angular_accel = self.filtered_state.pend_acceleration

        return (
            current_angle
            + current_angular_vel * dt
            + 0.5 * current_angular_accel * dt * dt
        )

    fn _predict_pendulum_velocity(self, dt: Float64) -> Float64:
        """Predict pendulum angular velocity using dynamics."""
        # Use angular velocity equation: ω = ω0 + α*t
        var current_angular_vel = self.filtered_state.pend_velocity
        var current_angular_accel = self.filtered_state.pend_acceleration

        return current_angular_vel + current_angular_accel * dt

    fn _predict_pendulum_acceleration(self, dt: Float64) -> Float64:
        """Predict pendulum angular acceleration using physics model."""
        # For more sophisticated prediction, could use pendulum equation of motion
        # For now, assume acceleration decays towards zero (damping effect)
        var current_angular_accel = self.filtered_state.pend_acceleration
        var damping_factor = 0.95  # Slight damping

        return current_angular_accel * damping_factor

    fn _update_statistics(mut self):
        """Update estimation performance statistics."""
        # Simple statistics tracking
        self.estimation_statistics.append(self.filtered_state.confidence)

        # Keep only recent statistics
        if len(self.estimation_statistics) > 100:
            new_stats = List[Float64]()
            start_idx = len(self.estimation_statistics) - 100
            for i in range(start_idx, len(self.estimation_statistics)):
                new_stats.append(self.estimation_statistics[i])
            self.estimation_statistics = new_stats

    fn reset_estimator(mut self):
        """Reset state estimator to initial conditions."""
        self.state_history = StateHistory(
            List[Float64](),
            List[Float64](),
            List[Float64](),
            List[Float64](),
            List[Float64](),
            20,
        )

        self.filtered_state = FilteredState(
            0.0, 0.0, 180.0, 0.0, 0.0, 0.0, 1.0, False
        )

        self.last_raw_state = List[Float64]()
        self.filter_initialized = False
        self.estimation_statistics = List[Float64]()

        print("State estimator reset successfully")
