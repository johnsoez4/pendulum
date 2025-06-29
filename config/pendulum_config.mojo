"""
Configuration constants and parameters for the pendulum project.

This module defines all system constants, physical parameters, and configuration
settings used throughout the pendulum digital twin and control system.
"""

# Physical System Constants
alias MAX_ACTUATOR_TRAVEL = 4.0  # inches - physical actuator limit
alias ACTUATOR_CENTER_POSITION = 0.0  # inches - center/home position
alias MAX_CONTROL_VOLTAGE = 5.0  # volts - motor voltage limit
alias MIN_CONTROL_VOLTAGE = -5.0  # volts - motor voltage limit

# Pendulum Physical Parameters
alias PENDULUM_UPRIGHT_ANGLE = 0.0  # degrees - inverted/upright position
alias PENDULUM_HANGING_ANGLE = 180.0  # degrees - hanging down position
alias MAX_PENDULUM_VELOCITY = 1000.0  # deg/s - maximum observed velocity
alias PENDULUM_FULL_ROTATION = 360.0  # degrees - full rotation range

# Control System Parameters
alias TARGET_SAMPLE_RATE = 25.0  # Hz - control system update rate
alias SAMPLE_INTERVAL_MS = 40.0  # milliseconds - target sample interval
alias CONTROL_TIMEOUT_MS = 100.0  # milliseconds - control timeout
alias MAX_CONTROL_ITERATIONS = 1000  # maximum control loop iterations

# State Classification Thresholds
alias INVERTED_ANGLE_THRESHOLD = 10.0  # degrees - consider inverted if |angle| < 10°
alias HANGING_ANGLE_THRESHOLD = 170.0  # degrees - consider hanging if |angle| > 170°
alias LOW_VELOCITY_THRESHOLD = 50.0  # deg/s - consider stationary if |vel| < 50°
alias HIGH_VELOCITY_THRESHOLD = 500.0  # deg/s - consider high energy if |vel| > 500°

# Digital Twin Model Parameters
alias MODEL_INPUT_DIM = 4  # [la_pos, pend_vel, pend_pos, cmd_volts]
alias MODEL_OUTPUT_DIM = 3  # [next_la_pos, next_pend_vel, next_pend_pos]
alias MODEL_HIDDEN_LAYERS = 3  # number of hidden layers
alias MODEL_HIDDEN_SIZE = 128  # neurons per hidden layer
alias MODEL_LEARNING_RATE = 0.001  # training learning rate
alias MODEL_BATCH_SIZE = 64  # training batch size
alias MODEL_MAX_EPOCHS = 1000  # maximum training epochs

# Data Processing Parameters
alias DATA_VALIDATION_TOLERANCE = 0.01  # tolerance for data validation
alias OUTLIER_DETECTION_SIGMA = 3.0  # standard deviations for outlier detection
alias DATA_SMOOTHING_WINDOW = 5  # samples for data smoothing
alias MISSING_DATA_THRESHOLD = 0.05  # maximum fraction of missing data allowed

# Control Algorithm Parameters
alias CONTROL_HORIZON = 10  # steps - model predictive control horizon
alias CONTROL_WEIGHT_POSITION = 1.0  # weight for position error in cost function
alias CONTROL_WEIGHT_VELOCITY = 0.1  # weight for velocity error in cost function
alias CONTROL_WEIGHT_EFFORT = 0.01  # weight for control effort in cost function
alias CONTROL_CONVERGENCE_TOL = 0.001  # convergence tolerance for optimization

# Safety Parameters
alias SAFETY_ACTUATOR_MARGIN = 0.1  # inches - safety margin from actuator limits
alias SAFETY_VOLTAGE_MARGIN = 0.1  # volts - safety margin from voltage limits
alias SAFETY_VELOCITY_LIMIT = 800.0  # deg/s - safety limit for pendulum velocity
alias EMERGENCY_STOP_THRESHOLD = 900.0  # deg/s - emergency stop if exceeded

# File Paths and Data Configuration
alias DEFAULT_DATA_FILE = "data/sample_data.csv"
alias MODEL_SAVE_PATH = "models/"
alias LOG_FILE_PATH = "logs/"
alias CONFIG_FILE_PATH = "config/"

# Logging and Debug Configuration
alias LOG_LEVEL_DEBUG = 0
alias LOG_LEVEL_INFO = 1
alias LOG_LEVEL_WARNING = 2
alias LOG_LEVEL_ERROR = 3
alias DEFAULT_LOG_LEVEL = LOG_LEVEL_INFO

# Performance and Resource Limits
alias MAX_MEMORY_USAGE_MB = 1024  # MB - maximum memory usage
alias MAX_CPU_USAGE_PERCENT = 80  # % - maximum CPU usage
alias REAL_TIME_DEADLINE_MS = 40  # ms - real-time deadline for control
alias MAX_PREDICTION_HORIZON = 100  # steps - maximum prediction horizon


struct PendulumConfig:
    """
    Configuration container for pendulum system parameters.

    Provides centralized access to all system configuration parameters
    with validation and default value management.
    """

    var sample_rate: Float64
    var max_actuator_travel: Float64
    var max_control_voltage: Float64
    var inverted_threshold: Float64
    var hanging_threshold: Float64
    var model_learning_rate: Float64
    var control_horizon: Int
    var log_level: Int

    fn __init__(out self):
        """Initialize with default configuration values."""
        self.sample_rate = TARGET_SAMPLE_RATE
        self.max_actuator_travel = MAX_ACTUATOR_TRAVEL
        self.max_control_voltage = MAX_CONTROL_VOLTAGE
        self.inverted_threshold = INVERTED_ANGLE_THRESHOLD
        self.hanging_threshold = HANGING_ANGLE_THRESHOLD
        self.model_learning_rate = MODEL_LEARNING_RATE
        self.control_horizon = CONTROL_HORIZON
        self.log_level = DEFAULT_LOG_LEVEL

    fn validate_config(self) -> Bool:
        """
        Validate configuration parameters for consistency and safety.

        Returns:
            True if configuration is valid, False otherwise.
        """
        # Check physical constraints
        if self.max_actuator_travel <= 0.0 or self.max_actuator_travel > 10.0:
            return False

        if self.max_control_voltage <= 0.0 or self.max_control_voltage > 10.0:
            return False

        # Check angle thresholds
        if self.inverted_threshold < 0.0 or self.inverted_threshold > 90.0:
            return False

        if self.hanging_threshold < 90.0 or self.hanging_threshold > 180.0:
            return False

        # Check model parameters
        if self.model_learning_rate <= 0.0 or self.model_learning_rate > 1.0:
            return False

        if self.control_horizon <= 0 or self.control_horizon > 100:
            return False

        # Check sample rate
        if self.sample_rate <= 0.0 or self.sample_rate > 1000.0:
            return False

        return True

    fn get_actuator_limits(self) -> (Float64, Float64):
        """
        Get actuator position limits with safety margins.

        Returns:
            Tuple of (min_position, max_position) in inches.
        """
        margin = SAFETY_ACTUATOR_MARGIN
        return (
            -self.max_actuator_travel + margin,
            self.max_actuator_travel - margin,
        )

    fn get_voltage_limits(self) -> (Float64, Float64):
        """
        Get control voltage limits with safety margins.

        Returns:
            Tuple of (min_voltage, max_voltage) in volts.
        """
        margin = SAFETY_VOLTAGE_MARGIN
        return (
            -self.max_control_voltage + margin,
            self.max_control_voltage - margin,
        )

    fn is_inverted_state(self, angle: Float64) -> Bool:
        """
        Check if pendulum angle represents inverted state.

        Args:
            angle: Pendulum angle in degrees.

        Returns:
            True if angle is within inverted threshold.
        """
        return abs(angle) < self.inverted_threshold

    fn is_hanging_state(self, angle: Float64) -> Bool:
        """
        Check if pendulum angle represents hanging state.

        Args:
            angle: Pendulum angle in degrees.

        Returns:
            True if angle is within hanging threshold.
        """
        return abs(angle) > self.hanging_threshold


struct PendulumConfigUtils:
    """Utility functions for pendulum configuration management."""

    @staticmethod
    fn get_default_config() -> PendulumConfig:
        """
        Get default pendulum configuration.

        Returns:
            PendulumConfig with default values.
        """
        return PendulumConfig()

    @staticmethod
    fn validate_physical_limits(
        actuator_pos: Float64, control_voltage: Float64
    ) -> Bool:
        """
        Validate that physical values are within safe operating limits.

        Args:
            actuator_pos: Linear actuator position in inches.
            control_voltage: Control voltage in volts.

        Returns:
            True if values are within safe limits.
        """
        if abs(actuator_pos) > MAX_ACTUATOR_TRAVEL:
            return False

        if abs(control_voltage) > MAX_CONTROL_VOLTAGE:
            return False

        return True
