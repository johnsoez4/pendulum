"""
Physics model and calculations for inverted pendulum system.

This module provides physics-informed models of pendulum dynamics including
equations of motion, energy calculations, and constraint handling for the
digital twin development.
"""

from math import sin, cos, sqrt, abs, pi, atan2
from collections import List

# Import project configuration
from config.pendulum_config import (
    MAX_ACTUATOR_TRAVEL,
    MAX_CONTROL_VOLTAGE,
    PENDULUM_FULL_ROTATION,
    MAX_PENDULUM_VELOCITY,
    SAFETY_ACTUATOR_MARGIN,
    SAFETY_VOLTAGE_MARGIN,
)

# Physical constants (typical values for inverted pendulum)
alias GRAVITY = 9.81                    # m/s^2 - gravitational acceleration
alias PENDULUM_LENGTH = 0.3             # meters - pendulum length (estimated)
alias PENDULUM_MASS = 0.1               # kg - pendulum mass (estimated)
alias CART_MASS = 1.0                   # kg - cart/actuator mass (estimated)
alias FRICTION_COEFFICIENT = 0.01       # friction coefficient
alias ACTUATOR_GAIN = 0.02              # m/V - actuator gain (estimated)
alias INCHES_TO_METERS = 0.0254         # conversion factor
alias DEGREES_TO_RADIANS = pi / 180.0   # conversion factor

struct PendulumState:
    """
    Complete state of the pendulum system.
    
    Represents the full state vector including position, velocity,
    and derived quantities for physics calculations.
    """
    
    var cart_position: Float64      # Cart position (meters)
    var cart_velocity: Float64      # Cart velocity (m/s)
    var pendulum_angle: Float64     # Pendulum angle from vertical (radians)
    var pendulum_velocity: Float64  # Pendulum angular velocity (rad/s)
    var control_force: Float64      # Applied control force (N)
    var timestamp: Float64          # Time (seconds)
    
    fn __init__(out self, cart_pos: Float64 = 0.0, cart_vel: Float64 = 0.0,
                pend_angle: Float64 = 0.0, pend_vel: Float64 = 0.0,
                force: Float64 = 0.0, time: Float64 = 0.0):
        """Initialize pendulum state."""
        self.cart_position = cart_pos
        self.cart_velocity = cart_vel
        self.pendulum_angle = pend_angle
        self.pendulum_velocity = pend_vel
        self.control_force = force
        self.timestamp = time
    
    fn from_data_sample(la_pos_inches: Float64, pend_vel_deg_s: Float64,
                       pend_pos_deg: Float64, cmd_volts: Float64) -> Self:
        """
        Create state from experimental data sample.
        
        Args:
            la_pos_inches: Linear actuator position in inches
            pend_vel_deg_s: Pendulum velocity in degrees/second
            pend_pos_deg: Pendulum position in degrees
            cmd_volts: Command voltage in volts
            
        Returns:
            PendulumState in SI units
        """
        # Convert units to SI
        cart_pos = la_pos_inches * INCHES_TO_METERS
        pend_angle = pend_pos_deg * DEGREES_TO_RADIANS
        pend_vel = pend_vel_deg_s * DEGREES_TO_RADIANS
        force = cmd_volts * ACTUATOR_GAIN  # Approximate force from voltage
        
        return PendulumState(cart_pos, 0.0, pend_angle, pend_vel, force, 0.0)
    
    fn to_data_format(self) -> (Float64, Float64, Float64, Float64):
        """
        Convert state back to experimental data format.
        
        Returns:
            Tuple of (la_pos_inches, pend_vel_deg_s, pend_pos_deg, cmd_volts)
        """
        la_pos = self.cart_position / INCHES_TO_METERS
        pend_vel = self.pendulum_velocity / DEGREES_TO_RADIANS
        pend_pos = self.pendulum_angle / DEGREES_TO_RADIANS
        cmd_volts = self.control_force / ACTUATOR_GAIN
        
        return (la_pos, pend_vel, pend_pos, cmd_volts)
    
    fn is_inverted(self, threshold_deg: Float64 = 10.0) -> Bool:
        """Check if pendulum is in inverted state."""
        angle_deg = abs(self.pendulum_angle / DEGREES_TO_RADIANS)
        return angle_deg < threshold_deg
    
    fn is_hanging(self, threshold_deg: Float64 = 170.0) -> Bool:
        """Check if pendulum is in hanging state."""
        angle_deg = abs(self.pendulum_angle / DEGREES_TO_RADIANS)
        return angle_deg > threshold_deg
    
    fn total_energy(self) -> Float64:
        """
        Calculate total energy of the pendulum system.
        
        Returns:
            Total energy in Joules
        """
        # Kinetic energy of cart
        cart_ke = 0.5 * CART_MASS * self.cart_velocity * self.cart_velocity
        
        # Kinetic energy of pendulum
        pend_ke = 0.5 * PENDULUM_MASS * PENDULUM_LENGTH * PENDULUM_LENGTH * self.pendulum_velocity * self.pendulum_velocity
        
        # Potential energy of pendulum
        height = PENDULUM_LENGTH * (1.0 - cos(self.pendulum_angle))
        pend_pe = PENDULUM_MASS * GRAVITY * height
        
        return cart_ke + pend_ke + pend_pe

struct PendulumPhysics:
    """
    Physics model for inverted pendulum system.
    
    Implements the equations of motion, energy calculations, and physics-based
    constraints for the pendulum system.
    """
    
    var pendulum_length: Float64
    var pendulum_mass: Float64
    var cart_mass: Float64
    var friction: Float64
    var gravity: Float64
    
    fn __init__(out self, length: Float64 = PENDULUM_LENGTH,
                pend_mass: Float64 = PENDULUM_MASS,
                cart_mass: Float64 = CART_MASS,
                friction: Float64 = FRICTION_COEFFICIENT):
        """Initialize physics model with system parameters."""
        self.pendulum_length = length
        self.pendulum_mass = pend_mass
        self.cart_mass = cart_mass
        self.friction = friction
        self.gravity = GRAVITY
    
    fn equations_of_motion(self, state: PendulumState) -> (Float64, Float64, Float64, Float64):
        """
        Compute derivatives of state variables using equations of motion.
        
        Args:
            state: Current pendulum state
            
        Returns:
            Tuple of (cart_accel, cart_vel, pend_accel, pend_vel)
        """
        # Extract state variables
        theta = state.pendulum_angle
        theta_dot = state.pendulum_velocity
        x_dot = state.cart_velocity
        force = state.control_force
        
        # Precompute trigonometric functions
        sin_theta = sin(theta)
        cos_theta = cos(theta)
        
        # System parameters
        m_p = self.pendulum_mass
        m_c = self.cart_mass
        l = self.pendulum_length
        g = self.gravity
        b = self.friction
        
        # Total mass
        total_mass = m_c + m_p
        
        # Denominator for equations
        denominator = total_mass - m_p * cos_theta * cos_theta
        
        # Cart acceleration
        numerator_x = force - m_p * l * theta_dot * theta_dot * sin_theta + m_p * g * sin_theta * cos_theta - b * x_dot
        x_ddot = numerator_x / denominator
        
        # Pendulum angular acceleration
        numerator_theta = -force * cos_theta + total_mass * g * sin_theta - m_p * l * theta_dot * theta_dot * sin_theta * cos_theta + b * x_dot * cos_theta
        theta_ddot = numerator_theta / (l * denominator)
        
        return (x_ddot, x_dot, theta_ddot, theta_dot)
    
    fn integrate_step(self, state: PendulumState, dt: Float64) -> PendulumState:
        """
        Integrate equations of motion for one time step using RK4.
        
        Args:
            state: Current state
            dt: Time step (seconds)
            
        Returns:
            New state after integration
        """
        # RK4 integration
        k1 = self.equations_of_motion(state)
        
        # Intermediate state for k2
        state_k2 = PendulumState(
            state.cart_position + 0.5 * dt * k1[1],
            state.cart_velocity + 0.5 * dt * k1[0],
            state.pendulum_angle + 0.5 * dt * k1[3],
            state.pendulum_velocity + 0.5 * dt * k1[2],
            state.control_force,
            state.timestamp + 0.5 * dt
        )
        k2 = self.equations_of_motion(state_k2)
        
        # Intermediate state for k3
        state_k3 = PendulumState(
            state.cart_position + 0.5 * dt * k2[1],
            state.cart_velocity + 0.5 * dt * k2[0],
            state.pendulum_angle + 0.5 * dt * k2[3],
            state.pendulum_velocity + 0.5 * dt * k2[2],
            state.control_force,
            state.timestamp + 0.5 * dt
        )
        k3 = self.equations_of_motion(state_k3)
        
        # Intermediate state for k4
        state_k4 = PendulumState(
            state.cart_position + dt * k3[1],
            state.cart_velocity + dt * k3[0],
            state.pendulum_angle + dt * k3[3],
            state.pendulum_velocity + dt * k3[2],
            state.control_force,
            state.timestamp + dt
        )
        k4 = self.equations_of_motion(state_k4)
        
        # Final integration
        new_cart_pos = state.cart_position + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        new_cart_vel = state.cart_velocity + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        new_pend_angle = state.pendulum_angle + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
        new_pend_vel = state.pendulum_velocity + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        
        return PendulumState(new_cart_pos, new_cart_vel, new_pend_angle, new_pend_vel,
                           state.control_force, state.timestamp + dt)
    
    fn validate_physics_constraints(self, state: PendulumState) -> Bool:
        """
        Validate that state satisfies physical constraints.
        
        Args:
            state: State to validate
            
        Returns:
            True if state is physically valid
        """
        # Check cart position limits
        cart_pos_inches = state.cart_position / INCHES_TO_METERS
        if abs(cart_pos_inches) > MAX_ACTUATOR_TRAVEL:
            return False
        
        # Check velocity limits
        pend_vel_deg_s = abs(state.pendulum_velocity / DEGREES_TO_RADIANS)
        if pend_vel_deg_s > MAX_PENDULUM_VELOCITY:
            return False
        
        # Check control force limits
        cmd_volts = abs(state.control_force / ACTUATOR_GAIN)
        if cmd_volts > MAX_CONTROL_VOLTAGE:
            return False
        
        # Check energy conservation (approximate)
        energy = state.total_energy()
        if energy < 0.0 or energy > 100.0:  # Reasonable energy bounds
            return False
        
        return True
    
    fn compute_linearized_model(self, equilibrium: PendulumState) -> List[List[Float64]]:
        """
        Compute linearized model around equilibrium point.
        
        Args:
            equilibrium: Equilibrium state for linearization
            
        Returns:
            State matrix A for linearized system dx/dt = Ax + Bu
        """
        # For inverted pendulum, linearize around upright position
        theta_eq = equilibrium.pendulum_angle
        
        # System parameters
        m_p = self.pendulum_mass
        m_c = self.cart_mass
        l = self.pendulum_length
        g = self.gravity
        b = self.friction
        
        # Linearized system matrix (4x4)
        var A = List[List[Float64]]()
        
        # Row 1: d(cart_pos)/dt = cart_vel
        var row1 = List[Float64](0.0, 1.0, 0.0, 0.0)
        A.append(row1)
        
        # Row 2: d(cart_vel)/dt = ...
        total_mass = m_c + m_p
        a22 = -b / total_mass
        a23 = m_p * g / total_mass
        var row2 = List[Float64](0.0, a22, a23, 0.0)
        A.append(row2)
        
        # Row 3: d(pend_angle)/dt = pend_vel
        var row3 = List[Float64](0.0, 0.0, 0.0, 1.0)
        A.append(row3)
        
        # Row 4: d(pend_vel)/dt = ...
        a42 = b / (l * total_mass)
        a43 = g / l
        var row4 = List[Float64](0.0, a42, a43, 0.0)
        A.append(row4)
        
        return A

@staticmethod
fn create_physics_model() -> PendulumPhysics:
    """
    Create default physics model with standard parameters.
    
    Returns:
        PendulumPhysics with default parameters
    """
    return PendulumPhysics()

@staticmethod
fn normalize_angle(angle: Float64) -> Float64:
    """
    Normalize angle to [-pi, pi] range.
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle in [-pi, pi]
    """
    var normalized = angle
    while normalized > pi:
        normalized -= 2.0 * pi
    while normalized < -pi:
        normalized += 2.0 * pi
    return normalized
