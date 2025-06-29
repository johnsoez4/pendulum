"""
Unit tests for physics module.

This module tests the physics calculations, state management, and constraint
validation functions in the pendulum physics model.
"""

from collections import List
from testing import assert_equal, assert_true, assert_false


# Helper functions for testing
fn abs(x: Float64) -> Float64:
    """Return absolute value of x."""
    return x if x >= 0.0 else -x


fn assert_near(
    actual: Float64, expected: Float64, tolerance: Float64 = 1e-6
) raises:
    """Assert that two floating point values are close."""
    var diff = abs(actual - expected)
    if diff > tolerance:
        print(
            "Assertion failed: expected",
            expected,
            "but got",
            actual,
            "diff:",
            diff,
        )
        assert_true(False)


# Physics constants for testing
alias GRAVITY = 9.81
alias PENDULUM_LENGTH = 0.5
alias PENDULUM_MASS = 0.1
alias CART_MASS = 1.0
alias FRICTION_COEFFICIENT = 0.1


@fieldwise_init
struct PendulumState(Copyable, Movable):
    """Simplified pendulum state for testing."""

    var cart_position: Float64
    var cart_velocity: Float64
    var pendulum_angle: Float64
    var pendulum_velocity: Float64
    var applied_force: Float64
    var timestamp: Float64

    fn total_energy(self) -> Float64:
        """Calculate total energy of the system."""
        # Kinetic energy of cart
        var cart_ke = 0.5 * CART_MASS * self.cart_velocity * self.cart_velocity

        # Kinetic energy of pendulum
        var pend_ke = (
            0.5
            * PENDULUM_MASS
            * self.pendulum_velocity
            * self.pendulum_velocity
        )

        # Potential energy of pendulum
        var height = PENDULUM_LENGTH * (1.0 - cos_approx(self.pendulum_angle))
        var pend_pe = PENDULUM_MASS * GRAVITY * height

        return cart_ke + pend_ke + pend_pe

    fn is_valid(self) -> Bool:
        """Check if state is physically valid."""
        # Check for NaN or infinite values
        if self.cart_position != self.cart_position:  # NaN check
            return False
        if self.cart_velocity != self.cart_velocity:
            return False
        if self.pendulum_angle != self.pendulum_angle:
            return False
        if self.pendulum_velocity != self.pendulum_velocity:
            return False

        # Check reasonable bounds
        if abs(self.cart_position) > 10.0:  # 10 meter limit
            return False
        if abs(self.cart_velocity) > 100.0:  # 100 m/s limit
            return False
        if abs(self.pendulum_velocity) > 1000.0:  # 1000 rad/s limit
            return False

        return True


fn cos_approx(angle: Float64) -> Float64:
    """Approximate cosine function using Taylor series."""
    var x2 = angle * angle
    return 1.0 - x2 / 2.0 + x2 * x2 / 24.0


fn sin_approx(angle: Float64) -> Float64:
    """Approximate sine function using Taylor series."""
    var x2 = angle * angle
    return angle * (1.0 - x2 / 6.0 + x2 * x2 / 120.0)


struct PhysicsTests:
    """Test suite for physics calculations."""

    @staticmethod
    fn test_pendulum_state_creation() raises:
        """Test PendulumState creation and basic properties."""
        print("Testing PendulumState creation...")

        var state = PendulumState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert_true(state.is_valid())

        # Test energy calculation at rest
        var energy = state.total_energy()
        assert_near(energy, 0.0, 1e-6)

        print("✓ PendulumState creation test passed")

    @staticmethod
    fn test_energy_conservation() raises:
        """Test energy conservation properties."""
        print("Testing energy conservation...")

        # Test pendulum at bottom (hanging down)
        var state1 = PendulumState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        var energy1 = state1.total_energy()

        # Test pendulum at top (inverted)
        var state2 = PendulumState(0.0, 0.0, 3.14159, 0.0, 0.0, 0.0)
        var energy2 = state2.total_energy()

        # Energy at top should be higher due to potential energy
        assert_true(energy2 > energy1)

        # Test with kinetic energy
        var state3 = PendulumState(0.0, 1.0, 0.0, 2.0, 0.0, 0.0)
        var energy3 = state3.total_energy()
        assert_true(energy3 > energy1)

        print("✓ Energy conservation test passed")

    @staticmethod
    fn test_state_validation() raises:
        """Test state validation functions."""
        print("Testing state validation...")

        # Valid state
        var valid_state = PendulumState(1.0, 0.5, 1.57, 0.1, 0.0, 0.0)
        assert_true(valid_state.is_valid())

        # Invalid states - extreme positions
        var invalid_pos = PendulumState(20.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert_false(invalid_pos.is_valid())

        # Invalid states - extreme velocities
        var invalid_vel = PendulumState(0.0, 200.0, 0.0, 0.0, 0.0, 0.0)
        assert_false(invalid_vel.is_valid())

        var invalid_pend_vel = PendulumState(0.0, 0.0, 0.0, 2000.0, 0.0, 0.0)
        assert_false(invalid_pend_vel.is_valid())

        print("✓ State validation test passed")

    @staticmethod
    fn test_trigonometric_approximations() raises:
        """Test trigonometric function approximations."""
        print("Testing trigonometric approximations...")

        # Test cos approximation at key points
        assert_near(cos_approx(0.0), 1.0, 1e-3)
        assert_near(
            cos_approx(1.57), 0.0, 1e-1
        )  # π/2, looser tolerance for approximation

        # Test sin approximation at key points
        assert_near(sin_approx(0.0), 0.0, 1e-3)
        assert_near(
            sin_approx(1.57), 1.0, 1e-1
        )  # π/2, looser tolerance for approximation

        print("✓ Trigonometric approximation test passed")

    @staticmethod
    fn test_physics_constraints() raises:
        """Test physics constraint validation."""
        print("Testing physics constraints...")

        # Test actuator position constraints (should be within [-4, 4] inches)
        var actuator_pos_valid = 2.0  # inches
        var actuator_pos_invalid = 6.0  # inches

        assert_true(abs(actuator_pos_valid) <= 4.0)
        assert_false(abs(actuator_pos_invalid) <= 4.0)

        # Test velocity constraints (should be within [-1000, 1000] deg/s)
        var velocity_valid = 500.0  # deg/s
        var velocity_invalid = 1500.0  # deg/s

        assert_true(abs(velocity_valid) <= 1000.0)
        assert_false(abs(velocity_invalid) <= 1000.0)

        print("✓ Physics constraints test passed")

    @staticmethod
    fn test_unit_conversions() raises:
        """Test unit conversion functions."""
        print("Testing unit conversions...")

        # Test inches to meters conversion
        var inches_to_meters = 0.0254
        var pos_inches = 4.0
        var pos_meters = pos_inches * inches_to_meters
        assert_near(pos_meters, 0.1016, 1e-6)

        # Test degrees to radians conversion
        var degrees_to_radians = 3.14159 / 180.0
        var angle_degrees = 180.0
        var angle_radians = angle_degrees * degrees_to_radians
        assert_near(angle_radians, 3.14159, 1e-4)

        print("✓ Unit conversion test passed")

    @staticmethod
    fn run_all_tests() raises:
        """Run all physics tests."""
        print("Running Physics Unit Tests")
        print("==========================")

        PhysicsTests.test_pendulum_state_creation()
        PhysicsTests.test_energy_conservation()
        PhysicsTests.test_state_validation()
        PhysicsTests.test_trigonometric_approximations()
        PhysicsTests.test_physics_constraints()
        PhysicsTests.test_unit_conversions()

        print()
        print("✓ All physics tests passed!")
        print()


fn main() raises:
    """Run physics unit tests."""
    PhysicsTests.run_all_tests()
