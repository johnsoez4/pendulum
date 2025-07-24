"""
Shared physics constraints for pendulum neural networks.

This module centralizes constraint logic to ensure consistency across all
neural network implementations. It provides standardized constraint validation
and application functions that maintain physical realism in predictions.
"""

from collections import List

# Physics constraint constants
alias MAX_ACTUATOR_TRAVEL = 4.0  # inches - physical actuator limit
alias MAX_PENDULUM_VELOCITY = 1000.0  # deg/s - maximum observed velocity
alias PENDULUM_FULL_ROTATION = 360.0  # degrees - full rotation range


struct PhysicsConstraints:
    """
    Centralized physics constraint validation and application.

    This struct provides static methods for applying physical constraints
    to neural network predictions, ensuring realistic pendulum behavior.
    """

    @staticmethod
    fn apply_actuator_constraints(position: Float64) -> Float64:
        """
        Apply actuator position constraints.

        Constrains the linear actuator position to physically achievable limits
        based on the hardware specifications.

        Args:
            position: Predicted actuator position in inches.

        Returns:
            Constrained position within [-4.0, 4.0] inches.
        """
        return max(-MAX_ACTUATOR_TRAVEL, min(MAX_ACTUATOR_TRAVEL, position))

    @staticmethod
    fn apply_velocity_constraints(velocity: Float64) -> Float64:
        """
        Apply pendulum velocity constraints.

        Constrains the pendulum angular velocity to physically reasonable limits
        based on observed system behavior and safety considerations.

        Args:
            velocity: Predicted pendulum velocity in deg/s.

        Returns:
            Constrained velocity within [-1000.0, 1000.0] deg/s.
        """
        return max(-MAX_PENDULUM_VELOCITY, min(MAX_PENDULUM_VELOCITY, velocity))

    @staticmethod
    fn apply_angle_continuity(
        current_angle: Float64, predicted_angle: Float64
    ) -> Float64:
        """
        Handle angle wrapping and continuity.

        Ensures smooth angle transitions by handling wraparound at ±180 degrees
        and preventing sudden jumps in angle predictions.

        Args:
            current_angle: Current pendulum angle in degrees.
            predicted_angle: Predicted pendulum angle in degrees.

        Returns:
            Adjusted predicted angle with proper continuity.
        """
        angle_diff = predicted_angle - current_angle
        result = predicted_angle

        if abs(angle_diff) > 180.0:
            if angle_diff > 180.0:
                result -= PENDULUM_FULL_ROTATION
            elif angle_diff < -180.0:
                result += PENDULUM_FULL_ROTATION

        return result

    @staticmethod
    fn apply_all_constraints(
        input: List[Float64], prediction: List[Float64]
    ) -> List[Float64]:
        """
        Apply all physics constraints to neural network predictions.

        This is the main constraint application function that should be used
        by all neural network implementations to ensure consistent constraint
        enforcement across the system.

        Args:
            input: Input state vector [la_position, pend_velocity, pend_position, cmd_volts].
            prediction: Raw network prediction [next_la_position, next_pend_velocity, next_pend_position].

        Returns:
            Physics-constrained prediction vector.
        """
        constrained = List[Float64]()

        # Apply actuator position constraints
        constrained.append(Self.apply_actuator_constraints(prediction[0]))

        # Apply velocity constraints
        constrained.append(Self.apply_velocity_constraints(prediction[1]))

        # Apply angle continuity
        current_angle = input[2] if len(input) > 2 else 0.0
        constrained.append(
            Self.apply_angle_continuity(current_angle, prediction[2])
        )

        return constrained

    @staticmethod
    fn validate_constraints(prediction: List[Float64]) -> Bool:
        """
        Validate that predictions satisfy physical constraints.

        Args:
            prediction: Prediction vector to validate.

        Returns:
            True if all constraints are satisfied, False otherwise.
        """
        if len(prediction) < 3:
            return False

        # Check actuator position constraint
        if abs(prediction[0]) > MAX_ACTUATOR_TRAVEL:
            return False

        # Check velocity constraint
        if abs(prediction[1]) > MAX_PENDULUM_VELOCITY:
            return False

        return True

    @staticmethod
    fn compute_constraint_violation_penalty(
        prediction: List[Float64],
    ) -> Float64:
        """
        Compute penalty for constraint violations.

        This can be used in loss functions to penalize predictions that
        violate physical constraints during training.

        Args:
            prediction: Prediction vector to evaluate.

        Returns:
            Constraint violation penalty (0.0 if no violations).
        """
        penalty = 0.0

        if len(prediction) >= 3:
            # Actuator constraint penalty
            if abs(prediction[0]) > MAX_ACTUATOR_TRAVEL:
                penalty += 10.0

            # Velocity constraint penalty
            if abs(prediction[1]) > MAX_PENDULUM_VELOCITY:
                penalty += 5.0

        return penalty


fn apply_physics_constraints(
    input: List[Float64], prediction: List[Float64]
) -> List[Float64]:
    """
    Convenience function for applying physics constraints.

    This function provides a simple interface for constraint application
    that can be easily imported and used across the codebase.

    Args:
        input: Input state vector.
        prediction: Raw network prediction.

    Returns:
        Physics-constrained prediction vector.
    """
    return PhysicsConstraints.apply_all_constraints(input, prediction)


fn validate_physics_constraints(prediction: List[Float64]) -> Bool:
    """
    Convenience function for validating physics constraints.

    Args:
        prediction: Prediction vector to validate.

    Returns:
        True if all constraints are satisfied, False otherwise.
    """
    return PhysicsConstraints.validate_constraints(prediction)


fn compute_constraint_penalty(prediction: List[Float64]) -> Float64:
    """
    Convenience function for computing constraint violation penalties.

    Args:
        prediction: Prediction vector to evaluate.

    Returns:
        Constraint violation penalty (0.0 if no violations).
    """
    return PhysicsConstraints.compute_constraint_violation_penalty(prediction)
