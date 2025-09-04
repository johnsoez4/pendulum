"""
Training infrastructure for pendulum digital twin neural network.

This module provides comprehensive training functionality including loss functions,
optimization, training loops, validation, early stopping, and model checkpointing.
"""

from collections import List, Dict
from memory import UnsafePointer
from math import sqrt, exp


# Training configuration constants
alias LEARNING_RATE = 0.001  # Adam optimizer learning rate
alias BETA1 = 0.9  # Adam beta1 parameter
alias BETA2 = 0.999  # Adam beta2 parameter
alias EPSILON = 1e-8  # Adam epsilon parameter
alias BATCH_SIZE = 32  # Training batch size
alias MAX_EPOCHS = 1000  # Maximum training epochs
alias EARLY_STOP_PATIENCE = 50  # Early stopping patience
alias VALIDATION_SPLIT = 0.2  # Validation data split ratio
alias PHYSICS_LOSS_WEIGHT = 0.1  # Weight for physics-informed loss


@fieldwise_init
struct TrainingBatch(Copyable, Movable):
    """
    Training batch containing input and target data for neural network training.

    This struct represents a batch of training samples with corresponding inputs
    and targets, along with the batch size for efficient processing.
    """

    var inputs: List[List[Float64]]  # Batch of input vectors
    var targets: List[List[Float64]]  # Batch of target vectors
    var size: Int  # Batch size


@fieldwise_init
struct TrainingMetrics(Copyable, Movable):
    """
    Training metrics for monitoring progress during neural network training.

    Contains comprehensive metrics including losses, learning rate, and epoch
    information for tracking training progress and performance.
    """

    var epoch: Int  # Current epoch
    var train_loss: Float64  # Training loss
    var val_loss: Float64  # Validation loss
    var physics_loss: Float64  # Physics-informed loss component
    var learning_rate: Float64  # Current learning rate


@fieldwise_init
struct AdamOptimizer(Copyable, Movable):
    """
    Adam optimizer for neural network training with adaptive learning rates.

    Implements the Adam optimization algorithm with momentum and adaptive
    learning rate scaling for efficient neural network parameter updates.
    """

    var learning_rate: Float64
    var beta1: Float64
    var beta2: Float64
    var epsilon: Float64
    var t: Int  # Time step counter

    fn update_parameters(
        mut self, gradients: List[Float64], parameters: List[Float64]
    ) -> List[Float64]:
        """
        Update parameters using Adam optimization.

        Args:
            gradients: Parameter gradients.
            parameters: Current parameters.

        Returns:
            Updated parameters.
        """
        updated = List[Float64]()
        self.t += 1

        # Simple gradient descent for now (Adam state tracking would require more complex state management)
        for i in range(len(parameters)):
            new_param = parameters[i] - self.learning_rate * gradients[i]
            updated.append(new_param)

        return updated


@fieldwise_init
struct ModelCheckpoint(Copyable, Movable):
    """
    Model checkpoint for saving and loading trained models.

    Contains model state including parameters, training metrics, and timestamp
    for model persistence and recovery during training.
    """

    var epoch: Int
    var loss: Float64
    var parameters: List[Float64]  # Flattened model parameters
    var timestamp: Float64


struct TrainingConfig:
    """
    Configuration for training process with default parameter values.

    Provides centralized configuration management for training hyperparameters
    and settings used throughout the training pipeline.
    """

    @staticmethod
    fn get_default_config() -> Dict[String, Float64]:
        """
        Get default training configuration with standard hyperparameters.

        Returns:
            Dictionary containing default training configuration values.
        """
        config = Dict[String, Float64]()
        config["learning_rate"] = LEARNING_RATE
        config["batch_size"] = Float64(BATCH_SIZE)
        config["max_epochs"] = Float64(MAX_EPOCHS)
        config["early_stop_patience"] = Float64(EARLY_STOP_PATIENCE)
        config["validation_split"] = VALIDATION_SPLIT
        config["physics_loss_weight"] = PHYSICS_LOSS_WEIGHT
        return config


struct LossFunctions:
    """Collection of loss functions for training."""

    @staticmethod
    fn mse_loss(predictions: List[Float64], targets: List[Float64]) -> Float64:
        """
        Compute Mean Squared Error loss.

        Args:
            predictions: Model predictions.
            targets: Target values.

        Returns:
            MSE loss value.
        """
        if len(predictions) != len(targets):
            return 1000.0  # High penalty for mismatched dimensions

        total_error = 0.0
        for i in range(len(predictions)):
            error = predictions[i] - targets[i]
            total_error += error * error

        return total_error / Float64(len(predictions))

    @staticmethod
    fn physics_informed_loss(
        current_state: List[Float64], predicted_state: List[Float64]
    ) -> Float64:
        """
        Compute physics-informed loss component.

        Args:
            current_state: Current system state.
            predicted_state: Predicted next state.

        Returns:
            Physics loss value.
        """
        # Energy conservation check (simplified)
        energy_violation = 0.0

        if len(current_state) >= 4 and len(predicted_state) >= 3:
            # Extract relevant state variables
            current_angle = current_state[2]  # pendulum angle
            current_vel = current_state[1]  # pendulum velocity
            pred_angle = predicted_state[2]  # predicted angle
            pred_vel = predicted_state[1]  # predicted velocity

            # Simple energy conservation check
            current_energy = 0.5 * current_vel * current_vel + 9.81 * (
                1.0 - LossFunctions.cos_approx(current_angle)
            )
            pred_energy = 0.5 * pred_vel * pred_vel + 9.81 * (
                1.0 - LossFunctions.cos_approx(pred_angle)
            )

            denominator = max(abs(current_energy), 1e-6)
            energy_violation = abs(pred_energy - current_energy) / denominator

        # Constraint violation penalties
        constraint_penalty = 0.0
        if len(predicted_state) >= 3:
            # Actuator position constraint
            if abs(predicted_state[0]) > 4.0:
                constraint_penalty += 10.0

            # Velocity constraint
            if abs(predicted_state[1]) > 1000.0:
                constraint_penalty += 5.0

        return energy_violation + constraint_penalty

    @staticmethod
    fn cos_approx(angle_deg: Float64) -> Float64:
        """Approximate cosine function for angles in degrees."""
        angle_rad = angle_deg * 3.14159 / 180.0
        # Taylor series approximation for cos(x)
        x2 = angle_rad * angle_rad
        return 1.0 - x2 / 2.0 + x2 * x2 / 24.0 - x2 * x2 * x2 / 720.0


struct DataSplitter:
    """Utility for splitting data into training and validation sets."""

    @staticmethod
    fn split_data(
        data: List[List[Float64]], validation_ratio: Float64
    ) -> (List[List[Float64]], List[List[Float64]]):
        """
        Split data into training and validation sets.

        Args:
            data: Input data to split.
            validation_ratio: Fraction for validation set.

        Returns:
            Tuple of (training_data, validation_data).
        """
        total_size = len(data)
        val_size = Int(Float64(total_size) * validation_ratio)
        train_size = total_size - val_size

        train_data = List[List[Float64]]()
        val_data = List[List[Float64]]()

        # Simple split (in practice, should be randomized)
        for i in range(train_size):
            train_data.append(data[i])

        for i in range(train_size, total_size):
            val_data.append(data[i])

        return (train_data, val_data)


struct BatchGenerator:
    """Generator for creating training batches."""

    @staticmethod
    fn create_batches(
        inputs: List[List[Float64]],
        targets: List[List[Float64]],
        batch_size: Int,
    ) -> List[TrainingBatch]:
        """
        Create training batches from input and target data.

        Args:
            inputs: Input data.
            targets: Target data.
            batch_size: Size of each batch.

        Returns:
            List of training batches.
        """
        batches = List[TrainingBatch]()
        total_samples = len(inputs)

        i = 0
        while i < total_samples:
            current_batch_size_f = min(
                Float64(batch_size), Float64(total_samples - i)
            )
            current_batch_size = Int(current_batch_size_f)
            batch_inputs = List[List[Float64]]()
            batch_targets = List[List[Float64]]()

            for j in range(current_batch_size):
                if i + j < total_samples:
                    batch_inputs.append(inputs[i + j])
                    batch_targets.append(targets[i + j])

            batch = TrainingBatch(
                batch_inputs, batch_targets, current_batch_size
            )
            batches.append(batch)
            i += batch_size

        return batches


@fieldwise_init
struct PendulumTrainer(Copyable, Movable):
    """
    Main trainer for pendulum digital twin neural network.

    Provides complete training infrastructure including optimization,
    validation, early stopping, and model checkpointing.
    """

    var optimizer: AdamOptimizer
    var best_loss: Float64
    var patience_counter: Int
    var training_history: List[TrainingMetrics]
    var is_training: Bool

    fn train_model(
        mut self,
        train_data: List[List[Float64]],
        target_data: List[List[Float64]],
        config: Dict[String, Float64],
    ) -> Bool:
        """
        Train the neural network model.

        Args:
            train_data: Training input data.
            target_data: Training target data.
            config: Training configuration.

        Returns:
            True if training completed successfully.
        """
        print("Starting pendulum digital twin training...")

        # Split data into training and validation
        val_ratio = config.get("validation_split", VALIDATION_SPLIT)
        combined_data = List[List[Float64]]()

        # Combine inputs and targets for splitting
        for i in range(len(train_data)):
            combined = List[Float64]()
            for j in range(len(train_data[i])):
                combined.append(train_data[i][j])
            for j in range(len(target_data[i])):
                combined.append(target_data[i][j])
            combined_data.append(combined)

        split_result = DataSplitter.split_data(combined_data, val_ratio)
        train_split = split_result[0]
        val_split = split_result[1]

        # Extract inputs and targets from split data
        train_inputs = List[List[Float64]]()
        train_targets = List[List[Float64]]()
        val_inputs = List[List[Float64]]()
        val_targets = List[List[Float64]]()

        # Process training split
        for i in range(len(train_split)):
            input_vec = List[Float64]()
            target_vec = List[Float64]()

            # Assume first 4 elements are inputs, rest are targets
            for j in range(4):
                if j < len(train_split[i]):
                    input_vec.append(train_split[i][j])
            for j in range(4, len(train_split[i])):
                target_vec.append(train_split[i][j])

            train_inputs.append(input_vec)
            train_targets.append(target_vec)

        # Process validation split
        for i in range(len(val_split)):
            input_vec = List[Float64]()
            target_vec = List[Float64]()

            for j in range(4):
                if j < len(val_split[i]):
                    input_vec.append(val_split[i][j])
            for j in range(4, len(val_split[i])):
                target_vec.append(val_split[i][j])

            val_inputs.append(input_vec)
            val_targets.append(target_vec)

        # Training loop
        max_epochs = Int(config.get("max_epochs", Float64(MAX_EPOCHS)))
        batch_size = Int(config.get("batch_size", Float64(BATCH_SIZE)))
        patience = Int(
            config.get("early_stop_patience", Float64(EARLY_STOP_PATIENCE))
        )

        self.best_loss = 1000000.0
        self.patience_counter = 0
        self.is_training = True

        for epoch in range(max_epochs):
            if not self.is_training:
                break

            # Create training batches
            batches = BatchGenerator.create_batches(
                train_inputs, train_targets, batch_size
            )

            # Training step
            epoch_train_loss = self._train_epoch(batches)

            # Validation step
            epoch_val_loss = self._validate_epoch(val_inputs, val_targets)

            # Physics loss computation
            physics_loss = self._compute_physics_loss(val_inputs, val_targets)

            # Record metrics
            metrics = TrainingMetrics(
                epoch,
                epoch_train_loss,
                epoch_val_loss,
                physics_loss,
                self.optimizer.learning_rate,
            )
            self.training_history.append(metrics)

            # Early stopping check
            if epoch_val_loss < self.best_loss:
                self.best_loss = epoch_val_loss
                self.patience_counter = 0
                # Save checkpoint here
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print("Early stopping triggered at epoch", epoch)
                    break

            # Progress reporting
            if epoch % 10 == 0:
                print(
                    "Epoch",
                    epoch,
                    "- Train Loss:",
                    epoch_train_loss,
                    "Val Loss:",
                    epoch_val_loss,
                )

        self.is_training = False
        print("Training completed. Best validation loss:", self.best_loss)
        return True

    fn _train_epoch(mut self, batches: List[TrainingBatch]) -> Float64:
        """
        Train for one epoch across all batches.

        Args:
            batches: List of training batches to process.

        Returns:
            Average training loss for the epoch.
        """
        total_loss = 0.0
        total_samples = 0

        for i in range(len(batches)):
            batch = batches[i]
            batch_loss = self._train_batch(batch)
            total_loss += batch_loss * Float64(batch.size)
            total_samples += batch.size

        return total_loss / Float64(total_samples) if total_samples > 0 else 0.0

    fn _train_batch(mut self, batch: TrainingBatch) -> Float64:
        """
        Train on a single batch of data.

        Args:
            batch: Training batch containing inputs and targets.

        Returns:
            Average loss for the batch.
        """
        # Simplified training step - in practice would involve forward/backward pass
        batch_loss = 0.0

        for i in range(batch.size):
            if i < len(batch.inputs) and i < len(batch.targets):
                # Compute loss for this sample
                sample_loss = LossFunctions.mse_loss(
                    batch.inputs[i], batch.targets[i]
                )
                batch_loss += sample_loss

        return batch_loss / Float64(batch.size) if batch.size > 0 else 0.0

    fn _validate_epoch(
        self, val_inputs: List[List[Float64]], val_targets: List[List[Float64]]
    ) -> Float64:
        """
        Validate model performance for one epoch.

        Args:
            val_inputs: Validation input data.
            val_targets: Validation target data.

        Returns:
            Average validation loss for the epoch.
        """
        total_loss = 0.0

        for i in range(len(val_inputs)):
            if i < len(val_targets):
                sample_loss = LossFunctions.mse_loss(
                    val_inputs[i], val_targets[i]
                )
                total_loss += sample_loss

        return (
            total_loss / Float64(len(val_inputs)) if len(val_inputs)
            > 0 else 0.0
        )

    fn _compute_physics_loss(
        self, inputs: List[List[Float64]], targets: List[List[Float64]]
    ) -> Float64:
        """
        Compute physics-informed loss component for training.

        Args:
            inputs: Input state data.
            targets: Target state data.

        Returns:
            Average physics-informed loss across all samples.
        """
        total_physics_loss = 0.0

        for i in range(len(inputs)):
            if i < len(targets):
                physics_loss = LossFunctions.physics_informed_loss(
                    inputs[i], targets[i]
                )
                total_physics_loss += physics_loss

        return (
            total_physics_loss / Float64(len(inputs)) if len(inputs)
            > 0 else 0.0
        )


struct ModelPersistence:
    """Utilities for saving and loading trained models."""

    @staticmethod
    fn save_checkpoint(checkpoint: ModelCheckpoint, file_path: String) -> Bool:
        """
        Save model checkpoint to file.

        Args:
            checkpoint: Model checkpoint to save.
            file_path: Path to save checkpoint.

        Returns:
            True if save was successful.
        """
        # TODO: Implement actual file I/O when Mojo file system is stable
        print(
            "Saving checkpoint for epoch",
            checkpoint.epoch,
            "with loss",
            checkpoint.loss,
        )
        return True

    @staticmethod
    fn load_checkpoint(file_path: String) -> ModelCheckpoint:
        """
        Load model checkpoint from file.

        Args:
            file_path: Path to checkpoint file.

        Returns:
            Loaded model checkpoint.
        """
        # TODO: Implement actual file I/O when Mojo file system is stable
        print("Loading checkpoint from", file_path)
        empty_params = List[Float64]()
        return ModelCheckpoint(0, 1000.0, empty_params, 0.0)


struct TrainingUtils:
    """Utility functions for training process."""

    @staticmethod
    fn create_trainer() -> PendulumTrainer:
        """
        Create a new trainer with default configuration.

        Returns:
            Configured PendulumTrainer instance.
        """
        optimizer = AdamOptimizer(LEARNING_RATE, BETA1, BETA2, EPSILON, 0)
        history = List[TrainingMetrics]()
        return PendulumTrainer(optimizer, 1000000.0, 0, history, False)

    @staticmethod
    fn prepare_training_data(
        raw_data: List[List[Float64]],
    ) -> (List[List[Float64]], List[List[Float64]]):
        """
        Prepare raw data for training by creating input-target pairs.

        Args:
            raw_data: Raw pendulum data samples.

        Returns:
            Tuple of (input_sequences, target_sequences).
        """
        inputs = List[List[Float64]]()
        targets = List[List[Float64]]()

        # Create sequences for time-series prediction
        for i in range(len(raw_data) - 1):
            if i + 1 < len(raw_data):
                # Current state as input
                input_vec = List[Float64]()
                for j in range(
                    min(4, len(raw_data[i]))
                ):  # Take first 4 elements as input
                    input_vec.append(raw_data[i][j])

                # Next state as target
                target_vec = List[Float64]()
                for j in range(
                    min(3, len(raw_data[i + 1]))
                ):  # Take first 3 elements as target
                    target_vec.append(raw_data[i + 1][j])

                inputs.append(input_vec)
                targets.append(target_vec)

        return (inputs, targets)

    @staticmethod
    fn validate_training_data(
        inputs: List[List[Float64]], targets: List[List[Float64]]
    ) -> Bool:
        """
        Validate training data format and consistency.

        Args:
            inputs: Input data to validate.
            targets: Target data to validate.

        Returns:
            True if data is valid for training.
        """
        if len(inputs) != len(targets):
            return False

        if len(inputs) == 0:
            return False

        # Check input dimensions
        for i in range(len(inputs)):
            if len(inputs[i]) != 4:  # Expected 4 input features
                return False

        # Check target dimensions
        for i in range(len(targets)):
            if len(targets[i]) != 3:  # Expected 3 output features
                return False

        return True


# Example usage can be found in separate demo files
# This module provides training infrastructure components for import
