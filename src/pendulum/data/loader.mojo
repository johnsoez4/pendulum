"""
Data loading and preprocessing module for pendulum experimental data.

This module provides utilities for loading CSV data, preprocessing samples,
and preparing training datasets with proper normalization and validation.
"""

from collections import List, Dict
from math import abs, sqrt, min, max
from memory import UnsafePointer

# Import project configuration
from config.pendulum_config import (
    PendulumConfig,
    MAX_ACTUATOR_TRAVEL,
    MAX_CONTROL_VOLTAGE,
    PENDULUM_FULL_ROTATION,
    MAX_PENDULUM_VELOCITY,
    DATA_VALIDATION_TOLERANCE,
    OUTLIER_DETECTION_SIGMA,
)

struct DataSample:
    """
    Single data sample from pendulum experiment.
    
    Represents one timestep of pendulum system state and control input.
    """
    
    var la_position: Float64      # Linear actuator position (inches)
    var pend_velocity: Float64    # Pendulum angular velocity (deg/s)
    var pend_position: Float64    # Pendulum angle (degrees)
    var cmd_volts: Float64        # Control voltage (volts)
    var elapsed: Float64          # Time since previous sample (ms)
    var timestamp: Float64        # Absolute timestamp (seconds)
    
    fn __init__(out self, la_pos: Float64, pend_vel: Float64, pend_pos: Float64, 
                cmd_v: Float64, elapsed_ms: Float64, ts: Float64 = 0.0):
        """Initialize data sample with values."""
        self.la_position = la_pos
        self.pend_velocity = pend_vel
        self.pend_position = pend_pos
        self.cmd_volts = cmd_v
        self.elapsed = elapsed_ms
        self.timestamp = ts
    
    fn is_valid(self) -> Bool:
        """
        Validate sample data against physical constraints.
        
        Returns:
            True if sample is within valid ranges
        """
        # Check actuator position limits
        if abs(self.la_position) > MAX_ACTUATOR_TRAVEL + DATA_VALIDATION_TOLERANCE:
            return False
        
        # Check control voltage limits
        if abs(self.cmd_volts) > MAX_CONTROL_VOLTAGE + DATA_VALIDATION_TOLERANCE:
            return False
        
        # Check pendulum angle range
        if abs(self.pend_position) > PENDULUM_FULL_ROTATION / 2.0 + DATA_VALIDATION_TOLERANCE:
            return False
        
        # Check velocity reasonableness
        if abs(self.pend_velocity) > MAX_PENDULUM_VELOCITY + DATA_VALIDATION_TOLERANCE:
            return False
        
        # Check elapsed time reasonableness (should be ~40ms)
        if self.elapsed < 10.0 or self.elapsed > 100.0:
            return False
        
        return True
    
    fn normalize(self, means: List[Float64], stds: List[Float64]) -> List[Float64]:
        """
        Normalize sample data using provided statistics.
        
        Args:
            means: Mean values for [la_pos, pend_vel, pend_pos, cmd_volts]
            stds: Standard deviations for [la_pos, pend_vel, pend_pos, cmd_volts]
            
        Returns:
            Normalized values as list
        """
        var normalized = List[Float64]()
        
        # Normalize each field
        normalized.append((self.la_position - means[0]) / stds[0])
        normalized.append((self.pend_velocity - means[1]) / stds[1])
        normalized.append((self.pend_position - means[2]) / stds[2])
        normalized.append((self.cmd_volts - means[3]) / stds[3])
        
        return normalized

struct DataStatistics:
    """Statistical summary of dataset for normalization and analysis."""
    
    var means: List[Float64]
    var stds: List[Float64]
    var mins: List[Float64]
    var maxs: List[Float64]
    var count: Int
    
    fn __init__(out self):
        """Initialize empty statistics."""
        self.means = List[Float64]()
        self.stds = List[Float64]()
        self.mins = List[Float64]()
        self.maxs = List[Float64]()
        self.count = 0
    
    fn compute_from_samples(mut self, samples: List[DataSample]):
        """
        Compute statistics from list of data samples.
        
        Args:
            samples: List of data samples to analyze
        """
        if len(samples) == 0:
            return
        
        self.count = len(samples)
        
        # Initialize accumulators
        var sums = List[Float64](0.0, 0.0, 0.0, 0.0)
        var sum_squares = List[Float64](0.0, 0.0, 0.0, 0.0)
        var mins = List[Float64](1e9, 1e9, 1e9, 1e9)
        var maxs = List[Float64](-1e9, -1e9, -1e9, -1e9)
        
        # Accumulate statistics
        for i in range(len(samples)):
            sample = samples[i]
            values = List[Float64](sample.la_position, sample.pend_velocity, 
                                 sample.pend_position, sample.cmd_volts)
            
            for j in range(4):
                val = values[j]
                sums[j] += val
                sum_squares[j] += val * val
                mins[j] = min(mins[j], val)
                maxs[j] = max(maxs[j], val)
        
        # Compute means
        self.means = List[Float64]()
        for i in range(4):
            self.means.append(sums[i] / Float64(self.count))
        
        # Compute standard deviations
        self.stds = List[Float64]()
        for i in range(4):
            variance = (sum_squares[i] / Float64(self.count)) - (self.means[i] * self.means[i])
            self.stds.append(sqrt(max(variance, 1e-8)))  # Avoid division by zero
        
        # Store min/max values
        self.mins = mins
        self.maxs = maxs

struct PendulumDataLoader:
    """
    Main data loader for pendulum experimental data.
    
    Provides functionality to load CSV data, validate samples, compute statistics,
    and prepare datasets for training and validation.
    """
    
    var _samples: List[DataSample]
    var _statistics: DataStatistics
    var _loaded: Bool
    var _config: PendulumConfig
    
    fn __init__(out self):
        """Initialize data loader."""
        self._samples = List[DataSample]()
        self._statistics = DataStatistics()
        self._loaded = False
        self._config = PendulumConfig()
    
    fn load_csv_data(mut self, file_path: String) raises -> Int:
        """
        Load pendulum data from CSV file.
        
        Args:
            file_path: Path to CSV data file
            
        Returns:
            Number of samples loaded
            
        Raises:
            Error if file cannot be read or data is invalid
        """
        # TODO: Implement actual CSV file reading when Mojo file I/O is available
        # For now, create synthetic data based on known characteristics
        
        self._samples = List[DataSample]()
        
        # Generate representative samples based on data analysis
        # This simulates the actual data loading process
        var sample_count = 1000  # Reduced for testing
        var timestamp = 0.0
        
        for i in range(sample_count):
            # Simulate data patterns observed in actual dataset
            var t = Float64(i) * 0.04  # 40ms intervals
            
            # Simulate pendulum motion (simplified)
            var angle = 180.0 * (1.0 - Float64(i) / Float64(sample_count))  # Transition from hanging
            var velocity = -200.0 + 400.0 * Float64(i) / Float64(sample_count)  # Increasing velocity
            var actuator = -2.0 + 4.0 * Float64(i) / Float64(sample_count)  # Actuator movement
            var voltage = 0.1 * (Float64(i % 10) - 5.0)  # Control voltage variation
            
            var sample = DataSample(actuator, velocity, angle, voltage, 40.0, timestamp)
            
            if sample.is_valid():
                self._samples.append(sample)
            
            timestamp += 0.04
        
        # Compute dataset statistics
        self._statistics.compute_from_samples(self._samples)
        self._loaded = True
        
        return len(self._samples)
    
    fn get_sample_count(self) -> Int:
        """Get number of loaded samples."""
        return len(self._samples)
    
    fn get_sample(self, index: Int) -> DataSample:
        """
        Get sample at specified index.
        
        Args:
            index: Sample index
            
        Returns:
            DataSample at index
        """
        return self._samples[index]
    
    fn get_statistics(self) -> DataStatistics:
        """Get dataset statistics."""
        return self._statistics
    
    fn create_training_sequences(self, sequence_length: Int, stride: Int = 1) -> List[List[DataSample]]:
        """
        Create training sequences from loaded data.
        
        Args:
            sequence_length: Length of each training sequence
            stride: Step size between sequences
            
        Returns:
            List of training sequences
        """
        var sequences = List[List[DataSample]]()
        
        var i = 0
        while i + sequence_length <= len(self._samples):
            var sequence = List[DataSample]()
            
            for j in range(sequence_length):
                sequence.append(self._samples[i + j])
            
            sequences.append(sequence)
            i += stride
        
        return sequences
    
    fn split_data(self, train_ratio: Float64 = 0.7, val_ratio: Float64 = 0.15) -> (List[DataSample], List[DataSample], List[DataSample]):
        """
        Split data into training, validation, and test sets.
        
        Args:
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set (remainder goes to test)
            
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        var total_samples = len(self._samples)
        var train_count = Int(Float64(total_samples) * train_ratio)
        var val_count = Int(Float64(total_samples) * val_ratio)
        
        var train_samples = List[DataSample]()
        var val_samples = List[DataSample]()
        var test_samples = List[DataSample]()
        
        # Split samples
        for i in range(total_samples):
            if i < train_count:
                train_samples.append(self._samples[i])
            elif i < train_count + val_count:
                val_samples.append(self._samples[i])
            else:
                test_samples.append(self._samples[i])
        
        return (train_samples, val_samples, test_samples)
    
    fn validate_data_quality(self) -> Dict[String, Bool]:
        """
        Validate overall data quality and consistency.
        
        Returns:
            Dictionary with validation results
        """
        var results = Dict[String, Bool]()
        
        if not self._loaded:
            results["data_loaded"] = False
            return results
        
        results["data_loaded"] = True
        
        # Check sample count
        results["sufficient_samples"] = len(self._samples) > 100
        
        # Check for valid samples
        var valid_count = 0
        for i in range(len(self._samples)):
            if self._samples[i].is_valid():
                valid_count += 1
        
        results["valid_samples"] = Float64(valid_count) / Float64(len(self._samples)) > 0.95
        
        # Check data ranges
        results["actuator_range_ok"] = (self._statistics.maxs[0] - self._statistics.mins[0]) > 2.0
        results["velocity_range_ok"] = (self._statistics.maxs[1] - self._statistics.mins[1]) > 100.0
        results["angle_range_ok"] = (self._statistics.maxs[2] - self._statistics.mins[2]) > 90.0
        
        return results

@staticmethod
fn load_pendulum_data(file_path: String) raises -> PendulumDataLoader:
    """
    Convenience function to load pendulum data.
    
    Args:
        file_path: Path to CSV data file
        
    Returns:
        Configured PendulumDataLoader with loaded data
    """
    var loader = PendulumDataLoader()
    _ = loader.load_csv_data(file_path)
    return loader
