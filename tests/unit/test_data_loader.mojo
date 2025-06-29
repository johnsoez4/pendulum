"""
Unit tests for data loading and processing module.

This module tests data loading, validation, preprocessing, and statistics
calculation functions.
"""

from collections import List
from testing import assert_equal, assert_true, assert_false

# Helper functions for testing
fn abs(x: Float64) -> Float64:
    """Return absolute value of x."""
    return x if x >= 0.0 else -x

fn assert_near(actual: Float64, expected: Float64, tolerance: Float64 = 1e-6):
    """Assert that two floating point values are close."""
    var diff = abs(actual - expected)
    if diff > tolerance:
        print("Assertion failed: expected", expected, "but got", actual, "diff:", diff)
        assert_true(False)

@fieldwise_init
struct DataSample(Copyable, Movable):
    """Test data sample structure."""
    var la_position: Float64      # Linear actuator position (inches)
    var pend_velocity: Float64    # Pendulum velocity (deg/s)
    var pend_position: Float64    # Pendulum position (degrees)
    var cmd_volts: Float64        # Command voltage (volts)
    var sample_rate: Float64      # Sample rate (Hz)
    var timestamp: Float64        # Timestamp (seconds)
    
    fn is_valid(self) -> Bool:
        """Check if data sample is valid."""
        # Check for NaN values
        if self.la_position != self.la_position:
            return False
        if self.pend_velocity != self.pend_velocity:
            return False
        if self.pend_position != self.pend_position:
            return False
        if self.cmd_volts != self.cmd_volts:
            return False
        
        # Check physical constraints
        if abs(self.la_position) > 10.0:  # 10 inch limit
            return False
        if abs(self.pend_velocity) > 2000.0:  # 2000 deg/s limit
            return False
        if abs(self.cmd_volts) > 15.0:  # 15V limit
            return False
        if self.sample_rate <= 0.0:
            return False
        if self.timestamp < 0.0:
            return False
        
        return True
    
    fn to_input_vector(self) -> List[Float64]:
        """Convert to neural network input format."""
        var input = List[Float64]()
        input.append(self.la_position)
        input.append(self.pend_velocity)
        input.append(self.pend_position)
        input.append(self.cmd_volts)
        return input

@fieldwise_init
struct DataStatistics(Copyable, Movable):
    """Statistics for data analysis."""
    var mean_la_pos: Float64
    var std_la_pos: Float64
    var mean_pend_vel: Float64
    var std_pend_vel: Float64
    var mean_pend_pos: Float64
    var std_pend_pos: Float64
    var mean_cmd_volts: Float64
    var std_cmd_volts: Float64
    var sample_count: Int
    
    fn compute_from_samples(mut self, samples: List[DataSample]):
        """Compute statistics from data samples."""
        if len(samples) == 0:
            return
        
        self.sample_count = len(samples)
        
        # Compute means
        var sum_la_pos = 0.0
        var sum_pend_vel = 0.0
        var sum_pend_pos = 0.0
        var sum_cmd_volts = 0.0
        
        for i in range(len(samples)):
            sum_la_pos += samples[i].la_position
            sum_pend_vel += samples[i].pend_velocity
            sum_pend_pos += samples[i].pend_position
            sum_cmd_volts += samples[i].cmd_volts
        
        var n = Float64(len(samples))
        self.mean_la_pos = sum_la_pos / n
        self.mean_pend_vel = sum_pend_vel / n
        self.mean_pend_pos = sum_pend_pos / n
        self.mean_cmd_volts = sum_cmd_volts / n
        
        # Compute standard deviations
        var sum_sq_la_pos = 0.0
        var sum_sq_pend_vel = 0.0
        var sum_sq_pend_pos = 0.0
        var sum_sq_cmd_volts = 0.0
        
        for i in range(len(samples)):
            var diff_la = samples[i].la_position - self.mean_la_pos
            var diff_vel = samples[i].pend_velocity - self.mean_pend_vel
            var diff_pos = samples[i].pend_position - self.mean_pend_pos
            var diff_volts = samples[i].cmd_volts - self.mean_cmd_volts
            
            sum_sq_la_pos += diff_la * diff_la
            sum_sq_pend_vel += diff_vel * diff_vel
            sum_sq_pend_pos += diff_pos * diff_pos
            sum_sq_cmd_volts += diff_volts * diff_volts
        
        self.std_la_pos = (sum_sq_la_pos / n) ** 0.5
        self.std_pend_vel = (sum_sq_pend_vel / n) ** 0.5
        self.std_pend_pos = (sum_sq_pend_pos / n) ** 0.5
        self.std_cmd_volts = (sum_sq_cmd_volts / n) ** 0.5

struct DataLoaderTests:
    """Test suite for data loading functionality."""
    
    @staticmethod
    fn test_data_sample_creation():
        """Test DataSample creation and validation."""
        print("Testing DataSample creation...")
        
        # Valid sample
        var valid_sample = DataSample(1.0, 100.0, 180.0, 2.0, 25.0, 1.0)
        assert_true(valid_sample.is_valid())
        
        # Invalid samples
        var invalid_pos = DataSample(15.0, 100.0, 180.0, 2.0, 25.0, 1.0)  # Position too large
        assert_false(invalid_pos.is_valid())
        
        var invalid_vel = DataSample(1.0, 3000.0, 180.0, 2.0, 25.0, 1.0)  # Velocity too large
        assert_false(invalid_vel.is_valid())
        
        var invalid_volts = DataSample(1.0, 100.0, 180.0, 20.0, 25.0, 1.0)  # Voltage too large
        assert_false(invalid_volts.is_valid())
        
        var invalid_rate = DataSample(1.0, 100.0, 180.0, 2.0, -1.0, 1.0)  # Negative sample rate
        assert_false(invalid_rate.is_valid())
        
        print("✓ DataSample creation test passed")
    
    @staticmethod
    fn test_input_vector_conversion():
        """Test conversion to neural network input format."""
        print("Testing input vector conversion...")
        
        var sample = DataSample(2.0, 150.0, 90.0, 1.5, 25.0, 2.0)
        var input_vector = sample.to_input_vector()
        
        assert_equal(len(input_vector), 4)
        assert_near(input_vector[0], 2.0)
        assert_near(input_vector[1], 150.0)
        assert_near(input_vector[2], 90.0)
        assert_near(input_vector[3], 1.5)
        
        print("✓ Input vector conversion test passed")
    
    @staticmethod
    fn test_statistics_computation():
        """Test statistics calculation from data samples."""
        print("Testing statistics computation...")
        
        # Create test samples
        var samples = List[DataSample]()
        samples.append(DataSample(0.0, 0.0, 180.0, 0.0, 25.0, 0.0))
        samples.append(DataSample(2.0, 100.0, 170.0, 1.0, 25.0, 1.0))
        samples.append(DataSample(-2.0, -100.0, 190.0, -1.0, 25.0, 2.0))
        
        var stats = DataStatistics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        stats.compute_from_samples(samples)
        
        # Check means
        assert_near(stats.mean_la_pos, 0.0, 1e-6)  # (0 + 2 - 2) / 3 = 0
        assert_near(stats.mean_pend_vel, 0.0, 1e-6)  # (0 + 100 - 100) / 3 = 0
        assert_near(stats.mean_pend_pos, 180.0, 1e-6)  # (180 + 170 + 190) / 3 = 180
        assert_near(stats.mean_cmd_volts, 0.0, 1e-6)  # (0 + 1 - 1) / 3 = 0
        
        # Check sample count
        assert_equal(stats.sample_count, 3)
        
        print("✓ Statistics computation test passed")
    
    @staticmethod
    fn test_data_validation():
        """Test data validation functions."""
        print("Testing data validation...")
        
        # Create mixed valid/invalid samples
        var samples = List[DataSample]()
        samples.append(DataSample(1.0, 50.0, 180.0, 1.0, 25.0, 1.0))    # Valid
        samples.append(DataSample(20.0, 50.0, 180.0, 1.0, 25.0, 2.0))   # Invalid position
        samples.append(DataSample(2.0, 50.0, 180.0, 1.0, 25.0, 3.0))    # Valid
        samples.append(DataSample(3.0, 5000.0, 180.0, 1.0, 25.0, 4.0))  # Invalid velocity
        
        var valid_count = 0
        for i in range(len(samples)):
            if samples[i].is_valid():
                valid_count += 1
        
        assert_equal(valid_count, 2)  # Only 2 valid samples
        
        print("✓ Data validation test passed")
    
    @staticmethod
    fn test_data_preprocessing():
        """Test data preprocessing and normalization."""
        print("Testing data preprocessing...")
        
        # Create samples with known statistics
        var samples = List[DataSample]()
        samples.append(DataSample(0.0, 0.0, 180.0, 0.0, 25.0, 0.0))
        samples.append(DataSample(4.0, 200.0, 0.0, 2.0, 25.0, 1.0))
        samples.append(DataSample(-4.0, -200.0, 360.0, -2.0, 25.0, 2.0))
        
        var stats = DataStatistics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        stats.compute_from_samples(samples)
        
        # Test normalization (z-score)
        var sample = samples[1]  # [4.0, 200.0, 0.0, 2.0]
        var input_vector = sample.to_input_vector()
        
        # Normalize using computed statistics
        var normalized = List[Float64]()
        normalized.append((input_vector[0] - stats.mean_la_pos) / stats.std_la_pos)
        normalized.append((input_vector[1] - stats.mean_pend_vel) / stats.std_pend_vel)
        normalized.append((input_vector[2] - stats.mean_pend_pos) / stats.std_pend_pos)
        normalized.append((input_vector[3] - stats.mean_cmd_volts) / stats.std_cmd_volts)
        
        # Check that normalized values are reasonable
        for i in range(len(normalized)):
            assert_true(abs(normalized[i]) < 10.0)  # Should be within reasonable range
        
        print("✓ Data preprocessing test passed")
    
    @staticmethod
    fn test_sequence_creation():
        """Test creation of training sequences."""
        print("Testing sequence creation...")
        
        # Create time series data
        var samples = List[DataSample]()
        for i in range(10):
            var t = Float64(i) * 0.04  # 40ms intervals
            var sample = DataSample(
                Float64(i) * 0.1,      # Increasing position
                Float64(i) * 10.0,     # Increasing velocity
                180.0 - Float64(i),    # Decreasing angle
                0.1 * Float64(i % 3),  # Varying voltage
                25.0, t
            )
            samples.append(sample)
        
        # Test sequence creation (input-target pairs)
        var sequence_length = 3
        var sequences = List[List[DataSample]]()
        
        for i in range(len(samples) - sequence_length):
            var sequence = List[DataSample]()
            for j in range(sequence_length):
                sequence.append(samples[i + j])
            sequences.append(sequence)
        
        # Check sequence properties
        var expected_sequences = len(samples) - sequence_length
        assert_equal(len(sequences), expected_sequences)
        assert_equal(len(sequences[0]), sequence_length)
        
        print("✓ Sequence creation test passed")
    
    @staticmethod
    fn run_all_tests():
        """Run all data loader tests."""
        print("Running Data Loader Unit Tests")
        print("==============================")
        
        DataLoaderTests.test_data_sample_creation()
        DataLoaderTests.test_input_vector_conversion()
        DataLoaderTests.test_statistics_computation()
        DataLoaderTests.test_data_validation()
        DataLoaderTests.test_data_preprocessing()
        DataLoaderTests.test_sequence_creation()
        
        print()
        print("✓ All data loader tests passed!")
        print()

fn main():
    """Run data loader unit tests."""
    DataLoaderTests.run_all_tests()
