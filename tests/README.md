# Pendulum Digital Twin - Test Suite

This directory contains a comprehensive test suite for the pendulum digital twin system, including unit tests, integration tests, and performance benchmarks.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_physics.mojo           # Physics model tests
│   ├── test_neural_network.mojo    # Neural network tests
│   └── test_data_loader.mojo       # Data loading tests
├── integration/             # Integration tests for system components
│   └── test_training_pipeline.mojo # End-to-end training tests
├── performance/             # Performance benchmarks
│   └── test_benchmarks.mojo        # Real-time performance tests
├── data/                    # Test data files
├── run_all_tests.mojo      # Comprehensive test runner
└── README.md               # This file
```

## Test Categories

### 1. Unit Tests (`unit/`)

**Physics Tests (`test_physics.mojo`)**
- PendulumState creation and validation
- Energy conservation calculations
- Physics constraint enforcement
- Trigonometric approximations
- Unit conversions (inches/meters, degrees/radians)

**Neural Network Tests (`test_neural_network.mojo`)**
- Network architecture validation
- Forward pass functionality
- Physics constraint integration
- Activation function accuracy
- Weight initialization
- Input/output validation

**Data Loader Tests (`test_data_loader.mojo`)**
- Data sample validation
- Statistics computation
- Data preprocessing and normalization
- Sequence creation for training
- Input vector conversion

### 2. Integration Tests (`integration/`)

**Training Pipeline Tests (`test_training_pipeline.mojo`)**
- End-to-end training workflow
- Data generation and validation
- Training loop execution
- Physics constraint enforcement
- Validation pipeline
- Complete system integration

### 3. Performance Tests (`performance/`)

**Benchmark Tests (`test_benchmarks.mojo`)**
- Inference latency measurement
- System throughput testing
- Memory usage analysis
- Batch processing performance
- Real-time control loop simulation
- 25 Hz frequency validation

## Running Tests

### Run All Tests
```bash
mojo build tests/run_all_tests.mojo
./run_all_tests
```

### Run Individual Test Suites
```bash
# Unit tests
mojo build tests/unit/test_physics.mojo && ./test_physics
mojo build tests/unit/test_neural_network.mojo && ./test_neural_network
mojo build tests/unit/test_data_loader.mojo && ./test_data_loader

# Integration tests
mojo build tests/integration/test_training_pipeline.mojo && ./test_training_pipeline

# Performance tests
mojo build tests/performance/test_benchmarks.mojo && ./test_benchmarks
```

## Test Requirements and Targets

### Functional Requirements
- **Physics Accuracy**: Energy conservation within 1% error
- **Constraint Enforcement**: 100% compliance with actuator/velocity limits
- **Data Validation**: >95% valid samples in datasets
- **Training Convergence**: Loss reduction and early stopping functionality

### Performance Requirements
- **Inference Latency**: <40ms per prediction (25 Hz requirement)
- **Throughput**: >25 predictions/second sustained
- **Memory Usage**: <100KB for neural network weights
- **Real-time Control**: 95% of target frequency achievement

### Physics Constraints
- **Actuator Position**: [-4, 4] inches
- **Pendulum Velocity**: [-1000, 1000] degrees/second
- **Control Voltage**: [-10, 10] volts
- **Energy Conservation**: <5% deviation in physics-informed loss

## Test Results Interpretation

### Success Criteria
- **All Unit Tests Pass**: Individual components function correctly
- **All Integration Tests Pass**: System components work together
- **Performance Targets Met**: Real-time requirements satisfied
- **Physics Compliance**: 100% constraint satisfaction

### Common Issues and Debugging

**Physics Test Failures**
- Check unit conversions (inches/meters, degrees/radians)
- Verify energy calculation accuracy
- Validate constraint boundary conditions

**Neural Network Test Failures**
- Verify weight initialization ranges
- Check activation function approximations
- Validate input/output dimensions

**Performance Test Failures**
- Profile inference latency bottlenecks
- Check memory allocation patterns
- Optimize constraint application

**Integration Test Failures**
- Verify data flow between components
- Check training loop convergence
- Validate end-to-end pipeline

## Test Data

### Synthetic Data Generation
Tests use physics-based synthetic data generation to ensure:
- Consistent test conditions
- Known ground truth values
- Comprehensive edge case coverage
- Reproducible results

### Test Data Characteristics
- **Sample Rate**: 25 Hz (40ms intervals)
- **Duration**: Variable (1-10 seconds for different tests)
- **State Space**: Full pendulum dynamics coverage
- **Constraints**: All physical limits represented

## Continuous Integration

### Automated Testing
The test suite is designed for automated execution in CI/CD pipelines:
- Fast execution (<30 seconds total)
- Clear pass/fail indicators
- Detailed error reporting
- Performance regression detection

### Test Coverage
- **Unit Tests**: 100% of core functions
- **Integration Tests**: All major workflows
- **Performance Tests**: All real-time requirements
- **Edge Cases**: Boundary conditions and error handling

## Contributing to Tests

### Adding New Tests
1. Follow existing test structure and naming conventions
2. Include both positive and negative test cases
3. Add performance benchmarks for new features
4. Update this README with new test descriptions

### Test Guidelines
- Use descriptive test names and comments
- Include assertion messages for failures
- Test edge cases and boundary conditions
- Validate both functionality and performance
- Ensure tests are deterministic and reproducible

## Performance Monitoring

### Benchmarking
Regular performance monitoring ensures:
- Real-time requirements are maintained
- Performance regressions are detected
- Optimization opportunities are identified
- System scalability is validated

### Metrics Tracked
- **Inference Latency**: Per-prediction timing
- **Memory Usage**: Peak and average consumption
- **Throughput**: Sustained prediction rate
- **CPU Utilization**: Processing efficiency

## Test Environment

### Requirements
- Mojo 25.5.0 or later
- GPU access (optional, for performance tests)
- Minimum 4GB RAM
- Linux/macOS environment

### Configuration
Tests are configured for:
- Development environment validation
- CI/CD pipeline execution
- Production readiness assessment
- Performance regression testing

---

For questions or issues with the test suite, please refer to the main project documentation or create an issue in the project repository.
