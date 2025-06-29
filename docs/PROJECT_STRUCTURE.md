# Inverted Pendulum Project Structure

## Directory Organization

```
pendulum/
├── src/pendulum/           # Main Mojo source code
│   ├── digital_twin/       # Digital twin implementation
│   ├── control/            # AI control algorithms
│   ├── data/               # Data processing utilities
│   └── utils/              # Common utilities
├── tests/pendulum/         # Test files
│   ├── test_digital_twin/  # Digital twin tests
│   ├── test_control/       # Control algorithm tests
│   └── test_utils/         # Utility tests
├── docs/                   # Documentation
├── examples/               # Example usage and demos
├── data/                   # Experimental data
└── presentation/           # Background materials
```

## File Naming Conventions

Following `mojo_syntax.md` guidelines:

### Source Files
- **Modules**: `snake_case.mojo` (e.g., `digital_twin.mojo`, `pendulum_model.mojo`)
- **Test Files**: `test_*.mojo` (e.g., `test_digital_twin.mojo`)
- **Main Entry**: `main.mojo` for executable modules

### Naming Standards
- **Structs**: `PascalCase` (e.g., `PendulumModel`, `ControlSystem`)
- **Functions**: `snake_case` (e.g., `predict_next_state`, `apply_control`)
- **Constants**: `UPPER_CASE` (e.g., `MAX_ACTUATOR_TRAVEL`, `SAMPLE_RATE`)
- **Variables**: `snake_case` (e.g., `pendulum_angle`, `control_voltage`)

## Module Organization

### src/pendulum/digital_twin/
- `pendulum_model.mojo` - Core digital twin model
- `data_processor.mojo` - Data preprocessing and analysis
- `trainer.mojo` - Model training infrastructure
- `validator.mojo` - Model validation and testing

### src/pendulum/control/
- `ai_controller.mojo` - Main AI control algorithm
- `safety_monitor.mojo` - Safety constraints and monitoring
- `optimization.mojo` - Control optimization algorithms
- `state_estimator.mojo` - State estimation utilities

### src/pendulum/data/
- `loader.mojo` - Data loading and parsing
- `preprocessor.mojo` - Data preprocessing utilities
- `analyzer.mojo` - Data analysis tools
- `visualizer.mojo` - Data visualization (if supported)

### src/pendulum/utils/
- `math_utils.mojo` - Mathematical utilities
- `physics.mojo` - Physics calculations and constants
- `config.mojo` - Configuration management
- `logger.mojo` - Logging utilities

## Import Patterns

Following mojo_syntax.md standards:

```mojo
# Standard library imports first
from sys.ffi import external_call
from memory import UnsafePointer
from collections import Dict
from testing import assert_equal, assert_true, assert_false

# Project imports with full paths from root
from src.pendulum.digital_twin import (
    PendulumModel,
    DataProcessor,
    ModelTrainer,
)
from src.pendulum.control import (
    AIController,
    SafetyMonitor,
)
from src.pendulum.utils import (
    MathUtils,
    PhysicsConstants,
    Logger,
)
```

## Development Workflow

### 1. Phase 1: Digital Twin
1. Implement data loading and preprocessing
2. Design and implement pendulum model architecture
3. Create training infrastructure
4. Train and validate model
5. Test model performance

### 2. Phase 2: Control System
1. Implement control algorithm framework
2. Design AI controller using digital twin
3. Add safety monitoring and constraints
4. Train controller in simulation
5. Integrate and test complete system

### 3. Testing Strategy
- Unit tests for each module
- Integration tests for system components
- Performance tests for real-time requirements
- Safety tests for constraint validation

## Documentation Standards

- Module-level docstrings for all files
- Function docstrings with parameters and return types
- Struct docstrings explaining purpose and usage
- Inline comments for complex algorithms
- README files for each major directory

## Configuration Management

- `config/` directory for configuration files
- Environment-specific settings
- Model hyperparameters
- System constants and limits
- GPU/MAX engine configuration

---

*This structure follows Mojo best practices and supports the two-phase development approach for the inverted pendulum project.*
