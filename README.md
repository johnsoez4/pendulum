# Inverted Pendulum AI Control System

A Mojo-based AI project for developing a digital twin and control system for an inverted pendulum using the MAX engine.

## Project Overview

This project implements a two-phase approach to AI-based pendulum control:

1. **Phase 1: Digital Twin Development** - AI model of pendulum dynamics using experimental data
2. **Phase 2: AI Control Algorithm** - Intelligent control system for achieving and maintaining inverted state

## Current Status

### ✅ Completed
- **Project Setup**: Complete directory structure and configuration
- **Requirements Documentation**: Comprehensive project requirements and specifications
- **Data Analysis**: Analysis of 10,101 experimental samples showing system characteristics
- **Physics Model**: Complete pendulum physics implementation with equations of motion
- **Data Loading**: Module for CSV data loading and preprocessing
- **Neural Network**: Basic architecture for digital twin (CPU-based)
- **Configuration**: System parameters and constants

### 🔄 In Progress
- **Neural Network Architecture**: Physics-informed network design
- **Training Infrastructure**: Model training and validation framework

### 📋 Planned
- **Model Training**: Train digital twin on experimental data
- **Model Validation**: Comprehensive testing and validation
- **Control Algorithm**: AI-based control system development
- **Integration Testing**: Complete system validation

## Key Findings from Data Analysis

- **Total Data**: 10,101 samples over 404 seconds (6.7 minutes)
- **Sample Rate**: 25 Hz (40ms intervals) - suitable for real-time control
- **System States**:
  - Inverted (|angle| ≤ 10°): 14.4% of data
  - Swinging (10° < |angle| ≤ 170°): 72.3% of data  
  - Hanging (|angle| > 170°): 13.3% of data
- **Dynamic Range**: ±955 deg/s maximum angular velocity
- **Control Authority**: Full ±4 inch actuator range, ±5V control voltage

## Architecture

### Directory Structure
```
pendulum/
├── src/pendulum/           # Main Mojo source code
│   ├── digital_twin/       # Digital twin implementation
│   ├── control/            # AI control algorithms
│   ├── data/               # Data processing utilities
│   └── utils/              # Common utilities
├── tests/pendulum/         # Test files
├── docs/                   # Documentation
├── config/                 # Configuration files
├── data/                   # Experimental data
└── presentation/           # Background materials
```

### Key Components

#### Data Processing (`src/pendulum/data/`)
- **loader.mojo**: CSV data loading and preprocessing
- **analyzer.mojo**: Data analysis and statistics

#### Physics Model (`src/pendulum/utils/`)
- **physics.mojo**: Complete pendulum dynamics model
- Equations of motion with RK4 integration
- Energy calculations and constraint validation
- Physics-informed constraints for AI training

#### Digital Twin (`src/pendulum/digital_twin/`)
- **neural_network.mojo**: Physics-informed neural network
- 4-input, 3-output architecture
- 3 hidden layers with 128 neurons each
- Physics constraints integrated into predictions

#### Configuration (`config/`)
- **pendulum_config.mojo**: System parameters and constants
- Physical limits and safety margins
- Model hyperparameters and training settings

## Technical Specifications

### Physical System
- **Actuator Range**: ±4 inches with limit switches
- **Pendulum Dynamics**: Full ±180° rotation capability
- **Control Voltage**: ±5V motor control range
- **Sample Rate**: 25 Hz real-time operation

### AI Model Architecture
- **Input**: [actuator_position, pendulum_velocity, pendulum_angle, control_voltage]
- **Output**: [next_actuator_position, next_pendulum_velocity, next_pendulum_angle]
- **Hidden Layers**: 3 layers × 128 neurons with tanh activation
- **Physics Integration**: Energy conservation and constraint handling

### Performance Targets
- **Digital Twin Accuracy**: <5% prediction error
- **Control Success Rate**: >90% inversion achievement
- **Stability Duration**: >30 seconds inverted state
- **Real-time Performance**: 25 Hz control loop

## Development Environment

### Requirements
- **Mojo**: Latest version (currently not installed - see setup guide)
- **MAX Engine**: For GPU acceleration (requires compatible GPU)
- **System**: Linux/macOS with sufficient memory

### Current Limitations
- **Mojo Installation**: Not currently available on development system
- **MAX Engine**: Requires GPU-enabled system for full functionality
- **File I/O**: Limited Mojo file operations (using Python for data analysis)

### Setup Instructions
1. Install Mojo: `sudo snap install mojo` or follow official docs
2. Verify installation: `mojo --version`
3. For MAX engine: Requires GPU-compatible system
4. See `docs/DEVELOPMENT_SETUP.md` for detailed instructions

## Usage

### Data Analysis (Python)
```bash
# Analyze experimental data
python3 -c "
import csv
# ... (see data analysis script in docs)
"
```

### Mojo Development (when available)
```bash
# Compile and run digital twin
mojo run src/pendulum/digital_twin/main.mojo

# Run tests
mojo test tests/pendulum/

# Train model
mojo run examples/train_digital_twin.mojo
```

## Documentation

- **[Requirements](requirements.md)**: Complete project requirements and specifications
- **[Data Analysis Report](docs/DATA_ANALYSIS_REPORT.md)**: Comprehensive data analysis
- **[Project Structure](docs/PROJECT_STRUCTURE.md)**: Code organization and conventions
- **[Development Setup](docs/DEVELOPMENT_SETUP.md)**: Environment setup instructions
- **[Background](docs/pendulum_background.md)**: Project context and history
- **[Methodology](docs/investigation_methodology.md)**: Research approach and methods

## Next Steps

### Immediate (Phase 1 Completion)
1. **Install Mojo** on development system
2. **Complete Training Infrastructure** with loss functions and optimization
3. **Implement Model Training** using experimental data
4. **Validate Digital Twin** accuracy and stability
5. **Performance Testing** for real-time requirements

### Future (Phase 2)
1. **Control Algorithm Design** using reinforcement learning or MPC
2. **Controller Training** in simulation environment
3. **Safety System Integration** with constraint handling
4. **Complete System Testing** and validation
5. **GPU Migration** for MAX engine utilization

## Success Criteria

### Phase 1: Digital Twin
- ✅ Data loading and preprocessing complete
- ✅ Physics model implemented
- ✅ Neural network architecture designed
- 🔄 Training infrastructure (in progress)
- 📋 Model training and validation (planned)
- 📋 <5% prediction error achievement (planned)

### Phase 2: Control System
- 📋 Control algorithm implementation
- 📋 >90% inversion success rate
- 📋 >30 second stability duration
- 📋 Real-time performance validation

## Contributing

This project follows Mojo best practices as defined in `mojo_syntax.md`:
- Use proper import patterns and naming conventions
- Include comprehensive docstrings and type annotations
- Implement physics-informed constraints
- Follow modular architecture principles

## License

This project is developed for educational and research purposes as part of the Modular Hackathon 2025.

---

*For detailed technical information, see the documentation in the `docs/` directory.*
