# Inverted Pendulum Digital Twin and AI Control System

## Project Overview

This project develops a Mojo-based AI system for an inverted pendulum, consisting of two main phases:
1. **Digital Twin Development**: AI-based model of pendulum apparatus using experimental data
2. **AI Control Algorithm**: Intelligent control system for achieving and maintaining inverted state

## Critical Project Parameters

### Physical System Specifications
- **Linear Actuator**: ±4 inch travel range with limit switches, center position = 0
- **Pendulum**: Angle measurement in degrees (upright/vertical = 0°, hanging = ±180°)
- **Control Signal**: Voltage applied to linear actuator motor (vdc)
- **Sampling Rate**: ~25Hz (40ms typical elapsed time between samples)

### Data Format (sample_data.csv)
- `la_position`: Linear actuator position (inches, range: ±4)
- `pend_velocity`: Pendulum angular velocity (degrees/second)
- `pend_position`: Pendulum angle (degrees, 0° = upright)
- `cmd_volts`: Control voltage to actuator motor (vdc)
- `elapsed`: Time since previous sample (milliseconds)

### Physical Constraints
- Actuator travel limits: ±4 inches with hard limit switches
- Pendulum dynamics: Natural frequency, damping, inertia characteristics
- Control voltage limits: Based on motor specifications
- Safety considerations: Prevent actuator overtravel and excessive forces

## Technical Requirements

### Mojo/MAX Integration
- **Language**: Mojo programming language following syntax guidelines in `mojo_syntax.md`
- **Engine**: MAX engine capabilities for AI/ML operations
- **GPU Requirement**: Compatible GPU needed for MAX engine (development transfer required)
- **Documentation**: Reference https://docs.modular.com/ and https://docs.modular.com/max/intro

### Development Environment
- **Workspace**: `/home/johnsoe1/dev/Modular/Hackathon_2025_06_27/pendulum/`
- **Source Structure**: All Mojo files in `src/` directory
- **Data Source**: `data/sample_data.csv` (10,101 data points)
- **Background**: `presentation/` directory contains original project context

## Implementation Plan

### Phase 1: Digital Twin Development

#### 1.1 Data Analysis and Preprocessing
- **Objective**: Understand pendulum dynamics from experimental data
- **Tasks**:
  - Load and analyze sample_data.csv (10,101 samples)
  - Identify system behavior patterns and transitions
  - Characterize pendulum states (hanging, swinging, inverted)
  - Extract key dynamic relationships
- **Deliverables**: Data analysis module, preprocessed datasets
- **Success Criteria**: Clear understanding of system dynamics and data quality

#### 1.2 Digital Twin Model Architecture
- **Objective**: Design AI model architecture for pendulum simulation
- **Tasks**:
  - Define state space representation (position, velocity, actuator state)
  - Design neural network architecture using MAX engine
  - Implement physics-informed constraints
  - Create model training infrastructure
- **Deliverables**: Model architecture, training framework
- **Success Criteria**: Scalable, physics-aware model design

#### 1.3 Model Training and Validation
- **Objective**: Train digital twin on experimental data
- **Tasks**:
  - Implement training loop with MAX engine optimization
  - Split data for training/validation/testing
  - Train model to predict next state given current state and control input
  - Validate model accuracy across different pendulum behaviors
- **Deliverables**: Trained digital twin model, validation metrics
- **Success Criteria**: <5% prediction error on validation data

#### 1.4 Digital Twin Testing
- **Objective**: Comprehensive testing of digital twin accuracy
- **Tasks**:
  - Test model on unseen data sequences
  - Evaluate long-term prediction stability
  - Test edge cases and boundary conditions
  - Performance benchmarking
- **Deliverables**: Test suite, performance reports
- **Success Criteria**: Stable predictions over 10+ second horizons

### Phase 2: AI Control Algorithm Development

#### 2.1 Control Problem Formulation
- **Objective**: Define control objectives and constraints
- **Tasks**:
  - Formulate inverted pendulum control as optimization problem
  - Define reward/cost functions for control performance
  - Specify safety constraints and operational limits
  - Design control architecture framework
- **Deliverables**: Control problem specification, safety framework
- **Success Criteria**: Well-defined, safe control objectives

#### 2.2 AI Controller Design
- **Objective**: Develop intelligent control algorithm
- **Tasks**:
  - Implement reinforcement learning or model predictive control
  - Use digital twin for controller training/simulation
  - Design controller to handle various initial conditions
  - Implement safety mechanisms and constraint handling
- **Deliverables**: AI controller implementation
- **Success Criteria**: Controller achieves inversion from multiple starting positions

#### 2.3 Controller Training and Optimization
- **Objective**: Train controller using digital twin
- **Tasks**:
  - Set up training environment with digital twin
  - Implement training algorithms (RL/MPC optimization)
  - Train controller on diverse scenarios
  - Optimize for robustness and performance
- **Deliverables**: Trained controller, training metrics
- **Success Criteria**: >90% success rate in simulation

#### 2.4 Integrated System Testing
- **Objective**: Test complete digital twin + controller system
- **Tasks**:
  - Integration testing of twin and controller
  - Test various initial conditions and disturbances
  - Validate safety mechanisms
  - Performance evaluation and optimization
- **Deliverables**: Integrated system, test results
- **Success Criteria**: Robust performance across test scenarios

## Success Criteria

### Phase 1 Success Metrics
- **Accuracy**: Digital twin prediction error <5% on validation data
- **Stability**: Stable predictions over 10+ second simulation horizons
- **Coverage**: Accurate modeling of all pendulum states (hanging, swinging, inverted)
- **Performance**: Real-time prediction capability (>25Hz)

### Phase 2 Success Metrics
- **Control Performance**: Achieve inverted state from arbitrary initial conditions
- **Success Rate**: >90% inversion success rate in simulation
- **Stability**: Maintain inverted state for >30 seconds
- **Safety**: No constraint violations or unsafe operations
- **Robustness**: Handle disturbances and model uncertainties

### Overall Project Success
- **Technical Achievement**: Working digital twin and control system
- **Mojo Integration**: Effective use of Mojo language and MAX engine
- **Documentation**: Comprehensive documentation and testing
- **Transferability**: System ready for GPU-based verification

## Risk Mitigation

### Technical Risks
- **Data Quality**: Validate data integrity and completeness
- **Model Convergence**: Implement multiple training strategies
- **Control Stability**: Extensive simulation testing before hardware
- **GPU Compatibility**: Plan for development environment transfer

### Project Risks
- **Scope Creep**: Focus on core objectives, document extensions
- **Timeline**: Prioritize Phase 1 completion before Phase 2
- **Resource Constraints**: Plan for GPU access requirements

## Next Steps

1. **Immediate**: Complete requirements documentation and project setup
2. **Phase 1 Start**: Begin data analysis and digital twin development
3. **Milestone Reviews**: Regular assessment of progress against success criteria
4. **GPU Transfer**: Plan transition to GPU-enabled development environment
5. **Testing Strategy**: Comprehensive validation before hardware integration

---

*This requirements document serves as the foundation for the inverted pendulum AI project, ensuring clear objectives, technical specifications, and success criteria for both development phases.*
