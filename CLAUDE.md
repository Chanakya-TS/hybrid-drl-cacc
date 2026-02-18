# Hybrid DRL-MPC Eco-Driving Controller Project

## Project Overview
This project implements a hybrid Deep Reinforcement Learning (DRL) and Model Predictive Control (MPC) framework for eco-driving in autonomous vehicles. The DRL agent learns to dynamically adjust MPC cost function weights to optimize energy consumption while maintaining safety and comfort.

## Persona
Act as a senior robotics engineer specializing in autonomous vehicle simulation and control. Code should be:
- Well-structured and modular
- Thoroughly commented with docstrings
- Following PEP 8 standards
- Using numpy for all numerical operations
- Type-annotated where appropriate

## Project Structure
```
/controllers          - All controller implementations (MPC, ACC, baselines)
/environment         - Simulation environment and Gymnasium wrapper
/utils              - Helper functions and utilities
/results            - Output data (CSV logs, plots, tables)
requirements.txt    - Python dependencies
train.py           - DRL agent training script
evaluate.py        - Controller evaluation script
generate_results.py - Results processing and visualization
```

## Key Technical Concepts

### Vehicle Dynamics (CARLA Simulator)
- **Simulation Platform**: CARLA Simulator (high-fidelity 3D vehicle physics)
- **State Variables**: [position, velocity, acceleration] from CARLA vehicle actor
- **Control Inputs**: throttle [0,1], brake [0,1], steering [optional for longitudinal-only]
- **Timestep**: Synchronous mode with fixed dt (e.g., 0.05s or 0.1s)
- **Physics**: Realistic engine dynamics, aerodynamic drag, rolling resistance, mass/inertia
- **Sensors**: Can access IMU, GPS, velocity, and other built-in sensors
- **Energy Model**: Use throttle position and vehicle speed to estimate power consumption

#### CARLA Implementation Details
- **World Settings**: Use synchronous mode for deterministic simulation
- **Vehicle Selection**: Choose appropriate vehicle blueprint (e.g., vehicle.tesla.model3)
- **Lead Vehicle**: Spawn second vehicle and control via Traffic Manager or custom trajectory
- **Scenario Setup**: Define route/waypoints for car-following scenario
- **Data Collection**: Extract velocity, position, throttle/brake from vehicle.get_velocity(), get_location(), get_control()

### MPC Controller
- Uses CasADi for optimization
- Prediction horizon: N steps (e.g., 20-50 steps)
- Decision variables: states X and controls U over horizon
- **Cost function components:**
  - w_v: velocity tracking weight (efficiency)
  - w_s: safety distance weight (safety)
  - w_u: control effort (smoothness/comfort) weight
- **Constraints:**
  - Simplified dynamics for prediction (kinematic approximation)
  - State limits (velocity bounds, e.g., 0-30 m/s)
  - Control limits (acceleration bounds, e.g., -3 to 2 m/s²)
  - Safety constraint: d(k) >= THW * v(k) where THW = 1.5-2.0s
- **Control Mapping:** MPC outputs desired acceleration → convert to CARLA throttle/brake
  - Positive acceleration → throttle control (PID or lookup table)
  - Negative acceleration → brake control

### DRL Integration
- Framework: stable-baselines3
- Algorithm: Soft Actor-Critic (SAC)
- Observation space: [ego_velocity, relative_velocity, distance_gap] (normalized)
- Action space: MPC weights [w_v, w_s, w_u] (bounded 0-1, normalized to sum to 1)
- Reward function: Based on energy efficiency, safety, and comfort

### Baseline Controllers
1. **Fixed-Weight MPC**: MPC with hand-tuned constant weights
2. **ACC (Adaptive Cruise Control)**: PID controller maintaining time headway
   - Setpoint: d_setpoint = THW * ego_velocity

## Implementation Guidelines

### Constants and Parameters
- Use clear, descriptive variable names
- Define magic numbers as named constants at module level
- Document units in comments (e.g., # m/s, # m/s^2)

### Safety
- Always validate safety distance constraints
- Include collision detection
- Log safety violations for analysis

### Data Logging
- Log all relevant time-series data during simulation
- Use consistent CSV format across all controllers
- Include headers: time, ego_velocity, lead_velocity, acceleration, distance_gap, (weights for hybrid)

### Testing and Validation
- Test scenario should be challenging: acceleration, cruise, hard braking, stop-and-go
- Ensure all controllers run on identical scenarios for fair comparison
- Validate that constraints are satisfied throughout simulation

## Metrics to Calculate
1. **Total Energy Consumption**:
   - Primary: Integral of (throttle * velocity) as power proxy
   - Alternative: Sum of positive acceleration work: m * Σ(max(a, 0) * v * dt)
2. **RMS Jerk**: Root mean square of acceleration derivative (comfort metric)
   - jerk = (a[t] - a[t-1]) / dt
   - RMS_jerk = sqrt(mean(jerk²))
3. **Minimum Time Headway**: Minimum d(t) / v_ego(t) during episode (safety metric)
4. **Average Speed**: Mean velocity over episode (efficiency indicator)
5. **Control Smoothness**: Standard deviation of throttle/brake changes

## Visualization Requirements
- **Figure 1**: Multi-panel plot (3 subplots) showing velocity, acceleration, and distance gap over time for all controllers
- **Figure 2**: Time series of DRL-chosen weights [w_v, w_s, w_u]
- Use clear legends, labels, and units
- Save as high-resolution PNG (300 dpi minimum)

## Common Pitfalls to Avoid
- Don't forget to normalize observations for DRL
- Ensure MPC weights sum to 1 after DRL outputs them
- Handle edge cases: zero velocity, lead vehicle stopping
- Check for numerical stability in optimization
- Validate that CasADi solver converges

### CARLA-Specific Pitfalls
- **Always use synchronous mode** with fixed time step for deterministic training
- **Clean up actors** on episode end (destroy ego and lead vehicles)
- **Handle CARLA connection loss** with try-except and reconnection logic
- **Throttle/Brake mapping**: Never apply both simultaneously (use if-else)
- **Vehicle spawning**: Check for collision-free spawn points
- **Lead vehicle control**: Use autopilot or set_target_velocity for consistent behavior
- **Sensor data timing**: Ensure sensor callbacks complete before next tick
- **Memory management**: CARLA can leak memory over long runs; periodic restarts may help

## Dependencies
```
numpy
matplotlib
pandas
scipy
stable-baselines3[extra]
gymnasium
casadi
carla==0.9.15  # Must match your CARLA server version
pygame  # For CARLA visualization (optional)
```

### CARLA Installation Notes
- Download CARLA from: https://github.com/carla-simulator/carla/releases
- Extract and run CarlaUE4.exe (Windows) or CarlaUE4.sh (Linux)
- Install Python API: `pip install carla==0.9.15` (match server version)
- Default connection: localhost:2000
- Recommended version: CARLA 0.9.13+ for stability

## Workflow
1. Setup project structure
2. Implement environment (CarFollowingEnv)
3. Implement MPC controller
4. Implement baseline controllers
5. Create Gymnasium wrapper
6. Train DRL agent
7. Evaluate all controllers
8. Generate results and visualizations

## Notes
- **CARLA Server**: Must be running before starting training/evaluation scripts
- **Rendering**: Can disable rendering in CARLA settings for faster training (use -RenderOffScreen flag)
- **Training Time**: DRL training may take hours; use reasonable timesteps (50k-200k)
- **Hardware**: CARLA is GPU-intensive; ensure adequate GPU memory (4GB+ recommended)
- **Debugging**: Use CARLA's spectator view to visualize what's happening
- **Results**: Should demonstrate clear advantages of hybrid approach over baselines
- **Reproducibility**: Set random seeds for CARLA, numpy, and stable-baselines3

## CARLA Quick Start Commands (Windows)
```bash
# Start CARLA server (in CARLA directory)
CarlaUE4.exe -quality-level=Low -fps=20

# Start CARLA without rendering (faster training)
CarlaUE4.exe -RenderOffScreen -quality-level=Low -fps=20

# In Python environment
pip install -r requirements.txt
python train.py
python evaluate.py
python generate_results.py
```
