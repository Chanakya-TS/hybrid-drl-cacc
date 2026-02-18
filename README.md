# Hybrid DRL-MPC Eco-Driving Controller

A hybrid Deep Reinforcement Learning (DRL) and Model Predictive Control (MPC) framework for eco-driving in autonomous vehicles using CARLA Simulator.

## Project Structure

```
.
├── controllers/          # Controller implementations
├── environment/         # CARLA-based simulation environment
│   └── car_following.py # Main car-following environment
├── utils/              # Helper utilities
├── results/            # Simulation results and plots
├── requirements.txt    # Python dependencies
├── CLAUDE.md          # Project guidelines
└── Action Plan.md     # Detailed implementation plan
```

## Setup

### 1. Install CARLA

Download CARLA 0.9.13 or later from:
https://github.com/carla-simulator/carla/releases

Extract to a convenient location.

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Ensure the `carla` package version matches your CARLA server version.

### 3. Start CARLA Server

**For development (with rendering):**
```bash
# Navigate to your CARLA directory
CarlaUE4.exe -quality-level=Low -fps=20
```

**For training (without rendering, faster):**
```bash
CarlaUE4.exe -RenderOffScreen -quality-level=Low -fps=20
```

## Phase 1: Environment Setup ✓

### Components Implemented:

1. **Project Structure** ✓
   - Created all necessary directories
   - Initialized Python packages
   - Created requirements.txt

2. **CARLA Environment Wrapper** ✓
   - `environment/car_following.py`: Complete CARLA-based car-following environment
   - Features:
     - Synchronous mode for deterministic simulation
     - Ego and lead vehicle spawning
     - Acceleration to throttle/brake mapping
     - Energy consumption tracking
     - Safety (collision) detection
     - Episode data logging
     - Customizable lead vehicle trajectory

### Testing the Environment

```python
from environment.car_following import CarFollowingEnv
import numpy as np

# Create test trajectory for lead vehicle
trajectory = np.concatenate([
    np.linspace(15, 15, 100),  # Constant velocity
    np.linspace(15, 8, 50),    # Deceleration
    np.linspace(8, 8, 100),    # Slow constant
    np.linspace(8, 18, 50),    # Acceleration
])

# Create environment
env = CarFollowingEnv(lead_vehicle_trajectory=trajectory)

# Reset environment
obs, info = env.reset()
print(f"Initial state: velocity={obs[0]:.2f} m/s, gap={obs[2]:.2f} m")

# Run simulation
for step in range(100):
    # Your controller here (simple example)
    acceleration = 0.5  # m/s²

    obs, reward, terminated, truncated, info = env.step(acceleration)

    if terminated or truncated:
        break

# Clean up
env.close()
```

Or run the built-in test:
```bash
python -m environment.car_following
```

## Next Steps

### Phase 2: Controller Implementation
- [ ] Implement MPC controller with CasADi
- [ ] Implement baseline controllers (Fixed MPC, ACC)

### Phase 3: DRL Integration
- [ ] Create Gymnasium wrapper for stable-baselines3
- [ ] Implement training script

### Phase 4 & 5: Experimentation
- [ ] Create evaluation script
- [ ] Generate results and visualizations

## Key Parameters

### Environment Settings
- `dt`: Simulation timestep (default: 0.05s)
- `MAX_ACCELERATION`: 2.0 m/s²
- `MAX_BRAKING`: 3.0 m/s²
- `TIME_HEADWAY`: 1.8 seconds (safety constraint)
- `MAX_VELOCITY`: 30.0 m/s

### CARLA Connection
- Default host: `localhost`
- Default port: `2000`
- Timeout: `10.0` seconds

## Troubleshooting

### CARLA Connection Issues
- Ensure CARLA server is running before starting scripts
- Check firewall settings if connecting to remote CARLA
- Increase timeout if needed: `client.set_timeout(20.0)`

### Memory Issues
- CARLA can be memory-intensive
- Use `-RenderOffScreen` flag for training
- Restart CARLA server periodically during long runs
- Ensure proper actor cleanup with `env.close()`

### Spawn Failures
- Check that the map has valid spawn points
- Try different spawn point indices
- Ensure sufficient space for both vehicles

## References

- CARLA Documentation: https://carla.readthedocs.io/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- CasADi: https://web.casadi.org/

## License

This project is for research and educational purposes.
