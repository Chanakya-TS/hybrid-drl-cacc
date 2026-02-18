# DRL Training Guide for Hybrid DRL-MPC Eco-Driving

This guide explains how to train, monitor, and evaluate the Deep Reinforcement Learning agent for the hybrid DRL-MPC eco-driving controller.

## Overview

The DRL agent learns to dynamically adjust MPC cost function weights based on driving conditions:

| Weight | Controls | When to Prioritize |
|--------|----------|-------------------|
| `w_velocity` | Speed tracking | Highway cruise, catching up |
| `w_safety` | Following distance | Close to lead vehicle, high speed |
| `w_comfort` | Smooth acceleration | Stop-and-go, passenger comfort |

## Prerequisites

1. **CARLA Server Running**
   ```bash
   start_carla.bat
   ```

2. **Dependencies Installed**
   ```bash
   pip install stable-baselines3[extra] tensorboard
   ```

3. **Verify Environment**
   ```bash
   python -c "from environment.gym_wrapper import HybridMPCEnv; print('OK')"
   ```

## Training Commands

### Basic Training
```bash
python train.py
```
Default: 50,000 timesteps with varied training trajectory.

### Custom Training
```bash
# Train for more timesteps
python train.py --timesteps 100000

# Train on specific scenario
python train.py --scenario multi_phase --timesteps 50000

# Resume from checkpoint
python train.py --resume models/sac_varied_20241220_final.zip --timesteps 25000
```

### All Options
```
--timesteps, -t      Total training timesteps (default: 50000)
--scenario, -s       Scenario to train on (default: varied trajectory)
--resume, -r         Path to checkpoint to resume from
--checkpoint-freq    Save frequency (default: 10000)
--save-path          Model save directory (default: models)
--log-dir            Log directory (default: logs)
```

## Monitoring Training

### TensorBoard (Recommended)
```bash
# In a separate terminal
tensorboard --logdir logs/tensorboard
```
Open browser: http://localhost:6006

**Key Metrics to Watch:**
- `rollout/ep_rew_mean` - Average episode reward (should increase)
- `rollout/ep_len_mean` - Episode length (should stay high, no early terminations)
- `train/loss` - Training loss (should decrease and stabilize)
- `train/entropy_loss` - Exploration entropy

### Console Output
Training prints progress every 1000 steps:
```
Step 5000: Episodes=8, Mean Reward (last 10)=45.23, Mean Length=625
```

## Training Duration Guidelines

| Timesteps | Duration* | Use Case |
|-----------|----------|----------|
| 10,000 | ~15 min | Quick test |
| 50,000 | ~1 hour | Initial training |
| 100,000 | ~2 hours | Good performance |
| 200,000 | ~4 hours | Best results |

*Approximate, depends on hardware and CARLA performance.

## SAC Hyperparameters

Default hyperparameters in `train.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 3e-4 | Adam optimizer learning rate |
| `buffer_size` | 100,000 | Replay buffer size |
| `batch_size` | 256 | Mini-batch size |
| `tau` | 0.005 | Soft update coefficient |
| `gamma` | 0.99 | Discount factor |
| `ent_coef` | auto | Entropy coefficient (auto-tuned) |

### Tuning Tips
- **Lower `learning_rate`** (1e-4) if training is unstable
- **Increase `buffer_size`** if you have RAM (improves sample diversity)
- **Lower `gamma`** (0.95) for more reactive behavior

## Reward Function

The agent receives rewards based on:

```python
reward = (
    - energy_cost           # Penalize throttle usage
    - safety_penalty        # Penalize THW < 1.5s
    - comfort_penalty       # Penalize high jerk
    + velocity_bonus        # Reward target velocity tracking
    + safety_bonus          # Small reward for safe distance
)

# Collision terminates episode with -100 penalty
```

**Reward Weights (in gym_wrapper.py):**
- `REWARD_ENERGY_WEIGHT = 1.0`
- `REWARD_SAFETY_WEIGHT = 2.0`
- `REWARD_COMFORT_WEIGHT = 0.5`

## Output Files

After training:
```
models/
├── sac_varied_20241220_123456_10000_steps.zip   # Checkpoint
├── sac_varied_20241220_123456_20000_steps.zip   # Checkpoint
└── sac_varied_20241220_123456_final.zip         # Final model

logs/
├── tensorboard/                                  # TensorBoard logs
└── monitor.csv                                   # Episode statistics
```

## Evaluation

After training, evaluate against baselines:

```bash
# Evaluate on all DRL-advantage scenarios
python evaluate.py --model models/sac_varied_20241220_final.zip --drl-scenarios

# Evaluate on specific scenario
python evaluate.py --model models/sac_varied_20241220_final.zip --scenario multi_phase

# Evaluate on all scenarios
python evaluate.py --model models/sac_varied_20241220_final.zip --all
```

## Troubleshooting

### Training Crashes / CARLA Disconnects
- Restart CARLA and resume from last checkpoint:
  ```bash
  python train.py --resume models/sac_varied_*_steps.zip
  ```

### Poor Performance After Training
- Train longer (more timesteps)
- Try training on specific challenging scenario first
- Adjust reward weights in `gym_wrapper.py`

### Very Negative Rewards
- Check if collisions are happening (episode lengths very short)
- Reduce initial learning rate
- Ensure CARLA is running smoothly (check FPS)

### TensorBoard Not Updating
- Wait for at least one episode to complete
- Refresh browser or restart TensorBoard
- Check `logs/tensorboard/` folder exists

## Training Strategies

### Strategy 1: Varied Training (Default)
Train on mixed trajectory with multiple driving patterns:
```bash
python train.py --timesteps 100000
```
Best for general-purpose agent.

### Strategy 2: Curriculum Learning
Start with simple scenario, progress to complex:
```bash
python train.py --scenario highway_cruise --timesteps 20000
python train.py --resume models/sac_highway_cruise_final.zip --scenario multi_phase --timesteps 50000
```

### Strategy 3: Scenario-Specific
Train separate models for different conditions:
```bash
python train.py --scenario traffic_waves --timesteps 50000
python train.py --scenario emergency_braking --timesteps 50000
```

## Expected Results

After successful training, the DRL-MPC agent should:
- Achieve **5-15% lower energy** than Fixed-MPC on multi-phase scenarios
- Maintain **similar or better safety** (time headway)
- Adapt weights based on situation:
  - High `w_safety` when close to lead vehicle
  - High `w_comfort` during stop-and-go
  - High `w_velocity` during steady cruise

## Quick Start Example

```bash
# Terminal 1: Start CARLA
start_carla.bat

# Terminal 2: Start TensorBoard
tensorboard --logdir logs/tensorboard

# Terminal 3: Train
python train.py --timesteps 50000

# After training completes:
python evaluate.py --model models/sac_varied_*_final.zip --drl-scenarios
python generate_results.py
```
