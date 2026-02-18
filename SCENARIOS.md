# Test Scenarios for Eco-Driving Controller Evaluation

This document describes the test scenarios used to evaluate and compare different controllers (DRL-MPC Hybrid, Fixed-Weight MPC, and ACC) for eco-driving in autonomous vehicles.

## Quick Reference

| Scenario | Duration | Purpose |
|----------|----------|---------|
| multi_phase | 45s | Adaptive weight adjustment across driving phases |
| predictable_pattern | 40s | Pattern anticipation and prediction |
| traffic_waves | 50s | Oscillation damping capability |
| varying_distance | 65s | Distance-aware optimization |
| mixed_speed | 60s | Speed-dependent weight tuning |
| emergency_braking | 45s | Safety response and recovery |
| eco_coasting | 60s | Energy-efficient deceleration |
| highway_cruise | 30s | Baseline steady-state driving |
| aggressive_driver | 35s | Robustness to erratic behavior |

---

## DRL-Advantage Scenarios

These scenarios are specifically designed to demonstrate where a DRL-based adaptive controller outperforms fixed-weight approaches.

### 1. Multi-Phase Driving (`multi_phase`)

**Duration:** 45 seconds

**Description:** Combines multiple driving conditions in sequence, requiring different optimization priorities in each phase.

**Phases:**
1. Highway cruise at 22 m/s (8s) - Energy efficiency priority
2. Approaching traffic slowdown (6s) - Anticipatory braking
3. Stop-and-go traffic with 3 cycles (18s) - Comfort priority
4. Traffic clearing acceleration (5s) - Balanced approach
5. Recovery cruise (8s) - Return to efficiency

**Why DRL Excels:**
- Fixed-weight MPC uses the same weights throughout, which is suboptimal
- DRL learns to shift priorities: comfort during stop-and-go, efficiency during cruise
- Demonstrates adaptive behavior across different driving conditions

---

### 2. Predictable Pattern (`predictable_pattern`)

**Duration:** 40 seconds

**Description:** Lead vehicle follows a regular sinusoidal speed pattern (15 m/s mean, ±5 m/s amplitude, 10s period).

**Pattern:** `velocity = 15 + 5 * sin(2π * t / 10)`

**Why DRL Excels:**
- Fixed MPC only reacts to current state
- DRL can learn the periodic pattern and act anticipatively
- Enables preemptive coasting before slowdowns, saving energy
- Tests learning and prediction capabilities

---

### 3. Traffic Waves (`traffic_waves`)

**Duration:** 50 seconds

**Description:** Simulates realistic traffic wave propagation with multiple frequency components, mimicking highway congestion patterns.

**Components:**
- Primary wave: 8s period, 3 m/s amplitude
- Secondary wave: 12s period, 2 m/s amplitude
- Tertiary wave: 5s period, 1.5 m/s amplitude
- Smoothed noise for realism

**Why DRL Excels:**
- Naive car-following amplifies oscillations (accordion effect)
- DRL can learn optimal damping strategies
- Reduces energy waste from unnecessary acceleration/braking
- Important for real-world traffic flow improvement

---

### 4. Varying Distance (`varying_distance`)

**Duration:** 65 seconds

**Description:** Lead vehicle alternates between pulling away quickly and slowing down significantly, creating varying gap distances.

**Behavior Pattern:**
- Lead accelerates away → large gap develops
- Lead maintains high speed → ego falls behind
- Lead brakes hard → gap closes rapidly
- Pattern repeats with variations

**Why DRL Excels:**
- When far from lead: should prioritize energy (don't chase aggressively)
- When close to lead: should prioritize safety
- DRL learns distance-dependent weight adjustment
- Fixed weights can't adapt to changing gap conditions

---

### 5. Mixed Speed Regime (`mixed_speed`)

**Duration:** 60 seconds

**Description:** Alternates between high-speed highway driving (25-28 m/s) and low-speed urban conditions (5-10 m/s).

**Segments:**
- High-speed highway (10s)
- Transition to low speed (5s)
- Low-speed urban (12s)
- Return to highway (5s)
- Repeat pattern

**Why DRL Excels:**
- At high speeds: safety weight should increase (less reaction time)
- At low speeds: comfort matters more, energy efficiency is key
- DRL learns speed-appropriate optimization weights
- Demonstrates context-aware control adaptation

---

## Additional Scenarios

These scenarios test specific capabilities and edge cases.

### 6. Emergency Braking (`emergency_braking`)

**Duration:** 45 seconds

**Description:** Multiple sudden braking events of varying severity, followed by recovery phases.

**Events:**
1. Hard brake to near-stop (20→2 m/s)
2. Moderate brake (18→8 m/s)
3. Severe brake to full stop (20→0.5 m/s)

**Tests:**
- Safety response time and effectiveness
- Collision avoidance capability
- Optimal post-emergency recovery strategy
- Smooth transition back to normal following

---

### 7. Eco-Coasting (`eco_coasting`)

**Duration:** 60 seconds

**Description:** Scenarios with known deceleration points, simulating approaches to traffic signals or known slowdown zones.

**Opportunities:**
- Long gradual coast (8s deceleration window)
- Medium coast (6s window)
- Short coast (3s window - less time to react)

**Tests:**
- Ability to initiate coasting early for energy savings
- Predictive deceleration vs reactive braking
- Energy recovery potential through regenerative braking

---

### 8. Highway Cruise (`highway_cruise`)

**Duration:** 30 seconds

**Description:** Steady high-speed driving with minor variations. Used as a baseline scenario.

**Profile:** Constant 22 m/s with small sinusoidal variation (±1 m/s)

**Purpose:**
- Baseline energy consumption measurement
- Steady-state controller performance
- Reference for comparing other scenarios

---

### 9. Aggressive Driver (`aggressive_driver`)

**Duration:** 35 seconds

**Description:** Erratic lead vehicle with sudden accelerations, hard braking, and unpredictable speed changes.

**Behaviors:**
- Sudden acceleration bursts (18→28 m/s in 1.5s)
- Hard braking events (30→5 m/s in 1.5s)
- Rapid speed oscillations
- Unpredictable timing

**Tests:**
- Controller robustness to disturbances
- Comfort maintenance under stress
- Safety margin preservation
- Avoiding overreaction to erratic inputs

---

## Usage

### Command Line

```bash
# List all scenarios
python test_controllers.py --list

# Run specific scenario
python test_controllers.py --scenario multi_phase

# Run all DRL-advantage scenarios
python test_controllers.py --drl

# Run all scenarios
python test_controllers.py --all

# Interactive menu
python test_controllers.py
```

### In Code

```python
from utils.scenarios import get_scenario, get_drl_advantage_scenarios

# Get single scenario
scenario = get_scenario('multi_phase')
print(f"Duration: {scenario.duration_s}s")
print(f"Steps: {scenario.num_steps}")
trajectory = scenario.trajectory

# Get all DRL scenarios
scenarios = get_drl_advantage_scenarios()
for name, config in scenarios.items():
    print(f"{name}: {config.description}")
```

---

## Metrics Evaluated

Each scenario evaluates controllers on:

| Metric | Description | Unit |
|--------|-------------|------|
| Total Energy | Integral of throttle × velocity | - |
| Energy per km | Normalized energy consumption | per km |
| RMS Jerk | Ride comfort (acceleration smoothness) | m/s³ |
| Min Time Headway | Closest following distance | seconds |
| THW Violation Rate | Safety constraint violations | % |
| Average Velocity | Travel efficiency | m/s |

---

## Expected Results

For DRL-advantage scenarios, a well-trained DRL agent should:

1. **Multi-phase**: Lower overall energy with similar safety
2. **Predictable pattern**: Smoother velocity profile, less jerk
3. **Traffic waves**: Damped oscillations, reduced energy
4. **Varying distance**: Efficient gap management
5. **Mixed speed**: Appropriate behavior in each regime

The degree of improvement depends on training quality and hyperparameter tuning.
