# Research Log — Hybrid DRL-MPC Eco-Driving Controller

## 2025-02-25: 3-Lead Vehicle MPC Expansion

### Motivation
The CARLA environment already spawns 3 lead vehicles in a platoon, but only the closest lead's state was used by the MPC and observation space. Expanding to all 3 leads enables the MPC to anticipate cascading braking events and drive more efficiently.

### Changes Made

**Observation Space Expansion (3-dim → 10-dim)**
- Old: `[ego_velocity, relative_velocity, distance_gap]`
- New: `[ego_velocity, rel_vel_1, dist_1, accel_1, rel_vel_2, dist_2, accel_2, rel_vel_3, dist_3, accel_3]`
- Lead accelerations computed via finite difference: `a_i = (v_i(t) - v_i(t-1)) / dt`

**MPC Reformulation (2-state → 4-state)**
- Old states: `[v_ego, d]` (velocity + distance to lead 1)
- New states: `[v_ego, d1, d2, d3]` (velocity + distances to all 3 leads)
- Parameters: `3N + 12` (was `N + 10`), where N = prediction horizon
- Dynamics for each lead: `d_i[k+1] = d_i[k] + (v_lead_i[k] - v_ego[k]) * dt`
- Lead velocity predictions: constant by default, but projected forward if lead is decelerating (`a < -0.5 m/s²`)

**Anticipatory Cost Terms (new in cost function)**
- Primary safety on d1 (unchanged): asymmetric penalty `w_s * max(0, -(d1 - d_desired))²`
- Lead 2 decel penalty: `w_s * 0.05 * max(0, v_lead1 - v_lead2)²` — penalizes when lead 2 is slower than lead 1
- Lead 3 decel penalty: `w_s * 0.02 * max(0, v_lead2 - v_lead3)²`
- Anticipatory speed penalty: `w_s * 0.03 * max(0, v_ego - v_lead2)²` — discourages maintaining high speed when downstream vehicles are slower

**DRL Observation Space (6-dim → 13-dim)**
- 10 normalized env values + 3 MPC weights
- Added normalization constant: `MAX_ACCELERATION = 5.0 m/s²`

**Files Modified:** `car_following.py`, `mpc_controller.py`, `gym_wrapper.py`, `evaluate.py`, `test_controllers.py`, `test_phase1.py`

---

## 2025-02-25: Reward Function Redesign (v2)

### Problem
After training with 5k timesteps, the DRL agent learned a degenerate policy: maintain low constant speed, trailing far behind the lead vehicle. This minimized energy cost and jerk while always receiving the safe-distance bonus (+0.1/step).

### Root Cause Analysis
The original reward had misaligned incentives:
| Component | Signal | Problem |
|-----------|--------|---------|
| Energy | `-1.0 * throttle * v * dt` | Going slow = near-zero penalty |
| Safety | `+0.1` if THW > 1.5s | Free reward for large gap |
| Comfort | `-0.5 * jerk²` | Constant speed = zero jerk |
| Velocity | `+0.1 * exp(-|v - v_target|/5)` | Too weak, decays slowly |

The agent maximized reward by going slow: energy ≈ 0, safety = +0.1/step, comfort = 0, velocity ≈ +0.06/step. No penalty for trailing far behind.

### Reward v2 Design
| Component | Formula | Weight | Purpose |
|-----------|---------|--------|---------|
| Following distance | `exp(-Δd² / 200)` where `Δd = \|d - d_desired\|` | 1.0 | PRIMARY: Gaussian centered on `d_desired = 1.8v + 5` |
| Velocity matching | `exp(-Δv² / 18)` where `Δv = \|v_ego - v_lead\|` | 0.5 | Match lead speed, not fixed target |
| Safety | `-(1.0 - THW)²` if THW < 1.0s | 2.0 | Hard penalty only in danger zone |
| Energy | `-throttle * v * dt` | 0.3 | Demoted to secondary signal |
| Comfort | `-jerk² * dt` | 0.005 | Light penalty, doesn't dominate |

Key design decisions:
- Following distance is **symmetric** (penalizes too-far AND too-close equally)
- Velocity matching tracks **lead vehicle**, not a fixed 20 m/s target
- Safety penalty threshold lowered from 1.5s to 1.0s THW (less conservative trigger)
- Energy weight reduced from 1.0 to 0.3 so car-following dominates

### Training Run
- Model: `sac_varied_20260225_123943_final.zip` (5k steps, old reward — degenerate)

---

## 2025-02-25: Evaluation Metrics Update

### THW Violation Rate — Upper Bound Added
The original violation rate only counted steps where THW < 1.5s (too close). The degenerate agent showed 0.0% violations despite avg THW of 102s (trailing far behind), which was misleading.

**Updated `calculate_safety_metrics()` in `utils/metrics.py`:**
- `too_close_rate`: fraction of steps with THW < 1.5s
- `too_far_rate`: fraction of steps with THW > 5.0s
- `violation_rate`: combined (too close OR too far)

Comparison tables in `evaluate.py` and `test_controllers.py` now show the breakdown.

### First Evaluation Results (Reward v1 — degenerate agent)
```
Metric                              DRL-MPC    Fixed-MPC          ACC
Total Energy                          10.16       152.94       174.80
Energy per km                        137.80       432.81       452.10
RMS Jerk (m/s³)                      0.6360       1.7697       5.0235
Min Time Headway (s)                   3.77         2.19         2.21
Avg Time Headway (s)                 101.92         3.75         3.84
THW Violation Rate                     0.0%         0.0%         0.0%
Avg Velocity (m/s)                     1.64         7.85         8.59
```
DRL agent avg velocity 1.64 m/s confirms degenerate slow-crawl policy. Low energy and jerk are meaningless when not actually following the lead.

---

## 2025-02-25: Multi-Scenario Training

### Problem
Training only used a single "varied" trajectory (1130 steps). The agent never saw the other 9 defined scenarios, limiting generalization.

### Solution — `ScenarioRandomizer` Wrapper
Added a Gymnasium wrapper in `train.py` that randomly selects a scenario from a pool of 10 on each `reset()`:
- 9 defined scenarios: `multi_phase`, `predictable_pattern`, `traffic_waves`, `varying_distance`, `mixed_speed`, `emergency_braking`, `eco_coasting`, `highway_cruise`, `aggressive_driver`
- 1 custom `varied_training` trajectory
- Logs scenario name to console: `[Episode 3] Scenario: traffic_waves (1000 steps, 50s)`
- Draws scenario name as green debug text in CARLA world (visible in spectator view)
- Ensures each episode runs the full trajectory (no early truncation)
- Fixed `max_episode_steps` to match trajectory length (was capped at 1000, cutting off the last 130 steps of varied trajectory)

### Current Training Approach
- **Sampling**: Uniform random across all 10 scenarios
- **Timesteps**: 50k (current run in progress)
- **Algorithm**: SAC with default hyperparameters

### Planned Improvement — Curriculum Learning (not yet implemented)
Uniform random is suboptimal: the agent gets hard scenarios (emergency braking) before learning basic following. Proposed 3-phase curriculum:
1. **Warmup (~20% of steps)**: Only `highway_cruise` + `predictable_pattern` — learn basic car-following
2. **Main (~50% of steps)**: Weighted random favoring 5 DRL-advantage scenarios (`multi_phase`, `traffic_waves`, `varying_distance`, `mixed_speed`, `emergency_braking`)
3. **Generalization (~30% of steps)**: All 10 scenarios equally

---

## 2025-02-25: 50k Evaluation Results (Reward v2, Uniform Random Training)

### Aggregate Results
```
Scenario                     DRL-MPC  Fixed-MPC        ACC     Winner
----------------------------------------------------------------------
multi_phase                   179.32     169.31     175.02  Fixed-MPC
predictable_pattern           178.94     164.29     171.45  Fixed-MPC
traffic_waves                 304.88     281.88     298.27  Fixed-MPC
varying_distance              396.19     366.90     389.39  Fixed-MPC
mixed_speed                   296.30     274.87     289.62  Fixed-MPC
emergency_braking             164.02     153.65     161.09  Fixed-MPC
eco_coasting                  238.41     226.56     231.63  Fixed-MPC
highway_cruise                215.53     198.68     214.07  Fixed-MPC
aggressive_driver             170.12     162.23     171.10  Fixed-MPC
----------------------------------------------------------------------
TOTAL WINS                         0          9          0

DRL-MPC vs Fixed-MPC: -7.1% energy
DRL-MPC vs ACC: -2.0% energy
```

### Analysis
- DRL-MPC lost all 9 scenarios to Fixed-MPC, using ~7% more energy
- The agent is following the lead vehicle now (not degenerate), but weight adjustments are hurting rather than helping
- **RMS Jerk**: DRL-MPC produces 5–7 m/s³ vs Fixed-MPC ~0.1 m/s³ — extremely jerky

### Root Cause — Weight Oscillation
The DRL changes MPC weights **every step** (every 0.05s / 20 Hz). Rapid weight changes between consecutive steps (e.g., `w_s` jumping 0.8 → 0.2) cause the MPC to produce wildly different acceleration commands, resulting in:
- High jerk (5–7x worse than fixed MPC)
- Increased energy consumption from oscillating throttle/brake
- The weight perturbations add noise to an already well-tuned fixed-weight MPC

### Additional Factors
- **Insufficient training**: 50k steps ≈ 50 episodes ≈ 5 per scenario — SAC barely explores weight space
- **No curriculum**: Agent faces hard scenarios (emergency braking) before learning basics

---

## 2025-02-25: Implemented Improvements for Next Training Run

### 1. Weight Smoothing Penalty (Reward v3) — IMPLEMENTED
Added penalty for large weight changes between consecutive DRL actions in `gym_wrapper.py`:
- Tracks `self.previous_weights` across steps (reset to `[0.33, 0.34, 0.33]` each episode)
- Penalty: `reward -= 0.5 * ||w(t) - w(t-1)||²` (L2 squared norm of weight change)
- Weight 0.5 chosen to be significant but not dominant over the primary following-distance reward (~1.0)
- Updated `_compute_reward()` signature to accept `current_weights` parameter

**Expected effect**: Agent should learn to make smooth, gradual weight transitions rather than rapid oscillations. This directly addresses the 5–7x jerk issue vs fixed MPC.

### 2. Curriculum Learning — IMPLEMENTED
Replaced `ScenarioRandomizer` with `CurriculumScheduler` in `train.py`:

| Phase | Steps | Scenarios | Purpose |
|-------|-------|-----------|---------|
| 1 — Warmup | 0–20% (0–40k) | `highway_cruise`, `predictable_pattern` | Learn stable car-following |
| 2 — Main | 20–70% (40k–140k) | 70% DRL-advantage, 30% all | Learn weight adaptation |
| 3 — Generalization | 70–100% (140k–200k) | All 10 scenarios equally | Refine across patterns |

Key implementation details:
- `CurriculumScheduler(gym.Wrapper)` tracks `cumulative_steps` across episodes
- Phase boundaries computed from `total_timesteps` at construction
- Console logging includes phase info: `[Episode 5] Phase 1 (Warmup) | Scenario: highway_cruise | Cumulative: 3200/200000`
- CARLA debug text shows phase and scenario name
- DRL-advantage scenarios: `multi_phase`, `predictable_pattern`, `traffic_waves`, `varying_distance`, `mixed_speed`
- Easy scenarios: `highway_cruise`, `predictable_pattern`

### 3. Training Duration — IMPLEMENTED
- Default timesteps increased from 50k to 200k in both `train()` and CLI `--timesteps` default
- 200k steps ≈ 200+ episodes ≈ 20+ per scenario
- Run name prefix changed from `sac_varied_` to `sac_curriculum_`

### Full Reward v3 Summary
| Component | Formula | Weight | Purpose |
|-----------|---------|--------|---------|
| Following distance | `exp(-Δd² / 200)` | 1.0 | PRIMARY: match desired gap |
| Velocity matching | `exp(-Δv² / 18)` | 0.5 | Match lead speed |
| Safety | `-(1.0 - THW)²` if THW < 1.0s | 2.0 | Hard penalty in danger zone |
| Energy | `-throttle * v * dt` | 0.3 | Secondary efficiency signal |
| Comfort | `-jerk² * dt` | 0.005 | Light jerk penalty |
| **Weight smoothing** | **`-‖w(t) - w(t-1)‖²`** | **0.5** | **Applied once per DRL decision** |

---

## 2025-02-26: 200k Training Results (Curriculum + Weight Smoothing)

### Results
- DRL-MPC won 1/9 scenarios (emergency_braking), Fixed-MPC won 8/9
- DRL-MPC vs Fixed-MPC: **-4.2% energy** (improved from -7.1% at 50k)
- DRL-MPC roughly on par with ACC (+0.4%)

### Analysis
Curriculum learning + weight smoothing penalty improved the gap (7.1% → 4.2%), but the DRL still loses to Fixed-MPC. **Root cause identified: DRL still decides every 0.05s (20Hz)**. Even with the smoothing penalty, changing MPC weights 20× within the MPC's own 1s prediction horizon creates a fundamentally noisy optimization landscape. Fixed-MPC wins precisely because its weights *never* change.

---

## 2025-02-26: Temporal Abstraction (Action Repeat) — IMPLEMENTED

### Core Insight
The MPC controller has a prediction horizon of ~1s (20 steps × 0.05s). Changing its cost function weights faster than the horizon itself is counterproductive — the optimizer can't converge to a consistent strategy. The DRL should operate on a slower timescale (~1s) while the MPC runs at full 20Hz.

### Implementation: `action_repeat = 20` (1.0s per DRL decision)

**`gym_wrapper.py` changes:**
- Added `action_repeat` parameter (default 20, configurable)
- `step()` now runs `action_repeat` CARLA sub-steps per DRL decision:
  1. DRL sets new MPC weights
  2. Weight smoothing penalty computed ONCE (not per sub-step)
  3. Inner loop: MPC computes acceleration → CARLA steps → reward accumulates
  4. MPC weights held constant for all 20 sub-steps
  5. Returns accumulated reward + observation from final sub-step
- Sub-step data (velocities, accelerations, throttles, etc.) collected and returned in `info['sub_step_data']` for full-resolution evaluation logging
- `_compute_reward()` simplified: removed `current_weights` parameter (weight smoothing handled in `step()`)

**`train.py` changes:**
- `CurriculumScheduler.step()` now increments step counters by `info['sim_steps']` instead of `+1`, ensuring episode boundaries align with sim-level trajectory lengths

**`evaluate.py` changes:**
- `run_drl_episode()` unpacks `info['sub_step_data']` to log at full 20Hz resolution (same as baselines), ensuring fair metrics comparison
- Distinguishes agent steps vs sim steps in logging

### Effect on Episode Structure
For a 1000-sim-step scenario (50s):
- **Before**: 1000 DRL decisions, 1000 weight changes, 1000 data points
- **After**: 50 DRL decisions, 50 weight changes, 1000 data points (from sub-steps)
- MPC has stable weights for 20 consecutive solves → consistent acceleration trajectory
- Jerk from weight oscillation should drop to near zero

### Why This Should Work
| Aspect | Before (20Hz DRL) | After (1Hz DRL) |
|--------|-------------------|-----------------|
| DRL decisions per episode | ~1000 | ~50 |
| Weight changes per second | 20 | 1 |
| MPC consistency window | 0.05s (1 solve) | 1.0s (20 solves) |
| Jerk source | Weight oscillation | Only driving dynamics |
| Learning problem | 1000 sequential decisions | 50 strategic decisions |
| SAC exploration | Noisy, oscillatory | Meaningful weight space |

### Next Steps
- Train 200k timesteps with action_repeat=20 + curriculum + weight smoothing
- Compare jerk, energy, and safety metrics against Fixed-MPC and ACC
- If jerk is now comparable to Fixed-MPC, the remaining question is whether DRL can find *better* weights than fixed [0.33, 0.34, 0.33]

---

## 2025-02-26: 200k Training Results (Action Repeat + Curriculum + Weight Smoothing)

### Training Run
- Model: `200k-high_jerk_w_20260226_130001`
- Training crashed at episode 2011 (~93,723 timesteps), last checkpoint at 90,000 steps
- `ep_len_mean = 45.7` — short episodes suggest frequent early terminations (collisions)
- `ep_rew_mean = 1.21e+03`, `actor_loss = -2.48e+03`, `ent_coef = 0.247`

### Evaluation Results
```
Scenario                     DRL-MPC  Fixed-MPC        ACC     Winner
----------------------------------------------------------------------
multi_phase                   173.76     169.30     174.78  Fixed-MPC
predictable_pattern           171.24     164.30     171.56  Fixed-MPC
traffic_waves                 299.04     281.87     298.38  Fixed-MPC
varying_distance              389.68     366.91     389.33  Fixed-MPC
mixed_speed                   288.94     274.89     289.63  Fixed-MPC
emergency_braking             157.95     157.62     161.68  Fixed-MPC
eco_coasting                  230.67     226.56     230.99  Fixed-MPC
highway_cruise                213.97     198.68     213.99  Fixed-MPC
aggressive_driver             174.30     162.19     171.19  Fixed-MPC
----------------------------------------------------------------------
TOTAL WINS                         0          9          0
```

### Root Cause Analysis — Reward-Evaluation Mismatch

The fundamental problem: **the DRL optimizes for car-following distance, but the evaluation ranks by total energy consumption**. The reward function and evaluation metric are measuring different things.

**Per-sub-step reward magnitude comparison (at typical conditions: throttle=0.4, velocity=15 m/s):**
| Component | Per-step magnitude | Effective weight | Role in reward |
|-----------|-------------------|-----------------|----------------|
| Following distance | ~1.0 (Gaussian peak) | **1.0** | PRIMARY |
| Velocity matching | ~0.5 | 0.5 | Secondary |
| Safety penalty | varies (only < 1.0s THW) | 2.0 | Hard constraint |
| Energy cost | ~0.09 (`0.3 × 0.4 × 15 × 0.05`) | 0.3 | **~10x weaker** |
| Comfort (jerk) | varies | 0.005 | Negligible |

Energy is **~10x weaker** than following distance in the reward signal. The DRL learned excellent car-following but has almost no incentive to reduce `throttle × velocity`.

**Why Fixed-MPC wins by accident:**
Fixed-MPC uses weights `(w_v=0.25, w_s=0.35, w_c=0.40)`. The high comfort weight (`w_c=0.40`) strongly penalizes acceleration magnitude (`w_c * a²`), which indirectly reduces throttle usage and thus energy consumption. Fixed-MPC is energy-efficient as a *side effect* of its comfort-heavy tuning.

**Additional factor — MPC energy term is coupled to w_velocity:**
In the MPC cost function, the energy penalty is `w_v * 0.1 * max(0, a)²` (line 193 of `mpc_controller.py`). This means the DRL cannot independently control the energy penalty — increasing `w_velocity` for better efficiency also increases velocity tracking, which can increase throttle to match lead speed. The two objectives fight each other through the same weight.

### Fix — Reward v4 (Energy-Primary)

Restructured reward to make energy the PRIMARY signal, matching the evaluation metric:

| Component | Reward v3 (old) | Reward v4 (new) | Change |
|-----------|----------------|----------------|--------|
| Energy | `-0.3 × throttle × v × dt` (raw) | **`-2.0 × throttle × (v / MAX_V)`** (normalized) | **6.7× stronger**, normalized to [0,1] |
| Following distance | `1.0 × exp(...)` | **`0.5 × exp(...)`** | Halved, now secondary |
| Velocity matching | `0.5 × exp(...)` | **`0.3 × exp(...)`** | Reduced |
| Safety | `2.0 × penalty` | **`3.0 × penalty`** | Increased (prevent energy-greedy crashes) |
| Comfort (jerk) | `0.005 × jerk² × dt` | **`0.1 × jerk² × dt`** | Slightly increased |
| Weight smoothing | `0.5 × ‖Δw‖²` | **`0.15 × ‖Δw‖²`** | Reduced (allow more exploration) |

**Key design decision — energy normalization:**
Old: `throttle × velocity × dt` → varies wildly with speed (0.01 at low speed, 1.5 at high speed)
New: `throttle × (velocity / MAX_VELOCITY)` → bounded in [0, 1], stable gradient signal

**Per-step magnitude check (throttle=0.4, velocity=15 m/s, at desired distance):**
- Energy penalty: `-2.0 × 0.4 × 0.5 = -0.40`
- Following reward: `+0.5 × 1.0 = +0.50`
- Velocity matching: `+0.3 × 1.0 = +0.30`
- Net: `+0.40` → energy is meaningful but doesn't cause degenerate slow-crawl

**Coasting incentive (throttle=0.0 vs throttle=0.4):**
- With throttle: net ≈ +0.40/step
- Without throttle: net ≈ +0.80/step
- Coasting advantage: +0.40/step × 20 sub-steps = **+8.0 per DRL decision**

### Fix — Resume Logic (`train.py`)

Found two bugs in the training resume path:
1. `model.learn()` was missing `reset_num_timesteps=False` — on resume, SB3 restarted its internal counter at 0, training for another full 200k steps instead of the remaining ~110k
2. `CurriculumScheduler.cumulative_steps` was not synced on resume — curriculum restarted from Phase 1 instead of continuing at the correct phase

Both fixed: `reset_num_timesteps=False` passed to `model.learn()`, and `cumulative_steps` synced from `model.num_timesteps` on resume.

### Files Modified
- `environment/gym_wrapper.py`: Reward v4 (energy-primary), new class constants, normalized energy cost
- `train.py`: Resume logic fixes (`reset_num_timesteps`, curriculum sync)

### Next Steps
- Retrain from scratch with reward v4 (200k timesteps) — old checkpoint learned wrong objective
- Monitor `ep_len_mean` during training — should be longer if safety penalty prevents collisions
- If DRL still loses: consider adding a 4th MPC weight (`w_energy`) to decouple energy from velocity tracking

---

## 2025-02-27: Reward v4 Evaluation — Still Losing, Structural Fix Applied

### Evaluation Results (Reward v4, retrained)
```
Scenario                     DRL-MPC  Fixed-MPC        ACC     Winner
----------------------------------------------------------------------
multi_phase                   176.72     169.37     174.68  Fixed-MPC
predictable_pattern           171.58     164.30     170.82  Fixed-MPC
traffic_waves                 299.19     281.88     298.29  Fixed-MPC
varying_distance              390.01     366.81     389.16  Fixed-MPC
mixed_speed                   290.89     274.90     289.81  Fixed-MPC
emergency_braking             158.50     157.62     160.82  Fixed-MPC
eco_coasting                  230.82     226.56     231.53  Fixed-MPC
highway_cruise                216.32     198.68     213.94  Fixed-MPC
aggressive_driver             172.78     162.23     171.18  Fixed-MPC
----------------------------------------------------------------------
TOTAL WINS                         0          9          0

DRL-MPC vs Fixed-MPC: -5.0% energy
DRL-MPC vs ACC: -0.3% energy
```

### Analysis — Why Reward Retuning Alone Wasn't Enough

DRL-MPC energy values barely changed from the previous run (within ~1-3 units per scenario). Fixed-MPC values are nearly identical (deterministic baseline). The slight differences confirm a different model was evaluated, but the learned policy is equally bad for energy.

**Root cause is structural, not reward-related:**

1. **Fixed-MPC weights are already near-optimal.** The hand-tuned weights `(w_v=0.25, w_s=0.35, w_c=0.40)` produce energy-efficient behavior because high `w_comfort` (0.40) strongly penalizes acceleration magnitude (`w_c * a²`), which reduces throttle as a side effect. The DRL cannot easily discover that `w_comfort ≈ 0.40` is the sweet spot.

2. **The DRL searches from a bad starting point.** With absolute action space `[0,1]³` and SAC's Gaussian exploration, the agent explores randomly across the entire weight simplex. Most sampled weights are worse than Fixed-MPC. The agent has to discover `(0.25, 0.35, 0.40)` from scratch AND learn when to deviate.

3. **Weight transitions cost energy.** Every time the DRL changes weights, the MPC gets a different optimization landscape, causing transient sub-optimal control. Even if the new weights are slightly better in steady-state, the transition cost can negate the benefit.

4. **Previous TensorBoard showed plateau at 5k steps.** The old reward curve flatlined at ~1,250 reward after 5k steps. The agent found a "good enough" following policy early and had no gradient to improve further. The reward v4 changes likely shifted the plateau value but not the convergence behavior.

### Fix — Residual Policy Learning

Instead of the DRL choosing absolute weights, it now outputs **small deltas** on top of Fixed-MPC's baseline weights. This is a well-established technique in robotics RL for improving upon strong base controllers.

**Action space change:**
| Property | Before (absolute) | After (residual) |
|---|---|---|
| Action space | `[0, 1]³` | `[-0.15, +0.15]³` |
| Action meaning | Raw MPC weights | **Deltas** from `(0.25, 0.35, 0.40)` |
| Zero action result | `(0.33, 0.33, 0.33)` (bad) | `(0.25, 0.35, 0.40)` (**= Fixed-MPC**) |
| SAC initial policy | Random weights | **Near-zero deltas → matches Fixed-MPC** |
| Search space | Entire weight simplex | Small neighborhood around optimum |
| Worst-case floor | Much worse than Fixed-MPC | Bounded: within ±0.15 of Fixed-MPC |

**Why this should work:**
- SAC's initial Gaussian policy outputs near-zero actions → DRL matches Fixed-MPC on day 1 of training
- The agent only needs to learn *when deviations help*, not the entire weight landscape
- Smaller action space → faster convergence, less destructive exploration
- Guaranteed performance floor: even random deltas of ±0.15 can't stray far from good weights

**Concrete example — what the DRL could learn:**
- Eco-coasting scenario: `Δw_c = +0.15` → total `w_c = 0.55` → stronger coast preference → less throttle
- Emergency braking: `Δw_s = +0.15` → total `w_s = 0.50` → earlier, smoother braking → less recovery throttle
- Highway cruise: `Δw_c = +0.10, Δw_v = -0.05` → minimize control effort at steady state
- All cases: zero delta is always safe — DRL defaults to Fixed-MPC when uncertain

**Weight computation in `step()`:**
```python
raw_weights = BASE_WEIGHTS + action          # [0.25, 0.35, 0.40] + deltas
raw_weights = clip(raw_weights, 0.05, 0.8)   # Prevent degenerate weights
target_weights = raw_weights / sum(raw_weights)  # Normalize to sum to 1
```

### Other Changes
- MPC controller now initializes at `(0.25, 0.35, 0.40)` instead of `(0.33, 0.34, 0.33)`
- `previous_weights` in reset starts at `BASE_WEIGHTS` instead of `(0.33, 0.34, 0.33)`

### Files Modified
- `environment/gym_wrapper.py`: Residual action space, BASE_WEIGHTS constant, delta-based step logic

### Next Steps
- Retrain from scratch with residual policy (200k timesteps)
- Expected: DRL-MPC should match Fixed-MPC early in training, then gradually improve
- TensorBoard: reward curve should start higher (baseline performance from day 1) and have room to grow
- If DRL learns zero deltas everywhere → proves Fixed-MPC is already optimal (valid result)
- If DRL learns scenario-dependent deltas → demonstrates adaptive advantage (desired result)
