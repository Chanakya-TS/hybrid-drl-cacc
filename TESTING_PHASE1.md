# Phase 1 Testing Guide

Quick guide to test the CARLA-based car-following environment.

## Prerequisites

### 1. Install CARLA
- Download CARLA 0.9.13+ from: https://github.com/carla-simulator/carla/releases
- Extract to `C:\CARLA_0.9.15\` (or your preferred location)
- Requires: GPU with 4GB+ VRAM, 8GB+ RAM

### 2. Install Python Dependencies
```bash
cd D:\Root\College\EcoCar\Research
pip install -r requirements.txt
```

## Running the Test

### Step 1: Start CARLA Server
Open a terminal and run:
```bash
cd C:\CARLA_0.9.15\WindowsNoEditor
CarlaUE4.exe -quality-level=Low -fps=20
```
**Keep this window open!** Wait for CARLA to fully load (~30-60 seconds).

### Step 2: Run Test Script
In another terminal:
```bash
cd D:\Root\College\EcoCar\Research
python test_phase1.py
```
Press ENTER when prompted.

## Expected Results

The test will:
- Spawn ego and lead vehicles in CARLA
- Run a 10-second simulation
- Display real-time metrics every second
- Show summary statistics

**Success indicators:**
- ✅ No collisions (Gap > 2m)
- ✅ Safe time headway (THW > 1.5s)
- ✅ Smooth control (RMS jerk < 2.0)
- ✅ Test completes without errors

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Failed to connect" | Verify CARLA server is running |
| "Spawn failed" | Restart CARLA server |
| CARLA crashes | Use `-RenderOffScreen` flag |
| Import errors | Run `pip install -r requirements.txt` |
| Slow performance | Use `-quality-level=Low` |

## Quick Alternative Test

For a minimal test without the full script:
```bash
python -m environment.car_following
```

## What's Being Tested

✅ CARLA connection and synchronous mode
✅ Vehicle spawning and control
✅ Acceleration → throttle/brake mapping
✅ State observation (velocity, gap, etc.)
✅ Energy tracking
✅ Collision detection
✅ Data logging

## Next Steps

After successful test:
- **Phase 2:** Implement MPC controller
- **Phase 3:** DRL integration
- **Phase 4:** Evaluation and results

## Key Metrics

- **EgoVel:** Ego vehicle speed (m/s)
- **Gap:** Distance between vehicles (m)
- **THW:** Time headway - should stay > 1.5s
- **Energy:** Cumulative consumption
- **RMS Jerk:** Comfort metric (lower is better)

---

**Need help?** Check `CLAUDE.md` or `Action Plan.md` for details.
