"""
Test Scenarios for Eco-Driving Controller Evaluation.

This module provides various driving scenarios designed to test and compare
different controller strategies (DRL, MPC, ACC).

Key scenarios for demonstrating DRL advantages:
1. Multi-phase driving - requires adaptive weight adjustment
2. Predictable patterns - tests anticipation capability
3. Traffic waves - tests smoothing/damping ability
4. Distance-dependent - tests state-aware optimization
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ScenarioConfig:
    """Configuration for a test scenario."""
    name: str
    description: str
    duration_s: float
    trajectory: np.ndarray
    dt: float = 0.05

    @property
    def num_steps(self) -> int:
        return len(self.trajectory)


def create_multi_phase_scenario(dt: float = 0.05) -> ScenarioConfig:
    """
    Multi-Phase Driving Scenario (MOST IMPORTANT FOR DRL).

    Combines multiple driving conditions where optimal MPC weights
    should change based on the current phase:

    Phase 1: Highway cruise (20 m/s) - Energy efficiency priority
    Phase 2: Approaching traffic - Safety priority, anticipate slowdown
    Phase 3: Stop-and-go traffic - Comfort priority, smooth following
    Phase 4: Traffic clearing - Balanced acceleration
    Phase 5: Recovery cruise - Energy efficiency again

    DRL Advantage: Can learn to adjust weights per phase.
    Fixed MPC: Uses same weights, suboptimal in some phases.
    """
    trajectory = np.concatenate([
        # Phase 1: Highway cruise at 22 m/s (8 seconds)
        np.ones(int(8 / dt)) * 22.0,

        # Phase 2: Gradual slowdown approaching traffic (6 seconds)
        np.linspace(22.0, 8.0, int(6 / dt)),

        # Phase 3: Stop-and-go traffic - 3 cycles (18 seconds total)
        # Cycle 1
        np.linspace(8.0, 2.0, int(2 / dt)),   # Slow down
        np.ones(int(1 / dt)) * 2.0,            # Near stop
        np.linspace(2.0, 10.0, int(3 / dt)),   # Speed up
        # Cycle 2
        np.linspace(10.0, 3.0, int(2 / dt)),
        np.ones(int(1 / dt)) * 3.0,
        np.linspace(3.0, 12.0, int(3 / dt)),
        # Cycle 3
        np.linspace(12.0, 4.0, int(2 / dt)),
        np.ones(int(1 / dt)) * 4.0,
        np.linspace(4.0, 15.0, int(3 / dt)),

        # Phase 4: Traffic clearing - accelerate (5 seconds)
        np.linspace(15.0, 22.0, int(5 / dt)),

        # Phase 5: Recovery cruise (8 seconds)
        np.ones(int(8 / dt)) * 22.0,
    ])

    return ScenarioConfig(
        name="multi_phase",
        description="Highway → Traffic → Stop-and-go → Recovery",
        duration_s=len(trajectory) * dt,
        trajectory=trajectory,
        dt=dt
    )


def create_predictable_pattern_scenario(dt: float = 0.05) -> ScenarioConfig:
    """
    Predictable Sinusoidal Pattern Scenario.

    Lead vehicle follows a regular sinusoidal speed pattern.
    Tests whether DRL can learn to anticipate the pattern
    and coast/accelerate preemptively.

    DRL Advantage: Can learn the pattern and act anticipatively.
    Fixed MPC: Only reacts to current state.
    """
    duration = 40.0  # seconds
    t = np.arange(0, duration, dt)

    # Sinusoidal pattern: 15 m/s mean, ±5 m/s amplitude, 10s period
    mean_speed = 15.0
    amplitude = 5.0
    period = 10.0

    trajectory = mean_speed + amplitude * np.sin(2 * np.pi * t / period)

    return ScenarioConfig(
        name="predictable_pattern",
        description="Sinusoidal speed pattern (tests anticipation)",
        duration_s=duration,
        trajectory=trajectory,
        dt=dt
    )


def create_traffic_waves_scenario(dt: float = 0.05) -> ScenarioConfig:
    """
    Traffic Waves Scenario.

    Simulates realistic traffic wave propagation with varying
    frequency and amplitude. Tests whether controller can
    dampen oscillations rather than amplify them.

    DRL Advantage: Can learn optimal damping strategy.
    Fixed MPC: May amplify waves with wrong weight settings.
    """
    duration = 50.0
    t = np.arange(0, duration, dt)

    # Base speed
    base_speed = 18.0

    # Multiple wave components (realistic traffic)
    wave1 = 3.0 * np.sin(2 * np.pi * t / 8.0)   # 8s period, 3 m/s amplitude
    wave2 = 2.0 * np.sin(2 * np.pi * t / 12.0)  # 12s period, 2 m/s amplitude
    wave3 = 1.5 * np.sin(2 * np.pi * t / 5.0)   # 5s period, 1.5 m/s amplitude

    # Add some noise for realism
    noise = 0.5 * np.random.randn(len(t))
    noise = np.convolve(noise, np.ones(10)/10, mode='same')  # Smooth noise

    trajectory = base_speed + wave1 + wave2 + wave3 + noise
    trajectory = np.clip(trajectory, 5.0, 25.0)  # Keep within bounds

    return ScenarioConfig(
        name="traffic_waves",
        description="Multi-frequency traffic oscillations",
        duration_s=duration,
        trajectory=trajectory,
        dt=dt
    )


def create_varying_distance_scenario(dt: float = 0.05) -> ScenarioConfig:
    """
    Varying Distance Scenario.

    Lead vehicle alternates between pulling away and slowing down,
    creating varying distance gaps. Tests distance-aware optimization.

    When far: Should prioritize energy (don't chase aggressively)
    When close: Should prioritize safety

    DRL Advantage: Learns distance-dependent weight adjustment.
    """
    trajectory = np.concatenate([
        # Start together at moderate speed (5s)
        np.ones(int(5 / dt)) * 15.0,

        # Lead accelerates away (6s) - ego should not chase aggressively
        np.linspace(15.0, 25.0, int(3 / dt)),
        np.ones(int(3 / dt)) * 25.0,

        # Lead maintains high speed (5s) - ego falls behind
        np.ones(int(5 / dt)) * 25.0,

        # Lead slows significantly (4s) - gap closes
        np.linspace(25.0, 10.0, int(4 / dt)),

        # Lead goes very slow (5s) - ego catches up, safety critical
        np.ones(int(5 / dt)) * 10.0,

        # Lead speeds up again (4s)
        np.linspace(10.0, 20.0, int(4 / dt)),

        # Cruise together (5s)
        np.ones(int(5 / dt)) * 20.0,

        # Repeat pattern
        np.linspace(20.0, 28.0, int(3 / dt)),
        np.ones(int(4 / dt)) * 28.0,
        np.linspace(28.0, 8.0, int(5 / dt)),
        np.ones(int(4 / dt)) * 8.0,
        np.linspace(8.0, 18.0, int(4 / dt)),
        np.ones(int(4 / dt)) * 18.0,
    ])

    return ScenarioConfig(
        name="varying_distance",
        description="Lead pulls away and slows down repeatedly",
        duration_s=len(trajectory) * dt,
        trajectory=trajectory,
        dt=dt
    )


def create_mixed_speed_scenario(dt: float = 0.05) -> ScenarioConfig:
    """
    Mixed Speed Regime Scenario.

    Alternates between high-speed highway and low-speed urban driving.
    Optimal weights differ significantly between regimes.

    High speed: Safety weight should be higher (less reaction time)
    Low speed: Comfort weight can be higher, energy matters more

    DRL Advantage: Learns speed-dependent optimization.
    """
    trajectory = np.concatenate([
        # High-speed highway (10s at 25-28 m/s)
        np.ones(int(3 / dt)) * 25.0,
        np.linspace(25.0, 28.0, int(2 / dt)),
        np.ones(int(3 / dt)) * 28.0,
        np.linspace(28.0, 25.0, int(2 / dt)),

        # Transition to low speed (5s)
        np.linspace(25.0, 8.0, int(5 / dt)),

        # Low-speed urban (12s at 5-10 m/s)
        np.ones(int(3 / dt)) * 8.0,
        np.linspace(8.0, 5.0, int(2 / dt)),
        np.ones(int(2 / dt)) * 5.0,
        np.linspace(5.0, 10.0, int(2 / dt)),
        np.ones(int(3 / dt)) * 10.0,

        # Transition back to high speed (5s)
        np.linspace(10.0, 25.0, int(5 / dt)),

        # High-speed again (10s)
        np.ones(int(4 / dt)) * 25.0,
        np.linspace(25.0, 22.0, int(3 / dt)),
        np.ones(int(3 / dt)) * 22.0,

        # Another low-speed segment (8s)
        np.linspace(22.0, 6.0, int(4 / dt)),
        np.ones(int(4 / dt)) * 6.0,

        # Final acceleration (5s)
        np.linspace(6.0, 20.0, int(5 / dt)),
    ])

    return ScenarioConfig(
        name="mixed_speed",
        description="Alternating highway and urban speeds",
        duration_s=len(trajectory) * dt,
        trajectory=trajectory,
        dt=dt
    )


def create_emergency_braking_scenario(dt: float = 0.05) -> ScenarioConfig:
    """
    Emergency Braking and Recovery Scenario.

    Tests response to sudden braking events and optimal recovery.
    Multiple emergency events with varying severity.

    DRL Advantage: Learns optimal post-emergency recovery strategy.
    """
    trajectory = np.concatenate([
        # Cruise (5s)
        np.ones(int(5 / dt)) * 20.0,

        # EMERGENCY 1: Hard brake to near-stop (2s)
        np.linspace(20.0, 2.0, int(2 / dt)),

        # Hold at low speed (2s)
        np.ones(int(2 / dt)) * 2.0,

        # Recovery acceleration (4s)
        np.linspace(2.0, 18.0, int(4 / dt)),

        # Cruise (4s)
        np.ones(int(4 / dt)) * 18.0,

        # EMERGENCY 2: Moderate brake (2s)
        np.linspace(18.0, 8.0, int(2 / dt)),

        # Brief hold (1s)
        np.ones(int(1 / dt)) * 8.0,

        # Quick recovery (3s)
        np.linspace(8.0, 20.0, int(3 / dt)),

        # Cruise (5s)
        np.ones(int(5 / dt)) * 20.0,

        # EMERGENCY 3: Severe - full stop (3s)
        np.linspace(20.0, 0.5, int(3 / dt)),

        # Stopped (3s)
        np.ones(int(3 / dt)) * 0.5,

        # Gradual recovery (6s)
        np.linspace(0.5, 18.0, int(6 / dt)),

        # Final cruise (5s)
        np.ones(int(5 / dt)) * 18.0,
    ])

    return ScenarioConfig(
        name="emergency_braking",
        description="Multiple emergency braking events with recovery",
        duration_s=len(trajectory) * dt,
        trajectory=trajectory,
        dt=dt
    )


def create_eco_coasting_scenario(dt: float = 0.05) -> ScenarioConfig:
    """
    Eco-Coasting Opportunity Scenario.

    Scenario with known deceleration points (like approaching
    a red light that will turn green). Tests ability to coast
    optimally rather than brake late.

    DRL Advantage: Can learn optimal coasting initiation timing.
    """
    trajectory = np.concatenate([
        # Cruise (6s)
        np.ones(int(6 / dt)) * 20.0,

        # Long gradual coast opportunity (8s) - optimal eco-driving
        np.linspace(20.0, 5.0, int(8 / dt)),

        # Brief low speed (2s) - "traffic light"
        np.ones(int(2 / dt)) * 5.0,

        # Accelerate (4s)
        np.linspace(5.0, 22.0, int(4 / dt)),

        # Cruise (5s)
        np.ones(int(5 / dt)) * 22.0,

        # Another coasting opportunity (6s)
        np.linspace(22.0, 8.0, int(6 / dt)),

        # Low speed (3s)
        np.ones(int(3 / dt)) * 8.0,

        # Accelerate (4s)
        np.linspace(8.0, 18.0, int(4 / dt)),

        # Cruise (4s)
        np.ones(int(4 / dt)) * 18.0,

        # Short coast (3s) - less time to react
        np.linspace(18.0, 6.0, int(3 / dt)),

        # Hold (2s)
        np.ones(int(2 / dt)) * 6.0,

        # Final acceleration (4s)
        np.linspace(6.0, 20.0, int(4 / dt)),

        # End cruise (3s)
        np.ones(int(3 / dt)) * 20.0,
    ])

    return ScenarioConfig(
        name="eco_coasting",
        description="Coasting opportunities for energy-efficient driving",
        duration_s=len(trajectory) * dt,
        trajectory=trajectory,
        dt=dt
    )


def create_highway_cruise_scenario(dt: float = 0.05) -> ScenarioConfig:
    """
    Simple Highway Cruise Scenario (Baseline).

    Constant high-speed driving with minor variations.
    Used as a baseline comparison.
    """
    duration = 30.0
    t = np.arange(0, duration, dt)

    # Mostly constant with small realistic variations
    base_speed = 22.0
    small_variation = 1.0 * np.sin(2 * np.pi * t / 20.0)

    trajectory = base_speed + small_variation

    return ScenarioConfig(
        name="highway_cruise",
        description="Steady highway driving (baseline)",
        duration_s=duration,
        trajectory=trajectory,
        dt=dt
    )


def create_aggressive_driver_scenario(dt: float = 0.05) -> ScenarioConfig:
    """
    Aggressive Lead Driver Scenario.

    Lead vehicle drives erratically with sudden accelerations
    and hard braking. Tests robustness and comfort maintenance.
    """
    trajectory = np.concatenate([
        # Normal start (3s)
        np.ones(int(3 / dt)) * 18.0,

        # Sudden acceleration (1.5s)
        np.linspace(18.0, 28.0, int(1.5 / dt)),

        # Brief high speed (2s)
        np.ones(int(2 / dt)) * 28.0,

        # Hard brake (1s)
        np.linspace(28.0, 12.0, int(1 / dt)),

        # Erratic (4s)
        np.linspace(12.0, 20.0, int(1 / dt)),
        np.linspace(20.0, 10.0, int(1.5 / dt)),
        np.linspace(10.0, 22.0, int(1.5 / dt)),

        # Aggressive acceleration (1.5s)
        np.linspace(22.0, 30.0, int(1.5 / dt)),

        # Hold (2s)
        np.ones(int(2 / dt)) * 30.0,

        # Very hard brake (1.5s)
        np.linspace(30.0, 5.0, int(1.5 / dt)),

        # Slow (3s)
        np.ones(int(3 / dt)) * 5.0,

        # Quick acceleration (2s)
        np.linspace(5.0, 25.0, int(2 / dt)),

        # More erratic behavior (6s)
        np.linspace(25.0, 15.0, int(1 / dt)),
        np.linspace(15.0, 28.0, int(2 / dt)),
        np.linspace(28.0, 18.0, int(1.5 / dt)),
        np.ones(int(1.5 / dt)) * 18.0,

        # Calm down (5s)
        np.linspace(18.0, 20.0, int(2 / dt)),
        np.ones(int(3 / dt)) * 20.0,
    ])

    return ScenarioConfig(
        name="aggressive_driver",
        description="Erratic lead vehicle behavior",
        duration_s=len(trajectory) * dt,
        trajectory=trajectory,
        dt=dt
    )


# ============================================================================
# Scenario Registry and Utilities
# ============================================================================

ALL_SCENARIOS = {
    'multi_phase': create_multi_phase_scenario,
    'predictable_pattern': create_predictable_pattern_scenario,
    'traffic_waves': create_traffic_waves_scenario,
    'varying_distance': create_varying_distance_scenario,
    'mixed_speed': create_mixed_speed_scenario,
    'emergency_braking': create_emergency_braking_scenario,
    'eco_coasting': create_eco_coasting_scenario,
    'highway_cruise': create_highway_cruise_scenario,
    'aggressive_driver': create_aggressive_driver_scenario,
}

# Scenarios specifically designed to show DRL advantages
DRL_ADVANTAGE_SCENARIOS = [
    'multi_phase',
    'predictable_pattern',
    'traffic_waves',
    'varying_distance',
    'mixed_speed',
]


def get_scenario(name: str, dt: float = 0.05) -> ScenarioConfig:
    """
    Get a scenario by name.

    Args:
        name: Scenario name (see ALL_SCENARIOS keys)
        dt: Timestep in seconds

    Returns:
        ScenarioConfig with trajectory and metadata
    """
    if name not in ALL_SCENARIOS:
        available = ', '.join(ALL_SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{name}'. Available: {available}")

    return ALL_SCENARIOS[name](dt)


def get_all_scenarios(dt: float = 0.05) -> Dict[str, ScenarioConfig]:
    """Get all scenarios as a dictionary."""
    return {name: func(dt) for name, func in ALL_SCENARIOS.items()}


def get_drl_advantage_scenarios(dt: float = 0.05) -> Dict[str, ScenarioConfig]:
    """Get scenarios specifically designed to show DRL advantages."""
    return {name: ALL_SCENARIOS[name](dt) for name in DRL_ADVANTAGE_SCENARIOS}


def print_scenario_summary():
    """Print a summary of all available scenarios."""
    print("\n" + "=" * 70)
    print("AVAILABLE TEST SCENARIOS")
    print("=" * 70)

    print("\n--- DRL Advantage Scenarios (Recommended for Research) ---")
    for name in DRL_ADVANTAGE_SCENARIOS:
        scenario = ALL_SCENARIOS[name]()
        print(f"\n  {name}:")
        print(f"    Description: {scenario.description}")
        print(f"    Duration: {scenario.duration_s:.1f}s ({scenario.num_steps} steps)")

    print("\n--- Additional Scenarios ---")
    for name in ALL_SCENARIOS:
        if name not in DRL_ADVANTAGE_SCENARIOS:
            scenario = ALL_SCENARIOS[name]()
            print(f"\n  {name}:")
            print(f"    Description: {scenario.description}")
            print(f"    Duration: {scenario.duration_s:.1f}s ({scenario.num_steps} steps)")

    print("\n" + "=" * 70)


# Test script
if __name__ == "__main__":
    print_scenario_summary()

    # Plot a sample scenario
    import matplotlib.pyplot as plt

    scenario = get_scenario('multi_phase')
    t = np.arange(len(scenario.trajectory)) * scenario.dt

    plt.figure(figsize=(12, 4))
    plt.plot(t, scenario.trajectory, 'b-', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Lead Vehicle Velocity (m/s)')
    plt.title(f"Scenario: {scenario.name} - {scenario.description}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/scenario_preview.png', dpi=150)
    plt.show()

    print(f"\nScenario '{scenario.name}' preview saved to results/scenario_preview.png")
