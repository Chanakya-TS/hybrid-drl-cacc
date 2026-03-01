"""
EPA Drive Cycle Scenarios for Eco-Driving Controller Evaluation.

This module wraps the three standardized EPA drive cycles (UDDS, HWFET, US06)
as ScenarioConfig objects for training and evaluation of the DRL-MPC controller.

EPA cycles replace hand-crafted scenarios for research credibility and
reproducibility.  Each cycle is sourced from official EPA data and interpolated
to the simulation timestep via utils.drive_cycles.
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass

from utils.drive_cycles import get_cycle_trajectory, CYCLE_INFO


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


# =============================================================================
# EPA Cycle Scenario Creators
# =============================================================================

def create_udds_scenario(dt: float = 0.05) -> ScenarioConfig:
    """
    UDDS (Urban Dynamometer Driving Schedule) scenario.

    City driving with frequent stops, accelerations, and decelerations.
    Duration: ~1369s, peak 56.7 mph (25.4 m/s).

    DRL Advantage: Highly variable speed profile requires adaptive weight
    adjustment between stop-and-go and cruising segments.
    """
    trajectory, duration_s = get_cycle_trajectory('udds', dt)
    info = CYCLE_INFO['udds']
    return ScenarioConfig(
        name='udds',
        description=f"{info['full_name']} — city driving ({duration_s:.0f}s)",
        duration_s=duration_s,
        trajectory=trajectory,
        dt=dt,
    )


def create_hwfet_scenario(dt: float = 0.05) -> ScenarioConfig:
    """
    HWFET (Highway Fuel Economy Test) scenario.

    Highway driving with relatively steady cruising speeds.
    Duration: ~765s, peak 59.9 mph (26.8 m/s).

    Easiest cycle for car-following; used as warmup in curriculum learning.
    """
    trajectory, duration_s = get_cycle_trajectory('hwfet', dt)
    info = CYCLE_INFO['hwfet']
    return ScenarioConfig(
        name='hwfet',
        description=f"{info['full_name']} — highway driving ({duration_s:.0f}s)",
        duration_s=duration_s,
        trajectory=trajectory,
        dt=dt,
    )


def create_us06_scenario(dt: float = 0.05) -> ScenarioConfig:
    """
    US06 (Supplemental FTP) scenario.

    Aggressive driving with high speeds and rapid accelerations/decelerations.
    Duration: ~600s, peak 80.3 mph (35.9 m/s).

    DRL Advantage: Extreme speed variations and high velocities demand
    aggressive, situation-aware weight adjustments.
    """
    trajectory, duration_s = get_cycle_trajectory('us06', dt)
    info = CYCLE_INFO['us06']
    return ScenarioConfig(
        name='us06',
        description=f"{info['full_name']} — aggressive driving ({duration_s:.0f}s)",
        duration_s=duration_s,
        trajectory=trajectory,
        dt=dt,
    )


# =============================================================================
# Scenario Registry
# =============================================================================

ALL_SCENARIOS = {
    'udds': create_udds_scenario,
    'hwfet': create_hwfet_scenario,
    'us06': create_us06_scenario,
}

# Scenarios where DRL weight adaptation provides the most benefit
DRL_ADVANTAGE_SCENARIOS = ['udds', 'us06']

# Steady-state scenarios suitable for curriculum warmup
EASY_SCENARIOS = ['hwfet']


# =============================================================================
# Public API
# =============================================================================

def get_scenario(name: str, dt: float = 0.05) -> ScenarioConfig:
    """
    Get a scenario by name.

    Args:
        name: Scenario name ('udds', 'hwfet', or 'us06')
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
    print("AVAILABLE EPA DRIVE CYCLE SCENARIOS")
    print("=" * 70)

    print("\n--- DRL Advantage Scenarios ---")
    for name in DRL_ADVANTAGE_SCENARIOS:
        scenario = ALL_SCENARIOS[name]()
        info = CYCLE_INFO[name]
        print(f"\n  {name}:")
        print(f"    Description: {scenario.description}")
        print(f"    Duration: {scenario.duration_s:.0f}s ({scenario.num_steps} steps)")
        print(f"    Peak speed: {info['peak_speed_mph']} mph ({info['peak_speed_ms']:.1f} m/s)")
        print(f"    Category: {info['category']}")

    print("\n--- Easy / Warmup Scenarios ---")
    for name in EASY_SCENARIOS:
        scenario = ALL_SCENARIOS[name]()
        info = CYCLE_INFO[name]
        print(f"\n  {name}:")
        print(f"    Description: {scenario.description}")
        print(f"    Duration: {scenario.duration_s:.0f}s ({scenario.num_steps} steps)")
        print(f"    Peak speed: {info['peak_speed_mph']} mph ({info['peak_speed_ms']:.1f} m/s)")
        print(f"    Category: {info['category']}")

    print("\n" + "=" * 70)


# Test script
if __name__ == "__main__":
    print_scenario_summary()

    # Plot a sample scenario
    import matplotlib.pyplot as plt

    scenario = get_scenario('udds')
    t = np.arange(len(scenario.trajectory)) * scenario.dt

    plt.figure(figsize=(14, 4))
    plt.plot(t, scenario.trajectory, 'b-', linewidth=1.0)
    plt.xlabel('Time (s)')
    plt.ylabel('Lead Vehicle Velocity (m/s)')
    plt.title(f"Scenario: {scenario.name} — {scenario.description}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/scenario_preview.png', dpi=150)
    plt.show()

    print(f"\nScenario '{scenario.name}' preview saved to results/scenario_preview.png")
