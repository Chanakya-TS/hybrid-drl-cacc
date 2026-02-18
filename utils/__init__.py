"""
Utilities package for Hybrid DRL-MPC Eco-Driving Framework.

This package contains helper functions and utilities.
"""

from utils.metrics import (
    calculate_energy_consumption,
    calculate_positive_acceleration_energy,
    calculate_rms_jerk,
    calculate_safety_metrics,
    calculate_velocity_metrics,
    calculate_control_metrics,
    calculate_all_metrics,
    create_comparison_table,
    print_metrics_summary
)

from utils.scenarios import (
    ScenarioConfig,
    get_scenario,
    get_all_scenarios,
    get_drl_advantage_scenarios,
    print_scenario_summary,
    ALL_SCENARIOS,
    DRL_ADVANTAGE_SCENARIOS,
)

__all__ = [
    # Metrics
    'calculate_energy_consumption',
    'calculate_positive_acceleration_energy',
    'calculate_rms_jerk',
    'calculate_safety_metrics',
    'calculate_velocity_metrics',
    'calculate_control_metrics',
    'calculate_all_metrics',
    'create_comparison_table',
    'print_metrics_summary',
    # Scenarios
    'ScenarioConfig',
    'get_scenario',
    'get_all_scenarios',
    'get_drl_advantage_scenarios',
    'print_scenario_summary',
    'ALL_SCENARIOS',
    'DRL_ADVANTAGE_SCENARIOS',
]
