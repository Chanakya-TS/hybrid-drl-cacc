"""
Metrics Calculation Utilities for Eco-Driving Evaluation.

This module provides functions to calculate key performance metrics
for comparing different controllers:
- Energy consumption
- Safety (time headway violations)
- Comfort (jerk/acceleration smoothness)
- Efficiency (average speed)
"""

import numpy as np
from typing import Dict, List, Optional
import pandas as pd


def calculate_energy_consumption(
    throttle: np.ndarray,
    velocity: np.ndarray,
    dt: float
) -> Dict[str, float]:
    """
    Calculate total energy consumption using throttle as power proxy.

    Energy proxy: integral of (throttle * velocity * dt)
    This approximates mechanical power output.

    Args:
        throttle: Array of throttle values [0, 1]
        velocity: Array of velocity values (m/s)
        dt: Timestep in seconds

    Returns:
        Dictionary with energy metrics:
        - total_energy: Total energy consumed (proxy units)
        - avg_power: Average power (proxy units)
        - energy_per_km: Energy per kilometer traveled
    """
    # Power proxy: throttle * velocity
    power = throttle * velocity

    # Total energy (integral)
    total_energy = np.sum(power) * dt

    # Average power
    avg_power = np.mean(power)

    # Distance traveled
    distance = np.sum(velocity) * dt  # meters

    # Energy per kilometer
    energy_per_km = total_energy / (distance / 1000.0 + 1e-6)

    return {
        'total_energy': float(total_energy),
        'avg_power': float(avg_power),
        'distance_m': float(distance),
        'distance_km': float(distance / 1000.0),
        'energy_per_km': float(energy_per_km)
    }


def calculate_positive_acceleration_energy(
    acceleration: np.ndarray,
    velocity: np.ndarray,
    mass: float,
    dt: float
) -> Dict[str, float]:
    """
    Calculate energy using positive acceleration work.

    Energy = m * sum(max(a, 0) * v * dt)
    This measures mechanical work done to accelerate the vehicle.

    Args:
        acceleration: Array of acceleration values (m/s²)
        velocity: Array of velocity values (m/s)
        mass: Vehicle mass (kg)
        dt: Timestep in seconds

    Returns:
        Dictionary with energy metrics
    """
    # Only count positive acceleration (energy spent accelerating)
    positive_accel = np.maximum(acceleration, 0.0)

    # Work done: F * d = m * a * v * dt
    work = mass * positive_accel * velocity * dt

    # Total energy
    total_energy = np.sum(work)

    # Energy for deceleration (regenerative potential)
    negative_accel = np.minimum(acceleration, 0.0)
    regen_potential = mass * np.abs(negative_accel) * velocity * dt
    total_regen = np.sum(regen_potential)

    return {
        'total_energy_J': float(total_energy),
        'total_energy_kJ': float(total_energy / 1000.0),
        'regen_potential_J': float(total_regen),
        'regen_potential_kJ': float(total_regen / 1000.0),
        'net_energy_kJ': float((total_energy - total_regen * 0.7) / 1000.0)  # Assume 70% regen efficiency
    }


def calculate_rms_jerk(
    acceleration: np.ndarray,
    dt: float
) -> Dict[str, float]:
    """
    Calculate RMS jerk (rate of change of acceleration).

    Jerk = da/dt
    Lower jerk indicates smoother, more comfortable driving.

    Args:
        acceleration: Array of acceleration values (m/s²)
        dt: Timestep in seconds

    Returns:
        Dictionary with jerk metrics
    """
    if len(acceleration) < 2:
        return {
            'rms_jerk': 0.0,
            'max_jerk': 0.0,
            'avg_abs_jerk': 0.0
        }

    # Calculate jerk (derivative of acceleration)
    jerk = np.diff(acceleration) / dt

    # RMS jerk
    rms_jerk = np.sqrt(np.mean(jerk ** 2))

    # Maximum absolute jerk
    max_jerk = np.max(np.abs(jerk))

    # Average absolute jerk
    avg_abs_jerk = np.mean(np.abs(jerk))

    return {
        'rms_jerk': float(rms_jerk),
        'max_jerk': float(max_jerk),
        'avg_abs_jerk': float(avg_abs_jerk),
        'jerk_std': float(np.std(jerk))
    }


def calculate_safety_metrics(
    distance_gap: np.ndarray,
    ego_velocity: np.ndarray,
    min_thw_threshold: float = 1.5,
    max_thw_threshold: float = 5.0
) -> Dict[str, float]:
    """
    Calculate safety-related metrics.

    Time Headway (THW) = distance / velocity
    Lower THW values indicate potentially unsafe following.
    Upper THW values indicate trailing too far (poor car-following).

    Args:
        distance_gap: Array of distance gap values (m)
        ego_velocity: Array of ego velocity values (m/s)
        min_thw_threshold: Minimum safe THW threshold (s)
        max_thw_threshold: Maximum acceptable THW threshold (s)

    Returns:
        Dictionary with safety metrics
    """
    # Avoid division by zero
    safe_velocity = np.maximum(ego_velocity, 0.1)

    # Calculate time headway
    time_headway = distance_gap / safe_velocity

    # Minimum THW (most dangerous moment)
    min_thw = np.min(time_headway)

    # Average THW
    avg_thw = np.mean(time_headway)

    # THW violations — too close (below min threshold)
    too_close = time_headway < min_thw_threshold
    too_close_count = int(np.sum(too_close))
    too_close_rate = too_close_count / len(time_headway)

    # THW violations — too far (above max threshold)
    too_far = time_headway > max_thw_threshold
    too_far_count = int(np.sum(too_far))
    too_far_rate = too_far_count / len(time_headway)

    # Combined violation rate (either too close or too far)
    violations = too_close | too_far
    violation_count = int(np.sum(violations))
    violation_rate = violation_count / len(time_headway)

    # Minimum distance gap
    min_distance = np.min(distance_gap)

    return {
        'min_time_headway': float(min_thw),
        'avg_time_headway': float(avg_thw),
        'thw_std': float(np.std(time_headway)),
        'violation_count': violation_count,
        'violation_rate': float(violation_rate),
        'too_close_count': too_close_count,
        'too_close_rate': float(too_close_rate),
        'too_far_count': too_far_count,
        'too_far_rate': float(too_far_rate),
        'min_distance_gap': float(min_distance),
        'avg_distance_gap': float(np.mean(distance_gap))
    }


def calculate_velocity_metrics(
    ego_velocity: np.ndarray,
    lead_velocity: np.ndarray,
    target_velocity: float
) -> Dict[str, float]:
    """
    Calculate velocity-related metrics.

    Args:
        ego_velocity: Array of ego velocity values (m/s)
        lead_velocity: Array of lead velocity values (m/s)
        target_velocity: Desired cruising velocity (m/s)

    Returns:
        Dictionary with velocity metrics
    """
    # Average velocities
    avg_ego_velocity = np.mean(ego_velocity)
    avg_lead_velocity = np.mean(lead_velocity)

    # Velocity tracking error
    velocity_error = ego_velocity - target_velocity
    rmse_velocity = np.sqrt(np.mean(velocity_error ** 2))

    # Relative velocity statistics
    relative_velocity = lead_velocity - ego_velocity
    avg_rel_velocity = np.mean(relative_velocity)

    # Velocity smoothness (std of velocity changes)
    velocity_changes = np.diff(ego_velocity)
    velocity_smoothness = np.std(velocity_changes)

    return {
        'avg_ego_velocity': float(avg_ego_velocity),
        'avg_lead_velocity': float(avg_lead_velocity),
        'max_ego_velocity': float(np.max(ego_velocity)),
        'min_ego_velocity': float(np.min(ego_velocity)),
        'velocity_rmse': float(rmse_velocity),
        'avg_relative_velocity': float(avg_rel_velocity),
        'velocity_smoothness': float(velocity_smoothness)
    }


def calculate_control_metrics(
    throttle: np.ndarray,
    brake: np.ndarray,
    acceleration: np.ndarray
) -> Dict[str, float]:
    """
    Calculate control-related metrics.

    Args:
        throttle: Array of throttle values [0, 1]
        brake: Array of brake values [0, 1]
        acceleration: Array of acceleration values (m/s²)

    Returns:
        Dictionary with control metrics
    """
    # Control smoothness
    throttle_changes = np.diff(throttle)
    brake_changes = np.diff(brake)

    throttle_smoothness = np.std(throttle_changes)
    brake_smoothness = np.std(brake_changes)

    # Control effort
    avg_throttle = np.mean(throttle)
    avg_brake = np.mean(brake)

    # Acceleration statistics
    avg_acceleration = np.mean(acceleration)
    accel_std = np.std(acceleration)

    # Positive/negative acceleration time
    accel_time = np.sum(acceleration > 0.1) / len(acceleration)
    decel_time = np.sum(acceleration < -0.1) / len(acceleration)
    coast_time = 1.0 - accel_time - decel_time

    return {
        'avg_throttle': float(avg_throttle),
        'avg_brake': float(avg_brake),
        'throttle_smoothness': float(throttle_smoothness),
        'brake_smoothness': float(brake_smoothness),
        'avg_acceleration': float(avg_acceleration),
        'acceleration_std': float(accel_std),
        'max_acceleration': float(np.max(acceleration)),
        'max_braking': float(np.min(acceleration)),
        'accel_time_ratio': float(accel_time),
        'decel_time_ratio': float(decel_time),
        'coast_time_ratio': float(coast_time)
    }


def calculate_all_metrics(
    episode_data: Dict[str, List],
    dt: float,
    target_velocity: float = 20.0,
    vehicle_mass: float = 1800.0
) -> Dict[str, Dict]:
    """
    Calculate all metrics from episode data.

    Args:
        episode_data: Dictionary containing time series data from simulation
        dt: Timestep in seconds
        target_velocity: Target cruising velocity (m/s)
        vehicle_mass: Vehicle mass in kg

    Returns:
        Dictionary containing all metric categories
    """
    # Convert to numpy arrays
    ego_velocity = np.array(episode_data['ego_velocity'])
    lead_velocity = np.array(episode_data.get('lead_velocity', ego_velocity))
    distance_gap = np.array(episode_data['distance_gap'])
    acceleration = np.array(episode_data['acceleration'])
    throttle = np.array(episode_data['throttle'])
    brake = np.array(episode_data['brake'])

    # Calculate all metric categories
    energy_throttle = calculate_energy_consumption(throttle, ego_velocity, dt)
    energy_accel = calculate_positive_acceleration_energy(acceleration, ego_velocity, vehicle_mass, dt)
    jerk = calculate_rms_jerk(acceleration, dt)
    safety = calculate_safety_metrics(distance_gap, ego_velocity)
    velocity = calculate_velocity_metrics(ego_velocity, lead_velocity, target_velocity)
    control = calculate_control_metrics(throttle, brake, acceleration)

    # Episode summary
    episode_summary = {
        'duration_s': float(len(ego_velocity) * dt),
        'total_steps': len(ego_velocity),
        'collision': bool(np.min(distance_gap) < 2.0)
    }

    return {
        'energy_throttle': energy_throttle,
        'energy_mechanical': energy_accel,
        'comfort': jerk,
        'safety': safety,
        'velocity': velocity,
        'control': control,
        'summary': episode_summary
    }


def create_comparison_table(
    results: Dict[str, Dict],
    metrics_to_compare: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create a comparison table from multiple controller results.

    Args:
        results: Dictionary mapping controller names to their metrics
        metrics_to_compare: List of specific metrics to include (None = all)

    Returns:
        pandas DataFrame with comparison
    """
    if metrics_to_compare is None:
        metrics_to_compare = [
            'energy_throttle.total_energy',
            'energy_mechanical.total_energy_kJ',
            'comfort.rms_jerk',
            'safety.min_time_headway',
            'safety.violation_rate',
            'velocity.avg_ego_velocity',
            'control.avg_throttle'
        ]

    rows = []
    for controller_name, metrics in results.items():
        row = {'Controller': controller_name}
        for metric_path in metrics_to_compare:
            parts = metric_path.split('.')
            value = metrics
            for part in parts:
                value = value.get(part, 'N/A')
            row[metric_path] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index('Controller', inplace=True)

    return df


def print_metrics_summary(metrics: Dict, controller_name: str = "Controller"):
    """
    Print a formatted summary of metrics.

    Args:
        metrics: Dictionary of calculated metrics
        controller_name: Name of the controller for display
    """
    print(f"\n{'='*60}")
    print(f"METRICS SUMMARY: {controller_name}")
    print('=' * 60)

    print("\n--- Energy ---")
    e = metrics['energy_throttle']
    print(f"  Total Energy (throttle proxy): {e['total_energy']:.2f}")
    print(f"  Distance traveled: {e['distance_km']:.2f} km")
    print(f"  Energy per km: {e['energy_per_km']:.2f}")

    em = metrics['energy_mechanical']
    print(f"  Mechanical Energy: {em['total_energy_kJ']:.2f} kJ")

    print("\n--- Comfort ---")
    c = metrics['comfort']
    print(f"  RMS Jerk: {c['rms_jerk']:.3f} m/s³")
    print(f"  Max Jerk: {c['max_jerk']:.3f} m/s³")

    print("\n--- Safety ---")
    s = metrics['safety']
    print(f"  Min Time Headway: {s['min_time_headway']:.2f} s")
    print(f"  Avg Time Headway: {s['avg_time_headway']:.2f} s")
    print(f"  THW Violations: {s['violation_count']} ({s['violation_rate']*100:.1f}%)")
    print(f"    Too Close (<1.5s): {s['too_close_count']} ({s['too_close_rate']*100:.1f}%)")
    print(f"    Too Far  (>5.0s):  {s['too_far_count']} ({s['too_far_rate']*100:.1f}%)")
    print(f"  Min Distance Gap: {s['min_distance_gap']:.2f} m")

    print("\n--- Velocity ---")
    v = metrics['velocity']
    print(f"  Avg Ego Velocity: {v['avg_ego_velocity']:.2f} m/s ({v['avg_ego_velocity']*3.6:.1f} km/h)")
    print(f"  Velocity RMSE: {v['velocity_rmse']:.2f} m/s")

    print("\n--- Control ---")
    ctrl = metrics['control']
    print(f"  Avg Throttle: {ctrl['avg_throttle']:.3f}")
    print(f"  Accel/Decel/Coast: {ctrl['accel_time_ratio']*100:.1f}% / "
          f"{ctrl['decel_time_ratio']*100:.1f}% / {ctrl['coast_time_ratio']*100:.1f}%")

    print("\n--- Summary ---")
    summary = metrics['summary']
    print(f"  Duration: {summary['duration_s']:.1f} s")
    print(f"  Collision: {summary['collision']}")
    print('=' * 60)


# Test script
if __name__ == "__main__":
    print("Testing metrics utilities...")

    # Create dummy episode data
    n_steps = 100
    dt = 0.1

    dummy_data = {
        'time': np.arange(n_steps) * dt,
        'ego_velocity': 15.0 + 3.0 * np.sin(np.linspace(0, 4*np.pi, n_steps)),
        'lead_velocity': 18.0 + 2.0 * np.sin(np.linspace(0, 4*np.pi, n_steps)),
        'distance_gap': 25.0 + 5.0 * np.sin(np.linspace(0, 2*np.pi, n_steps)),
        'acceleration': np.random.randn(n_steps) * 0.5,
        'throttle': np.clip(np.random.rand(n_steps) * 0.5, 0, 1),
        'brake': np.clip(np.random.rand(n_steps) * 0.2, 0, 1)
    }

    # Calculate all metrics
    metrics = calculate_all_metrics(dummy_data, dt)

    # Print summary
    print_metrics_summary(metrics, "Test Controller")

    print("\nMetrics test completed!")
