"""
Test script for MPC and ACC controllers in CARLA.

This script runs both controllers on various test scenarios
and compares their performance using calculated metrics.

Usage:
    python test_controllers.py                    # Interactive menu
    python test_controllers.py --scenario multi_phase
    python test_controllers.py --all              # Run all scenarios
    python test_controllers.py --list             # List available scenarios
"""

import numpy as np
import sys
import os
import argparse
import logging
import time
from typing import Dict, Tuple, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
from environment.car_following import CarFollowingEnv
from controllers.mpc_controller import MPCController, FixedWeightMPC
from controllers.acc_controller import ACCController
from utils.metrics import calculate_all_metrics, print_metrics_summary
from utils.scenarios import (
    get_scenario,
    get_all_scenarios,
    get_drl_advantage_scenarios,
    print_scenario_summary,
    ALL_SCENARIOS,
    DRL_ADVANTAGE_SCENARIOS,
    ScenarioConfig
)


def run_controller_episode(
    env: CarFollowingEnv,
    controller,
    controller_name: str,
    max_steps: int = 500,
    verbose: bool = True
) -> Tuple[Dict, Dict]:
    """
    Run a single episode with the given controller.

    Args:
        env: CARLA environment
        controller: MPC or ACC controller instance
        controller_name: Name for logging
        max_steps: Maximum steps to run
        verbose: Whether to print progress

    Returns:
        episode_data: Recorded time series data
        metrics: Calculated metrics
    """
    if verbose:
        logger.info(f"Running episode with {controller_name}...")

    # Reset environment
    obs, info = env.reset(seed=42)
    controller.reset()

    # Data collection
    episode_data = {
        'time': [],
        'ego_velocity': [],
        'lead_velocity': [],
        'distance_gap': [],
        'relative_velocity': [],
        'acceleration': [],
        'throttle': [],
        'brake': [],
        'time_headway': []
    }

    step = 0
    done = False

    while step < max_steps and not done:
        # Extract observation
        ego_velocity, relative_velocity, distance_gap = obs
        lead_velocity = ego_velocity + relative_velocity

        # Compute control action
        if isinstance(controller, (MPCController, FixedWeightMPC)):
            acceleration, ctrl_info = controller.compute_control(
                ego_velocity=ego_velocity,
                lead_velocity=lead_velocity,
                distance_gap=distance_gap
            )
        elif isinstance(controller, ACCController):
            acceleration, ctrl_info = controller.compute_control(
                ego_velocity=ego_velocity,
                lead_velocity=lead_velocity,
                distance_gap=distance_gap
            )
        else:
            raise ValueError(f"Unknown controller type: {type(controller)}")

        # Step environment
        obs, reward, terminated, truncated, info = env.step(acceleration)
        done = terminated or truncated

        # Record data
        episode_data['time'].append(step * env.dt)
        episode_data['ego_velocity'].append(ego_velocity)
        episode_data['lead_velocity'].append(lead_velocity)
        episode_data['distance_gap'].append(distance_gap)
        episode_data['relative_velocity'].append(relative_velocity)
        episode_data['acceleration'].append(acceleration)
        episode_data['throttle'].append(info.get('throttle', 0.0))
        episode_data['brake'].append(info.get('brake', 0.0))
        episode_data['time_headway'].append(info.get('time_headway', 0.0))

        # Progress logging
        if verbose and step % 100 == 0:
            logger.info(
                f"  Step {step}: vel={ego_velocity:.1f} m/s, "
                f"gap={distance_gap:.1f} m, THW={info.get('time_headway', 0):.2f} s"
            )

        step += 1

        if terminated:
            logger.warning(f"  Episode terminated (collision) at step {step}")
            break

    if verbose:
        logger.info(f"  Episode completed: {step} steps")

    # Calculate metrics
    metrics = calculate_all_metrics(
        episode_data=episode_data,
        dt=env.dt,
        target_velocity=20.0,
        vehicle_mass=1800.0
    )

    return episode_data, metrics


def print_comparison(mpc_metrics: Dict, acc_metrics: Dict):
    """Print comparison table between MPC and ACC."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Metric':<35} {'MPC':>15} {'ACC':>15}")
    print("-" * 65)

    comparisons = [
        ('Total Energy', 'energy_throttle', 'total_energy', '.2f'),
        ('Energy per km', 'energy_throttle', 'energy_per_km', '.2f'),
        ('RMS Jerk (m/s³)', 'comfort', 'rms_jerk', '.4f'),
        ('Min Time Headway (s)', 'safety', 'min_time_headway', '.2f'),
        ('Avg Time Headway (s)', 'safety', 'avg_time_headway', '.2f'),
        ('THW Violation Rate (%)', 'safety', 'violation_rate', '.1%'),
        ('Avg Velocity (m/s)', 'velocity', 'avg_ego_velocity', '.2f'),
        ('Avg Throttle', 'control', 'avg_throttle', '.3f'),
    ]

    for name, category, key, fmt in comparisons:
        mpc_val = mpc_metrics[category][key]
        acc_val = acc_metrics[category][key]
        print(f"{name:<35} {mpc_val:>15{fmt}} {acc_val:>15{fmt}}")

    print()

    # Determine winner for key metrics
    print("Analysis:")
    if mpc_metrics['energy_throttle']['total_energy'] < acc_metrics['energy_throttle']['total_energy']:
        energy_diff = (1 - mpc_metrics['energy_throttle']['total_energy'] / acc_metrics['energy_throttle']['total_energy']) * 100
        print(f"  + MPC is {energy_diff:.1f}% more energy efficient")
    else:
        energy_diff = (1 - acc_metrics['energy_throttle']['total_energy'] / mpc_metrics['energy_throttle']['total_energy']) * 100
        print(f"  + ACC is {energy_diff:.1f}% more energy efficient")

    if mpc_metrics['comfort']['rms_jerk'] < acc_metrics['comfort']['rms_jerk']:
        jerk_diff = (1 - mpc_metrics['comfort']['rms_jerk'] / acc_metrics['comfort']['rms_jerk']) * 100
        print(f"  + MPC is {jerk_diff:.1f}% smoother (lower jerk)")
    else:
        jerk_diff = (1 - acc_metrics['comfort']['rms_jerk'] / mpc_metrics['comfort']['rms_jerk']) * 100
        print(f"  + ACC is {jerk_diff:.1f}% smoother (lower jerk)")

    if mpc_metrics['safety']['min_time_headway'] > acc_metrics['safety']['min_time_headway']:
        print(f"  + MPC maintains safer following distance")
    else:
        print(f"  + ACC maintains safer following distance")


def run_scenario_test(
    scenario_name: str,
    env: Optional[CarFollowingEnv] = None,
    save_results: bool = False
) -> Dict:
    """
    Run a single scenario test with both controllers.

    Args:
        scenario_name: Name of the scenario to run
        env: Existing environment (will create if None)
        save_results: Whether to save results to CSV

    Returns:
        Dictionary with results for both controllers
    """
    # Get scenario
    scenario = get_scenario(scenario_name)
    print("\n" + "=" * 70)
    print(f"SCENARIO: {scenario.name}")
    print(f"Description: {scenario.description}")
    print(f"Duration: {scenario.duration_s:.1f}s ({scenario.num_steps} steps)")
    print("=" * 70)

    # Create environment if needed
    created_env = False
    if env is None:
        env = CarFollowingEnv(
            carla_host='localhost',
            carla_port=2000,
            dt=scenario.dt,
            lead_vehicle_trajectory=scenario.trajectory,
            max_episode_steps=scenario.num_steps,
            map_name='Town04'
        )
        created_env = True
    else:
        # Update trajectory for existing env
        env.lead_trajectory = scenario.trajectory
        env.max_episode_steps = scenario.num_steps

    # Create controllers
    mpc_controller = FixedWeightMPC(
        dt=scenario.dt,
        horizon=30,
        w_velocity=0.25,
        w_safety=0.35,
        w_comfort=0.40,
        target_velocity=20.0
    )

    acc_controller = ACCController(
        time_headway=1.8,
        min_distance=5.0,
        target_velocity=20.0,
        dt=scenario.dt
    )

    results = {}

    try:
        # Run MPC controller
        print("\n" + "-" * 50)
        print("Testing Fixed-Weight MPC Controller")
        print("-" * 50)
        mpc_data, mpc_metrics = run_controller_episode(
            env, mpc_controller, "MPC", max_steps=scenario.num_steps
        )
        results['MPC'] = {
            'data': mpc_data,
            'metrics': mpc_metrics
        }
        print_metrics_summary(mpc_metrics, "Fixed-Weight MPC")

        # Small pause between tests
        time.sleep(2.0)

        # Run ACC controller
        print("\n" + "-" * 50)
        print("Testing ACC Baseline Controller")
        print("-" * 50)
        acc_data, acc_metrics = run_controller_episode(
            env, acc_controller, "ACC", max_steps=scenario.num_steps
        )
        results['ACC'] = {
            'data': acc_data,
            'metrics': acc_metrics
        }
        print_metrics_summary(acc_metrics, "ACC Baseline")

        # Print comparison
        print_comparison(mpc_metrics, acc_metrics)

        # Save results if requested
        if save_results:
            import pandas as pd
            os.makedirs('results', exist_ok=True)

            for name, data in [('MPC', mpc_data), ('ACC', acc_data)]:
                filename = f'results/{scenario_name}_{name.lower()}_data.csv'
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
                logger.info(f"Saved to {filename}")

    finally:
        if created_env:
            env.close()

    return results


def run_all_scenarios(save_results: bool = True, drl_only: bool = True) -> Dict[str, Dict]:
    """Run scenarios and compile results.

    Args:
        save_results: Whether to save CSV files
        drl_only: If True, only run DRL-advantage scenarios. If False, run ALL.
    """
    if drl_only:
        scenarios = list(DRL_ADVANTAGE_SCENARIOS)
        title = "RUNNING ALL DRL-ADVANTAGE SCENARIOS"
    else:
        scenarios = list(ALL_SCENARIOS.keys())
        title = "RUNNING ALL SCENARIOS"

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    all_results = {}

    # Create environment once
    first_scenario = get_scenario(scenarios[0])
    env = CarFollowingEnv(
        carla_host='localhost',
        carla_port=2000,
        dt=first_scenario.dt,
        lead_vehicle_trajectory=first_scenario.trajectory,
        max_episode_steps=first_scenario.num_steps,
        map_name='Town04'
    )

    try:
        for i, scenario_name in enumerate(scenarios):
            print(f"\n[{i+1}/{len(scenarios)}] Running scenario: {scenario_name}")
            results = run_scenario_test(scenario_name, env=env, save_results=save_results)
            all_results[scenario_name] = results

            # Pause between scenarios
            if i < len(scenarios) - 1:
                print("\nPausing 3 seconds before next scenario...")
                time.sleep(3.0)

    finally:
        env.close()

    # Print aggregate summary
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS ACROSS ALL SCENARIOS")
    print("=" * 70)
    print()
    print(f"{'Scenario':<25} {'MPC Energy':>12} {'ACC Energy':>12} {'Winner':>10}")
    print("-" * 60)

    mpc_wins = 0
    acc_wins = 0

    for scenario_name, results in all_results.items():
        mpc_energy = results['MPC']['metrics']['energy_throttle']['total_energy']
        acc_energy = results['ACC']['metrics']['energy_throttle']['total_energy']
        winner = "MPC" if mpc_energy < acc_energy else "ACC"
        if winner == "MPC":
            mpc_wins += 1
        else:
            acc_wins += 1
        print(f"{scenario_name:<25} {mpc_energy:>12.2f} {acc_energy:>12.2f} {winner:>10}")

    print("-" * 60)
    print(f"{'TOTAL WINS':<25} {mpc_wins:>12} {acc_wins:>12}")

    return all_results


def interactive_menu():
    """Show interactive menu for scenario selection."""
    print("\n" + "=" * 70)
    print("CONTROLLER COMPARISON TEST: MPC vs ACC")
    print("=" * 70)
    print()
    print("Prerequisites:")
    print("1. CARLA server must be running (use start_carla.bat)")
    print("2. Make sure you have installed all dependencies")
    print()

    input("Press ENTER when CARLA is ready...")

    while True:
        print("\n" + "-" * 50)
        print("SELECT TEST SCENARIO")
        print("-" * 50)
        print()
        print("DRL Advantage Scenarios (Recommended):")
        for i, name in enumerate(DRL_ADVANTAGE_SCENARIOS, 1):
            scenario = get_scenario(name)
            print(f"  {i}. {name:<25} ({scenario.duration_s:.0f}s)")

        print()
        print("Additional Scenarios:")
        other_scenarios = [s for s in ALL_SCENARIOS if s not in DRL_ADVANTAGE_SCENARIOS]
        for i, name in enumerate(other_scenarios, len(DRL_ADVANTAGE_SCENARIOS) + 1):
            scenario = get_scenario(name)
            print(f"  {i}. {name:<25} ({scenario.duration_s:.0f}s)")

        print()
        print("  A. Run ALL scenarios (9 total)")
        print("  D. Run DRL-advantage scenarios only (5 total)")
        print("  Q. Quit")
        print()

        choice = input("Enter choice: ").strip().upper()

        if choice == 'Q':
            print("Exiting...")
            break
        elif choice == 'A':
            run_all_scenarios(save_results=True, drl_only=False)
        elif choice == 'D':
            run_all_scenarios(save_results=True, drl_only=True)
        else:
            try:
                idx = int(choice) - 1
                all_scenario_names = list(DRL_ADVANTAGE_SCENARIOS) + other_scenarios
                if 0 <= idx < len(all_scenario_names):
                    scenario_name = all_scenario_names[idx]
                    run_scenario_test(scenario_name, save_results=True)
                else:
                    print("Invalid choice. Try again.")
            except ValueError:
                print("Invalid input. Enter a number or Q.")

        # Ask to continue
        print()
        cont = input("Run another scenario? (y/n): ").strip().lower()
        if cont != 'y':
            break

    print("\nTest session complete.")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test MPC and ACC controllers on various scenarios"
    )
    parser.add_argument(
        '--scenario', '-s',
        type=str,
        help='Scenario name to run (e.g., multi_phase, traffic_waves)'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run ALL scenarios (9 total)'
    )
    parser.add_argument(
        '--drl', '-d',
        action='store_true',
        help='Run DRL-advantage scenarios only (5 total)'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available scenarios'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        default=True,
        help='Save results to CSV (default: True)'
    )

    args = parser.parse_args()

    if args.list:
        print_scenario_summary()
        return

    if args.all:
        print("\n" + "=" * 70)
        print("RUNNING ALL SCENARIOS")
        print("=" * 70)
        print()
        print("Prerequisites:")
        print("1. CARLA server must be running (use start_carla.bat)")
        print()
        input("Press ENTER when CARLA is ready...")
        run_all_scenarios(save_results=args.save, drl_only=False)

    elif args.drl:
        print("\n" + "=" * 70)
        print("RUNNING DRL-ADVANTAGE SCENARIOS")
        print("=" * 70)
        print()
        print("Prerequisites:")
        print("1. CARLA server must be running (use start_carla.bat)")
        print()
        input("Press ENTER when CARLA is ready...")
        run_all_scenarios(save_results=args.save, drl_only=True)

    elif args.scenario:
        if args.scenario not in ALL_SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {', '.join(ALL_SCENARIOS.keys())}")
            return

        print("\n" + "=" * 70)
        print(f"RUNNING SCENARIO: {args.scenario}")
        print("=" * 70)
        print()
        print("Prerequisites:")
        print("1. CARLA server must be running (use start_carla.bat)")
        print()
        input("Press ENTER when CARLA is ready...")
        run_scenario_test(args.scenario, save_results=args.save)

    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()
