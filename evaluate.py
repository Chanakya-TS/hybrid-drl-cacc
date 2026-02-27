"""
Evaluation Script for Hybrid DRL-MPC Eco-Driving Controller.

This script evaluates and compares three controllers:
1. Hybrid DRL-MPC (trained agent)
2. Fixed-Weight MPC (baseline)
3. ACC (baseline)

Usage:
    python evaluate.py --model models/sac_final.zip
    python evaluate.py --model models/sac_final.zip --scenario multi_phase
    python evaluate.py --model models/sac_final.zip --all
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

# Stable-baselines3
from stable_baselines3 import SAC

# Project imports
from environment.car_following import CarFollowingEnv
from environment.gym_wrapper import HybridMPCEnv
from controllers.mpc_controller import MPCController, FixedWeightMPC
from controllers.acc_controller import ACCController
from utils.metrics import calculate_all_metrics, print_metrics_summary
from utils.scenarios import (
    get_scenario,
    get_all_scenarios,
    DRL_ADVANTAGE_SCENARIOS,
    ALL_SCENARIOS
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_drl_episode(
    env: HybridMPCEnv,
    model: SAC,
    scenario_name: str
) -> Tuple[Dict, Dict, np.ndarray]:
    """
    Run a single episode with the trained DRL agent.

    Returns:
        episode_data: Time series data
        metrics: Calculated metrics
        weight_history: History of MPC weights chosen by DRL
    """
    logger.info("Running DRL-MPC controller...")

    obs, info = env.reset(seed=42)

    episode_data = {
        'time': [],
        'ego_velocity': [],
        'lead_velocity': [],
        'distance_gap': [],
        'relative_velocity': [],
        'acceleration': [],
        'throttle': [],
        'brake': [],
        'time_headway': [],
        'w_velocity': [],
        'w_safety': [],
        'w_comfort': []
    }

    sim_step = 0
    agent_step = 0
    done = False
    total_reward = 0.0

    while not done:
        # Get action from trained model
        action, _ = model.predict(obs, deterministic=True)

        # Step environment (runs action_repeat sub-steps internally)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Extract sub-step data for full-resolution logging
        mpc_weights = info.get('mpc_weights', [0.33, 0.34, 0.33])
        sub_data = info.get('sub_step_data', {})

        n_subs = info.get('sim_steps', 1)
        for i in range(n_subs):
            episode_data['time'].append(sim_step * env.dt)
            episode_data['ego_velocity'].append(
                sub_data.get('ego_velocities', [0])[i] if i < len(sub_data.get('ego_velocities', [])) else 0
            )
            episode_data['lead_velocity'].append(
                sub_data.get('lead_velocities', [0])[i] if i < len(sub_data.get('lead_velocities', [])) else 0
            )
            episode_data['distance_gap'].append(
                sub_data.get('distance_gaps', [0])[i] if i < len(sub_data.get('distance_gaps', [])) else 0
            )
            ego_v = sub_data.get('ego_velocities', [0])[i] if i < len(sub_data.get('ego_velocities', [])) else 0
            lead_v = sub_data.get('lead_velocities', [0])[i] if i < len(sub_data.get('lead_velocities', [])) else 0
            episode_data['relative_velocity'].append(lead_v - ego_v)
            episode_data['acceleration'].append(
                sub_data.get('accelerations', [0])[i] if i < len(sub_data.get('accelerations', [])) else 0
            )
            episode_data['throttle'].append(
                sub_data.get('throttles', [0])[i] if i < len(sub_data.get('throttles', [])) else 0
            )
            episode_data['brake'].append(
                sub_data.get('brakes', [0])[i] if i < len(sub_data.get('brakes', [])) else 0
            )
            episode_data['time_headway'].append(
                sub_data.get('time_headways', [0])[i] if i < len(sub_data.get('time_headways', [])) else 0
            )
            # Weights are constant across all sub-steps (that's the point)
            episode_data['w_velocity'].append(mpc_weights[0])
            episode_data['w_safety'].append(mpc_weights[1])
            episode_data['w_comfort'].append(mpc_weights[2])
            sim_step += 1

        agent_step += 1

        if agent_step % 10 == 0:
            logger.info(
                f"  Agent step {agent_step} (sim {sim_step}): "
                f"reward={reward:.2f}, weights={mpc_weights}"
            )

    logger.info(
        f"  Episode completed: {agent_step} agent steps, "
        f"{sim_step} sim steps, total reward: {total_reward:.2f}"
    )

    # Calculate metrics at simulation-level dt (full resolution)
    metrics = calculate_all_metrics(
        episode_data=episode_data,
        dt=env.dt,
        target_velocity=20.0,
        vehicle_mass=1800.0
    )

    weight_history = np.array([
        episode_data['w_velocity'],
        episode_data['w_safety'],
        episode_data['w_comfort']
    ]).T

    return episode_data, metrics, weight_history


def run_baseline_episode(
    env: CarFollowingEnv,
    controller,
    controller_name: str
) -> Tuple[Dict, Dict]:
    """
    Run a single episode with a baseline controller (MPC or ACC).

    Returns:
        episode_data: Time series data
        metrics: Calculated metrics
    """
    logger.info(f"Running {controller_name} controller...")

    obs, info = env.reset(seed=42)
    controller.reset()

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

    while not done:
        # Extract 10-dim observation
        ego_velocity = obs[0]
        rel_vel_1, distance_1, accel_1 = obs[1], obs[2], obs[3]
        rel_vel_2, distance_2, accel_2 = obs[4], obs[5], obs[6]
        rel_vel_3, distance_3, accel_3 = obs[7], obs[8], obs[9]

        lead_velocity_1 = ego_velocity + rel_vel_1

        # Dispatch based on controller type
        if isinstance(controller, (MPCController, FixedWeightMPC)):
            acceleration, _ = controller.compute_control(
                ego_velocity=ego_velocity,
                lead_velocities=[
                    lead_velocity_1,
                    ego_velocity + rel_vel_2,
                    ego_velocity + rel_vel_3
                ],
                distance_gaps=[distance_1, distance_2, distance_3],
                lead_accelerations=[accel_1, accel_2, accel_3]
            )
        else:
            # ACC or other baseline: single lead
            acceleration, _ = controller.compute_control(
                ego_velocity=ego_velocity,
                lead_velocity=lead_velocity_1,
                distance_gap=distance_1
            )

        # Step environment
        obs, _, terminated, truncated, info = env.step(acceleration)
        done = terminated or truncated

        # Record data
        episode_data['time'].append(step * env.dt)
        episode_data['ego_velocity'].append(ego_velocity)
        episode_data['lead_velocity'].append(lead_velocity_1)
        episode_data['distance_gap'].append(distance_1)
        episode_data['relative_velocity'].append(rel_vel_1)
        episode_data['acceleration'].append(acceleration)
        episode_data['throttle'].append(info.get('throttle', 0))
        episode_data['brake'].append(info.get('brake', 0))
        episode_data['time_headway'].append(info.get('time_headway', 0))

        step += 1

    logger.info(f"  Episode completed: {step} steps")

    # Calculate metrics
    metrics = calculate_all_metrics(
        episode_data=episode_data,
        dt=env.dt,
        target_velocity=20.0,
        vehicle_mass=1800.0
    )

    return episode_data, metrics


def evaluate_scenario(
    scenario_name: str,
    model_path: str,
    save_results: bool = True,
    results_dir: str = "results"
) -> Dict:
    """
    Evaluate all controllers on a single scenario.

    Returns:
        Dictionary with results for all controllers
    """
    scenario = get_scenario(scenario_name)

    print("\n" + "=" * 70)
    print(f"EVALUATING SCENARIO: {scenario_name}")
    print(f"Description: {scenario.description}")
    print(f"Duration: {scenario.duration_s:.1f}s ({scenario.num_steps} steps)")
    print("=" * 70)

    results = {}

    # Create CARLA environment for baselines
    carla_env = CarFollowingEnv(
        carla_host='localhost',
        carla_port=2000,
        dt=scenario.dt,
        lead_vehicle_trajectory=scenario.trajectory,
        max_episode_steps=scenario.num_steps,
        map_name='Town04'
    )

    # Create hybrid environment for DRL
    hybrid_env = HybridMPCEnv(
        carla_host='localhost',
        carla_port=2000,
        dt=scenario.dt,
        mpc_horizon=20,
        max_episode_steps=scenario.num_steps,
        target_velocity=20.0,
        lead_trajectory=scenario.trajectory,
        map_name='Town04'
    )

    try:
        # 1. Evaluate DRL-MPC
        print("\n" + "-" * 50)
        print("Testing DRL-MPC (Hybrid) Controller")
        print("-" * 50)

        if os.path.exists(model_path):
            model = SAC.load(model_path)
            drl_data, drl_metrics, weight_history = run_drl_episode(
                hybrid_env, model, scenario_name
            )
            results['DRL-MPC'] = {
                'data': drl_data,
                'metrics': drl_metrics,
                'weight_history': weight_history
            }
            print_metrics_summary(drl_metrics, "DRL-MPC (Hybrid)")
        else:
            logger.warning(f"Model not found: {model_path}")
            results['DRL-MPC'] = None

        time.sleep(2.0)

        # 2. Evaluate Fixed-Weight MPC
        print("\n" + "-" * 50)
        print("Testing Fixed-Weight MPC Controller")
        print("-" * 50)

        mpc_controller = FixedWeightMPC(
            dt=scenario.dt,
            horizon=30,
            w_velocity=0.25,
            w_safety=0.35,
            w_comfort=0.40,
            target_velocity=20.0
        )

        mpc_data, mpc_metrics = run_baseline_episode(
            carla_env, mpc_controller, "Fixed MPC"
        )
        results['Fixed-MPC'] = {
            'data': mpc_data,
            'metrics': mpc_metrics
        }
        print_metrics_summary(mpc_metrics, "Fixed-Weight MPC")

        time.sleep(2.0)

        # 3. Evaluate ACC
        print("\n" + "-" * 50)
        print("Testing ACC Baseline Controller")
        print("-" * 50)

        acc_controller = ACCController(
            time_headway=1.8,
            min_distance=5.0,
            target_velocity=20.0,
            dt=scenario.dt
        )

        acc_data, acc_metrics = run_baseline_episode(
            carla_env, acc_controller, "ACC"
        )
        results['ACC'] = {
            'data': acc_data,
            'metrics': acc_metrics
        }
        print_metrics_summary(acc_metrics, "ACC Baseline")

        # Print comparison
        print_comparison(results)

        # Save results
        if save_results:
            save_scenario_results(scenario_name, results, results_dir)

    finally:
        carla_env.close()
        hybrid_env.close()

    return results


def print_comparison(results: Dict):
    """Print comparison table for all controllers."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    controllers = [name for name in results if results[name] is not None]

    if len(controllers) == 0:
        print("No results to compare")
        return

    # Header
    header = f"{'Metric':<30}"
    for name in controllers:
        header += f" {name:>12}"
    print(header)
    print("-" * 70)

    # Metrics to compare
    comparisons = [
        ('Total Energy', 'energy_throttle', 'total_energy', '.2f'),
        ('Energy per km', 'energy_throttle', 'energy_per_km', '.2f'),
        ('RMS Jerk (m/s³)', 'comfort', 'rms_jerk', '.4f'),
        ('Min Time Headway (s)', 'safety', 'min_time_headway', '.2f'),
        ('Avg Time Headway (s)', 'safety', 'avg_time_headway', '.2f'),
        ('THW Violation Rate', 'safety', 'violation_rate', '.1%'),
        ('  Too Close (<1.5s)', 'safety', 'too_close_rate', '.1%'),
        ('  Too Far (>5.0s)', 'safety', 'too_far_rate', '.1%'),
        ('Avg Velocity (m/s)', 'velocity', 'avg_ego_velocity', '.2f'),
    ]

    for name, category, key, fmt in comparisons:
        row = f"{name:<30}"
        for ctrl in controllers:
            val = results[ctrl]['metrics'][category][key]
            row += f" {val:>12{fmt}}"
        print(row)

    print()

    # Determine winners
    print("Analysis:")

    # Energy winner
    energy_vals = {
        name: results[name]['metrics']['energy_throttle']['total_energy']
        for name in controllers
    }
    energy_winner = min(energy_vals, key=energy_vals.get)
    print(f"  + Most energy efficient: {energy_winner}")

    # Comfort winner
    jerk_vals = {
        name: results[name]['metrics']['comfort']['rms_jerk']
        for name in controllers
    }
    comfort_winner = min(jerk_vals, key=jerk_vals.get)
    print(f"  + Smoothest ride (lowest jerk): {comfort_winner}")

    # Safety winner
    thw_vals = {
        name: results[name]['metrics']['safety']['min_time_headway']
        for name in controllers
    }
    safety_winner = max(thw_vals, key=thw_vals.get)
    print(f"  + Safest following distance: {safety_winner}")


def save_scenario_results(
    scenario_name: str,
    results: Dict,
    results_dir: str
):
    """Save scenario results to files."""
    os.makedirs(results_dir, exist_ok=True)

    for ctrl_name, ctrl_results in results.items():
        if ctrl_results is None:
            continue

        # Save time series data
        data_file = os.path.join(
            results_dir,
            f"{scenario_name}_{ctrl_name.lower().replace('-', '_')}_data.csv"
        )
        df = pd.DataFrame(ctrl_results['data'])
        df.to_csv(data_file, index=False)
        logger.info(f"Saved: {data_file}")

        # Save metrics
        metrics_file = os.path.join(
            results_dir,
            f"{scenario_name}_{ctrl_name.lower().replace('-', '_')}_metrics.json"
        )
        with open(metrics_file, 'w') as f:
            # Convert numpy types for JSON serialization
            metrics_serializable = {}
            for cat, cat_metrics in ctrl_results['metrics'].items():
                metrics_serializable[cat] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in cat_metrics.items()
                }
            json.dump(metrics_serializable, f, indent=2)

    logger.info(f"Results saved to {results_dir}/")


def evaluate_all_scenarios(
    model_path: str,
    drl_only: bool = True,
    save_results: bool = True,
    results_dir: str = "results"
) -> Dict[str, Dict]:
    """
    Evaluate all controllers on multiple scenarios.

    Args:
        model_path: Path to trained DRL model
        drl_only: If True, only evaluate DRL-advantage scenarios
        save_results: Whether to save results
        results_dir: Directory for results

    Returns:
        Dictionary with results for all scenarios
    """
    if drl_only:
        scenarios = list(DRL_ADVANTAGE_SCENARIOS)
    else:
        scenarios = list(ALL_SCENARIOS.keys())

    print("\n" + "=" * 70)
    print(f"EVALUATING {len(scenarios)} SCENARIOS")
    print("=" * 70)

    all_results = {}

    for i, scenario_name in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}] {scenario_name}")
        results = evaluate_scenario(
            scenario_name=scenario_name,
            model_path=model_path,
            save_results=save_results,
            results_dir=results_dir
        )
        all_results[scenario_name] = results

        if i < len(scenarios) - 1:
            print("\nPausing 3 seconds before next scenario...")
            time.sleep(3.0)

    # Print aggregate summary
    print_aggregate_summary(all_results)

    return all_results


def print_aggregate_summary(all_results: Dict[str, Dict]):
    """Print aggregate results across all scenarios."""
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS ACROSS ALL SCENARIOS")
    print("=" * 70)
    print()

    # Collect wins
    wins = {'DRL-MPC': 0, 'Fixed-MPC': 0, 'ACC': 0}
    energy_data = []

    for scenario_name, results in all_results.items():
        scenario_energy = {}
        for ctrl_name in ['DRL-MPC', 'Fixed-MPC', 'ACC']:
            if results.get(ctrl_name) is not None:
                energy = results[ctrl_name]['metrics']['energy_throttle']['total_energy']
                scenario_energy[ctrl_name] = energy

        if scenario_energy:
            winner = min(scenario_energy, key=scenario_energy.get)
            wins[winner] += 1
            energy_data.append({
                'scenario': scenario_name,
                **scenario_energy,
                'winner': winner
            })

    # Print table
    print(f"{'Scenario':<25} {'DRL-MPC':>10} {'Fixed-MPC':>10} {'ACC':>10} {'Winner':>10}")
    print("-" * 70)

    for row in energy_data:
        line = f"{row['scenario']:<25}"
        for ctrl in ['DRL-MPC', 'Fixed-MPC', 'ACC']:
            if ctrl in row:
                line += f" {row[ctrl]:>10.2f}"
            else:
                line += f" {'N/A':>10}"
        line += f" {row['winner']:>10}"
        print(line)

    print("-" * 70)
    print(f"{'TOTAL WINS':<25} {wins['DRL-MPC']:>10} {wins['Fixed-MPC']:>10} {wins['ACC']:>10}")
    print()

    # Calculate improvements
    if energy_data:
        drl_energies = [r['DRL-MPC'] for r in energy_data if 'DRL-MPC' in r]
        mpc_energies = [r['Fixed-MPC'] for r in energy_data if 'Fixed-MPC' in r]
        acc_energies = [r['ACC'] for r in energy_data if 'ACC' in r]

        if drl_energies and mpc_energies:
            avg_improvement_mpc = np.mean([
                (m - d) / m * 100
                for d, m in zip(drl_energies, mpc_energies)
            ])
            print(f"DRL-MPC vs Fixed-MPC: {avg_improvement_mpc:+.1f}% energy")

        if drl_energies and acc_energies:
            avg_improvement_acc = np.mean([
                (a - d) / a * 100
                for d, a in zip(drl_energies, acc_energies)
            ])
            print(f"DRL-MPC vs ACC: {avg_improvement_acc:+.1f}% energy")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate DRL-MPC controller against baselines"
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained DRL model (.zip file)'
    )
    parser.add_argument(
        '--scenario', '-s',
        type=str,
        default=None,
        help='Specific scenario to evaluate'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Evaluate all scenarios'
    )
    parser.add_argument(
        '--drl-scenarios',
        action='store_true',
        help='Evaluate DRL-advantage scenarios only'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory for results (default: results)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to files'
    )

    args = parser.parse_args()

    # Check model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("HYBRID DRL-MPC CONTROLLER EVALUATION")
    print("=" * 70)
    print()
    print("Prerequisites:")
    print("1. CARLA server must be running (use start_carla.bat)")
    print()
    print(f"Model: {args.model}")
    print()

    input("Press ENTER when CARLA is ready...")

    if args.scenario:
        if args.scenario not in ALL_SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {', '.join(ALL_SCENARIOS.keys())}")
            sys.exit(1)
        evaluate_scenario(
            scenario_name=args.scenario,
            model_path=args.model,
            save_results=not args.no_save,
            results_dir=args.results_dir
        )
    elif args.drl_scenarios:
        evaluate_all_scenarios(
            model_path=args.model,
            drl_only=True,
            save_results=not args.no_save,
            results_dir=args.results_dir
        )
    else:
        # Default: run all scenarios
        evaluate_all_scenarios(
            model_path=args.model,
            drl_only=False,
            save_results=not args.no_save,
            results_dir=args.results_dir
        )


if __name__ == "__main__":
    main()
