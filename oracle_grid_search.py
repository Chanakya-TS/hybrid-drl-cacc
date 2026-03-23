"""
Oracle Grid Search for Optimal MPC Weights per EPA Drive Cycle.

Runs a grid search over the MPC weight simplex (w_v, w_s, w_c) for each
EPA cycle (UDDS, HWFET, US06) to find the energy-optimal weights per scenario.

Purpose:
    1. Prove that optimal weights DIFFER across scenarios (justifies adaptation)
    2. Establish an oracle upper bound for DRL-MPC performance
    3. Show the gap between Fixed-MPC and oracle (room for DRL improvement)

Usage:
    python oracle_grid_search.py                   # Default 10-point grid
    python oracle_grid_search.py --resolution 15   # Finer 15-point grid
    python oracle_grid_search.py --scenario udds    # Single scenario
"""

import os
import sys
import argparse
import json
import time
import logging
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from environment.car_following import CarFollowingEnv
from controllers.mpc_controller import MPCController
from utils.metrics import calculate_all_metrics
from utils.scenarios import get_scenario, ALL_SCENARIOS

logging.basicConfig(
    level=logging.WARNING,  # Suppress MPC solver spam
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_weight_grid(resolution: int = 10) -> List[Tuple[float, float, float]]:
    """
    Generate a grid of (w_v, w_s, w_c) triples on the weight simplex.

    Points are sampled uniformly on the simplex: w_v + w_s + w_c = 1,
    with a minimum of 0.05 per weight to avoid degenerate MPC behavior.

    Args:
        resolution: Number of divisions per axis (total points ~ resolution²/2)

    Returns:
        List of (w_v, w_s, w_c) tuples summing to 1.0
    """
    grid = []
    min_w = 0.05
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            w_v = min_w + (1.0 - 3 * min_w) * i / resolution
            w_s = min_w + (1.0 - 3 * min_w) * j / resolution
            w_c = min_w + (1.0 - 3 * min_w) * k / resolution
            # Normalize to exactly 1.0
            total = w_v + w_s + w_c
            grid.append((w_v / total, w_s / total, w_c / total))
    return grid


def run_fixed_mpc_episode(
    scenario_name: str,
    w_velocity: float,
    w_safety: float,
    w_comfort: float,
    dt: float = 0.05
) -> Dict:
    """
    Run a single episode with fixed MPC weights and return metrics.

    Args:
        scenario_name: EPA cycle name
        w_velocity, w_safety, w_comfort: MPC weight triple
        dt: Simulation timestep

    Returns:
        Dictionary with energy and key metrics
    """
    scenario = get_scenario(scenario_name, dt)

    env = CarFollowingEnv(
        dt=scenario.dt,
        lead_vehicle_trajectory=scenario.trajectory,
        max_episode_steps=scenario.num_steps,
    )

    mpc = MPCController(
        dt=scenario.dt,
        horizon=20,
        w_velocity=w_velocity,
        w_safety=w_safety,
        w_comfort=w_comfort,
        target_velocity=20.0
    )

    obs, _ = env.reset(seed=42)
    mpc.reset()

    episode_data = {
        'ego_velocity': [], 'lead_velocity': [], 'distance_gap': [],
        'acceleration': [], 'throttle': [], 'brake': [], 'time_headway': []
    }

    done = False
    while not done:
        ego_velocity = obs[0]
        lead_velocities = [
            ego_velocity + obs[1],
            ego_velocity + obs[4],
            ego_velocity + obs[7]
        ]
        distance_gaps = [obs[2], obs[5], obs[8]]
        lead_accelerations = [obs[3], obs[6], obs[9]]

        acceleration, _ = mpc.compute_control(
            ego_velocity=ego_velocity,
            lead_velocities=lead_velocities,
            distance_gaps=distance_gaps,
            lead_accelerations=lead_accelerations
        )

        obs, _, terminated, truncated, info = env.step(acceleration)
        done = terminated or truncated

        episode_data['ego_velocity'].append(ego_velocity)
        episode_data['lead_velocity'].append(lead_velocities[0])
        episode_data['distance_gap'].append(distance_gaps[0])
        episode_data['acceleration'].append(acceleration)
        episode_data['throttle'].append(info.get('throttle', 0.0))
        episode_data['brake'].append(info.get('brake', 0.0))
        episode_data['time_headway'].append(info.get('time_headway', float('inf')))

    env.close()

    metrics = calculate_all_metrics(
        episode_data=episode_data,
        dt=scenario.dt,
        target_velocity=20.0,
        vehicle_mass=1800.0
    )

    return {
        'total_energy': metrics['energy_throttle']['total_energy'],
        'energy_per_km': metrics['energy_throttle']['energy_per_km'],
        'rms_jerk': metrics['comfort']['rms_jerk'],
        'min_thw': metrics['safety']['min_time_headway'],
        'avg_velocity': metrics['velocity']['avg_ego_velocity'],
        'collision': metrics['summary']['collision'],
    }


def run_grid_search(
    scenario_name: str,
    grid: List[Tuple[float, float, float]],
    dt: float = 0.05
) -> pd.DataFrame:
    """
    Run grid search for a single scenario.

    Args:
        scenario_name: EPA cycle name
        grid: List of weight triples
        dt: Simulation timestep

    Returns:
        DataFrame with results for each weight combination
    """
    results = []
    total = len(grid)

    print(f"\n{'='*70}")
    print(f"GRID SEARCH: {scenario_name.upper()} ({total} weight combinations)")
    print(f"{'='*70}")

    start_time = time.time()

    for idx, (w_v, w_s, w_c) in enumerate(grid):
        metrics = run_fixed_mpc_episode(scenario_name, w_v, w_s, w_c, dt)

        results.append({
            'w_velocity': round(w_v, 4),
            'w_safety': round(w_s, 4),
            'w_comfort': round(w_c, 4),
            **metrics
        })

        # Progress update every 10%
        if (idx + 1) % max(1, total // 10) == 0 or idx == total - 1:
            elapsed = time.time() - start_time
            eta = elapsed / (idx + 1) * (total - idx - 1)
            best_so_far = min(r['total_energy'] for r in results if not r['collision'])
            print(
                f"  [{idx+1:>4}/{total}] "
                f"w=({w_v:.2f},{w_s:.2f},{w_c:.2f}) "
                f"E={metrics['total_energy']:.1f} "
                f"best={best_so_far:.1f} "
                f"ETA={eta:.0f}s"
            )

    df = pd.DataFrame(results)
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    return df


def analyze_results(
    all_results: Dict[str, pd.DataFrame],
    fixed_weights: Tuple[float, float, float] = (0.25, 0.35, 0.40)
) -> Dict:
    """
    Analyze grid search results: find oracle weights, compare to Fixed-MPC.

    Args:
        all_results: Dict mapping scenario name to results DataFrame
        fixed_weights: The Fixed-MPC baseline weights

    Returns:
        Summary dictionary
    """
    print("\n" + "=" * 70)
    print("ORACLE ANALYSIS RESULTS")
    print("=" * 70)

    summary = {}

    for scenario_name, df in all_results.items():
        # Filter out collisions
        safe = df[~df['collision']]
        if safe.empty:
            print(f"\n{scenario_name}: ALL COMBINATIONS CAUSED COLLISIONS")
            continue

        # Find oracle (minimum energy with no collision)
        oracle_idx = safe['total_energy'].idxmin()
        oracle = safe.loc[oracle_idx]

        # Find Fixed-MPC result (closest to fixed_weights)
        distances = (
            (df['w_velocity'] - fixed_weights[0])**2 +
            (df['w_safety'] - fixed_weights[1])**2 +
            (df['w_comfort'] - fixed_weights[2])**2
        )
        fixed_idx = distances.idxmin()
        fixed = df.loc[fixed_idx]

        improvement = (fixed['total_energy'] - oracle['total_energy']) / fixed['total_energy'] * 100

        summary[scenario_name] = {
            'oracle_weights': (oracle['w_velocity'], oracle['w_safety'], oracle['w_comfort']),
            'oracle_energy': oracle['total_energy'],
            'fixed_energy': fixed['total_energy'],
            'improvement_pct': improvement,
            'oracle_jerk': oracle['rms_jerk'],
            'fixed_jerk': fixed['rms_jerk'],
            'oracle_min_thw': oracle['min_thw'],
        }

        print(f"\n--- {scenario_name.upper()} ---")
        print(f"  Oracle weights: w_v={oracle['w_velocity']:.3f}, w_s={oracle['w_safety']:.3f}, w_c={oracle['w_comfort']:.3f}")
        print(f"  Oracle energy:  {oracle['total_energy']:.2f}")
        print(f"  Fixed energy:   {fixed['total_energy']:.2f}")
        print(f"  Improvement:    {improvement:+.1f}%")
        print(f"  Oracle jerk:    {oracle['rms_jerk']:.4f} m/s³")
        print(f"  Oracle min THW: {oracle['min_thw']:.2f} s")

    # Key question: do oracle weights differ across scenarios?
    print("\n" + "=" * 70)
    print("KEY FINDING: Do optimal weights differ across scenarios?")
    print("=" * 70)

    if len(summary) >= 2:
        oracle_weights_list = [v['oracle_weights'] for v in summary.values()]
        scenarios = list(summary.keys())

        # Compute pairwise weight distances
        max_dist = 0
        for i in range(len(scenarios)):
            for j in range(i + 1, len(scenarios)):
                dist = np.sqrt(sum(
                    (a - b)**2
                    for a, b in zip(oracle_weights_list[i], oracle_weights_list[j])
                ))
                max_dist = max(max_dist, dist)
                print(f"  {scenarios[i]} vs {scenarios[j]}: weight distance = {dist:.4f}")

        if max_dist > 0.05:
            print(f"\n  >> YES — optimal weights are scenario-dependent (max distance: {max_dist:.4f})")
            print(f"  >> This justifies adaptive weight tuning via DRL.")
        else:
            print(f"\n  >> NO — optimal weights are similar across scenarios (max distance: {max_dist:.4f})")
            print(f"  >> Fixed-MPC is near-optimal; adaptation has limited value.")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (for paper)")
    print("=" * 70)
    print(f"{'Scenario':<12} {'Oracle w_v':>10} {'Oracle w_s':>10} {'Oracle w_c':>10} "
          f"{'Oracle E':>10} {'Fixed E':>10} {'Improve':>10}")
    print("-" * 74)
    for name, s in summary.items():
        w = s['oracle_weights']
        print(f"{name:<12} {w[0]:>10.3f} {w[1]:>10.3f} {w[2]:>10.3f} "
              f"{s['oracle_energy']:>10.2f} {s['fixed_energy']:>10.2f} "
              f"{s['improvement_pct']:>+9.1f}%")

    avg_improvement = np.mean([s['improvement_pct'] for s in summary.values()])
    print("-" * 74)
    print(f"{'AVG':>54} {avg_improvement:>+9.1f}%")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Oracle grid search for optimal MPC weights per EPA cycle"
    )
    parser.add_argument(
        '--resolution', '-r', type=int, default=10,
        help='Grid resolution (default 10, ~66 points per scenario)'
    )
    parser.add_argument(
        '--scenario', '-s', type=str, default=None,
        help='Run single scenario (udds, hwfet, us06). Default: all'
    )
    parser.add_argument(
        '--results-dir', type=str, default='results/oracle',
        help='Directory for results'
    )
    args = parser.parse_args()

    # Generate weight grid
    grid = generate_weight_grid(args.resolution)
    print(f"Generated {len(grid)} weight combinations (resolution={args.resolution})")

    # Select scenarios
    if args.scenario:
        scenarios = [args.scenario]
    else:
        scenarios = list(ALL_SCENARIOS.keys())

    # Run grid search
    all_results = {}
    for scenario_name in scenarios:
        df = run_grid_search(scenario_name, grid)
        all_results[scenario_name] = df

    # Analyze
    summary = analyze_results(all_results)

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for scenario_name, df in all_results.items():
        csv_path = os.path.join(args.results_dir, f"grid_{scenario_name}_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    summary_path = os.path.join(args.results_dir, f"oracle_summary_{timestamp}.json")
    # Convert tuples to lists for JSON
    json_summary = {}
    for k, v in summary.items():
        json_summary[k] = {
            **v,
            'oracle_weights': list(v['oracle_weights'])
        }
    with open(summary_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
