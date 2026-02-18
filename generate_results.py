"""
Results Processing and Visualization for Hybrid DRL-MPC Eco-Driving.

This script generates plots and summary tables from evaluation results.

Usage:
    python generate_results.py                    # Process all results in results/
    python generate_results.py --scenario multi_phase
"""

import os
import sys
import argparse
import glob
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10


def load_scenario_results(scenario_name: str, results_dir: str = "results") -> Dict:
    """
    Load results for a scenario from CSV and JSON files.

    Returns:
        Dictionary with data and metrics for each controller
    """
    results = {}
    controllers = ['drl_mpc', 'fixed_mpc', 'acc']
    controller_names = ['DRL-MPC', 'Fixed-MPC', 'ACC']

    for ctrl, name in zip(controllers, controller_names):
        data_file = os.path.join(results_dir, f"{scenario_name}_{ctrl}_data.csv")
        metrics_file = os.path.join(results_dir, f"{scenario_name}_{ctrl}_metrics.json")

        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            results[name] = {'data': df}

            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    results[name]['metrics'] = json.load(f)

    return results


def plot_scenario_comparison(
    scenario_name: str,
    results: Dict,
    save_path: Optional[str] = None
):
    """
    Generate multi-panel comparison plot for a scenario.

    Creates Figure 1 from the research: velocity, acceleration, and distance gap.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)

    colors = {
        'DRL-MPC': '#2E86AB',
        'Fixed-MPC': '#A23B72',
        'ACC': '#F18F01'
    }

    # Panel 1: Velocity
    ax1 = fig.add_subplot(gs[0])
    for name, data in results.items():
        if 'data' in data:
            df = data['data']
            ax1.plot(df['time'], df['ego_velocity'],
                    label=f'{name} (ego)', color=colors.get(name, 'gray'), linewidth=1.5)

    # Add lead velocity (from first controller that has it)
    for name, data in results.items():
        if 'data' in data and 'lead_velocity' in data['data'].columns:
            df = data['data']
            ax1.plot(df['time'], df['lead_velocity'],
                    label='Lead Vehicle', color='black', linewidth=1.5, linestyle='--')
            break

    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title(f'Scenario: {scenario_name}')
    ax1.legend(loc='upper right')
    ax1.set_xlim(left=0)

    # Panel 2: Acceleration
    ax2 = fig.add_subplot(gs[1])
    for name, data in results.items():
        if 'data' in data:
            df = data['data']
            ax2.plot(df['time'], df['acceleration'],
                    label=name, color=colors.get(name, 'gray'), linewidth=1.0, alpha=0.8)

    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.legend(loc='upper right')
    ax2.set_xlim(left=0)

    # Panel 3: Distance Gap
    ax3 = fig.add_subplot(gs[2])
    for name, data in results.items():
        if 'data' in data:
            df = data['data']
            ax3.plot(df['time'], df['distance_gap'],
                    label=name, color=colors.get(name, 'gray'), linewidth=1.5)

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Distance Gap (m)')
    ax3.legend(loc='upper right')
    ax3.set_xlim(left=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_drl_weights(
    scenario_name: str,
    results: Dict,
    save_path: Optional[str] = None
):
    """
    Generate Figure 2: DRL weight adjustments over time.
    """
    if 'DRL-MPC' not in results or 'data' not in results['DRL-MPC']:
        print("No DRL-MPC data available for weight plot")
        return None

    df = results['DRL-MPC']['data']

    if 'w_velocity' not in df.columns:
        print("No weight data in DRL-MPC results")
        return None

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(df['time'], 0, df['w_velocity'],
                   alpha=0.7, label='Velocity (w_v)', color='#2E86AB')
    ax.fill_between(df['time'], df['w_velocity'],
                   df['w_velocity'] + df['w_safety'],
                   alpha=0.7, label='Safety (w_s)', color='#E94F37')
    ax.fill_between(df['time'], df['w_velocity'] + df['w_safety'],
                   df['w_velocity'] + df['w_safety'] + df['w_comfort'],
                   alpha=0.7, label='Comfort (w_c)', color='#44AF69')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Weight')
    ax.set_title(f'DRL-MPC Weight Adaptation - {scenario_name}')
    ax.legend(loc='upper right')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_metrics_comparison(
    all_results: Dict[str, Dict],
    save_path: Optional[str] = None
):
    """
    Generate bar chart comparing metrics across scenarios.
    """
    scenarios = list(all_results.keys())
    controllers = ['DRL-MPC', 'Fixed-MPC', 'ACC']

    # Collect energy data
    energy_data = {ctrl: [] for ctrl in controllers}
    for scenario in scenarios:
        for ctrl in controllers:
            if ctrl in all_results[scenario] and all_results[scenario][ctrl]:
                metrics = all_results[scenario][ctrl].get('metrics', {})
                energy = metrics.get('energy_throttle', {}).get('total_energy', np.nan)
                energy_data[ctrl].append(energy)
            else:
                energy_data[ctrl].append(np.nan)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(scenarios))
    width = 0.25

    colors = ['#2E86AB', '#A23B72', '#F18F01']

    for i, (ctrl, color) in enumerate(zip(controllers, colors)):
        bars = ax.bar(x + i * width, energy_data[ctrl], width,
                     label=ctrl, color=color, alpha=0.8)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Total Energy')
    ax.set_title('Energy Consumption Comparison Across Scenarios')
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def generate_summary_table(
    all_results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate summary table of all results.
    """
    rows = []

    for scenario, results in all_results.items():
        row = {'Scenario': scenario}

        for ctrl in ['DRL-MPC', 'Fixed-MPC', 'ACC']:
            if ctrl in results and results[ctrl]:
                metrics = results[ctrl].get('metrics', {})
                energy = metrics.get('energy_throttle', {}).get('total_energy', np.nan)
                jerk = metrics.get('comfort', {}).get('rms_jerk', np.nan)
                thw = metrics.get('safety', {}).get('min_time_headway', np.nan)

                row[f'{ctrl} Energy'] = energy
                row[f'{ctrl} Jerk'] = jerk
                row[f'{ctrl} Min THW'] = thw
            else:
                row[f'{ctrl} Energy'] = np.nan
                row[f'{ctrl} Jerk'] = np.nan
                row[f'{ctrl} Min THW'] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

    return df


def generate_latex_table(df: pd.DataFrame, save_path: Optional[str] = None) -> str:
    """
    Generate LaTeX table from results DataFrame.
    """
    latex = df.to_latex(index=False, float_format="%.2f")

    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"Saved: {save_path}")

    return latex


def process_all_results(results_dir: str = "results", output_dir: str = "results/figures"):
    """
    Process all results and generate visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all scenarios with results
    data_files = glob.glob(os.path.join(results_dir, "*_data.csv"))
    scenarios = set()

    for f in data_files:
        basename = os.path.basename(f)
        # Extract scenario name (everything before _drl_mpc_, _fixed_mpc_, or _acc_)
        for suffix in ['_drl_mpc_data.csv', '_fixed_mpc_data.csv', '_acc_data.csv']:
            if basename.endswith(suffix):
                scenario = basename.replace(suffix, '')
                scenarios.add(scenario)
                break

    if not scenarios:
        print("No results found in", results_dir)
        return

    print(f"Found {len(scenarios)} scenarios: {', '.join(scenarios)}")
    print()

    all_results = {}

    for scenario in scenarios:
        print(f"Processing: {scenario}")
        results = load_scenario_results(scenario, results_dir)

        if results:
            all_results[scenario] = results

            # Generate scenario comparison plot
            plot_scenario_comparison(
                scenario,
                results,
                save_path=os.path.join(output_dir, f"{scenario}_comparison.png")
            )

            # Generate DRL weights plot
            plot_drl_weights(
                scenario,
                results,
                save_path=os.path.join(output_dir, f"{scenario}_weights.png")
            )

            plt.close('all')

    # Generate aggregate visualizations
    if all_results:
        print("\nGenerating aggregate results...")

        # Metrics comparison chart
        plot_metrics_comparison(
            all_results,
            save_path=os.path.join(output_dir, "energy_comparison.png")
        )

        # Summary table
        summary_df = generate_summary_table(
            all_results,
            save_path=os.path.join(results_dir, "summary_table.csv")
        )

        # LaTeX table
        generate_latex_table(
            summary_df,
            save_path=os.path.join(results_dir, "summary_table.tex")
        )

        # Print summary
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(summary_df.to_string(index=False))

        plt.close('all')

    print(f"\nFigures saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate results and visualizations"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing result files (default: results)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/figures',
        help='Directory for output figures (default: results/figures)'
    )
    parser.add_argument(
        '--scenario', '-s',
        type=str,
        default=None,
        help='Process specific scenario only'
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("GENERATING RESULTS AND VISUALIZATIONS")
    print("=" * 70)
    print()

    if args.scenario:
        # Process single scenario
        results = load_scenario_results(args.scenario, args.results_dir)
        if results:
            os.makedirs(args.output_dir, exist_ok=True)
            plot_scenario_comparison(
                args.scenario,
                results,
                save_path=os.path.join(args.output_dir, f"{args.scenario}_comparison.png")
            )
            plot_drl_weights(
                args.scenario,
                results,
                save_path=os.path.join(args.output_dir, f"{args.scenario}_weights.png")
            )
            plt.show()
        else:
            print(f"No results found for scenario: {args.scenario}")
    else:
        # Process all results
        process_all_results(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
