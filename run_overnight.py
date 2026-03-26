"""
Overnight Experiment Runner for DRL-MPC Paper.

Runs all experiments needed for the paper in sequence:
  1. Oracle grid search (optimal weights per EPA cycle)
  2. Main training: residual policy + curriculum (3 seeds)
  3. Ablation: absolute action space (1 seed)
  4. Ablation: no curriculum / uniform random (1 seed)
  5. Ablation: action_repeat=1 (every-step decisions, 1 seed)
  6. Evaluate all trained models on all EPA cycles

Usage:
    conda activate ecocar
    python run_overnight.py                # Full run (~8-12 hours)
    python run_overnight.py --quick        # Quick test (10k steps, resolution 5)
    python run_overnight.py --skip-oracle  # Skip grid search if already done
    python run_overnight.py --eval-only    # Only evaluate existing models

Output:
    results/oracle/       — Grid search CSVs and oracle summary
    models/               — All trained model .zip files
    results/eval/         — Evaluation results for all models
    results/overnight_summary.json — Final summary of all experiments
"""

import os
import sys
import json
import time
import argparse
import subprocess
import glob
from datetime import datetime


# Python executable — use current interpreter (should be ecocar env)
PYTHON = sys.executable

# Project root
ROOT = os.path.dirname(os.path.abspath(__file__))


def run_command(cmd: list, description: str, log_file: str = None) -> int:
    """
    Run a subprocess command with logging.

    Args:
        cmd: Command and arguments
        description: Human-readable description
        log_file: Optional file to capture stdout/stderr

    Returns:
        Return code (0 = success)
    """
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"  Command: {' '.join(cmd)}")
    if log_file:
        print(f"  Log: {log_file}")
    print(f"{'='*70}\n")

    start = time.time()

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd, cwd=ROOT,
                stdout=f, stderr=subprocess.STDOUT,
                text=True
            )
    else:
        result = subprocess.run(cmd, cwd=ROOT)

    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {description} ({elapsed/60:.1f} min)")

    return result.returncode


def run_oracle(resolution: int = 10) -> int:
    """Step 1: Oracle grid search."""
    return run_command(
        [PYTHON, "oracle_grid_search.py", "--resolution", str(resolution)],
        f"Oracle grid search (resolution={resolution})",
        log_file="results/oracle/oracle_log.txt"
    )


def run_training(
    name: str,
    timesteps: int,
    seed: int = 42,
    no_residual: bool = False,
    no_curriculum: bool = False,
    action_repeat: int = 20
) -> int:
    """Run a single training experiment."""
    cmd = [
        PYTHON, "train.py",
        "--name", name,
        "--timesteps", str(timesteps),
        "--action-repeat", str(action_repeat),
        "--seed", str(seed),
        "--checkpoint-freq", str(max(10000, timesteps // 10)),
    ]
    if no_residual:
        cmd.append("--no-residual")
    if no_curriculum:
        cmd.append("--no-curriculum")

    desc = f"Train: {name} (seed={seed}, {timesteps/1000:.0f}k steps)"
    return run_command(cmd, desc, log_file=f"logs/{name}_seed{seed}.txt")


def run_evaluation(model_path: str, results_dir: str) -> int:
    """Evaluate a trained model on all EPA cycles."""
    model_name = os.path.basename(model_path).replace('.zip', '')
    return run_command(
        [PYTHON, "evaluate.py",
         "--model", model_path,
         "--all",
         "--results-dir", results_dir],
        f"Evaluate: {model_name}",
        log_file=f"{results_dir}/{model_name}_eval_log.txt"
    )


def find_latest_model(pattern: str) -> str:
    """Find the most recently created model matching a glob pattern."""
    matches = glob.glob(os.path.join(ROOT, "models", pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Run all paper experiments overnight")
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run (10k steps, coarse grid)')
    parser.add_argument('--skip-oracle', action='store_true',
                        help='Skip oracle grid search')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate existing models in models/')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Override training timesteps')
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of seeds for main model (default: 1)')
    args = parser.parse_args()

    # Configuration
    # At ~11 fps (agent steps/s), 300k steps ≈ 7.5 hours per run.
    # Budget: oracle (~1h) + main (7.5h) + ablation (7.5h) + eval (~0.5h) ≈ 16.5h
    if args.quick:
        timesteps = 10000
        oracle_resolution = 5
        n_seeds = 1
    else:
        timesteps = args.timesteps or 300000
        oracle_resolution = 10
        n_seeds = args.seeds

    start_time = time.time()
    results = {}

    print("\n" + "#" * 70)
    print("#  OVERNIGHT EXPERIMENT RUNNER — DRL-MPC Paper")
    print(f"#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  Timesteps: {timesteps/1000:.0f}k | Seeds: {n_seeds} | Quick: {args.quick}")
    print("#" * 70)

    # =========================================================================
    # Step 1: Oracle Grid Search
    # =========================================================================
    if not args.skip_oracle and not args.eval_only:
        rc = run_oracle(oracle_resolution)
        results['oracle'] = 'OK' if rc == 0 else 'FAILED'
    else:
        print("\n[SKIP] Oracle grid search")
        results['oracle'] = 'SKIPPED'

    # =========================================================================
    # Step 2: Main training — residual + curriculum (multiple seeds)
    # =========================================================================
    if not args.eval_only:
        for seed in range(n_seeds):
            rc = run_training(
                name=f"main_residual_s{seed}",
                timesteps=timesteps,
                seed=seed,
            )
            results[f'main_seed{seed}'] = 'OK' if rc == 0 else 'FAILED'

    # =========================================================================
    # Step 3: Ablation — absolute action space (no residual)
    # This is the most important ablation: proves the residual architecture
    # matters. The other ablations (no-curriculum, action_repeat=1) are
    # optional and can be run separately if time permits.
    # =========================================================================
    if not args.eval_only:
        rc = run_training(
            name="ablation_absolute",
            timesteps=timesteps,
            seed=42,
            no_residual=True,
        )
        results['ablation_absolute'] = 'OK' if rc == 0 else 'FAILED'

    # =========================================================================
    # Step 6: Evaluate all models
    # =========================================================================
    eval_dir = "results/eval"
    os.makedirs(os.path.join(ROOT, eval_dir), exist_ok=True)

    # Find all models to evaluate
    model_patterns = [
        "main_residual_s*_final.zip",
        "ablation_absolute_*_final.zip",
    ]

    evaluated = []
    for pattern in model_patterns:
        model_path = find_latest_model(pattern)
        if model_path:
            rc = run_evaluation(model_path, eval_dir)
            model_name = os.path.basename(model_path)
            results[f'eval_{model_name}'] = 'OK' if rc == 0 else 'FAILED'
            evaluated.append(model_name)
        else:
            print(f"\n[SKIP] No model found for pattern: {pattern}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    total_time = time.time() - start_time

    print("\n" + "#" * 70)
    print("#  OVERNIGHT RUN COMPLETE")
    print(f"#  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  Total time: {total_time/3600:.1f} hours")
    print("#" * 70)

    print("\nResults:")
    for step, status in results.items():
        icon = "[+]" if status == 'OK' else "[-]" if status == 'FAILED' else "[~]"
        print(f"  {icon} {step}: {status}")

    print(f"\nModels evaluated: {len(evaluated)}")
    for m in evaluated:
        print(f"  - {m}")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_time_hours': total_time / 3600,
        'config': {
            'timesteps': timesteps,
            'n_seeds': n_seeds,
            'oracle_resolution': oracle_resolution,
        },
        'results': results,
        'evaluated_models': evaluated,
    }

    summary_path = os.path.join(ROOT, "results", "overnight_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Exit with error if any step failed
    if any(v == 'FAILED' for v in results.values()):
        print("\nWARNING: Some steps failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
