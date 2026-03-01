"""
DRL Training Script for Hybrid DRL-MPC Eco-Driving Controller.

This script trains a Soft Actor-Critic (SAC) agent to dynamically adjust
MPC cost function weights for optimal eco-driving performance.

Usage:
    python train.py                          # Train with default settings
    python train.py --timesteps 100000       # Train for specific timesteps
    python train.py --resume checkpoint.zip  # Resume from checkpoint
    python train.py --scenario udds           # Train on specific EPA cycle
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import gymnasium as gym

# Stable-baselines3 imports
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Project imports
from environment.gym_wrapper import HybridMPCEnv
from utils.scenarios import get_scenario, get_all_scenarios, DRL_ADVANTAGE_SCENARIOS, EASY_SCENARIOS
from utils.drive_cycles import get_cycle_trajectory, get_cycle_subsections

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback for logging training metrics.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Log metrics at specified frequency
        if self.n_calls % self.log_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_length = np.mean(self.episode_lengths[-10:])
                logger.info(
                    f"Step {self.n_calls}: "
                    f"Episodes={self.episode_count}, "
                    f"Mean Reward (last 10)={mean_reward:.2f}, "
                    f"Mean Length={mean_length:.0f}"
                )

        # Check for episode end
        if self.locals.get('dones') is not None:
            for idx, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][idx]
                    episode_reward = info.get('episode', {}).get('r', 0)
                    episode_length = info.get('episode', {}).get('l', 0)
                    if episode_reward != 0:
                        self.episode_rewards.append(episode_reward)
                        self.episode_lengths.append(episode_length)
                        self.episode_count += 1

        return True


class CurriculumScheduler(gym.Wrapper):
    """
    3-phase curriculum learning wrapper for EPA drive cycle scheduling.

    Progressively introduces harder drive cycles as training advances:

    Phase 1 — Warmup (0–20% of steps):
        HWFET only. Steady highway driving teaches stable car-following.

    Phase 2 — Main (20–70% of steps):
        UDDS + HWFET (70/30 mix). Stop-and-go city driving teaches
        adaptive weight adjustment.

    Phase 3 — Generalization (70–100% of steps):
        All 3 cycles equally (UDDS, HWFET, US06). Adds aggressive
        driving to generalize across all speed regimes.

    Uses subsectioned cycle windows (~300s each) for manageable episode
    lengths, with occasional full cycles in the all_pool.
    """

    def __init__(
        self,
        env: HybridMPCEnv,
        all_scenario_pool: list,
        drl_advantage_pool: list,
        easy_pool: list,
        total_timesteps: int
    ):
        """
        Args:
            env: The HybridMPCEnv to wrap
            all_scenario_pool: List of (trajectory, num_steps, name) for all scenarios
            drl_advantage_pool: Subset for DRL-advantage scenarios
            easy_pool: Subset for warmup (easy) scenarios
            total_timesteps: Total training timesteps (for phase boundaries)
        """
        super().__init__(env)
        self.all_pool = all_scenario_pool
        self.drl_advantage_pool = drl_advantage_pool
        self.easy_pool = easy_pool
        self.total_timesteps = total_timesteps

        # Phase boundaries (fraction of total_timesteps)
        self.phase1_end = int(total_timesteps * 0.20)
        self.phase2_end = int(total_timesteps * 0.70)

        # Tracking
        self.cumulative_steps = 0
        self.current_scenario_name = None
        self.current_num_steps = 0
        self.current_step = 0
        self.episode_num = 0
        self.current_phase = 1

        logger.info(f"CurriculumScheduler initialized:")
        logger.info(f"  Phase 1 (Warmup):         steps 0–{self.phase1_end} "
                     f"({len(easy_pool)} easy scenarios)")
        logger.info(f"  Phase 2 (Main):           steps {self.phase1_end}–{self.phase2_end} "
                     f"({len(drl_advantage_pool)} DRL-advantage + {len(all_scenario_pool)} all)")
        logger.info(f"  Phase 3 (Generalization): steps {self.phase2_end}–{total_timesteps} "
                     f"({len(all_scenario_pool)} all scenarios)")

    def _get_current_phase(self) -> int:
        """Determine curriculum phase from cumulative step count."""
        if self.cumulative_steps < self.phase1_end:
            return 1
        elif self.cumulative_steps < self.phase2_end:
            return 2
        else:
            return 3

    def _select_scenario(self, phase: int):
        """Select a scenario based on the current curriculum phase."""
        if phase == 1:
            # Warmup: only easy scenarios
            pool = self.easy_pool
        elif phase == 2:
            # Main: 70% DRL-advantage, 30% all scenarios
            if np.random.random() < 0.7:
                pool = self.drl_advantage_pool
            else:
                pool = self.all_pool
        else:
            # Generalization: all scenarios equally
            pool = self.all_pool

        idx = np.random.randint(len(pool))
        return pool[idx]

    def reset(self, **kwargs):
        self.current_phase = self._get_current_phase()
        trajectory, num_steps, name = self._select_scenario(self.current_phase)

        self.current_scenario_name = name
        self.current_num_steps = num_steps
        self.current_step = 0
        self.episode_num += 1

        # Swap trajectory and episode length on the CARLA env
        self.env.carla_env.lead_trajectory = trajectory
        self.env.carla_env.max_episode_steps = num_steps
        self.env.max_episode_steps = num_steps

        # Console output with phase info
        phase_names = {1: 'Warmup', 2: 'Main', 3: 'Generalization'}
        logger.info(
            f"[Episode {self.episode_num}] Phase {self.current_phase} "
            f"({phase_names[self.current_phase]}) | Scenario: {name} "
            f"({num_steps} steps, {num_steps * 0.05:.0f}s) | "
            f"Cumulative: {self.cumulative_steps}/{self.total_timesteps}"
        )

        obs, info = self.env.reset(**kwargs)

        # Draw scenario name in CARLA world above ego vehicle
        self._draw_scenario_label(name, num_steps)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # HybridMPCEnv runs action_repeat sim sub-steps per agent step.
        # Track sim-level steps for episode boundary control.
        sim_steps = info.get('sim_steps', 1)
        self.current_step += sim_steps
        self.cumulative_steps += sim_steps

        # Only allow truncation when the full trajectory has been played
        if truncated and self.current_step < self.current_num_steps:
            truncated = False

        # Force truncation when trajectory is complete (even if CARLA didn't)
        if not terminated and self.current_step >= self.current_num_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def _draw_scenario_label(self, name: str, num_steps: int):
        """Draw the current scenario name as debug text in CARLA."""
        try:
            import carla
            carla_env = self.env.carla_env
            if carla_env.ego_vehicle is not None and carla_env.world is not None:
                ego_loc = carla_env.ego_vehicle.get_location()
                label_loc = carla.Location(
                    x=ego_loc.x, y=ego_loc.y, z=ego_loc.z + 4.0
                )
                phase_names = {1: 'Warmup', 2: 'Main', 3: 'Generalize'}
                label = f"P{self.current_phase}({phase_names[self.current_phase]}): {name}"
                carla_env.world.debug.draw_string(
                    label_loc,
                    label,
                    draw_shadow=True,
                    color=carla.Color(0, 255, 0),
                    life_time=num_steps * 0.05 + 5.0
                )
        except Exception:
            pass  # Don't crash training if debug draw fails


def build_training_pools(
    dt: float = 0.05,
    window_s: float = 300.0,
    overlap_s: float = 30.0
) -> tuple:
    """
    Build subsectioned training pools from EPA drive cycles.

    Full EPA cycles are too long for training episodes (e.g. UDDS = 27,380
    steps at dt=0.05). This function splits each cycle into overlapping
    ~300s windows and also includes full cycles for occasional use.

    Args:
        dt: Simulation timestep
        window_s: Subsection window duration in seconds
        overlap_s: Overlap between windows in seconds

    Returns:
        (all_pool, drl_advantage_pool, easy_pool) — each is a list of
        (trajectory, num_steps, name) tuples
    """
    all_pool = []
    drl_advantage_pool = []
    easy_pool = []

    cycle_names = ['udds', 'hwfet', 'us06']

    for cycle_name in cycle_names:
        # Add subsections for training
        subsections = get_cycle_subsections(cycle_name, dt, window_s, overlap_s)
        for traj, dur, sub_name in subsections:
            entry = (traj, len(traj), sub_name)
            all_pool.append(entry)

            if cycle_name in DRL_ADVANTAGE_SCENARIOS:
                drl_advantage_pool.append(entry)
            if cycle_name in EASY_SCENARIOS:
                easy_pool.append(entry)

        # Also add the full cycle to the all_pool for occasional full-episode training
        full_traj, full_dur = get_cycle_trajectory(cycle_name, dt)
        all_pool.append((full_traj, len(full_traj), f"{cycle_name}_full"))

    return all_pool, drl_advantage_pool, easy_pool


def create_training_env(
    scenario_name: Optional[str] = None,
    max_episode_steps: int = 1000,
    total_timesteps: int = 500000,
    log_dir: str = "logs"
) -> gym.Env:
    """
    Create the training environment with curriculum learning over EPA cycles.

    When no scenario is specified, uses a 3-phase curriculum scheduler:
    - Phase 1 (Warmup, 0-20%): HWFET subsections for stable car-following
    - Phase 2 (Main, 20-70%): UDDS + HWFET for adaptive weight learning
    - Phase 3 (Generalization, 70-100%): All cycles including aggressive US06

    Args:
        scenario_name: Name of a single scenario to use (None for curriculum)
        max_episode_steps: Maximum steps per episode (used as initial default)
        total_timesteps: Total training timesteps (for curriculum phase boundaries)
        log_dir: Directory for logs

    Returns:
        Wrapped training environment
    """
    if scenario_name:
        # Single scenario mode
        scenario = get_scenario(scenario_name)
        trajectory = scenario.trajectory
        max_episode_steps = scenario.num_steps
        logger.info(f"Using scenario: {scenario_name} ({scenario.duration_s:.1f}s)")
    else:
        # Use the longest subsection as the initial trajectory (will be swapped on reset)
        all_pool, _, _ = build_training_pools()
        longest = max(all_pool, key=lambda x: x[1])
        trajectory = longest[0]
        max_episode_steps = longest[1]

    # Create environment
    env = HybridMPCEnv(
        carla_host='localhost',
        carla_port=2000,
        dt=0.05,
        mpc_horizon=20,
        max_episode_steps=max_episode_steps,
        target_velocity=20.0,
        lead_trajectory=trajectory,
        map_name='Town04'
    )

    # If no specific scenario, wrap with curriculum scheduler
    if not scenario_name:
        all_pool, drl_advantage_pool, easy_pool = build_training_pools()

        logger.info(f"Training with {len(all_pool)} trajectory segments (curriculum learning):")
        for traj, steps, name in all_pool:
            phase_tags = []
            # Determine which phase(s) this segment belongs to
            base_cycle = name.split('_')[0]
            if base_cycle in EASY_SCENARIOS:
                phase_tags.append("P1-warmup")
            if base_cycle in DRL_ADVANTAGE_SCENARIOS:
                phase_tags.append("P2-main")
            phase_tags.append("P3-all")
            logger.info(f"  {name}: {steps} steps ({steps*0.05:.1f}s) [{', '.join(phase_tags)}]")

        env = CurriculumScheduler(
            env,
            all_scenario_pool=all_pool,
            drl_advantage_pool=drl_advantage_pool,
            easy_pool=easy_pool,
            total_timesteps=total_timesteps
        )

    # Wrap with Monitor for logging
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    return env


def create_sac_model(
    env,
    learning_rate: float = 3e-4,
    buffer_size: int = 300000,
    batch_size: int = 256,
    tau: float = 0.005,
    gamma: float = 0.99,
    verbose: int = 1
) -> SAC:
    """
    Create and configure SAC model.

    Args:
        env: Training environment
        learning_rate: Learning rate
        buffer_size: Replay buffer size
        batch_size: Batch size for training
        tau: Soft update coefficient
        gamma: Discount factor
        verbose: Verbosity level

    Returns:
        Configured SAC model
    """
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_entropy='auto',
        verbose=verbose,
        tensorboard_log="logs/tensorboard/"
    )

    logger.info("SAC model created with parameters:")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Buffer size: {buffer_size}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gamma: {gamma}")
    logger.info(f"  Tau: {tau}")

    return model


def train(
    total_timesteps: int = 500000,
    scenario_name: Optional[str] = None,
    checkpoint_freq: int = 10000,
    resume_path: Optional[str] = None,
    save_path: str = "models",
    log_dir: str = "logs",
    model_name: Optional[str] = None
):
    """
    Main training function.

    Args:
        total_timesteps: Total training timesteps (default: 500k)
        scenario_name: Scenario to train on (None for curriculum learning)
        checkpoint_freq: Checkpoint save frequency
        resume_path: Path to resume training from
        save_path: Directory to save models
        log_dir: Directory for logs
        model_name: User-provided name for the model (optional)
    """
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create run name from user input or default
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_name:
        run_name = f"{model_name}_{timestamp}"
    else:
        run_name = f"sac_{scenario_name or 'curriculum'}_{timestamp}"

    logger.info("=" * 60)
    logger.info("HYBRID DRL-MPC TRAINING")
    logger.info("=" * 60)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Total timesteps: {total_timesteps}")
    logger.info(f"Checkpoint frequency: {checkpoint_freq}")

    # Create environment
    logger.info("\nCreating training environment...")
    env = create_training_env(
        scenario_name=scenario_name,
        total_timesteps=total_timesteps,
        log_dir=log_dir
    )

    try:
        # Create or load model
        reset_num_timesteps = True
        if resume_path and os.path.exists(resume_path):
            logger.info(f"\nResuming training from: {resume_path}")
            model = SAC.load(resume_path, env=env)
            reset_num_timesteps = False

            # Extract resumed timestep count from the loaded model
            resumed_steps = model.num_timesteps
            logger.info(f"Resumed at timestep {resumed_steps}/{total_timesteps}")

            # Sync curriculum scheduler so it starts at the correct phase
            if hasattr(env, 'env') and hasattr(env.env, 'cumulative_steps'):
                env.env.cumulative_steps = resumed_steps
                logger.info(f"Curriculum scheduler synced to step {resumed_steps} "
                            f"(Phase {env.env._get_current_phase()})")
        else:
            logger.info("\nCreating new SAC model...")
            model = create_sac_model(env)

        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=save_path,
            name_prefix=run_name
        )

        metrics_callback = TrainingMetricsCallback(log_freq=1000)

        callbacks = CallbackList([checkpoint_callback, metrics_callback])

        # Train
        logger.info("\nStarting training...")
        logger.info("Press Ctrl+C to stop training early (model will be saved)")
        print()

        start_time = time.time()

        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=10,
                progress_bar=True,
                reset_num_timesteps=reset_num_timesteps
            )
        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")

        training_time = time.time() - start_time
        logger.info(f"\nTraining completed in {training_time/60:.1f} minutes")

        # Save final model
        final_path = os.path.join(save_path, f"{run_name}_final.zip")
        model.save(final_path)
        logger.info(f"Final model saved to: {final_path}")

        # Print training summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total timesteps: {model.num_timesteps}")
        print(f"Episodes completed: {metrics_callback.episode_count}")
        if metrics_callback.episode_rewards:
            print(f"Mean episode reward: {np.mean(metrics_callback.episode_rewards):.2f}")
            print(f"Best episode reward: {max(metrics_callback.episode_rewards):.2f}")
        print(f"Training time: {training_time/60:.1f} minutes")
        print(f"Model saved to: {final_path}")
        print("=" * 60)

        return model

    finally:
        env.close()
        logger.info("Environment closed")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train DRL agent for hybrid DRL-MPC eco-driving"
    )
    parser.add_argument(
        '--timesteps', '-t',
        type=int,
        default=500000,
        help='Total training timesteps (default: 500000)'
    )
    parser.add_argument(
        '--scenario', '-s',
        type=str,
        default=None,
        help='EPA cycle to train on (udds/hwfet/us06, default: curriculum)'
    )
    parser.add_argument(
        '--resume', '-r',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=10000,
        help='Checkpoint save frequency (default: 10000)'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='models',
        help='Directory to save models (default: models)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for logs (default: logs)'
    )

    args = parser.parse_args()

    # Validate scenario if provided
    if args.scenario:
        all_scenarios = get_all_scenarios()
        if args.scenario not in all_scenarios:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available scenarios: {', '.join(all_scenarios.keys())}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("HYBRID DRL-MPC ECO-DRIVING TRAINING")
    print("=" * 60)
    print()
    print("Prerequisites:")
    print("1. CARLA server must be running (use start_carla.bat)")
    print("2. Ensure stable-baselines3 is installed")
    print()
    print("Training Configuration:")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Scenario: {args.scenario or 'curriculum (3-phase learning)'}")
    print(f"  Checkpoint freq: {args.checkpoint_freq}")
    print()

    model_name = input("Enter a name for this model: ").strip()
    if not model_name:
        model_name = None

    input("Press ENTER when CARLA is ready to start training...")

    train(
        total_timesteps=args.timesteps,
        scenario_name=args.scenario,
        checkpoint_freq=args.checkpoint_freq,
        resume_path=args.resume,
        save_path=args.save_path,
        log_dir=args.log_dir,
        model_name=model_name
    )


if __name__ == "__main__":
    main()
