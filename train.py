"""
DRL Training Script for Hybrid DRL-MPC Eco-Driving Controller.

This script trains a Soft Actor-Critic (SAC) agent to dynamically adjust
MPC cost function weights for optimal eco-driving performance.

Usage:
    python train.py                          # Train with default settings
    python train.py --timesteps 100000       # Train for specific timesteps
    python train.py --resume checkpoint.zip  # Resume from checkpoint
    python train.py --scenario multi_phase   # Train on specific scenario
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np

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
from utils.scenarios import get_scenario, get_all_scenarios, DRL_ADVANTAGE_SCENARIOS

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


def create_training_env(
    scenario_name: Optional[str] = None,
    max_episode_steps: int = 1000,
    log_dir: str = "logs"
) -> HybridMPCEnv:
    """
    Create the training environment.

    Args:
        scenario_name: Name of scenario to use (None for random)
        max_episode_steps: Maximum steps per episode
        log_dir: Directory for logs

    Returns:
        Wrapped training environment
    """
    # Get scenario trajectory
    if scenario_name:
        scenario = get_scenario(scenario_name)
        trajectory = scenario.trajectory
        max_episode_steps = scenario.num_steps
        logger.info(f"Using scenario: {scenario_name} ({scenario.duration_s:.1f}s)")
    else:
        # Create a varied training trajectory
        trajectory = create_varied_training_trajectory()
        logger.info("Using varied training trajectory")

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

    # Wrap with Monitor for logging
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    return env


def create_varied_training_trajectory() -> np.ndarray:
    """
    Create a varied training trajectory combining multiple patterns.

    This helps the agent learn to handle different driving situations.
    """
    dt = 0.05

    # Combine elements from different scenarios
    trajectory = np.concatenate([
        # Highway cruise with slight variations
        20.0 + 2.0 * np.sin(np.linspace(0, 2 * np.pi, 200)),

        # Gradual slowdown
        np.linspace(22.0, 10.0, 150),

        # Stop-and-go
        np.linspace(10.0, 3.0, 50),
        np.ones(30) * 3.0,
        np.linspace(3.0, 15.0, 80),

        # Traffic waves (sinusoidal)
        15.0 + 4.0 * np.sin(np.linspace(0, 4 * np.pi, 300)),

        # Emergency braking
        np.ones(50) * 20.0,
        np.linspace(20.0, 5.0, 40),
        np.ones(30) * 5.0,

        # Recovery
        np.linspace(5.0, 22.0, 100),

        # Final cruise
        np.ones(100) * 22.0,
    ])

    return trajectory


def create_sac_model(
    env,
    learning_rate: float = 3e-4,
    buffer_size: int = 100000,
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
    total_timesteps: int = 50000,
    scenario_name: Optional[str] = None,
    checkpoint_freq: int = 10000,
    resume_path: Optional[str] = None,
    save_path: str = "models",
    log_dir: str = "logs"
):
    """
    Main training function.

    Args:
        total_timesteps: Total training timesteps
        scenario_name: Scenario to train on (None for varied)
        checkpoint_freq: Checkpoint save frequency
        resume_path: Path to resume training from
        save_path: Directory to save models
        log_dir: Directory for logs
    """
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sac_{scenario_name or 'varied'}_{timestamp}"

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
        log_dir=log_dir
    )

    try:
        # Create or load model
        if resume_path and os.path.exists(resume_path):
            logger.info(f"\nResuming training from: {resume_path}")
            model = SAC.load(resume_path, env=env)
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
                progress_bar=True
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
        default=50000,
        help='Total training timesteps (default: 50000)'
    )
    parser.add_argument(
        '--scenario', '-s',
        type=str,
        default=None,
        help='Scenario to train on (default: varied training trajectory)'
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
    print(f"  Scenario: {args.scenario or 'varied (multiple patterns)'}")
    print(f"  Checkpoint freq: {args.checkpoint_freq}")
    print()

    input("Press ENTER when CARLA is ready to start training...")

    train(
        total_timesteps=args.timesteps,
        scenario_name=args.scenario,
        checkpoint_freq=args.checkpoint_freq,
        resume_path=args.resume,
        save_path=args.save_path,
        log_dir=args.log_dir
    )


if __name__ == "__main__":
    main()
