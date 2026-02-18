"""
Gymnasium Wrapper for Hybrid DRL-MPC Eco-Driving Environment.

This module wraps the CARLA CarFollowingEnv and MPC controller into a
Gymnasium-compatible environment for training DRL agents.

The DRL agent learns to dynamically adjust MPC cost function weights
based on the current driving situation.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import logging

from environment.car_following import CarFollowingEnv
from controllers.mpc_controller import MPCController

logger = logging.getLogger(__name__)


class HybridMPCEnv(gym.Env):
    """
    Gymnasium environment for hybrid DRL-MPC eco-driving control.

    The DRL agent observes the driving state and outputs MPC weight adjustments.
    The MPC controller then computes optimal acceleration using these weights.

    Observation Space:
        - ego_velocity (normalized): Current ego vehicle velocity
        - relative_velocity (normalized): Velocity difference with lead vehicle
        - distance_gap (normalized): Distance to lead vehicle
        - current_weights (3 values): Current MPC weights [w_v, w_s, w_c]

    Action Space:
        - weight_adjustments (3 values): New MPC weights [w_v, w_s, w_c]
          Continuous values in [0, 1], normalized to sum to 1
    """

    metadata = {'render_modes': ['human']}

    # Normalization constants
    MAX_VELOCITY = 30.0  # m/s
    MAX_REL_VELOCITY = 20.0  # m/s
    MAX_DISTANCE = 100.0  # m

    # Reward weights
    REWARD_ENERGY_WEIGHT = 1.0
    REWARD_SAFETY_WEIGHT = 2.0
    REWARD_COMFORT_WEIGHT = 0.5
    COLLISION_PENALTY = -100.0

    def __init__(
        self,
        carla_host: str = 'localhost',
        carla_port: int = 2000,
        dt: float = 0.05,
        mpc_horizon: int = 20,
        max_episode_steps: int = 1000,
        target_velocity: float = 20.0,
        lead_trajectory: Optional[np.ndarray] = None,
        map_name: str = 'Town04',
        render_mode: Optional[str] = None
    ):
        """
        Initialize the hybrid DRL-MPC environment.

        Args:
            carla_host: CARLA server hostname
            carla_port: CARLA server port
            dt: Simulation timestep
            mpc_horizon: MPC prediction horizon
            max_episode_steps: Maximum steps per episode
            target_velocity: Target cruising velocity
            lead_trajectory: Velocity profile for lead vehicle
            map_name: CARLA map name
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__()

        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.target_velocity = target_velocity
        self.render_mode = render_mode

        # Create CARLA environment
        self.carla_env = CarFollowingEnv(
            carla_host=carla_host,
            carla_port=carla_port,
            dt=dt,
            lead_vehicle_trajectory=lead_trajectory,
            max_episode_steps=max_episode_steps,
            map_name=map_name
        )

        # Create MPC controller
        self.mpc = MPCController(
            dt=dt,
            horizon=mpc_horizon,
            w_velocity=0.33,
            w_safety=0.34,
            w_comfort=0.33,
            target_velocity=target_velocity
        )

        # Define observation space
        # [ego_vel, rel_vel, distance, w_v, w_s, w_c] - all normalized to [0,1] or [-1,1]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Define action space: MPC weights [w_velocity, w_safety, w_comfort]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Episode tracking
        self.current_step = 0
        self.previous_velocity = 0.0
        self.previous_acceleration = 0.0
        self.total_energy = 0.0

        # Weight history for logging
        self.weight_history = []

        logger.info("HybridMPCEnv initialized")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            observation: Initial normalized observation
            info: Additional information
        """
        super().reset(seed=seed)

        # Reset CARLA environment
        carla_obs, carla_info = self.carla_env.reset(seed=seed)

        # Reset MPC controller
        self.mpc.reset()

        # Reset episode tracking
        self.current_step = 0
        self.previous_velocity = carla_obs[0]
        self.previous_acceleration = 0.0
        self.total_energy = 0.0
        self.weight_history = []

        # Build observation
        observation = self._build_observation(carla_obs)

        info = {
            'carla_info': carla_info,
            'mpc_weights': [self.mpc.w_velocity, self.mpc.w_safety, self.mpc.w_comfort]
        }

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.

        Args:
            action: New MPC weights [w_velocity, w_safety, w_comfort]

        Returns:
            observation: New normalized observation
            reward: Reward for this step
            terminated: Whether episode ended (collision)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        # Update MPC weights based on DRL action
        w_v, w_s, w_c = action
        self.mpc.set_weights(w_v, w_s, w_c)
        self.weight_history.append([self.mpc.w_velocity, self.mpc.w_safety, self.mpc.w_comfort])

        # Get current state from last observation
        # We need to get the raw observation from CARLA
        raw_obs = self.carla_env._get_observation()
        ego_velocity, relative_velocity, distance_gap = raw_obs

        lead_velocity = ego_velocity + relative_velocity

        # Compute acceleration using MPC
        acceleration, mpc_info = self.mpc.compute_control(
            ego_velocity=ego_velocity,
            lead_velocity=lead_velocity,
            distance_gap=distance_gap
        )

        # Step CARLA environment with computed acceleration
        carla_obs, _, terminated, truncated, carla_info = self.carla_env.step(acceleration)

        self.current_step += 1

        # Compute reward
        reward = self._compute_reward(carla_obs, acceleration, carla_info)

        # Build observation
        observation = self._build_observation(carla_obs)

        # Update tracking variables
        self.previous_acceleration = acceleration
        self.previous_velocity = carla_obs[0]

        info = {
            'carla_info': carla_info,
            'mpc_info': mpc_info,
            'mpc_weights': [self.mpc.w_velocity, self.mpc.w_safety, self.mpc.w_comfort],
            'acceleration': acceleration,
            'step': self.current_step
        }

        return observation, reward, terminated, truncated, info

    def _build_observation(self, carla_obs: np.ndarray) -> np.ndarray:
        """
        Build normalized observation for DRL agent.

        Args:
            carla_obs: Raw observation [ego_vel, rel_vel, distance]

        Returns:
            Normalized observation vector
        """
        ego_velocity, relative_velocity, distance_gap = carla_obs

        # Normalize observations
        norm_ego_vel = np.clip(ego_velocity / self.MAX_VELOCITY, 0.0, 1.0)
        norm_rel_vel = np.clip(relative_velocity / self.MAX_REL_VELOCITY, -1.0, 1.0)
        norm_distance = np.clip(distance_gap / self.MAX_DISTANCE, 0.0, 1.0)

        # Current weights (already normalized to sum to 1)
        w_v = self.mpc.w_velocity
        w_s = self.mpc.w_safety
        w_c = self.mpc.w_comfort

        observation = np.array([
            norm_ego_vel,
            norm_rel_vel,
            norm_distance,
            w_v,
            w_s,
            w_c
        ], dtype=np.float32)

        return observation

    def _compute_reward(
        self,
        carla_obs: np.ndarray,
        acceleration: float,
        carla_info: Dict
    ) -> float:
        """
        Compute reward for the current step.

        Reward components:
        1. Energy efficiency: Penalize throttle usage (proxy for energy)
        2. Safety: Penalize violations of time headway
        3. Comfort: Penalize high jerk (acceleration changes)

        Args:
            carla_obs: Current observation [ego_vel, rel_vel, distance]
            acceleration: Applied acceleration
            carla_info: Info from CARLA environment

        Returns:
            Scalar reward value
        """
        ego_velocity, relative_velocity, distance_gap = carla_obs

        # Check for collision
        if carla_info.get('collision', False):
            return self.COLLISION_PENALTY

        reward = 0.0

        # 1. Energy efficiency reward
        # Penalize throttle usage (proportional to power = throttle * velocity)
        throttle = carla_info.get('throttle', 0.0)
        energy_cost = throttle * ego_velocity * self.dt
        self.total_energy += energy_cost
        energy_reward = -self.REWARD_ENERGY_WEIGHT * energy_cost
        reward += energy_reward

        # 2. Safety reward
        # Penalize if time headway is below threshold
        time_headway = carla_info.get('time_headway', float('inf'))
        min_thw = 1.5  # Minimum acceptable time headway
        if time_headway < min_thw:
            safety_penalty = self.REWARD_SAFETY_WEIGHT * (min_thw - time_headway) ** 2
            reward -= safety_penalty
        else:
            # Small positive reward for maintaining safe distance
            reward += 0.1

        # 3. Comfort reward (jerk penalty)
        jerk = (acceleration - self.previous_acceleration) / self.dt
        comfort_penalty = self.REWARD_COMFORT_WEIGHT * (jerk ** 2) * 0.01
        reward -= comfort_penalty

        # 4. Velocity tracking bonus
        # Small reward for maintaining target velocity
        velocity_error = abs(ego_velocity - self.target_velocity)
        velocity_reward = 0.1 * np.exp(-velocity_error / 5.0)
        reward += velocity_reward

        return float(reward)

    def render(self):
        """Render the environment (handled by CARLA spectator camera)."""
        pass  # Rendering is handled by CARLA's spectator camera

    def close(self):
        """Clean up resources."""
        logger.info("Closing HybridMPCEnv")
        self.carla_env.close()

    def get_weight_history(self) -> np.ndarray:
        """Get history of MPC weights during episode."""
        return np.array(self.weight_history)


def create_training_trajectory() -> np.ndarray:
    """
    Create a challenging lead vehicle trajectory for training.

    Includes various driving scenarios:
    - Constant speed cruising
    - Gradual acceleration/deceleration
    - Hard braking
    - Stop and go

    Returns:
        Velocity profile array
    """
    trajectory = np.concatenate([
        # Phase 1: Highway cruise
        np.ones(100) * 22.0,  # 22 m/s (~79 km/h)

        # Phase 2: Gradual slowdown
        np.linspace(22.0, 15.0, 60),

        # Phase 3: Slow cruise
        np.ones(80) * 15.0,

        # Phase 4: Acceleration
        np.linspace(15.0, 25.0, 50),

        # Phase 5: Fast cruise
        np.ones(60) * 25.0,

        # Phase 6: Hard braking
        np.linspace(25.0, 8.0, 30),

        # Phase 7: Very slow
        np.ones(50) * 8.0,

        # Phase 8: Stop and go
        np.linspace(8.0, 0.0, 20),  # Stop
        np.ones(30) * 0.0,  # Stationary
        np.linspace(0.0, 12.0, 40),  # Go

        # Phase 9: Moderate cruise
        np.ones(80) * 18.0,

        # Phase 10: Final acceleration
        np.linspace(18.0, 22.0, 40),
        np.ones(60) * 22.0,
    ])

    return trajectory


# Test script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing HybridMPCEnv...")
    print("Make sure CARLA server is running!")
    print()

    # Create trajectory
    trajectory = create_training_trajectory()
    print(f"Training trajectory length: {len(trajectory)} steps")

    # Create environment
    env = HybridMPCEnv(
        lead_trajectory=trajectory,
        max_episode_steps=500
    )

    try:
        # Test reset
        obs, info = env.reset(seed=42)
        print(f"Initial observation: {obs}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print()

        # Run a few steps with random actions
        total_reward = 0.0
        for i in range(100):
            # Random action (random weights)
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if i % 20 == 0:
                print(f"Step {i}: reward={reward:.3f}, weights={info['mpc_weights']}")

            if terminated or truncated:
                print(f"Episode ended at step {i}")
                break

        print(f"\nTotal reward: {total_reward:.2f}")
        print(f"Total energy: {env.total_energy:.2f}")

        weight_history = env.get_weight_history()
        print(f"Weight history shape: {weight_history.shape}")

    finally:
        env.close()
        print("\nTest completed!")
