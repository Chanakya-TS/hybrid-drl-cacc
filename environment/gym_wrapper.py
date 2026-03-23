"""
Gymnasium Wrapper for Hybrid DRL-MPC Eco-Driving Environment.

This module wraps the CarFollowingEnv and MPC controller into a
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
    MAX_VELOCITY = 36.0  # m/s (~130 km/h, accommodates US06 peak)
    MAX_REL_VELOCITY = 20.0  # m/s
    MAX_DISTANCE = 100.0  # m
    MAX_ACCELERATION = 5.0  # m/s² (for normalizing lead accelerations)

    # Reward weights (energy-focused: energy is PRIMARY objective)
    REWARD_ENERGY_WEIGHT = 2.0       # Primary: penalize throttle*velocity (matches eval metric)
    REWARD_FOLLOWING_WEIGHT = 0.5    # Secondary: maintain desired following distance
    REWARD_VEL_MATCH_WEIGHT = 0.3    # Tertiary: match lead vehicle speed
    REWARD_SAFETY_WEIGHT = 3.0       # Hard penalty for THW < 1.0s
    REWARD_COMFORT_WEIGHT = 0.1      # Light jerk penalty
    WEIGHT_SMOOTH_PENALTY = 0.15     # Penalty for rapid weight changes (per DRL decision)
    COLLISION_PENALTY = -100.0

    # Residual policy: DRL outputs deltas on top of these Fixed-MPC baseline weights.
    # Zero delta = matches Fixed-MPC exactly. DRL only deviates when it finds improvement.
    BASE_WEIGHTS = np.array([0.25, 0.35, 0.40])
    MAX_DELTA = 0.15  # Maximum weight adjustment per dimension

    # Default action repeat: 20 sub-steps × 0.05s = 1.0s per DRL decision
    DEFAULT_ACTION_REPEAT = 20

    def __init__(
        self,
        dt: float = 0.05,
        mpc_horizon: int = 20,
        max_episode_steps: int = 1000,
        target_velocity: float = 20.0,
        lead_trajectory: Optional[np.ndarray] = None,
        action_repeat: Optional[int] = None,
        residual: bool = True,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the hybrid DRL-MPC environment.

        Args:
            dt: Simulation timestep
            mpc_horizon: MPC prediction horizon
            max_episode_steps: Maximum sim steps per episode
            target_velocity: Target cruising velocity
            lead_trajectory: Velocity profile for lead vehicle
            action_repeat: Number of sim steps per DRL decision (default 20 = 1.0s).
                           MPC weights are held constant between decisions.
            residual: If True (default), DRL outputs weight deltas from BASE_WEIGHTS.
                      If False, DRL outputs absolute weights in [0, 1].
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__()

        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.target_velocity = target_velocity
        self.action_repeat = action_repeat or self.DEFAULT_ACTION_REPEAT
        self.residual = residual
        self.render_mode = render_mode

        # Create simulation environment
        self.carla_env = CarFollowingEnv(
            dt=dt,
            lead_vehicle_trajectory=lead_trajectory,
            max_episode_steps=max_episode_steps,
        )

        # Create MPC controller (initialized at Fixed-MPC baseline weights)
        self.mpc = MPCController(
            dt=dt,
            horizon=mpc_horizon,
            w_velocity=self.BASE_WEIGHTS[0],
            w_safety=self.BASE_WEIGHTS[1],
            w_comfort=self.BASE_WEIGHTS[2],
            target_velocity=target_velocity
        )

        # Define observation space (13 dims):
        # [ego_vel,
        #  rel_vel_1, dist_1, accel_1,   # lead 1
        #  rel_vel_2, dist_2, accel_2,   # lead 2
        #  rel_vel_3, dist_3, accel_3,   # lead 3
        #  w_v, w_s, w_c]
        self.observation_space = spaces.Box(
            low=np.array([
                0.0,                         # ego_vel
                -1.0, 0.0, -1.0,             # lead 1: rel_vel, dist, accel
                -1.0, 0.0, -1.0,             # lead 2
                -1.0, 0.0, -1.0,             # lead 3
                0.0, 0.0, 0.0                # weights
            ], dtype=np.float32),
            high=np.array([
                1.0,                         # ego_vel
                1.0, 1.0, 1.0,              # lead 1
                1.0, 1.0, 1.0,              # lead 2
                1.0, 1.0, 1.0,              # lead 3
                1.0, 1.0, 1.0               # weights
            ], dtype=np.float32),
            dtype=np.float32
        )

        # Define action space based on residual vs absolute mode
        if self.residual:
            # Residual: weight DELTAS from baseline [-MAX_DELTA, +MAX_DELTA]
            # Zero action = Fixed-MPC weights (guaranteed baseline performance)
            self.action_space = spaces.Box(
                low=np.full(3, -self.MAX_DELTA, dtype=np.float32),
                high=np.full(3, self.MAX_DELTA, dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Absolute: raw weights in [0, 1], normalized to sum to 1
            self.action_space = spaces.Box(
                low=np.full(3, 0.0, dtype=np.float32),
                high=np.full(3, 1.0, dtype=np.float32),
                dtype=np.float32
            )

        # Episode tracking
        self.current_step = 0       # sim-level step counter
        self.previous_velocity = 0.0
        self.previous_acceleration = 0.0
        self.total_energy = 0.0
        self.previous_weights = self.BASE_WEIGHTS.copy()  # Start at baseline

        # Weight history for logging
        self.weight_history = []

        decision_interval_s = self.action_repeat * self.dt
        mode = "residual" if self.residual else "absolute"
        logger.info(
            f"HybridMPCEnv initialized ({mode} policy, action_repeat={self.action_repeat}, "
            f"DRL decides every {decision_interval_s:.2f}s, "
            f"MPC runs at {1/self.dt:.0f}Hz)"
        )

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
        self.previous_weights = self.BASE_WEIGHTS.copy()
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
        Execute one DRL decision step with temporal abstraction.

        The DRL agent sets MPC weights ONCE, then MPC runs for `action_repeat`
        simulation sub-steps (default 20 × 0.05s = 1.0s) with those weights
        held constant. Rewards are accumulated across sub-steps.

        This prevents weight oscillation at 20Hz — the MPC gets a stable
        optimization landscape for ~1 second between weight updates.

        Args:
            action: Weight deltas [Δw_v, Δw_s, Δw_c] from baseline.
                    Zero action = Fixed-MPC baseline weights.

        Returns:
            observation: Normalized observation at end of repeat window
            reward: Accumulated reward over all sub-steps
            terminated: Whether episode ended (collision)
            truncated: Whether episode was truncated (max steps)
            info: Additional information including sim_steps count
        """
        # 1. Compute target weights from action
        if self.residual:
            # Residual policy: deltas on top of baseline
            raw_weights = self.BASE_WEIGHTS + np.array(action)
        else:
            # Absolute policy: raw weights from action
            raw_weights = np.array(action)
        raw_weights = np.clip(raw_weights, 0.05, 0.8)  # Keep weights in valid range
        # Normalize to sum to 1
        target_weights = raw_weights / raw_weights.sum()
        self.weight_history.append(target_weights.tolist())

        # 2. Compute weight smoothing penalty ONCE per agent decision
        weight_change = np.sum((target_weights - self.previous_weights) ** 2)
        total_reward = -self.WEIGHT_SMOOTH_PENALTY * weight_change

        # Save start weights for linear interpolation
        start_weights = self.previous_weights.copy()
        self.previous_weights = target_weights.copy()

        # 3. Run action_repeat sub-steps with LINEAR WEIGHT INTERPOLATION
        #    Instead of snapping to new weights (causing jerk spike at boundary),
        #    smoothly ramp from old weights to new weights across the window.
        terminated = False
        truncated = False
        sim_steps_taken = 0
        carla_obs = None
        carla_info = {}
        mpc_info = {}
        acceleration = 0.0

        # Collect per-sub-step data for evaluation logging at full resolution
        sub_step_data = {
            'ego_velocities': [],
            'accelerations': [],
            'throttles': [],
            'brakes': [],
            'distance_gaps': [],
            'time_headways': [],
            'lead_velocities': [],
        }

        for sub_idx in range(self.action_repeat):
            # Linear interpolation: blend from start_weights to target_weights
            blend = (sub_idx + 1) / self.action_repeat  # 0.05, 0.10, ..., 1.0
            interp_weights = (1.0 - blend) * start_weights + blend * target_weights
            self.mpc.set_weights(interp_weights[0], interp_weights[1], interp_weights[2])

            # Get current state from CARLA (10-dim observation)
            raw_obs = self.carla_env._get_observation()
            ego_velocity = raw_obs[0]

            # Extract all 3 lead vehicle states
            lead_velocities = [
                ego_velocity + raw_obs[1],
                ego_velocity + raw_obs[4],
                ego_velocity + raw_obs[7]
            ]
            distance_gaps = [raw_obs[2], raw_obs[5], raw_obs[8]]
            lead_accelerations = [raw_obs[3], raw_obs[6], raw_obs[9]]

            # Compute acceleration using MPC (weights change smoothly)
            acceleration, mpc_info = self.mpc.compute_control(
                ego_velocity=ego_velocity,
                lead_velocities=lead_velocities,
                distance_gaps=distance_gaps,
                lead_accelerations=lead_accelerations
            )

            # Step CARLA environment
            carla_obs, _, terminated, truncated, carla_info = self.carla_env.step(acceleration)

            self.current_step += 1
            sim_steps_taken += 1

            # Accumulate sub-step reward (without weight smoothing — already applied)
            sub_reward = self._compute_reward(carla_obs, acceleration, carla_info)
            total_reward += sub_reward

            # Record sub-step data for evaluation
            sub_step_data['ego_velocities'].append(carla_info.get('ego_velocity', ego_velocity))
            sub_step_data['accelerations'].append(acceleration)
            sub_step_data['throttles'].append(carla_info.get('throttle', 0.0))
            sub_step_data['brakes'].append(carla_info.get('brake', 0.0))
            sub_step_data['distance_gaps'].append(carla_info.get('distance_gap', distance_gaps[0]))
            sub_step_data['time_headways'].append(carla_info.get('time_headway', float('inf')))
            sub_step_data['lead_velocities'].append(lead_velocities[0])

            # Update tracking variables for next sub-step
            self.previous_acceleration = acceleration
            self.previous_velocity = carla_obs[0]

            if terminated or truncated:
                break

        # 4. Build observation from final sub-step state
        observation = self._build_observation(carla_obs)

        info = {
            'carla_info': carla_info,
            'mpc_info': mpc_info,
            'mpc_weights': target_weights.tolist(),
            'acceleration': acceleration,
            'step': self.current_step,
            'sim_steps': sim_steps_taken,
            'sub_step_data': sub_step_data
        }

        return observation, total_reward, terminated, truncated, info

    def _build_observation(self, carla_obs: np.ndarray) -> np.ndarray:
        """
        Build normalized 13-dim observation for DRL agent.

        Args:
            carla_obs: Raw 10-dim observation from CARLA
                [ego_vel, rel_vel_1, dist_1, accel_1,
                         rel_vel_2, dist_2, accel_2,
                         rel_vel_3, dist_3, accel_3]

        Returns:
            Normalized 13-dim observation vector
        """
        ego_velocity = carla_obs[0]
        norm_ego_vel = np.clip(ego_velocity / self.MAX_VELOCITY, 0.0, 1.0)

        # Normalize each lead's state triplet
        normalized_leads = []
        for i in range(3):
            base = 1 + i * 3
            rel_vel = carla_obs[base]
            dist = carla_obs[base + 1]
            accel = carla_obs[base + 2]

            norm_rel_vel = np.clip(rel_vel / self.MAX_REL_VELOCITY, -1.0, 1.0)
            norm_dist = np.clip(dist / self.MAX_DISTANCE, 0.0, 1.0)
            norm_accel = np.clip(accel / self.MAX_ACCELERATION, -1.0, 1.0)
            normalized_leads.extend([norm_rel_vel, norm_dist, norm_accel])

        # Current weights (already normalized to sum to 1)
        w_v = self.mpc.w_velocity
        w_s = self.mpc.w_safety
        w_c = self.mpc.w_comfort

        observation = np.array(
            [norm_ego_vel] + normalized_leads + [w_v, w_s, w_c],
            dtype=np.float32
        )

        return observation

    def _compute_reward(
        self,
        carla_obs: np.ndarray,
        acceleration: float,
        carla_info: Dict
    ) -> float:
        """
        Compute per-sub-step reward (called once per simulation step).

        Energy-focused reward: the evaluation metric is total_energy = Σ(throttle×velocity×dt),
        so energy efficiency is the PRIMARY optimization signal. Car-following and safety are
        constraints that must be satisfied, but energy savings determine the winner.

        Weight smoothing penalty is handled separately in step() and applied
        once per DRL decision, not per sub-step.

        Reward components (ordered by priority):
        1. Energy efficiency (PRIMARY — matches evaluation metric)
        2. Safety penalty for dangerous proximity (hard constraint)
        3. Car-following distance (secondary)
        4. Velocity matching with lead (tertiary)
        5. Comfort penalty (jerk, light)

        Args:
            carla_obs: Current 10-dim observation from CARLA
            acceleration: Applied acceleration
            carla_info: Info from CARLA environment

        Returns:
            Scalar reward value
        """
        ego_velocity = carla_obs[0]
        distance_gap = carla_obs[2]  # distance to closest lead

        # Check for collision
        if carla_info.get('collision', False):
            return self.COLLISION_PENALTY

        reward = 0.0

        # Lead vehicle velocity (from obs: rel_vel_1 = carla_obs[1])
        lead_velocity = ego_velocity + carla_obs[1]

        # 1. Energy efficiency (PRIMARY — normalized to [0,1] for stable gradients)
        # Matches evaluation metric: throttle * velocity
        # Normalized by MAX_VELOCITY so reward magnitude doesn't scale with speed
        throttle = carla_info.get('throttle', 0.0)
        normalized_energy = throttle * (ego_velocity / self.MAX_VELOCITY)
        reward -= self.REWARD_ENERGY_WEIGHT * normalized_energy

        # Track total energy for logging (raw, unnormalized)
        energy_cost = throttle * ego_velocity * self.dt
        self.total_energy += energy_cost

        # 2. Safety penalty (hard constraint — large penalty for dangerous proximity)
        time_headway = carla_info.get('time_headway', float('inf'))
        if time_headway < 1.0:
            # Strong quadratic penalty below 1.0s THW
            reward -= self.REWARD_SAFETY_WEIGHT * (1.0 - time_headway) ** 2

        # 3. Car-following distance reward (secondary)
        THW_TARGET = 1.8  # seconds
        MIN_DIST = 5.0  # meters
        desired_distance = THW_TARGET * ego_velocity + MIN_DIST

        # Gaussian-shaped: rewards maintaining desired distance, penalizes too close or too far
        distance_error = abs(distance_gap - desired_distance)
        following_reward = self.REWARD_FOLLOWING_WEIGHT * np.exp(
            -(distance_error ** 2) / (2 * 10.0 ** 2)
        )
        reward += following_reward

        # 4. Velocity matching reward (tertiary)
        vel_error = abs(ego_velocity - lead_velocity)
        vel_match_reward = self.REWARD_VEL_MATCH_WEIGHT * np.exp(
            -(vel_error ** 2) / (2 * 3.0 ** 2)
        )
        reward += vel_match_reward

        # 5. Comfort penalty (light jerk penalty)
        jerk = (acceleration - self.previous_acceleration) / self.dt
        reward -= self.REWARD_COMFORT_WEIGHT * (jerk ** 2) * self.dt

        return float(reward)

    def render(self):
        """Render the environment (no-op for lightweight sim)."""
        pass

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
