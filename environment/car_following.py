"""
Pure-Python Car-Following Environment for Eco-Driving Control.

This module implements a lightweight 1D longitudinal dynamics simulation
using only numpy. No external simulator (CARLA, etc.) is required.

Vehicle dynamics:
    v(k+1) = clip(v(k) + a_net * dt, 0, V_MAX)
    a_net  = a_cmd - (F_drag + F_roll) / m
    F_drag = C_d * v^2          (aerodynamic drag)
    F_roll = C_r * m * g        (rolling resistance)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplifiedCarFollowingEnv:
    """
    Pure-Python car-following environment for eco-driving experiments.

    Simulates an ego vehicle following a platoon of 3 lead vehicles using
    simple 1D longitudinal dynamics with aerodynamic drag and rolling
    resistance. Provides the same 10-dim observation and info dict as the
    original CARLA-based environment.
    """

    # Constants
    MAX_ACCELERATION = 2.0   # m/s^2
    MAX_BRAKING = 3.0        # m/s^2
    TIME_HEADWAY = 1.8       # seconds (safety constraint)
    INITIAL_DISTANCE = 20.0  # meters (to first lead vehicle)
    MAX_VELOCITY = 36.0      # m/s (~130 km/h)

    # Vehicle parameters (sedan-class)
    VEHICLE_MASS = 1800.0    # kg
    DRAG_COEFF = 0.4         # C_d * 0.5 * rho * A  (lumped aero constant)
    ROLL_COEFF = 0.01        # rolling resistance coefficient
    GRAVITY = 9.81           # m/s^2

    # Platoon geometry
    NUM_LEAD_VEHICLES = 3
    LEAD_VEHICLE_GAP = 15.0  # m between consecutive lead vehicles

    def __init__(
        self,
        dt: float = 0.05,
        lead_vehicle_trajectory: Optional[np.ndarray] = None,
        max_episode_steps: int = 1000,
    ):
        """
        Initialize the pure-Python car-following environment.

        Args:
            dt: Simulation timestep in seconds
            lead_vehicle_trajectory: Predefined velocity profile for the
                front lead vehicle (array of target speeds in m/s)
            max_episode_steps: Maximum steps per episode
        """
        self.dt = dt
        self.max_episode_steps = max_episode_steps

        # Lead vehicle trajectory (for the front lead vehicle)
        self.lead_trajectory = lead_vehicle_trajectory
        self.trajectory_idx = 0

        # Number of lead vehicles
        self.num_lead_vehicles = self.NUM_LEAD_VEHICLES

        # Ego state
        self.ego_position = 0.0
        self.ego_velocity = 0.0

        # Lead vehicle states (arrays of length num_lead_vehicles)
        # Index 0 = closest to ego, index -1 = front of platoon
        self.lead_positions = np.zeros(self.num_lead_vehicles)
        self.lead_velocities = np.zeros(self.num_lead_vehicles)

        # Previous lead velocities for acceleration estimation
        self.prev_lead_velocities = np.zeros(self.num_lead_vehicles)

        # ACC PID state for follower lead vehicles (indices 0 and 1).
        # Each follower maintains its own integral and previous-error.
        self._follower_acc_integral = np.zeros(self.num_lead_vehicles)
        self._follower_acc_prev_err = np.zeros(self.num_lead_vehicles)

        # ACC PID gains for follower lead vehicles
        self._acc_kp = 0.4
        self._acc_ki = 0.01
        self._acc_kd = 0.2
        self._acc_kp_vel = 0.5
        self._acc_integral_max = 10.0
        self._acc_thw = 1.8       # s
        self._acc_min_dist = 5.0  # m

        # Episode tracking
        self.current_step = 0
        self.episode_count = 0

        # Data logging
        self.episode_data = {
            'time': [],
            'ego_velocity': [],
            'ego_position': [],
            'lead_velocity': [],
            'lead_position': [],
            'distance_gap': [],
            'relative_velocity': [],
            'acceleration': [],
            'throttle': [],
            'brake': [],
            'energy': [],
            # Per-lead data for all 3 vehicles
            'distance_gap_1': [],
            'distance_gap_2': [],
            'distance_gap_3': [],
            'rel_vel_1': [],
            'rel_vel_2': [],
            'rel_vel_3': [],
            'lead_accel_1': [],
            'lead_accel_2': [],
            'lead_accel_3': []
        }

        self.total_energy = 0.0
        self.previous_velocity = 0.0

        logger.info("SimplifiedCarFollowingEnv initialized (pure-Python sim)")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility

        Returns:
            observation: 10-dim state (see _get_observation)
            info: Additional information dictionary
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset tracking variables
        self.current_step = 0
        self.trajectory_idx = 0
        self.total_energy = 0.0
        self.previous_velocity = 0.0
        self.prev_lead_velocities = np.zeros(self.num_lead_vehicles)
        self._follower_acc_integral = np.zeros(self.num_lead_vehicles)
        self._follower_acc_prev_err = np.zeros(self.num_lead_vehicles)

        # Clear episode data
        for key in self.episode_data:
            self.episode_data[key] = []

        # Initialize ego state (stationary at origin)
        self.ego_position = 0.0
        self.ego_velocity = 0.0

        # Initialize lead vehicles ahead of ego
        # Index 0 = closest, index 2 = farthest (front of platoon)
        for i in range(self.num_lead_vehicles):
            self.lead_positions[i] = (
                self.ego_position
                + self.INITIAL_DISTANCE
                + i * self.LEAD_VEHICLE_GAP
            )
            self.lead_velocities[i] = 0.0

        # Get initial observation
        observation = self._get_observation()

        info = {
            'episode': self.episode_count,
            'step': self.current_step
        }

        self.episode_count += 1
        logger.info(f"Episode {self.episode_count} started")

        return observation, info

    def step(self, acceleration: float) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one simulation step.

        Args:
            acceleration: Desired acceleration command (m/s^2)

        Returns:
            observation: 10-dim state (see _get_observation)
            reward: Always 0.0 (reward computed in gym wrapper)
            terminated: Whether episode has terminated (collision)
            truncated: Whether episode has been truncated (max steps)
            info: Additional information dictionary
        """
        current_velocity = self.ego_velocity

        # Map acceleration to throttle/brake
        throttle, brake = self._acceleration_to_control(acceleration, current_velocity)

        # --- Ego dynamics ---
        a_cmd = np.clip(acceleration, -self.MAX_BRAKING, self.MAX_ACCELERATION)
        drag_force = self.DRAG_COEFF * current_velocity ** 2
        roll_force = self.ROLL_COEFF * self.VEHICLE_MASS * self.GRAVITY
        a_net = a_cmd - (drag_force + roll_force) / self.VEHICLE_MASS
        self.ego_velocity = float(np.clip(
            current_velocity + a_net * self.dt, 0.0, self.MAX_VELOCITY
        ))
        self.ego_position += self.ego_velocity * self.dt

        # --- Update lead vehicles ---
        self._update_lead_vehicles()

        # Update step counter
        self.current_step += 1

        # Get new observation (10-dim)
        observation = self._get_observation()
        ego_velocity = observation[0]
        rel_vel_1, distance_1, accel_1 = observation[1], observation[2], observation[3]
        rel_vel_2, distance_2, accel_2 = observation[4], observation[5], observation[6]
        rel_vel_3, distance_3, accel_3 = observation[7], observation[8], observation[9]

        # Calculate energy consumption (same model as original)
        energy_increment = throttle * ego_velocity * self.dt
        self.total_energy += energy_increment

        # Calculate measured acceleration for logging
        measured_acceleration = (ego_velocity - self.previous_velocity) / self.dt
        self.previous_velocity = ego_velocity

        # Log data
        self._log_data(
            ego_velocity, rel_vel_1, distance_1,
            measured_acceleration, throttle, brake, energy_increment,
            lead_distances=[distance_1, distance_2, distance_3],
            lead_rel_vels=[rel_vel_1, rel_vel_2, rel_vel_3],
            lead_accels=[accel_1, accel_2, accel_3]
        )

        # Check termination conditions
        terminated = self._check_collision(distance_1)
        truncated = self.current_step >= self.max_episode_steps

        # Prepare info dictionary (same keys as original)
        info = {
            'step': self.current_step,
            'ego_velocity': ego_velocity,
            'distance_gap': distance_1,
            'time_headway': distance_1 / (ego_velocity + 1e-6),
            'total_energy': self.total_energy,
            'throttle': throttle,
            'brake': brake,
            'collision': terminated
        }

        return observation, 0.0, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """
        Get current 10-dim observation.

        Returns:
            observation: [ego_velocity,
                          rel_vel_1, distance_1, accel_1,
                          rel_vel_2, distance_2, accel_2,
                          rel_vel_3, distance_3, accel_3]
        """
        lead_states = self._get_all_lead_states(self.ego_velocity)
        observation = np.array(
            [self.ego_velocity] + lead_states, dtype=np.float32
        )
        return observation

    def _get_all_lead_states(self, ego_velocity: float) -> List[float]:
        """
        Compute relative velocity, distance gap, and acceleration for
        all lead vehicles.

        Args:
            ego_velocity: Current ego vehicle speed (m/s)

        Returns:
            Flat list [rel_vel_1, dist_1, accel_1, ...]
        """
        states: List[float] = []

        for i in range(self.num_lead_vehicles):
            lead_vel = float(self.lead_velocities[i])
            distance_gap = float(self.lead_positions[i] - self.ego_position)
            distance_gap = max(distance_gap, 0.0)  # clamp negative

            rel_vel = lead_vel - ego_velocity

            # Acceleration from finite difference
            accel = (lead_vel - float(self.prev_lead_velocities[i])) / self.dt

            # Update stored velocity for next step
            self.prev_lead_velocities[i] = lead_vel

            states.extend([rel_vel, distance_gap, accel])

        return states

    # ------------------------------------------------------------------
    # Control mapping
    # ------------------------------------------------------------------

    def _acceleration_to_control(
        self,
        acceleration: float,
        current_velocity: float
    ) -> Tuple[float, float]:
        """
        Map desired acceleration to throttle/brake values.

        Args:
            acceleration: Desired acceleration (m/s^2)
            current_velocity: Current vehicle velocity (m/s)

        Returns:
            throttle: Throttle value [0, 1]
            brake: Brake value [0, 1]
        """
        acceleration = np.clip(acceleration, -self.MAX_BRAKING, self.MAX_ACCELERATION)

        if acceleration > 0:
            throttle = np.clip(acceleration / self.MAX_ACCELERATION, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-acceleration / self.MAX_BRAKING, 0.0, 1.0)

        # Don't apply throttle near max velocity
        if current_velocity > self.MAX_VELOCITY * 0.95:
            throttle = 0.0

        return float(throttle), float(brake)

    # ------------------------------------------------------------------
    # Lead vehicle dynamics
    # ------------------------------------------------------------------

    def _update_lead_vehicles(self):
        """
        Update all lead vehicles in the platoon for one timestep.

        The front vehicle (index num_lead_vehicles-1) directly follows the
        drive-cycle trajectory with a proportional controller.

        Follower vehicles (indices 0 and 1) use ACC-style PID control to
        car-follow the vehicle directly ahead of them, producing realistic
        propagation delay through the platoon.
        """
        # Determine target velocity for the front vehicle from trajectory
        if self.lead_trajectory is None:
            front_target_velocity = 15.0  # m/s default
        else:
            if self.trajectory_idx < len(self.lead_trajectory):
                front_target_velocity = self.lead_trajectory[self.trajectory_idx]
                self.trajectory_idx += 1
            else:
                front_target_velocity = self.lead_trajectory[-1]

        # Update from front to back so each follower sees its leader's
        # already-updated position/velocity for this timestep.
        for i in range(self.num_lead_vehicles - 1, -1, -1):
            current_vel = float(self.lead_velocities[i])

            if i == self.num_lead_vehicles - 1:
                # ----- Front vehicle: P-control to track trajectory -----
                vel_error = front_target_velocity - current_vel
                desired_accel = float(np.clip(0.3 * vel_error, -2.0, 1.5))
            else:
                # ----- Follower: ACC PID car-following -----
                leader_vel = float(self.lead_velocities[i + 1])
                gap = float(
                    self.lead_positions[i + 1] - self.lead_positions[i]
                )

                # Desired gap = THW * follower_speed + min_distance
                d_desired = self._acc_thw * current_vel + self._acc_min_dist
                dist_err = gap - d_desired  # positive = too far

                # PID on distance error
                p_term = self._acc_kp * dist_err

                self._follower_acc_integral[i] += dist_err * self.dt
                self._follower_acc_integral[i] = float(np.clip(
                    self._follower_acc_integral[i],
                    -self._acc_integral_max,
                    self._acc_integral_max
                ))
                i_term = self._acc_ki * self._follower_acc_integral[i]

                d_term = self._acc_kd * (
                    dist_err - self._follower_acc_prev_err[i]
                ) / self.dt
                self._follower_acc_prev_err[i] = dist_err

                # Velocity matching
                v_term = self._acc_kp_vel * (leader_vel - current_vel)

                desired_accel = p_term + i_term + d_term + v_term

                # Safety override: hard brake if dangerously close
                if gap < self._acc_min_dist:
                    desired_accel = -self.MAX_BRAKING

                desired_accel = float(np.clip(
                    desired_accel, -self.MAX_BRAKING, self.MAX_ACCELERATION
                ))

            # Apply 1D dynamics (same resistive forces as ego)
            drag = self.DRAG_COEFF * current_vel ** 2
            roll = self.ROLL_COEFF * self.VEHICLE_MASS * self.GRAVITY
            a_net = desired_accel - (drag + roll) / self.VEHICLE_MASS

            new_vel = float(np.clip(
                current_vel + a_net * self.dt, 0.0, self.MAX_VELOCITY
            ))
            self.lead_velocities[i] = new_vel
            self.lead_positions[i] += new_vel * self.dt

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def _check_collision(self, distance_gap: float) -> bool:
        """
        Check if collision has occurred.

        Args:
            distance_gap: Current distance gap (m)

        Returns:
            True if collision detected
        """
        COLLISION_THRESHOLD = 2.0
        return distance_gap < COLLISION_THRESHOLD

    # ------------------------------------------------------------------
    # Data logging
    # ------------------------------------------------------------------

    def _log_data(
        self,
        ego_velocity: float,
        relative_velocity: float,
        distance_gap: float,
        acceleration: float,
        throttle: float,
        brake: float,
        energy_increment: float,
        lead_distances: Optional[List[float]] = None,
        lead_rel_vels: Optional[List[float]] = None,
        lead_accels: Optional[List[float]] = None
    ):
        """Log episode data for later analysis."""
        self.episode_data['time'].append(self.current_step * self.dt)
        self.episode_data['ego_velocity'].append(ego_velocity)
        self.episode_data['lead_velocity'].append(ego_velocity + relative_velocity)
        self.episode_data['distance_gap'].append(distance_gap)
        self.episode_data['relative_velocity'].append(relative_velocity)
        self.episode_data['acceleration'].append(acceleration)
        self.episode_data['throttle'].append(throttle)
        self.episode_data['brake'].append(brake)
        self.episode_data['energy'].append(energy_increment)

        # Log per-lead data
        if lead_distances is not None:
            for i in range(3):
                self.episode_data[f'distance_gap_{i+1}'].append(lead_distances[i])
        if lead_rel_vels is not None:
            for i in range(3):
                self.episode_data[f'rel_vel_{i+1}'].append(lead_rel_vels[i])
        if lead_accels is not None:
            for i in range(3):
                self.episode_data[f'lead_accel_{i+1}'].append(lead_accels[i])

        # Log positions
        self.episode_data['ego_position'].append(self.ego_position)
        self.episode_data['lead_position'].append(float(self.lead_positions[0]))

    def get_episode_data(self) -> Dict:
        """
        Get logged episode data.

        Returns:
            episode_data: Dictionary of logged time series
        """
        return self.episode_data.copy()

    # ------------------------------------------------------------------
    # Cleanup (no-ops for compatibility)
    # ------------------------------------------------------------------

    def destroy(self):
        """No-op (no actors to destroy in pure-Python sim)."""
        pass

    def close(self):
        """Close the environment (reset internal state)."""
        logger.info("Closing SimplifiedCarFollowingEnv")
        self.destroy()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


# Alias so existing imports (from environment.car_following import CarFollowingEnv) work
CarFollowingEnv = SimplifiedCarFollowingEnv


# Test script
if __name__ == "__main__":
    # Create a simple test trajectory
    test_trajectory = np.concatenate([
        np.linspace(15, 15, 100),  # Constant velocity
        np.linspace(15, 8, 50),    # Deceleration
        np.linspace(8, 8, 100),    # Slow constant
        np.linspace(8, 18, 50),    # Acceleration
        np.linspace(18, 18, 100)   # Fast constant
    ])

    # Create environment
    env = CarFollowingEnv(lead_vehicle_trajectory=test_trajectory)

    try:
        # Test reset
        obs, info = env.reset()
        print(f"Initial observation ({len(obs)} dims): {obs}")

        # Run a few steps
        for i in range(10):
            ego_vel = obs[0]
            rel_vel_1, distance_1 = obs[1], obs[2]
            target_vel = ego_vel + rel_vel_1
            accel = 0.5 * (target_vel - ego_vel)

            obs, reward, terminated, truncated, info = env.step(accel)
            print(
                f"Step {i+1}: vel={obs[0]:.2f}, "
                f"d1={obs[2]:.2f}, d2={obs[5]:.2f}, d3={obs[8]:.2f}, "
                f"THW={info['time_headway']:.2f}"
            )

            if terminated or truncated:
                break

        print("Test completed successfully!")

    finally:
        env.close()
