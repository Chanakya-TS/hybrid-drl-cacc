"""
CARLA-based Car-Following Environment for Eco-Driving Control.

This module implements a realistic car-following simulation using the CARLA simulator
with high-fidelity vehicle dynamics.
"""

import numpy as np
import carla
import time
from typing import Optional, Tuple, List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarFollowingEnv:
    """
    CARLA-based car-following environment for eco-driving experiments.

    The environment spawns an ego vehicle and a lead vehicle in CARLA,
    and provides an interface for controlling the ego vehicle while
    tracking energy consumption, safety metrics, and comfort metrics.
    """

    # Constants
    MAX_ACCELERATION = 2.0  # m/s²
    MAX_BRAKING = 3.0  # m/s²
    TIME_HEADWAY = 1.8  # seconds (safety constraint)
    INITIAL_DISTANCE = 20.0  # meters (to first lead vehicle)
    MAX_VELOCITY = 36.0  # m/s (~130 km/h, accommodates US06 peak)

    def __init__(
        self,
        carla_host: str = 'localhost',
        carla_port: int = 2000,
        dt: float = 0.05,
        lead_vehicle_trajectory: Optional[np.ndarray] = None,
        max_episode_steps: int = 1000,
        vehicle_blueprint: str = 'vehicle.tesla.model3',
        map_name: str = 'Town04'
    ):
        """
        Initialize the CARLA-based car-following environment.

        Args:
            carla_host: CARLA server hostname
            carla_port: CARLA server port
            dt: Simulation timestep in seconds (fixed_delta_seconds)
            lead_vehicle_trajectory: Predefined velocity profile for lead vehicle
            max_episode_steps: Maximum steps per episode
            vehicle_blueprint: CARLA vehicle blueprint name
            map_name: CARLA map to load (Town04 has highway, Town06 has long highway)
        """
        self.host = carla_host
        self.port = carla_port
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.vehicle_blueprint_name = vehicle_blueprint
        self.map_name = map_name

        # CARLA connection
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.map: Optional[carla.Map] = None
        self.blueprint_library = None
        self.spawn_points: List[carla.Transform] = []

        # Actors
        self.ego_vehicle: Optional[carla.Vehicle] = None
        self.lead_vehicles: List[carla.Vehicle] = []  # Multiple lead vehicles (platoon)
        self.num_lead_vehicles = 3  # Number of lead vehicles in platoon
        self.lead_vehicle_gap = 15.0  # Gap between lead vehicles (m) - kept small for render distance

        # Lead vehicle trajectory (for the first/front lead vehicle)
        self.lead_trajectory = lead_vehicle_trajectory
        self.trajectory_idx = 0

        # Episode tracking
        self.current_step = 0
        self.episode_count = 0

        # Previous lead velocities for acceleration estimation (one per lead)
        self.prev_lead_velocities = [0.0] * self.num_lead_vehicles

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

        # Connection state
        self._connected = False

        logger.info("CarFollowingEnv initialized")

    def connect(self) -> bool:
        """
        Connect to CARLA server and setup synchronous mode.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to CARLA at {self.host}:{self.port}")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(30.0)

            # Load the specified map (Town04 has highway loop)
            current_map = self.client.get_world().get_map().name
            if self.map_name not in current_map:
                logger.info(f"Loading map: {self.map_name}")
                self.world = self.client.load_world(self.map_name)
                time.sleep(2.0)  # Wait for map to load
            else:
                logger.info(f"Map {self.map_name} already loaded")
                self.world = self.client.get_world()

            # Enable synchronous mode with increased actor distance
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.dt
            settings.actor_active_distance = 500.0  # Increase actor active distance (default ~200m)
            self.world.apply_settings(settings)

            logger.info(f"Synchronous mode enabled with dt={self.dt}s")

            # Get blueprint library, map, and spawn points
            self.blueprint_library = self.world.get_blueprint_library()
            self.map = self.world.get_map()
            self.spawn_points = self.map.get_spawn_points()

            self._connected = True
            logger.info("Successfully connected to CARLA")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to CARLA: {e}")
            self._connected = False
            return False

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

        # Connect if not already connected
        if not self._connected:
            if not self.connect():
                raise RuntimeError("Failed to connect to CARLA server")

        # Destroy existing vehicles
        self.destroy()

        # Reset tracking variables
        self.current_step = 0
        self.trajectory_idx = 0
        self.total_energy = 0.0
        self.previous_velocity = 0.0
        self.prev_lead_velocities = [0.0] * self.num_lead_vehicles

        # Clear episode data
        for key in self.episode_data:
            self.episode_data[key] = []

        # Spawn vehicles
        self._spawn_vehicles()

        # Tick once to initialize
        self.world.tick()
        time.sleep(0.1)

        # Position spectator camera behind ego vehicle
        self._update_spectator()

        # Get initial observation
        observation = self._get_observation()

        info = {
            'episode': self.episode_count,
            'step': self.current_step
        }

        self.episode_count += 1
        logger.info(f"Episode {self.episode_count} started")

        return observation, info

    def _spawn_vehicles(self):
        """Spawn ego vehicle and platoon of lead vehicles along the road."""
        try:
            # Get vehicle blueprint
            vehicle_bp = self.blueprint_library.find(self.vehicle_blueprint_name)

            # Choose a suitable spawn point (prefer straight roads)
            if len(self.spawn_points) == 0:
                raise RuntimeError("No spawn points available in map")

            # For Town04, spawn points 0-50 are typically on the highway
            spawn_idx = min(10, len(self.spawn_points) - 1)
            ego_spawn_point = self.spawn_points[spawn_idx]

            # Spawn ego vehicle
            self.ego_vehicle = self.world.spawn_actor(vehicle_bp, ego_spawn_point)
            logger.info(f"Ego vehicle spawned at {ego_spawn_point.location}")

            # Get waypoint at ego position to follow the road
            ego_waypoint = self.map.get_waypoint(
                ego_spawn_point.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )

            # Spawn lead vehicles along the road using waypoints
            self.lead_vehicles = []
            current_waypoint = ego_waypoint

            for i in range(self.num_lead_vehicles):
                # Get waypoint ahead along the road
                distance_ahead = self.INITIAL_DISTANCE if i == 0 else self.lead_vehicle_gap
                waypoints_ahead = current_waypoint.next(distance_ahead)

                if not waypoints_ahead:
                    logger.warning(f"No waypoint found for lead vehicle {i+1}")
                    break

                lead_waypoint = waypoints_ahead[0]

                # Create spawn transform from waypoint
                lead_spawn_point = carla.Transform(
                    carla.Location(
                        x=lead_waypoint.transform.location.x,
                        y=lead_waypoint.transform.location.y,
                        z=lead_waypoint.transform.location.z + 0.5
                    ),
                    lead_waypoint.transform.rotation
                )

                # Spawn lead vehicle
                lead_vehicle = self.world.spawn_actor(vehicle_bp, lead_spawn_point)
                self.lead_vehicles.append(lead_vehicle)
                logger.info(f"Lead vehicle {i+1} spawned on road")

                # Update current waypoint for next vehicle
                current_waypoint = lead_waypoint

            logger.info(f"Platoon created with {len(self.lead_vehicles)} lead vehicles")

        except Exception as e:
            logger.error(f"Failed to spawn vehicles: {e}")
            self.destroy()
            raise

    def step(self, acceleration: float) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            acceleration: Desired acceleration command (m/s²)

        Returns:
            observation: 10-dim state (see _get_observation)
            reward: Reward for this step (computed elsewhere, return 0.0 here)
            terminated: Whether episode has terminated (collision)
            truncated: Whether episode has been truncated (max steps)
            info: Additional information dictionary
        """
        # Get current ego velocity for control mapping
        current_velocity = self._get_vehicle_speed(self.ego_vehicle)

        # Map acceleration to CARLA control
        throttle, brake = self._acceleration_to_control(acceleration, current_velocity)

        # Compute steering to follow road
        steer = self._compute_steering(self.ego_vehicle)

        # Apply control to ego vehicle
        control = carla.VehicleControl(
            throttle=float(throttle),
            brake=float(brake),
            steer=float(steer)
        )
        self.ego_vehicle.apply_control(control)

        # Update all lead vehicles in platoon
        self._update_lead_vehicles()

        # Tick simulation
        self.world.tick()

        # Update spectator camera to follow ego vehicle
        self._update_spectator()

        # Update step counter
        self.current_step += 1

        # Get new observation (10-dim)
        observation = self._get_observation()
        ego_velocity = observation[0]
        rel_vel_1, distance_1, accel_1 = observation[1], observation[2], observation[3]
        rel_vel_2, distance_2, accel_2 = observation[4], observation[5], observation[6]
        rel_vel_3, distance_3, accel_3 = observation[7], observation[8], observation[9]

        # Calculate energy consumption
        energy_increment = throttle * ego_velocity * self.dt
        self.total_energy += energy_increment

        # Calculate acceleration for logging
        measured_acceleration = (ego_velocity - self.previous_velocity) / self.dt
        self.previous_velocity = ego_velocity

        # Log data (includes all 3 leads)
        self._log_data(
            ego_velocity, rel_vel_1, distance_1,
            measured_acceleration, throttle, brake, energy_increment,
            lead_distances=[distance_1, distance_2, distance_3],
            lead_rel_vels=[rel_vel_1, rel_vel_2, rel_vel_3],
            lead_accels=[accel_1, accel_2, accel_3]
        )

        # Check termination conditions (closest lead only)
        terminated = self._check_collision(distance_1)
        truncated = self.current_step >= self.max_episode_steps

        # Prepare info dictionary
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

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation from CARLA for all 3 lead vehicles.

        Returns:
            observation: 10-dim array
                [ego_velocity,
                 rel_vel_1, distance_1, accel_1,   # closest lead
                 rel_vel_2, distance_2, accel_2,   # second lead
                 rel_vel_3, distance_3, accel_3]   # third lead
        """
        ego_velocity = self._get_vehicle_speed(self.ego_vehicle)  # m/s
        lead_states = self._get_all_lead_states(ego_velocity)

        observation = np.array([ego_velocity] + lead_states, dtype=np.float32)
        return observation

    def _get_all_lead_states(self, ego_velocity: float) -> List[float]:
        """
        Compute relative velocity, distance gap, and acceleration for all lead vehicles.

        Args:
            ego_velocity: Current ego vehicle speed (m/s)

        Returns:
            Flat list [rel_vel_1, dist_1, accel_1, rel_vel_2, dist_2, accel_2, ...]
        """
        ego_location = self.ego_vehicle.get_location()
        states = []

        for i in range(self.num_lead_vehicles):
            if i < len(self.lead_vehicles):
                vehicle = self.lead_vehicles[i]
                lead_velocity = self._get_vehicle_speed(vehicle)

                # Euclidean distance from ego to this lead
                lead_location = vehicle.get_location()
                distance_gap = float(np.sqrt(
                    (lead_location.x - ego_location.x)**2 +
                    (lead_location.y - ego_location.y)**2
                ))

                # Relative velocity (positive = lead pulling away)
                rel_vel = lead_velocity - ego_velocity

                # Acceleration from finite difference
                accel = (lead_velocity - self.prev_lead_velocities[i]) / self.dt

                # Update stored velocity for next step
                self.prev_lead_velocities[i] = lead_velocity
            else:
                # Fallback if fewer vehicles than expected
                rel_vel = 0.0
                distance_gap = 100.0
                accel = 0.0

            states.extend([rel_vel, distance_gap, accel])

        return states

    def _get_vehicle_speed(self, vehicle: carla.Vehicle) -> float:
        """
        Extract scalar speed from vehicle.

        Args:
            vehicle: CARLA vehicle actor

        Returns:
            speed: Scalar speed in m/s
        """
        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return float(speed)

    def _get_distance_between_vehicles(self) -> float:
        """
        Calculate distance gap to closest lead vehicle.

        Returns:
            distance_gap: Distance in meters to closest lead vehicle
        """
        if len(self.lead_vehicles) == 0:
            return 100.0  # Default large distance

        ego_location = self.ego_vehicle.get_location()

        # Get distance to closest lead vehicle (first in list)
        closest_lead = self.lead_vehicles[0]
        lead_location = closest_lead.get_location()

        # Calculate Euclidean distance
        distance = np.sqrt(
            (lead_location.x - ego_location.x)**2 +
            (lead_location.y - ego_location.y)**2
        )

        return float(distance)

    def _acceleration_to_control(
        self,
        acceleration: float,
        current_velocity: float
    ) -> Tuple[float, float]:
        """
        Map desired acceleration to CARLA throttle/brake commands.

        Args:
            acceleration: Desired acceleration (m/s²)
            current_velocity: Current vehicle velocity (m/s)

        Returns:
            throttle: Throttle value [0, 1]
            brake: Brake value [0, 1]
        """
        # Clip acceleration to limits
        acceleration = np.clip(acceleration, -self.MAX_BRAKING, self.MAX_ACCELERATION)

        if acceleration > 0:
            # Positive acceleration: apply throttle
            throttle = np.clip(acceleration / self.MAX_ACCELERATION, 0.0, 1.0)
            brake = 0.0
        else:
            # Negative acceleration: apply brake
            throttle = 0.0
            brake = np.clip(-acceleration / self.MAX_BRAKING, 0.0, 1.0)

        # Don't apply throttle if near max velocity
        if current_velocity > self.MAX_VELOCITY * 0.95:
            throttle = 0.0

        return float(throttle), float(brake)

    def _compute_steering(self, vehicle: carla.Vehicle, lookahead: float = 5.0) -> float:
        """
        Compute steering angle to follow the road using waypoints.

        Uses a simple pure pursuit-like approach to steer towards
        the next waypoint on the road.

        Args:
            vehicle: The vehicle to compute steering for
            lookahead: Distance ahead to look for waypoint (m)

        Returns:
            steer: Steering value [-1, 1]
        """
        # Get current vehicle transform
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle_transform.location

        # Get the waypoint at the vehicle's current location
        current_waypoint = self.map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if current_waypoint is None:
            return 0.0

        # Get waypoint ahead for steering target
        waypoints_ahead = current_waypoint.next(lookahead)
        if not waypoints_ahead:
            return 0.0

        target_waypoint = waypoints_ahead[0]
        target_location = target_waypoint.transform.location

        # Calculate steering angle
        # Vector from vehicle to target
        dx = target_location.x - vehicle_location.x
        dy = target_location.y - vehicle_location.y

        # Vehicle's forward direction (yaw in radians)
        vehicle_yaw = np.radians(vehicle_transform.rotation.yaw)

        # Angle to target in world frame
        target_angle = np.arctan2(dy, dx)

        # Angle difference (how much we need to turn)
        angle_diff = target_angle - vehicle_yaw

        # Normalize to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # Convert to steering command [-1, 1]
        # Typical max steering angle is ~70 degrees = 1.22 radians
        max_steer_angle = 1.22
        steer = np.clip(angle_diff / max_steer_angle, -1.0, 1.0)

        return float(steer)

    def _update_lead_vehicles(self):
        """Update all lead vehicles in the platoon."""
        if len(self.lead_vehicles) == 0:
            return

        # Get target velocity from trajectory for the front vehicle
        if self.lead_trajectory is None:
            front_target_velocity = 15.0  # m/s default
        else:
            if self.trajectory_idx < len(self.lead_trajectory):
                front_target_velocity = self.lead_trajectory[self.trajectory_idx]
                self.trajectory_idx += 1
            else:
                front_target_velocity = self.lead_trajectory[-1]

        # Update all vehicles from front to back
        for i in range(len(self.lead_vehicles) - 1, -1, -1):
            vehicle = self.lead_vehicles[i]

            if i == len(self.lead_vehicles) - 1:
                # Front vehicle: follow trajectory velocity
                target_velocity = front_target_velocity
            else:
                # Following vehicles: follow the vehicle in front
                leader = self.lead_vehicles[i + 1]
                leader_velocity = self._get_vehicle_speed(leader)

                # Calculate distance to leader
                follower_loc = vehicle.get_location()
                leader_loc = leader.get_location()
                distance = np.sqrt(
                    (leader_loc.x - follower_loc.x)**2 +
                    (leader_loc.y - follower_loc.y)**2
                )

                # Target velocity based on car-following
                follower_velocity = self._get_vehicle_speed(vehicle)
                desired_distance = self.TIME_HEADWAY * follower_velocity + 10.0
                distance_error = distance - desired_distance

                # Blend leader velocity with distance correction
                target_velocity = leader_velocity + 0.2 * distance_error
                target_velocity = np.clip(target_velocity, 0.0, self.MAX_VELOCITY)

            # Apply control
            self._control_vehicle_smooth(vehicle, target_velocity)

    def _control_vehicle_smooth(self, vehicle: carla.Vehicle, target_velocity: float):
        """Apply smooth velocity control to a vehicle with proper steering."""
        current_velocity = self._get_vehicle_speed(vehicle)

        # Gentle velocity control to avoid oscillations
        velocity_error = target_velocity - current_velocity
        kp = 0.3  # Reduced from 0.5 for smoother control

        # Clip acceleration more conservatively
        desired_accel = np.clip(kp * velocity_error, -2.0, 1.5)

        throttle, brake = self._acceleration_to_control(desired_accel, current_velocity)

        # Speed-adaptive lookahead: longer at higher speeds for better curve handling
        # At 20 m/s, lookahead = 10m; at 5 m/s, lookahead = 5m
        lookahead = max(5.0, 0.5 * current_velocity)
        steer = self._compute_steering(vehicle, lookahead=lookahead)

        control = carla.VehicleControl(
            throttle=float(throttle),
            brake=float(brake),
            steer=float(steer)
        )
        vehicle.apply_control(control)

    def _check_collision(self, distance_gap: float) -> bool:
        """
        Check if collision has occurred.

        Args:
            distance_gap: Current distance gap (m)

        Returns:
            collision: True if collision detected
        """
        # Collision threshold: 2 meters
        COLLISION_THRESHOLD = 2.0
        return distance_gap < COLLISION_THRESHOLD

    def _validate_lead_vehicles(self) -> List[Dict]:
        """
        Validate that all lead vehicles are still active and on the road.

        Returns:
            List of dictionaries with info about each lead vehicle
        """
        lead_info = []
        ego_location = self.ego_vehicle.get_location()

        for i, vehicle in enumerate(self.lead_vehicles):
            try:
                # Check if vehicle is still valid by accessing its location
                loc = vehicle.get_location()
                vel = self._get_vehicle_speed(vehicle)

                # Calculate distance from ego
                dist_from_ego = np.sqrt(
                    (loc.x - ego_location.x)**2 +
                    (loc.y - ego_location.y)**2
                )

                # Check if on road
                waypoint = self.map.get_waypoint(loc, project_to_road=True)
                on_road = waypoint is not None

                # Distance from waypoint center (lane position)
                if waypoint:
                    wp_loc = waypoint.transform.location
                    lane_offset = np.sqrt(
                        (loc.x - wp_loc.x)**2 + (loc.y - wp_loc.y)**2
                    )
                else:
                    lane_offset = -1.0

                info = {
                    'id': i,
                    'x': loc.x,
                    'y': loc.y,
                    'z': loc.z,
                    'velocity': vel,
                    'dist_from_ego': dist_from_ego,
                    'on_road': on_road,
                    'lane_offset': lane_offset,
                    'valid': True
                }

            except Exception as e:
                info = {'id': i, 'valid': False, 'error': str(e)}

            lead_info.append(info)

        return lead_info

    def _update_spectator(self):
        """
        Update spectator camera to follow the ego vehicle from behind.

        Provides a third-person view for visualization during simulation.
        """
        if self.ego_vehicle is None or self.world is None:
            return

        # Get ego vehicle transform
        ego_transform = self.ego_vehicle.get_transform()

        # Calculate camera position: behind and above the vehicle
        # Offset: 15 meters behind, 8 meters above (to see platoon)
        forward_vector = ego_transform.get_forward_vector()
        camera_location = carla.Location(
            x=ego_transform.location.x - 15.0 * forward_vector.x,
            y=ego_transform.location.y - 15.0 * forward_vector.y,
            z=ego_transform.location.z + 10.0
        )

        # Camera looks at the platoon (pitch down more to see all vehicles)
        camera_rotation = carla.Rotation(
            pitch=-20.0,
            yaw=ego_transform.rotation.yaw,
            roll=0.0
        )

        # Set spectator transform
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(camera_location, camera_rotation))

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

        # Get positions
        if self.ego_vehicle is not None:
            ego_loc = self.ego_vehicle.get_location()
            self.episode_data['ego_position'].append(ego_loc.x)
        if len(self.lead_vehicles) > 0:
            lead_loc = self.lead_vehicles[0].get_location()
            self.episode_data['lead_position'].append(lead_loc.x)

    def get_episode_data(self) -> Dict:
        """
        Get logged episode data.

        Returns:
            episode_data: Dictionary of logged time series
        """
        return self.episode_data.copy()

    def destroy(self):
        """Destroy all spawned actors and clean up."""
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None
            logger.debug("Ego vehicle destroyed")

        # Destroy all lead vehicles in platoon
        for i, vehicle in enumerate(self.lead_vehicles):
            if vehicle is not None:
                vehicle.destroy()
                logger.debug(f"Lead vehicle {i+1} destroyed")
        self.lead_vehicles = []
        logger.debug("All lead vehicles destroyed")

    def close(self):
        """Close the environment and restore CARLA settings."""
        logger.info("Closing CarFollowingEnv")

        # Destroy actors
        self.destroy()

        # Restore asynchronous mode
        if self.world is not None:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
                logger.info("Asynchronous mode restored")
            except Exception as e:
                logger.warning(f"Failed to restore async mode: {e}")

        self._connected = False

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


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
            # Simple proportional controller using closest lead
            ego_vel = obs[0]
            rel_vel_1, distance_1 = obs[1], obs[2]
            target_vel = ego_vel + rel_vel_1  # Match lead 1 velocity
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
