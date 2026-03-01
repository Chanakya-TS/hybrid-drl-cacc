"""
Adaptive Cruise Control (ACC) Baseline Controller.

This module implements a classic ACC controller using PID control
for maintaining a safe time headway behind a lead vehicle.
Used as a baseline for comparison with the hybrid DRL-MPC approach.
"""

import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ACCController:
    """
    Adaptive Cruise Control using PID for car-following.

    Maintains a desired time headway (THW) to the lead vehicle
    using proportional-integral-derivative control.
    """

    # Physical constants
    MAX_VELOCITY = 36.0  # m/s (~130 km/h, accommodates US06 peak)
    MIN_VELOCITY = 0.0  # m/s
    MAX_ACCELERATION = 2.0  # m/s²
    MAX_BRAKING = -3.0  # m/s²

    def __init__(
        self,
        time_headway: float = 1.8,
        min_distance: float = 5.0,
        target_velocity: float = 20.0,
        kp_distance: float = 0.4,
        ki_distance: float = 0.01,
        kd_distance: float = 0.2,
        kp_velocity: float = 0.5,
        dt: float = 0.05
    ):
        """
        Initialize the ACC controller.

        Args:
            time_headway: Desired time headway in seconds
            min_distance: Minimum distance gap at any speed (m)
            target_velocity: Desired cruising velocity when no lead vehicle (m/s)
            kp_distance: Proportional gain for distance control
            ki_distance: Integral gain for distance control
            kd_distance: Derivative gain for distance control
            kp_velocity: Proportional gain for velocity matching
            dt: Timestep in seconds
        """
        self.time_headway = time_headway
        self.min_distance = min_distance
        self.target_velocity = target_velocity
        self.dt = dt

        # PID gains
        self.kp_distance = kp_distance
        self.ki_distance = ki_distance
        self.kd_distance = kd_distance
        self.kp_velocity = kp_velocity

        # PID state
        self.integral_error = 0.0
        self.previous_error = 0.0

        # Anti-windup limits
        self.integral_max = 10.0

        # Statistics
        self.step_count = 0

        logger.info(
            f"ACCController initialized: THW={time_headway}s, "
            f"target_vel={target_velocity} m/s"
        )

    def compute_control(
        self,
        ego_velocity: float,
        lead_velocity: float,
        distance_gap: float
    ) -> Tuple[float, Dict]:
        """
        Compute acceleration command using PID control.

        Args:
            ego_velocity: Current ego vehicle velocity (m/s)
            lead_velocity: Current lead vehicle velocity (m/s)
            distance_gap: Current distance gap (m)

        Returns:
            acceleration: Acceleration command (m/s²)
            info: Dictionary with controller information
        """
        self.step_count += 1

        # Compute desired distance based on time headway
        d_desired = self.time_headway * ego_velocity + self.min_distance

        # Distance error (positive = too far, negative = too close)
        distance_error = distance_gap - d_desired

        # Velocity error (relative velocity)
        velocity_error = lead_velocity - ego_velocity

        # PID control for distance
        # Proportional term
        p_term = self.kp_distance * distance_error

        # Integral term with anti-windup
        self.integral_error += distance_error * self.dt
        self.integral_error = np.clip(
            self.integral_error,
            -self.integral_max,
            self.integral_max
        )
        i_term = self.ki_distance * self.integral_error

        # Derivative term
        d_term = self.kd_distance * (distance_error - self.previous_error) / self.dt
        self.previous_error = distance_error

        # Velocity matching term
        v_term = self.kp_velocity * velocity_error

        # Combine control terms
        acceleration = p_term + i_term + d_term + v_term

        # If far from lead vehicle or no lead, use cruise control
        if distance_gap > 100.0:
            # Pure velocity control to reach target
            acceleration = self.kp_velocity * (self.target_velocity - ego_velocity)

        # Clip to physical limits
        acceleration = np.clip(acceleration, self.MAX_BRAKING, self.MAX_ACCELERATION)

        # Safety override: hard brake if too close
        min_safe_distance = self.min_distance
        if distance_gap < min_safe_distance:
            # Emergency braking
            acceleration = self.MAX_BRAKING
            logger.warning(f"ACC: Emergency braking! Gap={distance_gap:.1f}m")

        info = {
            'desired_distance': d_desired,
            'distance_error': distance_error,
            'velocity_error': velocity_error,
            'p_term': p_term,
            'i_term': i_term,
            'd_term': d_term,
            'v_term': v_term
        }

        return float(acceleration), info

    def reset(self):
        """Reset controller state for new episode."""
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.step_count = 0

    def get_stats(self) -> Dict:
        """Get controller statistics."""
        return {
            'step_count': self.step_count,
            'time_headway': self.time_headway,
            'target_velocity': self.target_velocity,
            'pid_gains': {
                'kp_distance': self.kp_distance,
                'ki_distance': self.ki_distance,
                'kd_distance': self.kd_distance,
                'kp_velocity': self.kp_velocity
            }
        }


# Test script
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO)

    print("Testing ACC Controller...")

    # Create controller
    acc = ACCController(
        time_headway=1.8,
        target_velocity=20.0,
        dt=0.1
    )

    # Simulate car-following scenario
    dt = 0.1
    n_steps = 200

    # Lead vehicle velocity profile
    lead_velocities = np.concatenate([
        np.ones(50) * 20.0,  # Cruise
        np.linspace(20.0, 10.0, 30),  # Brake
        np.ones(50) * 10.0,  # Slow cruise
        np.linspace(10.0, 25.0, 30),  # Accelerate
        np.ones(40) * 25.0  # Fast cruise
    ])

    # Initial conditions
    ego_velocity = 15.0
    distance_gap = 30.0

    # Data logging
    ego_velocities = [ego_velocity]
    distances = [distance_gap]
    accelerations = []

    for i in range(n_steps):
        lead_vel = lead_velocities[min(i, len(lead_velocities) - 1)]

        # Compute control
        accel, info = acc.compute_control(ego_velocity, lead_vel, distance_gap)
        accelerations.append(accel)

        # Update state
        ego_velocity += accel * dt
        ego_velocity = np.clip(ego_velocity, 0.0, 30.0)

        rel_vel = lead_vel - ego_velocity
        distance_gap += rel_vel * dt

        ego_velocities.append(ego_velocity)
        distances.append(distance_gap)

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    time = np.arange(n_steps + 1) * dt

    axes[0].plot(time, ego_velocities, label='Ego', linewidth=2)
    axes[0].plot(time[:len(lead_velocities)], lead_velocities[:len(time)], label='Lead', linestyle='--')
    axes[0].set_ylabel('Velocity (m/s)')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title('ACC Controller Test')

    axes[1].plot(time[:-1], accelerations, linewidth=2, color='orange')
    axes[1].set_ylabel('Acceleration (m/s²)')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(True)

    axes[2].plot(time, distances, linewidth=2, color='green')
    desired_d = [acc.time_headway * v + acc.min_distance for v in ego_velocities]
    axes[2].plot(time, desired_d, 'r--', label='Desired distance')
    axes[2].set_ylabel('Distance Gap (m)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('acc_test.png', dpi=150)
    plt.show()

    print("ACC test completed!")
