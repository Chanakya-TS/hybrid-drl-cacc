"""
Test script for Phase 1: CARLA Environment Wrapper

This script tests the CarFollowingEnv with a simple proportional controller.
"""

import numpy as np
import sys
from environment.car_following import CarFollowingEnv


def create_test_trajectory():
    """
    Create a challenging test trajectory for the lead vehicle.

    Includes: cruise, deceleration, slow cruise, acceleration, fast cruise, braking
    """
    trajectory = np.concatenate([
        np.linspace(15, 15, 100),   # Cruise at 15 m/s (54 km/h)
        np.linspace(15, 8, 50),     # Decelerate to 8 m/s
        np.linspace(8, 8, 100),     # Slow cruise
        np.linspace(8, 20, 60),     # Accelerate to 20 m/s (72 km/h)
        np.linspace(20, 20, 100),   # Fast cruise
        np.linspace(20, 5, 80),     # Brake to 5 m/s
        np.linspace(5, 5, 50),      # Very slow cruise
    ])
    return trajectory


def simple_proportional_controller(obs, kp_velocity=0.3, kp_distance=0.1):
    """
    Simple proportional controller for testing.

    Args:
        obs: Observation [ego_velocity, relative_velocity, distance_gap]
        kp_velocity: Gain for velocity matching
        kp_distance: Gain for distance regulation

    Returns:
        acceleration: Desired acceleration (m/s²)
    """
    ego_velocity, relative_velocity, distance_gap = obs

    # Target: match lead vehicle velocity and maintain safe distance
    lead_velocity = ego_velocity + relative_velocity

    # Desired distance based on time headway
    desired_distance = 1.8 * ego_velocity + 10.0  # THW * v + buffer
    distance_error = distance_gap - desired_distance

    # Control law
    velocity_term = kp_velocity * relative_velocity
    distance_term = kp_distance * distance_error

    acceleration = velocity_term + distance_term

    # Clip to reasonable limits
    acceleration = np.clip(acceleration, -2.0, 1.5)

    return acceleration


def main():
    """Run the Phase 1 test."""
    print("="*60)
    print("PHASE 1 TEST: CARLA Environment Wrapper")
    print("="*60)
    print()

    # Check if CARLA is running
    print("Prerequisites:")
    print("1. CARLA server must be running")
    print("2. Default connection: localhost:2000")
    print()
    print("To start CARLA server:")
    print("  CarlaUE4.exe -quality-level=Low -fps=20")
    print()

    input("Press ENTER when CARLA server is ready...")
    print()

    # Create test trajectory
    print("Creating test trajectory...")
    trajectory = create_test_trajectory()
    print(f"  Trajectory length: {len(trajectory)} steps")
    print(f"  Duration: {len(trajectory) * 0.05:.1f} seconds")
    print()

    # Create environment
    print("Creating CarFollowingEnv...")
    try:
        env = CarFollowingEnv(
            carla_host='localhost',
            carla_port=2000,
            dt=0.05,
            lead_vehicle_trajectory=trajectory,
            max_episode_steps=500
        )
        print("  Environment created successfully!")
    except Exception as e:
        print(f"  ERROR: Failed to create environment: {e}")
        print()
        print("Make sure CARLA server is running!")
        sys.exit(1)

    print()

    try:
        # Reset environment
        print("Resetting environment (spawning vehicles)...")
        obs, info = env.reset(seed=42)
        print("  Vehicles spawned successfully!")
        print(f"  Initial state:")
        print(f"    Ego velocity: {obs[0]:.2f} m/s ({obs[0]*3.6:.1f} km/h)")
        print(f"    Relative velocity: {obs[1]:.2f} m/s")
        print(f"    Distance gap: {obs[2]:.2f} m")
        print()

        # Run simulation
        print("Running simulation with simple controller...")
        print("-"*60)
        print(f"{'Step':>5} | {'Time':>6} | {'EgoVel':>7} | {'Gap':>7} | {'THW':>6} | {'Energy':>8}")
        print("-"*60)

        step = 0
        total_energy = 0.0

        while step < 200:  # Run for 200 steps (10 seconds at dt=0.05)
            # Simple proportional controller
            acceleration = simple_proportional_controller(obs)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(acceleration)

            # Extract info
            ego_vel = info['ego_velocity']
            distance_gap = info['distance_gap']
            time_headway = info['time_headway']
            total_energy = info['total_energy']

            # Print every 20 steps (1 second)
            if step % 20 == 0:
                print(f"{step:5d} | {step*0.05:6.2f} | {ego_vel:7.2f} | "
                      f"{distance_gap:7.2f} | {time_headway:6.2f} | {total_energy:8.2f}")

            # Check termination
            if terminated:
                print()
                print("  COLLISION DETECTED! Episode terminated.")
                break

            if truncated:
                print()
                print("  Max steps reached. Episode truncated.")
                break

            step += 1

        print("-"*60)
        print()

        # Get episode data
        episode_data = env.get_episode_data()

        # Calculate summary statistics
        print("Episode Summary:")
        print(f"  Total steps: {len(episode_data['time'])}")
        print(f"  Duration: {episode_data['time'][-1]:.2f} seconds")
        print(f"  Total energy: {total_energy:.2f}")
        print(f"  Average ego velocity: {np.mean(episode_data['ego_velocity']):.2f} m/s")
        print(f"  Min distance gap: {np.min(episode_data['distance_gap']):.2f} m")
        print(f"  Max distance gap: {np.max(episode_data['distance_gap']):.2f} m")

        # Calculate min time headway (excluding very low velocities)
        valid_indices = np.array(episode_data['ego_velocity']) > 2.0
        if np.any(valid_indices):
            thw = np.array(episode_data['distance_gap'])[valid_indices] / \
                  (np.array(episode_data['ego_velocity'])[valid_indices] + 1e-6)
            print(f"  Min time headway: {np.min(thw):.2f} s")

        # Calculate RMS jerk
        if len(episode_data['acceleration']) > 1:
            jerk = np.diff(episode_data['acceleration']) / 0.05
            rms_jerk = np.sqrt(np.mean(jerk**2))
            print(f"  RMS jerk: {rms_jerk:.2f} m/s³")

        print()
        print("="*60)
        print("PHASE 1 TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        print()
        print("Next steps:")
        print("  - Phase 2: Implement MPC controller")
        print("  - Phase 3: Implement DRL integration")
        print()

    except KeyboardInterrupt:
        print()
        print("Test interrupted by user.")
    except Exception as e:
        print()
        print(f"ERROR during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print("Cleaning up...")
        env.close()
        print("Environment closed. Test complete.")


if __name__ == "__main__":
    main()
