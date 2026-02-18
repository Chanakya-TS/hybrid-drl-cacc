"""
Model Predictive Controller (MPC) for Eco-Driving Car-Following.

This module implements an MPC controller using CasADi for optimization.
The controller optimizes a weighted cost function balancing:
- Velocity tracking (efficiency)
- Safety distance maintenance
- Control smoothness (comfort)

The weights can be dynamically adjusted by a DRL agent for adaptive control.
"""

import numpy as np
import casadi as ca
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class MPCController:
    """
    Model Predictive Controller for car-following eco-driving.

    Uses CasADi to solve a nonlinear optimization problem at each timestep,
    computing optimal acceleration commands while respecting constraints.
    """

    # Physical constants
    MAX_VELOCITY = 30.0  # m/s (~108 km/h)
    MIN_VELOCITY = 0.0  # m/s
    MAX_ACCELERATION = 2.0  # m/s²
    MAX_BRAKING = -3.0  # m/s² (negative for deceleration)
    TIME_HEADWAY = 1.8  # seconds (safety constraint)
    MIN_DISTANCE = 5.0  # meters (minimum safe distance at any speed)

    def __init__(
        self,
        dt: float = 0.05,
        horizon: int = 20,
        w_velocity: float = 0.4,
        w_safety: float = 0.4,
        w_comfort: float = 0.2,
        target_velocity: float = 20.0
    ):
        """
        Initialize the MPC controller.

        Args:
            dt: Timestep in seconds
            horizon: Prediction horizon (number of steps)
            w_velocity: Weight for velocity tracking cost
            w_safety: Weight for safety distance cost
            w_comfort: Weight for control effort cost
            target_velocity: Desired cruising velocity (m/s)
        """
        self.dt = dt
        self.horizon = horizon
        self.target_velocity = target_velocity

        # Cost function weights (can be updated by DRL agent)
        self.w_velocity = w_velocity
        self.w_safety = w_safety
        self.w_comfort = w_comfort

        # Build the optimization problem
        self._build_optimizer()

        # Previous control for smoothness
        self.previous_acceleration = 0.0

        # Logging
        self.solve_count = 0
        self.solve_failures = 0

        logger.info(
            f"MPCController initialized: horizon={horizon}, dt={dt}, "
            f"weights=[v:{w_velocity}, s:{w_safety}, c:{w_comfort}]"
        )

    def _build_optimizer(self):
        """
        Build the CasADi optimization problem.

        State: [ego_velocity, distance_gap]
        Control: [acceleration]
        """
        N = self.horizon

        # Define symbolic variables
        # States: ego velocity (v) and distance gap (d)
        v = ca.SX.sym('v', N + 1)  # Ego velocity over horizon
        d = ca.SX.sym('d', N + 1)  # Distance gap over horizon
        a = ca.SX.sym('a', N)  # Acceleration (control) over horizon

        # Parameters (passed at each solve)
        # [v0, d0, v_rel0, v_lead0, v_lead_pred(0:N), a_prev, w_v, w_s, w_c, v_target]
        # 4 initial states + (N+1) lead predictions + 1 prev_a + 3 weights + 1 target = N + 10
        n_params = N + 10
        params = ca.SX.sym('params', n_params)

        # Extract parameters
        v0 = params[0]  # Initial ego velocity
        d0 = params[1]  # Initial distance gap
        v_rel0 = params[2]  # Initial relative velocity
        v_lead0 = params[3]  # Initial lead velocity

        # Lead vehicle velocity predictions (assume constant for simplicity)
        v_lead = params[4:4 + N + 1]

        # Previous acceleration (for jerk penalty)
        a_prev = params[4 + N + 1]

        # Weights
        w_v = params[4 + N + 2]
        w_s = params[4 + N + 3]
        w_c = params[4 + N + 4]

        # Target velocity
        v_target = params[4 + N + 5]

        # Build cost function
        cost = 0.0

        # Initial conditions constraints
        constraints = []
        constraints.append(v[0] - v0)
        constraints.append(d[0] - d0)

        for k in range(N):
            # ===== Dynamics constraints =====
            # Ego vehicle dynamics: v[k+1] = v[k] + a[k] * dt
            constraints.append(v[k + 1] - (v[k] + a[k] * self.dt))

            # Distance dynamics: d[k+1] = d[k] + (v_lead[k] - v[k]) * dt
            constraints.append(d[k + 1] - (d[k] + (v_lead[k] - v[k]) * self.dt))

            # ===== Cost function =====
            # 1. Velocity tracking cost (eco-driving: match lead velocity smoothly)
            # Penalize deviation from lead vehicle velocity (for car-following)
            rel_vel_error = v[k] - v_lead[k]
            cost += w_v * 0.5 * rel_vel_error ** 2

            # Also gently track target velocity when far from lead
            velocity_error = v[k] - v_target
            cost += w_v * 0.1 * velocity_error ** 2

            # 2. Safety cost (penalize being too close)
            # Desired distance based on time headway
            d_desired = self.TIME_HEADWAY * v[k] + self.MIN_DISTANCE
            safety_error = d[k] - d_desired
            # Only penalize if too close (asymmetric penalty)
            cost += w_s * ca.fmax(0, -safety_error) ** 2
            # Small reward for maintaining good distance (encourages coasting)
            cost += w_s * 0.01 * (d[k] - d_desired) ** 2

            # 3. Comfort cost (minimize acceleration magnitude)
            cost += w_c * a[k] ** 2

            # 4. Jerk penalty (MUCH stronger for smooth control)
            if k == 0:
                jerk = a[k] - a_prev
            else:
                jerk = a[k] - a[k - 1]
            cost += w_c * 2.0 * jerk ** 2  # Increased from 0.1 to 2.0

            # 5. Energy cost (penalize positive acceleration - eco-driving)
            cost += w_v * 0.1 * ca.fmax(0, a[k]) ** 2

        # Terminal cost (encourage reaching target velocity)
        cost += w_v * (v[N] - v_target) ** 2

        # Collect all decision variables
        opt_vars = ca.vertcat(
            ca.reshape(v, -1, 1),
            ca.reshape(d, -1, 1),
            ca.reshape(a, -1, 1)
        )

        # Constraints vector
        g = ca.vertcat(*constraints)

        # Variable bounds
        n_v = N + 1  # Number of velocity variables
        n_d = N + 1  # Number of distance variables
        n_a = N  # Number of acceleration variables
        n_vars = n_v + n_d + n_a

        # Lower and upper bounds for variables
        lbx = []
        ubx = []

        # Velocity bounds
        for _ in range(n_v):
            lbx.append(self.MIN_VELOCITY)
            ubx.append(self.MAX_VELOCITY)

        # Distance bounds (must be positive)
        for _ in range(n_d):
            lbx.append(0.0)
            ubx.append(200.0)  # Large upper bound

        # Acceleration bounds
        for _ in range(n_a):
            lbx.append(self.MAX_BRAKING)
            ubx.append(self.MAX_ACCELERATION)

        # Constraint bounds (all equality constraints = 0)
        n_constraints = len(constraints)
        lbg = [0.0] * n_constraints
        ubg = [0.0] * n_constraints

        # Create NLP problem
        nlp = {
            'x': opt_vars,
            'f': cost,
            'g': g,
            'p': params
        }

        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.tol': 1e-4
        }

        # Create solver
        self.solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, opts)

        # Store problem dimensions
        self.n_vars = n_vars
        self.n_constraints = n_constraints
        self.n_params = n_params
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg

        # Warm start solution
        self.x0 = None

        logger.debug("MPC optimizer built successfully")

    def set_weights(self, w_velocity: float, w_safety: float, w_comfort: float):
        """
        Update the cost function weights.

        Called by DRL agent to adapt the controller behavior.

        Args:
            w_velocity: Weight for velocity tracking (0-1)
            w_safety: Weight for safety distance (0-1)
            w_comfort: Weight for control effort (0-1)
        """
        # Normalize weights to sum to 1
        total = w_velocity + w_safety + w_comfort + 1e-8
        self.w_velocity = w_velocity / total
        self.w_safety = w_safety / total
        self.w_comfort = w_comfort / total

        logger.debug(
            f"Weights updated: v={self.w_velocity:.3f}, "
            f"s={self.w_safety:.3f}, c={self.w_comfort:.3f}"
        )

    def compute_control(
        self,
        ego_velocity: float,
        lead_velocity: float,
        distance_gap: float,
        lead_velocity_prediction: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict]:
        """
        Compute optimal acceleration using MPC.

        Args:
            ego_velocity: Current ego vehicle velocity (m/s)
            lead_velocity: Current lead vehicle velocity (m/s)
            distance_gap: Current distance gap (m)
            lead_velocity_prediction: Predicted lead velocity over horizon (optional)

        Returns:
            acceleration: Optimal acceleration command (m/s²)
            info: Dictionary with solver information
        """
        self.solve_count += 1

        # Relative velocity
        relative_velocity = lead_velocity - ego_velocity

        # Default lead velocity prediction (assume constant)
        if lead_velocity_prediction is None:
            lead_velocity_prediction = np.ones(self.horizon + 1) * lead_velocity

        # Build parameter vector
        params = np.zeros(self.n_params)
        params[0] = ego_velocity
        params[1] = distance_gap
        params[2] = relative_velocity
        params[3] = lead_velocity
        params[4:4 + self.horizon + 1] = lead_velocity_prediction[:self.horizon + 1]
        params[4 + self.horizon + 1] = self.previous_acceleration
        params[4 + self.horizon + 2] = self.w_velocity
        params[4 + self.horizon + 3] = self.w_safety
        params[4 + self.horizon + 4] = self.w_comfort
        params[4 + self.horizon + 5] = self.target_velocity

        # Initial guess (warm start or default)
        if self.x0 is None:
            # Default initialization
            x0 = np.zeros(self.n_vars)
            # Initialize velocities
            x0[:self.horizon + 1] = ego_velocity
            # Initialize distances
            x0[self.horizon + 1:2 * (self.horizon + 1)] = distance_gap
            # Initialize accelerations to zero
        else:
            x0 = self.x0

        # Solve the optimization problem
        try:
            sol = self.solver(
                x0=x0,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=params
            )

            # Check solver status
            stats = self.solver.stats()
            success = stats['success']

            if success:
                # Extract solution
                x_opt = np.array(sol['x']).flatten()

                # Get first acceleration command
                a_start = 2 * (self.horizon + 1)
                acceleration = float(x_opt[a_start])

                # Store solution for warm start
                self.x0 = x_opt

                # Update previous acceleration
                self.previous_acceleration = acceleration

                info = {
                    'success': True,
                    'cost': float(sol['f']),
                    'iterations': stats.get('iter_count', 0),
                    'predicted_velocity': x_opt[:self.horizon + 1],
                    'predicted_distance': x_opt[self.horizon + 1:2 * (self.horizon + 1)],
                    'planned_acceleration': x_opt[a_start:a_start + self.horizon]
                }

            else:
                # Solver failed - use fallback
                logger.warning(f"MPC solver failed: {stats.get('return_status', 'unknown')}")
                self.solve_failures += 1
                acceleration = self._fallback_control(ego_velocity, lead_velocity, distance_gap)
                info = {'success': False, 'fallback': True}

        except Exception as e:
            logger.error(f"MPC solver exception: {e}")
            self.solve_failures += 1
            acceleration = self._fallback_control(ego_velocity, lead_velocity, distance_gap)
            info = {'success': False, 'error': str(e)}

        # Clip acceleration to bounds
        acceleration = np.clip(acceleration, self.MAX_BRAKING, self.MAX_ACCELERATION)

        return acceleration, info

    def _fallback_control(
        self,
        ego_velocity: float,
        lead_velocity: float,
        distance_gap: float
    ) -> float:
        """
        Simple fallback controller when MPC fails.

        Uses a proportional controller for safety.
        """
        # Desired distance based on time headway
        d_desired = self.TIME_HEADWAY * ego_velocity + self.MIN_DISTANCE

        # Distance error
        d_error = distance_gap - d_desired

        # Velocity error
        v_error = lead_velocity - ego_velocity

        # Simple proportional control
        kp_distance = 0.3
        kp_velocity = 0.5

        acceleration = kp_distance * d_error + kp_velocity * v_error

        # Clip to safe bounds
        acceleration = np.clip(acceleration, self.MAX_BRAKING, self.MAX_ACCELERATION)

        return acceleration

    def reset(self):
        """Reset controller state for new episode."""
        self.previous_acceleration = 0.0
        self.x0 = None
        self.solve_count = 0
        self.solve_failures = 0

    def get_stats(self) -> Dict:
        """Get controller statistics."""
        return {
            'solve_count': self.solve_count,
            'solve_failures': self.solve_failures,
            'success_rate': (
                (self.solve_count - self.solve_failures) / max(1, self.solve_count)
            ),
            'weights': {
                'velocity': self.w_velocity,
                'safety': self.w_safety,
                'comfort': self.w_comfort
            }
        }


class FixedWeightMPC(MPCController):
    """
    MPC with fixed weights (baseline for comparison).

    This is the same as MPCController but emphasizes that weights are fixed.
    """

    def __init__(
        self,
        dt: float = 0.05,
        horizon: int = 20,
        w_velocity: float = 0.4,
        w_safety: float = 0.4,
        w_comfort: float = 0.2,
        target_velocity: float = 20.0
    ):
        super().__init__(
            dt=dt,
            horizon=horizon,
            w_velocity=w_velocity,
            w_safety=w_safety,
            w_comfort=w_comfort,
            target_velocity=target_velocity
        )
        logger.info("FixedWeightMPC initialized (baseline controller)")

    def set_weights(self, *args, **kwargs):
        """Override to prevent weight changes."""
        logger.warning("FixedWeightMPC: Weights are fixed and cannot be changed")


# Test script
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO)

    print("Testing MPC Controller...")

    # Create controller
    mpc = MPCController(
        dt=0.1,
        horizon=20,
        w_velocity=0.4,
        w_safety=0.4,
        w_comfort=0.2,
        target_velocity=20.0
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
        accel, info = mpc.compute_control(ego_velocity, lead_vel, distance_gap)
        accelerations.append(accel)

        # Update state (simple integration)
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
    axes[0].set_title('MPC Controller Test')

    axes[1].plot(time[:-1], accelerations, linewidth=2, color='orange')
    axes[1].set_ylabel('Acceleration (m/s²)')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(True)

    axes[2].plot(time, distances, linewidth=2, color='green')
    axes[2].axhline(y=mpc.TIME_HEADWAY * 20 + mpc.MIN_DISTANCE, color='red', linestyle='--', label='Safe distance')
    axes[2].set_ylabel('Distance Gap (m)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('mpc_test.png', dpi=150)
    plt.show()

    # Print stats
    stats = mpc.get_stats()
    print(f"\nMPC Stats: {stats}")
    print("MPC test completed!")
