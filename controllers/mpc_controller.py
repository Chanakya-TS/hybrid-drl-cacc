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

        State: [ego_velocity, distance_gap_1, distance_gap_2, distance_gap_3]
        Control: [acceleration]

        The MPC tracks distances to all 3 lead vehicles and uses anticipatory
        costs on leads 2 and 3 to enable smoother, earlier braking.
        """
        N = self.horizon

        # Define symbolic variables
        # States: ego velocity and 3 distance gaps
        v = ca.SX.sym('v', N + 1)   # Ego velocity over horizon
        d1 = ca.SX.sym('d1', N + 1)  # Distance gap to lead 1 (closest)
        d2 = ca.SX.sym('d2', N + 1)  # Distance gap to lead 2
        d3 = ca.SX.sym('d3', N + 1)  # Distance gap to lead 3
        a = ca.SX.sym('a', N)        # Acceleration (control) over horizon

        # Parameters layout:
        # [v0, d1_0, d2_0, d3_0,                        # 4 initial states
        #  v_lead1(0:N), v_lead2(0:N), v_lead3(0:N),    # 3*(N+1) lead velocity predictions
        #  a_prev, w_v, w_s, w_c, v_target]             # 5 scalars
        # Total: 4 + 3*(N+1) + 5 = 3N + 12
        n_params = 3 * N + 12
        params = ca.SX.sym('params', n_params)

        # Extract parameters
        v0 = params[0]     # Initial ego velocity
        d1_0 = params[1]   # Initial distance to lead 1
        d2_0 = params[2]   # Initial distance to lead 2
        d3_0 = params[3]   # Initial distance to lead 3

        # Lead vehicle velocity predictions for each lead
        idx = 4
        v_lead1 = params[idx:idx + N + 1]
        idx += N + 1
        v_lead2 = params[idx:idx + N + 1]
        idx += N + 1
        v_lead3 = params[idx:idx + N + 1]
        idx += N + 1

        # Scalar parameters
        a_prev = params[idx]       # Previous acceleration (for jerk penalty)
        w_v = params[idx + 1]      # Velocity tracking weight
        w_s = params[idx + 2]      # Safety weight
        w_c = params[idx + 3]      # Comfort weight
        v_target = params[idx + 4]  # Target velocity

        # Build cost function
        cost = 0.0

        # Initial conditions constraints
        constraints = []
        constraints.append(v[0] - v0)
        constraints.append(d1[0] - d1_0)
        constraints.append(d2[0] - d2_0)
        constraints.append(d3[0] - d3_0)

        for k in range(N):
            # ===== Dynamics constraints =====
            # Ego vehicle dynamics: v[k+1] = v[k] + a[k] * dt
            constraints.append(v[k + 1] - (v[k] + a[k] * self.dt))

            # Distance dynamics for each lead vehicle
            constraints.append(d1[k + 1] - (d1[k] + (v_lead1[k] - v[k]) * self.dt))
            constraints.append(d2[k + 1] - (d2[k] + (v_lead2[k] - v[k]) * self.dt))
            constraints.append(d3[k + 1] - (d3[k] + (v_lead3[k] - v[k]) * self.dt))

            # ===== Cost function =====
            # 1. Velocity tracking cost (eco-driving: match lead 1 velocity smoothly)
            rel_vel_error = v[k] - v_lead1[k]
            cost += w_v * 0.5 * rel_vel_error ** 2

            # Gently track target velocity
            velocity_error = v[k] - v_target
            cost += w_v * 0.1 * velocity_error ** 2

            # 2. Primary safety cost on lead 1 (closest)
            d1_desired = self.TIME_HEADWAY * v[k] + self.MIN_DISTANCE
            safety_error_1 = d1[k] - d1_desired
            # Asymmetric penalty: only penalize if too close
            cost += w_s * ca.fmax(0, -safety_error_1) ** 2
            # Small symmetric cost to maintain good distance
            cost += w_s * 0.01 * (d1[k] - d1_desired) ** 2

            # 3. Anticipatory safety cost on leads 2 and 3 (lighter weight)
            # Penalize if lead 2 is decelerating (v_lead2 dropping relative to v_lead1)
            # This encourages the ego to start braking earlier
            lead2_decel = v_lead1[k] - v_lead2[k]  # positive = lead2 slower than lead1
            cost += w_s * 0.05 * ca.fmax(0, lead2_decel) ** 2

            lead3_decel = v_lead2[k] - v_lead3[k]  # positive = lead3 slower than lead2
            cost += w_s * 0.02 * ca.fmax(0, lead3_decel) ** 2

            # If lead 2 or 3 are decelerating, penalize maintaining high speed
            # (encourages earlier, smoother braking)
            anticipated_decel = ca.fmax(0, v[k] - v_lead2[k])
            cost += w_s * 0.03 * anticipated_decel ** 2

            # 4. Comfort cost (minimize acceleration magnitude)
            cost += w_c * a[k] ** 2

            # 5. Jerk penalty (strong for smooth control)
            if k == 0:
                jerk = a[k] - a_prev
            else:
                jerk = a[k] - a[k - 1]
            cost += w_c * 2.0 * jerk ** 2

            # 6. Energy cost (penalize positive acceleration - eco-driving)
            cost += w_v * 0.1 * ca.fmax(0, a[k]) ** 2

        # Terminal cost (encourage reaching target velocity)
        cost += w_v * (v[N] - v_target) ** 2

        # Collect all decision variables
        opt_vars = ca.vertcat(
            ca.reshape(v, -1, 1),
            ca.reshape(d1, -1, 1),
            ca.reshape(d2, -1, 1),
            ca.reshape(d3, -1, 1),
            ca.reshape(a, -1, 1)
        )

        # Constraints vector
        g = ca.vertcat(*constraints)

        # Variable bounds
        n_v = N + 1   # Velocity variables
        n_d = N + 1   # Distance variables (per lead)
        n_a = N       # Acceleration variables
        n_vars = n_v + 3 * n_d + n_a

        lbx = []
        ubx = []

        # Velocity bounds
        for _ in range(n_v):
            lbx.append(self.MIN_VELOCITY)
            ubx.append(self.MAX_VELOCITY)

        # Distance bounds for d1, d2, d3 (must be positive)
        for _ in range(3 * n_d):
            lbx.append(0.0)
            ubx.append(500.0)  # Upper bound (d2, d3 can be larger)

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

        logger.debug("MPC optimizer built successfully (4-state, 3-lead formulation)")

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
        lead_velocities: Optional[list] = None,
        distance_gaps: Optional[list] = None,
        lead_accelerations: Optional[list] = None,
        lead_velocity: Optional[float] = None,
        distance_gap: Optional[float] = None
    ) -> Tuple[float, Dict]:
        """
        Compute optimal acceleration using MPC with 3-lead awareness.

        Accepts either the new multi-lead interface (lead_velocities, distance_gaps,
        lead_accelerations as lists of 3) or the legacy single-lead interface
        (lead_velocity, distance_gap) for backward compatibility.

        Args:
            ego_velocity: Current ego vehicle velocity (m/s)
            lead_velocities: Velocities of all 3 leads [v1, v2, v3] (m/s)
            distance_gaps: Distances to all 3 leads [d1, d2, d3] (m)
            lead_accelerations: Accelerations of all 3 leads [a1, a2, a3] (m/s²)
            lead_velocity: (legacy) Single lead velocity (m/s)
            distance_gap: (legacy) Single distance gap (m)

        Returns:
            acceleration: Optimal acceleration command (m/s²)
            info: Dictionary with solver information
        """
        self.solve_count += 1

        # Handle legacy single-lead interface
        if lead_velocities is None:
            lv = lead_velocity if lead_velocity is not None else ego_velocity
            dg = distance_gap if distance_gap is not None else 50.0
            lead_velocities = [lv, lv, lv]
            distance_gaps = [dg, dg + 15.0, dg + 30.0]
            lead_accelerations = [0.0, 0.0, 0.0]

        # Build lead velocity predictions (assume constant over horizon)
        N = self.horizon
        v_lead1_pred = np.ones(N + 1) * lead_velocities[0]
        v_lead2_pred = np.ones(N + 1) * lead_velocities[1]
        v_lead3_pred = np.ones(N + 1) * lead_velocities[2]

        # If leads are decelerating, project deceleration into predictions
        for i, (pred, accel) in enumerate([
            (v_lead1_pred, lead_accelerations[0]),
            (v_lead2_pred, lead_accelerations[1]),
            (v_lead3_pred, lead_accelerations[2])
        ]):
            if accel < -0.5:  # Only project significant deceleration
                for k in range(1, N + 1):
                    pred[k] = max(0.0, pred[k - 1] + accel * self.dt)

        # Build parameter vector: [v0, d1_0, d2_0, d3_0, v_lead1(N+1), v_lead2(N+1),
        #                           v_lead3(N+1), a_prev, w_v, w_s, w_c, v_target]
        params = np.zeros(self.n_params)
        params[0] = ego_velocity
        params[1] = distance_gaps[0]
        params[2] = distance_gaps[1]
        params[3] = distance_gaps[2]

        idx = 4
        params[idx:idx + N + 1] = v_lead1_pred
        idx += N + 1
        params[idx:idx + N + 1] = v_lead2_pred
        idx += N + 1
        params[idx:idx + N + 1] = v_lead3_pred
        idx += N + 1

        params[idx] = self.previous_acceleration
        params[idx + 1] = self.w_velocity
        params[idx + 2] = self.w_safety
        params[idx + 3] = self.w_comfort
        params[idx + 4] = self.target_velocity

        # Initial guess (warm start or default)
        if self.x0 is None:
            x0 = np.zeros(self.n_vars)
            n_state = N + 1
            # Initialize velocity
            x0[:n_state] = ego_velocity
            # Initialize d1, d2, d3
            x0[n_state:2 * n_state] = distance_gaps[0]
            x0[2 * n_state:3 * n_state] = distance_gaps[1]
            x0[3 * n_state:4 * n_state] = distance_gaps[2]
            # Accelerations default to zero
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

            stats = self.solver.stats()
            success = stats['success']

            if success:
                x_opt = np.array(sol['x']).flatten()

                # Acceleration starts after v, d1, d2, d3
                n_state = N + 1
                a_start = 4 * n_state
                acceleration = float(x_opt[a_start])

                # Store solution for warm start
                self.x0 = x_opt

                # Update previous acceleration
                self.previous_acceleration = acceleration

                info = {
                    'success': True,
                    'cost': float(sol['f']),
                    'iterations': stats.get('iter_count', 0),
                    'predicted_velocity': x_opt[:n_state],
                    'predicted_d1': x_opt[n_state:2 * n_state],
                    'predicted_d2': x_opt[2 * n_state:3 * n_state],
                    'predicted_d3': x_opt[3 * n_state:4 * n_state],
                    'planned_acceleration': x_opt[a_start:a_start + N]
                }

            else:
                logger.warning(f"MPC solver failed: {stats.get('return_status', 'unknown')}")
                self.solve_failures += 1
                acceleration = self._fallback_control(
                    ego_velocity, lead_velocities[0], distance_gaps[0]
                )
                info = {'success': False, 'fallback': True}

        except Exception as e:
            logger.error(f"MPC solver exception: {e}")
            self.solve_failures += 1
            acceleration = self._fallback_control(
                ego_velocity, lead_velocities[0], distance_gaps[0]
            )
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
    distance_gaps = [30.0, 45.0, 60.0]  # d1, d2, d3

    # Data logging
    ego_velocities = [ego_velocity]
    distances = [distance_gaps[0]]
    accelerations = []

    for i in range(n_steps):
        lead_vel = lead_velocities[min(i, len(lead_velocities) - 1)]
        # Assume leads 2 and 3 have same velocity as lead 1 for this test
        all_lead_vels = [lead_vel, lead_vel, lead_vel]

        # Compute control using new multi-lead interface
        accel, info = mpc.compute_control(
            ego_velocity=ego_velocity,
            lead_velocities=all_lead_vels,
            distance_gaps=distance_gaps,
            lead_accelerations=[0.0, 0.0, 0.0]
        )
        accelerations.append(accel)

        # Update state (simple integration)
        ego_velocity += accel * dt
        ego_velocity = np.clip(ego_velocity, 0.0, 30.0)

        for j in range(3):
            rel_vel_j = all_lead_vels[j] - ego_velocity
            distance_gaps[j] += rel_vel_j * dt

        ego_velocities.append(ego_velocity)
        distances.append(distance_gaps[0])

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
