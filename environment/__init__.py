"""
Environment package for Hybrid DRL-MPC Eco-Driving Framework.

This package contains:
- CarFollowingEnv: CARLA-based car-following simulation environment
- HybridMPCEnv: Gymnasium wrapper for DRL training
"""

from environment.car_following import CarFollowingEnv
from environment.gym_wrapper import HybridMPCEnv, create_training_trajectory

__all__ = ['CarFollowingEnv', 'HybridMPCEnv', 'create_training_trajectory']
