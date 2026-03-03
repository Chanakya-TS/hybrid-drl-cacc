"""
Environment package for Hybrid DRL-MPC Eco-Driving Framework.

This package contains:
- CarFollowingEnv: Pure-Python 1D longitudinal car-following simulation
- HybridMPCEnv: Gymnasium wrapper for DRL training
"""

from environment.car_following import CarFollowingEnv
from environment.gym_wrapper import HybridMPCEnv, create_training_trajectory

__all__ = ['CarFollowingEnv', 'HybridMPCEnv', 'create_training_trajectory']
