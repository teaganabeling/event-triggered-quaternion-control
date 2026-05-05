# rigid_body.py: dynamics functions for the rigid-body quaternion attitude control simulation
import numpy as np

from config import SimulationConfig, pack_state, unpack_state
from dynamics.quaternion import quat_normalize, quaternion_kinematics


def angular_acceleration(
    omega: np.ndarray,
    u: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """
    Rigid-body rotational dynamics:
        J * omega_dot = -omega * (J omega) + u
    s.t. 
        omega_dot = J^{-1} [ -omega * (J omega) + u ].
    """
    omega = np.asarray(omega, dtype=float)
    u = np.asarray(u, dtype=float)

    if omega.shape != (3,):
        raise ValueError(f"omega must have shape (3,), got {omega.shape}")
    if u.shape != (3,):
        raise ValueError(f"u must have shape (3,), got {u.shape}")

    J_omega = config.J @ omega
    coriolis_term = np.cross(omega, J_omega)
    omega_dot = config.J_inv @ (-coriolis_term + u)
    return omega_dot


def flow_rhs(
    t: float,
    x: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """
    Flow dynamics for the augmented hybrid state:
        x = [q0, q1, q2, q3, wx, wy, wz, ux, uy, uz]
    with
        q_dot     = 0.5 * q ⊗ [0, omega]
        omega_dot = J^{-1}(-omega x (J omega) + u)
        u_dot     = 0
    """
    _ = t  # included for solver compatibility

    q, omega, u = unpack_state(x)
    q = quat_normalize(q)

    q_dot = quaternion_kinematics(q, omega)
    omega_dot = angular_acceleration(omega, u, config)
    u_dot = np.zeros(3)

    return pack_state(q_dot, omega_dot, u_dot)


def flow_rhs_with_control(
    t: float,
    x_plant: np.ndarray,
    u: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """
    Plant-only flow dynamics for the state
        x_plant = [q0, q1, q2, q3, wx, wy, wz]
    when a control input u is supplied externally.
    Useful later for continuous-control stepping or alternative integrators.
    """
    _ = t

    x_plant = np.asarray(x_plant, dtype=float)
    u = np.asarray(u, dtype=float)

    if x_plant.shape != (7,):
        raise ValueError(f"x_plant must have shape (7,), got {x_plant.shape}")
    if u.shape != (3,):
        raise ValueError(f"u must have shape (3,), got {u.shape}")

    q = quat_normalize(x_plant[0:4])
    omega = x_plant[4:7]

    q_dot = quaternion_kinematics(q, omega)
    omega_dot = angular_acceleration(omega, u, config)

    return np.concatenate([q_dot, omega_dot])


def plant_energy(
    omega: np.ndarray,
    config: SimulationConfig,
) -> float:
    """
    Rotational kinetic energy:
        T = 0.5 * omega^T J omega
    """
    omega = np.asarray(omega, dtype=float)
    if omega.shape != (3,):
        raise ValueError(f"omega must have shape (3,), got {omega.shape}")

    return float(0.5 * omega @ config.J @ omega)