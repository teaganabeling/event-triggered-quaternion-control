# controller.py: control law and related functions for the rigid-body quaternion attitude control simulation
import numpy as np

from config import SimulationConfig
from dynamics.quaternion import attitude_error_vector


def continuous_control(
    q: np.ndarray,
    omega: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """
    Continuous-time quaternion PD controller:
        u_c(q, omega) = -K_R e_R - K_omega omega
    where e_R is the attitude error vector extracted from the error quaternion.
    """
    q = np.asarray(q, dtype=float)
    omega = np.asarray(omega, dtype=float)

    if q.shape != (4,):
        raise ValueError(f"q must have shape (4,), got {q.shape}")
    if omega.shape != (3,):
        raise ValueError(f"omega must have shape (3,), got {omega.shape}")

    e_R = attitude_error_vector(q, config.q_d)
    u = -config.K_R @ e_R - config.K_omega @ omega
    return u


def control_mismatch(
    q: np.ndarray,
    omega: np.ndarray,
    u_held: np.ndarray,
    config: SimulationConfig,
) -> float:
    """
    Norm of the mismatch between the ideal continuous-time control and the
    currently held control input.
    """
    q = np.asarray(q, dtype=float)
    omega = np.asarray(omega, dtype=float)
    u_held = np.asarray(u_held, dtype=float)

    if u_held.shape != (3,):
        raise ValueError(f"u_held must have shape (3,), got {u_held.shape}")

    u_desired = continuous_control(q, omega, config)
    return float(np.linalg.norm(u_desired - u_held))


def control_components(
    q: np.ndarray,
    omega: np.ndarray,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the individual pieces of the control law for debugging/analysis:

        proportional_term = -K_R e_R
        derivative_term   = -K_omega omega
        total_control     = proportional_term + derivative_term
    """
    q = np.asarray(q, dtype=float)
    omega = np.asarray(omega, dtype=float)

    if q.shape != (4,):
        raise ValueError(f"q must have shape (4,), got {q.shape}")
    if omega.shape != (3,):
        raise ValueError(f"omega must have shape (3,), got {omega.shape}")

    e_R = attitude_error_vector(q, config.q_d)
    proportional_term = -config.K_R @ e_R
    derivative_term = -config.K_omega @ omega
    total_control = proportional_term + derivative_term

    return proportional_term, derivative_term, total_control