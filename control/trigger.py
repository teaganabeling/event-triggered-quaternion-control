# trigger.py: event-triggering logic for the rigid-body attitude control problem
import numpy as np

from config import SimulationConfig
from control.controller import continuous_control, control_mismatch
from dynamics.quaternion import attitude_error_vector


def trigger_threshold(
    q: np.ndarray,
    omega: np.ndarray,
    config: SimulationConfig,
) -> float:
    """
    Compute the state-dependent trigger threshold

        sigma(||e_R||, ||omega||).
    """
    e_R = attitude_error_vector(q, config.q_d)
    e_norm = np.linalg.norm(e_R)
    omega_norm = np.linalg.norm(omega)
    return float(config.sigma(e_norm, omega_norm))


def trigger_value(
    q: np.ndarray,
    omega: np.ndarray,
    u_held: np.ndarray,
    config: SimulationConfig,
) -> float:
    """
    Compute the trigger function

        phi = ||u_c(q, omega) - u_held|| - sigma(||e_R||, ||omega||).
    """
    mismatch = control_mismatch(q, omega, u_held, config)
    threshold = trigger_threshold(q, omega, config)
    return float(mismatch - threshold)


def should_trigger(
    q: np.ndarray,
    omega: np.ndarray,
    u_held: np.ndarray,
    t: float,
    last_event_time: float,
    config: SimulationConfig,
) -> bool:
    """
    Return True if an event should be triggered.

    Trigger conditions:
    1. trigger function phi >= 0
    2. minimum inter-event time is satisfied
    """
    enough_time_elapsed = (t - last_event_time) >= config.tau_min
    phi = trigger_value(q, omega, u_held, config)

    return bool(enough_time_elapsed and phi >= 0.0)


def trigger_diagnostics(
    q: np.ndarray,
    omega: np.ndarray,
    u_held: np.ndarray,
    t: float,
    last_event_time: float,
    config: SimulationConfig,
) -> dict:
    """
    Return useful intermediate quantities for debugging and plotting.
    """
    e_R = attitude_error_vector(q, config.q_d)
    e_norm = float(np.linalg.norm(e_R))
    omega_norm = float(np.linalg.norm(omega))

    sigma_e = config.sigma_e_gain * e_norm
    sigma_w = config.sigma_w_gain * omega_norm
    sigma_raw = sigma_e + sigma_w
    sigma_val = max(sigma_raw, config.sigma_floor)

    u_desired = continuous_control(q, omega, config)
    mismatch = float(np.linalg.norm(u_desired - u_held))
    phi = mismatch - sigma_val
    enough_time_elapsed = (t - last_event_time) >= config.tau_min

    return {
        "e_R": e_R,
        "e_norm": e_norm,
        "omega_norm": omega_norm,
        "sigma_e": sigma_e,
        "sigma_w": sigma_w,
        "sigma_raw": sigma_raw,
        "sigma": sigma_val,
        "u_desired": u_desired,
        "mismatch": mismatch,
        "phi": phi,
        "enough_time_elapsed": enough_time_elapsed,
        "should_trigger": bool(enough_time_elapsed and phi >= 0.0),
    }