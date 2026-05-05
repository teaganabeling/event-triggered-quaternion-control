# continuous_sim.py: run the continuous-time baseline simulation for the rigid-body attitude control problem

"""
Runs the baseline PD simulation.

At every step:
    read current q, omega
    compute u_c(q, omega)
    integrate one step with that control
    renormalize quaternion
    log data

Outputs:
    time history
    quaternion history
    angular velocity history
    control history
    error history

"""

import numpy as np

from config import SimulationConfig, pack_state, unpack_state, initial_state
from control.controller import continuous_control
from dynamics.quaternion import quat_normalize, attitude_error_vector
from dynamics.rigid_body import flow_rhs_with_control


def rk4_step(
    x_plant: np.ndarray,
    u: np.ndarray,
    t: float,
    dt: float,
    config: SimulationConfig,
) -> np.ndarray:
    """
    One RK4 step for the plant-only state:
        x_plant = [q0, q1, q2, q3, wx, wy, wz]
    """
    k1 = flow_rhs_with_control(t, x_plant, u, config)
    k2 = flow_rhs_with_control(t + 0.5 * dt, x_plant + 0.5 * dt * k1, u, config)
    k3 = flow_rhs_with_control(t + 0.5 * dt, x_plant + 0.5 * dt * k2, u, config)
    k4 = flow_rhs_with_control(t + dt, x_plant + dt * k3, u, config)

    x_next = x_plant + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # renormalize quaternion after integration
    q_next = quat_normalize(x_next[0:4])
    omega_next = x_next[4:7]

    return np.concatenate([q_next, omega_next])


def run_continuous_sim(config: SimulationConfig) -> dict:
    """
    Run the continuous-time baseline simulation.

    Returns a dictionary of logged trajectories and metrics.
    """
    x0 = initial_state(config)
    q0, omega0, _ = unpack_state(x0)

    x_plant = np.concatenate([q0, omega0])

    t_values = []
    q_values = []
    omega_values = []
    u_values = []
    e_R_values = []
    e_norm_values = []
    omega_norm_values = []

    t = config.t0
    num_steps = int(np.floor((config.tf - config.t0) / config.dt)) + 1

    for _ in range(num_steps):
        q = x_plant[0:4]
        omega = x_plant[4:7]

        u = continuous_control(q, omega, config)
        e_R = attitude_error_vector(q, config.q_d)

        t_values.append(t)
        q_values.append(q.copy())
        omega_values.append(omega.copy())
        u_values.append(u.copy())
        e_R_values.append(e_R.copy())
        e_norm_values.append(np.linalg.norm(e_R))
        omega_norm_values.append(np.linalg.norm(omega))

        x_plant = rk4_step(x_plant, u, t, config.dt, config)
        t += config.dt

    return {
        "t": np.array(t_values),
        "q": np.array(q_values),
        "omega": np.array(omega_values),
        "u": np.array(u_values),
        "e_R": np.array(e_R_values),
        "e_norm": np.array(e_norm_values),
        "omega_norm": np.array(omega_norm_values),
    }