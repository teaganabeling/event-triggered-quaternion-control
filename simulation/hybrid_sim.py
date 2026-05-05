# hybrid_sim.py: run the event-triggered hybrid simulation for the rigid-body attitude control problem

"""
Psuedocode:

initialize state x
initialize u = u_c(q0, omega0)
last_event_time = 0
event_times = [0]

for each timestep:
    integrate flow one step with held u
    normalize quaternion
    compute ideal control u_c(q, omega)
    compute trigger value phi

    if phi >= 0 and (t - last_event_time) >= tau_min:
        u = u_c(q, omega)
        update state control component
        event_times.append(t)
        last_event_time = t

    log everything
    
"""

import numpy as np

from config import SimulationConfig, initial_state, pack_state, unpack_state
from control.controller import continuous_control
from control.trigger import trigger_diagnostics
from dynamics.quaternion import quat_normalize, attitude_error_vector
from dynamics.rigid_body import flow_rhs


def rk4_step_hybrid(
    x: np.ndarray,
    t: float,
    dt: float,
    config: SimulationConfig,
) -> np.ndarray:
    """
    One RK4 step for the full hybrid state:
        x = [q0, q1, q2, q3, wx, wy, wz, ux, uy, uz]
    with u held constant during the flow step.
    """
    k1 = flow_rhs(t, x, config)
    k2 = flow_rhs(t + 0.5 * dt, x + 0.5 * dt * k1, config)
    k3 = flow_rhs(t + 0.5 * dt, x + 0.5 * dt * k2, config)
    k4 = flow_rhs(t + dt, x + dt * k3, config)

    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    q_next, omega_next, u_next = unpack_state(x_next)
    q_next = quat_normalize(q_next)

    return pack_state(q_next, omega_next, u_next)


def run_hybrid_sim(config: SimulationConfig) -> dict:
    """
    Run the event-triggered hybrid simulation.
    """
    x = initial_state(config)

    # initialize held control with the continuous controller at the initial condition
    q0, omega0, _ = unpack_state(x)
    u0 = continuous_control(q0, omega0, config)
    x = pack_state(q0, omega0, u0)

    t_values = []
    q_values = []
    omega_values = []
    u_values = []
    e_R_values = []
    e_norm_values = []
    omega_norm_values = []

    phi_values = []
    sigma_values = []
    mismatch_values = []

    event_times = [config.t0]
    inter_event_times = []

    last_event_time = config.t0
    t = config.t0
    num_steps = int(np.floor((config.tf - config.t0) / config.dt)) + 1

    for _ in range(num_steps):
        q, omega, u_held = unpack_state(x)

        diag = trigger_diagnostics(q, omega, u_held, t, last_event_time, config)
        e_R = diag["e_R"]

        # log current state
        t_values.append(t)
        q_values.append(q.copy())
        omega_values.append(omega.copy())
        u_values.append(u_held.copy())
        e_R_values.append(e_R.copy())
        e_norm_values.append(diag["e_norm"])
        omega_norm_values.append(diag["omega_norm"])
        phi_values.append(diag["phi"])
        sigma_values.append(diag["sigma"])
        mismatch_values.append(diag["mismatch"])

        # flow step
        x = rk4_step_hybrid(x, t, config.dt, config)

        # check trigger after the flow step
        t_next = t + config.dt
        q_next, omega_next, u_next = unpack_state(x)
        diag_next = trigger_diagnostics(q_next, omega_next, u_next, t_next, last_event_time, config)

        if diag_next["should_trigger"]:
            u_updated = continuous_control(q_next, omega_next, config)
            x = pack_state(q_next, omega_next, u_updated)

            inter_event_times.append(t_next - last_event_time)
            event_times.append(t_next)
            last_event_time = t_next

        t = t_next

    return {
        "t": np.array(t_values),
        "q": np.array(q_values),
        "omega": np.array(omega_values),
        "u": np.array(u_values),
        "e_R": np.array(e_R_values),
        "e_norm": np.array(e_norm_values),
        "omega_norm": np.array(omega_norm_values),
        "phi": np.array(phi_values),
        "sigma": np.array(sigma_values),
        "mismatch": np.array(mismatch_values),
        "event_times": np.array(event_times),
        "inter_event_times": np.array(inter_event_times),
        "num_events": len(event_times),
    }