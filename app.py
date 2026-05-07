# app.py
# Streamlit webapp for the event-triggered quaternion attitude-control simulation

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from config import SimulationConfig
from simulation.continuous_sim import run_continuous_sim
from simulation.hybrid_sim import run_hybrid_sim
from web_animation import make_dashboard_animation_html


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="Event-Triggered Quaternion Attitude Control",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    .interpretation-box {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 0.75rem;
        border: 1px solid #e5e7eb;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helper functions
# ============================================================

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Normalize a scalar-first quaternion q = [q0, q1, q2, q3].
    If the norm is too small, return the identity quaternion.
    """
    norm_q = np.linalg.norm(q)

    if norm_q < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])

    q = q / norm_q

    # Optional sign convention: keep scalar part nonnegative.
    if q[0] < 0:
        q = -q

    return q


def summarize_results(continuous: dict, hybrid: dict) -> dict:
    """
    Compute summary metrics for the webapp.
    """
    continuous_steps = len(continuous["t"])

    hybrid_updates_including_initial = int(hybrid["num_events"])
    hybrid_triggered_updates_after_t0 = max(hybrid_updates_including_initial - 1, 0)

    update_reduction = 100.0 * (
        1.0 - hybrid_updates_including_initial / max(continuous_steps, 1)
    )

    inter_event_times = hybrid.get("inter_event_times", np.array([]))

    if len(inter_event_times) > 0:
        mean_inter_event = float(np.mean(inter_event_times))
        min_inter_event = float(np.min(inter_event_times))
        max_inter_event = float(np.max(inter_event_times))
    else:
        mean_inter_event = float("nan")
        min_inter_event = float("nan")
        max_inter_event = float("nan")

    return {
        "continuous_steps": continuous_steps,
        "hybrid_updates_including_initial": hybrid_updates_including_initial,
        "hybrid_triggered_updates_after_t0": hybrid_triggered_updates_after_t0,
        "update_reduction_percent": update_reduction,
        "final_continuous_error": float(continuous["e_norm"][-1]),
        "final_hybrid_error": float(hybrid["e_norm"][-1]),
        "final_continuous_omega": float(continuous["omega_norm"][-1]),
        "final_hybrid_omega": float(hybrid["omega_norm"][-1]),
        "mean_inter_event_time": mean_inter_event,
        "min_inter_event_time": min_inter_event,
        "max_inter_event_time": max_inter_event,
    }


@st.cache_data(show_spinner=False)
def run_cached_simulation(
    q_0_tuple,
    omega_0_tuple,
    tf,
    dt,
    k_R,
    k_omega,
    sigma_e_gain,
    sigma_w_gain,
    sigma_floor,
    tau_min,
):
    """
    Cached simulation wrapper.

    Streamlit caching works best when arguments are simple/hashable.
    So we pass tuples and floats, then rebuild SimulationConfig inside.
    """
    cfg = SimulationConfig(
        q_0=np.array(q_0_tuple, dtype=float),
        omega_0=np.array(omega_0_tuple, dtype=float),
        tf=float(tf),
        dt=float(dt),
        K_R=np.diag([k_R, k_R, k_R]),
        K_omega=np.diag([k_omega, k_omega, k_omega]),
        sigma_e_gain=float(sigma_e_gain),
        sigma_w_gain=float(sigma_w_gain),
        sigma_floor=float(sigma_floor),
        tau_min=float(tau_min),
    )

    continuous = run_continuous_sim(cfg)
    hybrid = run_hybrid_sim(cfg)
    summary = summarize_results(continuous, hybrid)

    return continuous, hybrid, summary


# ============================================================
# Sidebar inputs
# ============================================================

st.sidebar.title("Simulation Controls")

st.sidebar.header("Initial Quaternion Orientation")

st.sidebar.caption(
    "Quaternion convention: q = [q0, q1, q2, q3], where q0 is the scalar part."
)

q0_raw = st.sidebar.slider("q0 scalar", -1.0, 1.0, 0.9239, 0.01)
q1_raw = st.sidebar.slider("q1 x", -1.0, 1.0, 0.3827, 0.01)
q2_raw = st.sidebar.slider("q2 y", -1.0, 1.0, 0.0, 0.01)
q3_raw = st.sidebar.slider("q3 z", -1.0, 1.0, 0.0, 0.01)

q_raw = np.array([q0_raw, q1_raw, q2_raw, q3_raw], dtype=float)
q_0 = normalize_quaternion(q_raw)

if np.linalg.norm(q_raw) < 1e-8:
    st.sidebar.warning("Quaternion norm too small. Using identity quaternion.")

st.sidebar.write("Normalized quaternion:")
st.sidebar.code(np.array2string(q_0, precision=4), language="text")

st.sidebar.header("Initial Angular Velocity")

omega_x = st.sidebar.slider("Initial ω_x [rad/s]", -2.0, 2.0, 0.30, 0.05)
omega_y = st.sidebar.slider("Initial ω_y [rad/s]", -2.0, 2.0, -0.20, 0.05)
omega_z = st.sidebar.slider("Initial ω_z [rad/s]", -2.0, 2.0, 0.15, 0.05)

omega_0 = np.array([omega_x, omega_y, omega_z], dtype=float)

st.sidebar.header("Simulation Settings")

tf = st.sidebar.slider(
    "Final time tf [s]",
    min_value=2.0,
    max_value=20.0,
    value=10.0,
    step=1.0,
)

# Fixed timestep for web demo performance.
# Use dt = 0.001 locally for higher-resolution research figures.
dt = 0.005

# Fixed controller gains for public demo.
k_R = 4.0
k_omega = 2.0

# Fixed trigger parameters for public demo.
sigma_e_gain = 0.50
sigma_w_gain = 0.10
sigma_floor = 1e-3
tau_min = 0.05

st.sidebar.header("Display Settings")

animation_height = st.sidebar.slider(
    "Animation display height [px]",
    min_value=500,
    max_value=1200,
    value=850,
    step=50,
    help="Adjust this if the animation is too tall or too short for your screen.",
)


# ============================================================
# Main app
# ============================================================

st.title("Event-Triggered Quaternion Attitude Control Simulator")

st.markdown(
    """
This interactive demo compares two executions of the same quaternion PD attitude controller.

**Continuous execution:** the control torque is recomputed at every timestep.

**Event-triggered execution:** the control torque is updated only when the trigger condition is met, then held constant using zero-order hold.

The event-triggered system produces hybrid flow-event dynamics: the physical state flows continuously, while the held control input jumps at event times.
"""
)

q_0_tuple = tuple(float(v) for v in q_0)
omega_0_tuple = tuple(float(v) for v in omega_0)

with st.spinner("Running simulation..."):
    continuous, hybrid, summary = run_cached_simulation(
        q_0_tuple=q_0_tuple,
        omega_0_tuple=omega_0_tuple,
        tf=float(tf),
        dt=float(dt),
        k_R=float(k_R),
        k_omega=float(k_omega),
        sigma_e_gain=float(sigma_e_gain),
        sigma_w_gain=float(sigma_w_gain),
        sigma_floor=float(sigma_floor),
        tau_min=float(tau_min),
    )


# ============================================================
# Summary metrics
# ============================================================

st.subheader("Results Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Triggered updates after t=0",
    summary["hybrid_triggered_updates_after_t0"],
)

col2.metric(
    "Update reduction",
    f"{summary['update_reduction_percent']:.2f}%",
)

col3.metric(
    "Final hybrid attitude error",
    f"{summary['final_hybrid_error']:.2e}",
)

col4.metric(
    "Final hybrid angular velocity",
    f"{summary['final_hybrid_omega']:.2e}",
)

with st.expander("Detailed numerical summary"):
    st.write(
        {
            "continuous_steps": summary["continuous_steps"],
            "hybrid_updates_including_initial": summary["hybrid_updates_including_initial"],
            "hybrid_triggered_updates_after_t0": summary["hybrid_triggered_updates_after_t0"],
            "update_reduction_percent": summary["update_reduction_percent"],
            "mean_inter_event_time": summary["mean_inter_event_time"],
            "min_inter_event_time": summary["min_inter_event_time"],
            "max_inter_event_time": summary["max_inter_event_time"],
            "final_continuous_error": summary["final_continuous_error"],
            "final_hybrid_error": summary["final_hybrid_error"],
            "final_continuous_omega": summary["final_continuous_omega"],
            "final_hybrid_omega": summary["final_hybrid_omega"],
        }
    )

with st.expander("Current initial condition"):
    st.write(
        {
            "q_0_normalized": q_0.tolist(),
            "omega_0": omega_0.tolist(),
            "tf": tf,
            "dt": dt,
            "k_R": k_R,
            "k_omega": k_omega,
            "sigma_e_gain": sigma_e_gain,
            "sigma_w_gain": sigma_w_gain,
            "sigma_floor": sigma_floor,
            "tau_min": tau_min,
        }
    )


# ============================================================
# Animated dashboard
# ============================================================

st.subheader("Animated Flow–Event Dashboard")

st.markdown(
    """
The dashboard animates the continuous and event-triggered executions side by side.

- The **upper cube** shows continuous PD execution.
- The **lower cube** shows hybrid event-triggered execution.
- The hybrid cube flashes near event times.
- The attached plots show attitude error, angular velocity, trigger terms, and zero-order hold control input.
"""
)

# Fixed animation settings for a reliable poster/web demo.
# Higher stride = fewer frames = faster loading and smaller embedded animation.
ANIMATION_STRIDE = 100
ANIMATION_INTERVAL = 200

if st.button("Generate animated simulation", type="primary"):
    with st.spinner("Generating animation... this may take a few seconds."):
        animation_html = make_dashboard_animation_html(
            continuous=continuous,
            hybrid=hybrid,
            stride=ANIMATION_STRIDE,
            cube_size=0.8,
            interval=ANIMATION_INTERVAL,
        )

    responsive_animation_html = f"""
    <div style="
        width: 100%;
        max-width: 100%;
        overflow-x: auto;
        overflow-y: hidden;
        background-color: white;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        padding: 0.5rem;
    ">
        <div style="
            width: 100%;
            min-width: 700px;
        ">
            {animation_html}
        </div>
    </div>

    <style>
        img {{
            max-width: 100% !important;
            height: auto !important;
        }}

        video {{
            max-width: 100% !important;
            height: auto !important;
        }}

        canvas {{
            max-width: 100% !important;
            height: auto !important;
        }}

        svg {{
            max-width: 100% !important;
            height: auto !important;
        }}

        .animation {{
            width: 100% !important;
            height: auto !important;
        }}
    </style>
    """

    components.html(
        responsive_animation_html,
        height=int(animation_height),
        scrolling=True,
    )


# ============================================================
# Interpretation
# ============================================================

st.markdown(
    """
<div class="interpretation-box">

### How to interpret the results

The **attitude error graph** shows how quickly the spacecraft orientation approaches the desired attitude.
A decreasing attitude error means the controller is successfully stabilizing the orientation.

The **angular velocity graph** shows whether the body rates are being damped toward zero.
For a stabilized attitude, both the attitude error and angular velocity should decay.

The **control input graph** shows the held zero-order-hold torque used by the event-triggered controller.
Flat regions mean the controller is not being recomputed; jumps occur when a new event updates the control input.

The **trigger-condition graph** compares the control mismatch against the event threshold.
When the mismatch reaches the threshold, a new control update is triggered.

Together, these plots show whether the hybrid event-triggered controller stabilizes the system while using fewer control updates than continuous execution.

</div>
""",
    unsafe_allow_html=True,
)

st.caption(
    "Inputs are constrained to keep the simulation physically meaningful and robust for a live web demo."
)
