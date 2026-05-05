# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from simulator import run_hybrid_simulation, run_continuous_simulation

st.title("Event-Triggered Quaternion Attitude Control Simulator")

angle_deg = st.slider("Initial attitude error angle", 0, 180, 45)
axis = st.selectbox("Rotation axis", ["x", "y", "z"])
omega_x = st.slider("Initial ωx", -2.0, 2.0, 0.3)
omega_y = st.slider("Initial ωy", -2.0, 2.0, -0.2)
omega_z = st.slider("Initial ωz", -2.0, 2.0, 0.15)

T = st.slider("Simulation time", 1.0, 30.0, 20.0)
trigger_gain = st.slider("Trigger sensitivity", 0.05, 1.0, 0.5)

if st.button("Run Simulation"):
    results = run_hybrid_simulation(
        angle_deg=angle_deg,
        axis=axis,
        omega0=np.array([omega_x, omega_y, omega_z]),
        T=T,
        trigger_gain=trigger_gain,
    )

    st.write(f"Number of events: {results['num_events']}")
    st.write(f"Final attitude error: {results['final_error']:.3e}")

    fig, ax = plt.subplots()
    ax.plot(results["t"], results["eR_norm"])
    for tk in results["event_times"]:
        ax.axvline(tk, alpha=0.15)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\|e_R(t)\|$")
    st.pyplot(fig)