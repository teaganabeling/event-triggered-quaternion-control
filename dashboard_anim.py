import numpy as np
import matplotlib

try:
    matplotlib.use("MacOSX")
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config import SimulationConfig
from simulation.continuous_sim import run_continuous_sim
from simulation.hybrid_sim import run_hybrid_sim


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Convert scalar-first quaternion q = [q0, q1, q2, q3]
    to a 3x3 rotation matrix.
    """
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)

    q0, q1, q2, q3 = q

    return np.array([
        [1 - 2 * (q2**2 + q3**2),     2 * (q1*q2 - q0*q3),     2 * (q1*q3 + q0*q2)],
        [    2 * (q1*q2 + q0*q3), 1 - 2 * (q1**2 + q3**2),     2 * (q2*q3 - q0*q1)],
        [    2 * (q1*q3 - q0*q2),     2 * (q2*q3 + q0*q1), 1 - 2 * (q1**2 + q2**2)],
    ])


def cube_vertices(side_length: float = 1.0) -> np.ndarray:
    """
    Return 8 cube vertices centered at the origin.
    """
    s = side_length / 2.0
    return np.array([
        [-s, -s, -s],
        [ s, -s, -s],
        [ s,  s, -s],
        [-s,  s, -s],
        [-s, -s,  s],
        [ s, -s,  s],
        [ s,  s,  s],
        [-s,  s,  s],
    ])


def cube_faces(vertices: np.ndarray):
    """
    Return the 6 faces of the cube from the 8 vertices.
    """
    return [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
    ]


def rotate_vertices(vertices: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Rotate vertices by rotation matrix R.
    """
    return (R @ vertices.T).T


def nearest_event_distance(t_now: float, event_times: np.ndarray) -> float:
    """
    Distance in time to nearest event.
    """
    if len(event_times) == 0:
        return np.inf
    return float(np.min(np.abs(event_times - t_now)))


def draw_cube_panel(
    ax,
    q_now: np.ndarray,
    t_now: float,
    label: str,
    verts0: np.ndarray,
    event_times: np.ndarray | None = None,
    flash_window: float = 0.03,
):
    """
    Draw a cube plus inertial/body axes in a 3D axis.
    """
    ax.cla()

    R = quat_to_rotmat(q_now)
    verts_rot = rotate_vertices(verts0, R)
    faces = cube_faces(verts_rot)

    if event_times is not None:
        dt_event = nearest_event_distance(t_now, event_times)
        face_color = "orange" if dt_event < flash_window else "lightgray"
    else:
        face_color = "lightblue"

    poly = Poly3DCollection(
        faces,
        facecolors=face_color,
        edgecolors="k",
        linewidths=1.0,
        alpha=0.5,
    )
    ax.add_collection3d(poly)

    # Inertial axes (faint dashed)
    L_ref = 1.1
    ax.plot([0, L_ref], [0, 0], [0, 0], "--", color=(1, 0, 0, 0.20), linewidth=2)
    ax.plot([0, 0], [0, L_ref], [0, 0], "--", color=(0, 1, 0, 0.20), linewidth=2)
    ax.plot([0, 0], [0, 0], [0, L_ref], "--", color=(0, 0, 1, 0.20), linewidth=2)

    # Body-fixed axes
    body_x = R[:, 0]
    body_y = R[:, 1]
    body_z = R[:, 2]

    ax.plot([0, body_x[0]], [0, body_x[1]], [0, body_x[2]], color="r", linewidth=3)
    ax.plot([0, body_y[0]], [0, body_y[1]], [0, body_y[2]], color="g", linewidth=3)
    ax.plot([0, body_z[0]], [0, body_z[1]], [0, body_z[2]], color="b", linewidth=3)

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=35)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{label} | t = {t_now:.2f} s")

    if event_times is not None:
        num_events_so_far = int(np.sum(event_times <= t_now))
        ax.text2D(
            0.04,
            0.94,
            f"Events: {num_events_so_far}",
            transform=ax.transAxes,
        )


def animate_dashboard(
    continuous: dict,
    hybrid: dict,
    stride: int = 250,
    cube_size: float = 0.8,
):
    """
    Animate a dashboard with:
    - top-left: continuous PD cube
    - bottom-left: hybrid cube
    - top-middle: attitude error norm comparison
    - top-right: angular velocity norm comparison
    - bottom-middle: hybrid trigger terms
    - bottom-right: hybrid control input
    """
    t = hybrid["t"]
    q_c = continuous["q"]
    q_h = hybrid["q"]

    e_norm_c = continuous["e_norm"]
    e_norm_h = hybrid["e_norm"]

    omega_norm_c = continuous["omega_norm"]
    omega_norm_h = hybrid["omega_norm"]

    mismatch = hybrid["mismatch"]
    sigma = hybrid["sigma"]
    u = hybrid["u"]
    event_times = hybrid["event_times"]

    indices = list(range(0, len(t), stride))
    if indices[-1] != len(t) - 1:
        indices.append(len(t) - 1)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.6, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
    )

    ax3d_top = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d_bot = fig.add_subplot(gs[1, 0], projection="3d")
    ax_err = fig.add_subplot(gs[0, 1])
    ax_omega = fig.add_subplot(gs[0, 2])
    ax_trigger = fig.add_subplot(gs[1, 1])
    ax_control = fig.add_subplot(gs[1, 2])

    verts0 = cube_vertices(cube_size)

    e_max = max(np.max(e_norm_c), np.max(e_norm_h), 1e-8)
    omega_max = max(np.max(omega_norm_c), np.max(omega_norm_h), 1e-8)
    trig_max = max(np.max(mismatch), np.max(sigma), 1e-8)
    u_abs_max = max(np.max(np.abs(u)), 1e-8)

    def draw_error_panel(k_idx: int):
        ax_err.cla()
        i = indices[k_idx]

        ax_err.plot(t, e_norm_c, label="Continuous")
        ax_err.plot(t, e_norm_h, label="Hybrid")
        ax_err.plot(t[i], e_norm_c[i], "o", color="tab:blue")
        ax_err.plot(t[i], e_norm_h[i], "o", color="tab:orange")
        ax_err.axvline(t[i], color="k", linestyle="--", alpha=0.35)

        ax_err.set_title("Attitude Error Norm")
        ax_err.set_xlabel("t")
        ax_err.set_ylabel(r"$\|e_R\|$")
        ax_err.set_xlim(t[0], t[-1])
        ax_err.set_ylim(0, 1.05 * e_max)
        ax_err.legend()
        ax_err.grid(True)

    def draw_omega_panel(k_idx: int):
        ax_omega.cla()
        i = indices[k_idx]

        ax_omega.plot(t, omega_norm_c, label="Continuous")
        ax_omega.plot(t, omega_norm_h, label="Hybrid")
        ax_omega.plot(t[i], omega_norm_c[i], "o", color="tab:blue")
        ax_omega.plot(t[i], omega_norm_h[i], "o", color="tab:orange")
        ax_omega.axvline(t[i], color="k", linestyle="--", alpha=0.35)

        ax_omega.set_title("Angular Velocity Norm")
        ax_omega.set_xlabel("t")
        ax_omega.set_ylabel(r"$\|\omega\|$")
        ax_omega.set_xlim(t[0], t[-1])
        ax_omega.set_ylim(0, 1.05 * omega_max)
        ax_omega.legend()
        ax_omega.grid(True)

    def draw_trigger_panel(k_idx: int):
        ax_trigger.cla()
        i = indices[k_idx]

        ax_trigger.plot(t, mismatch, label=r"$\|u_c-u\|$")
        ax_trigger.plot(t, sigma, label=r"$\sigma(\|e_R\|,\|\omega\|)$")
        ax_trigger.plot(t[i], mismatch[i], "o", color="tab:blue")
        ax_trigger.plot(t[i], sigma[i], "o", color="tab:orange")
        ax_trigger.axvline(t[i], color="k", linestyle="--", alpha=0.35)

        ax_trigger.set_title("Hybrid Trigger Condition")
        ax_trigger.set_xlabel("t")
        ax_trigger.set_ylabel("Trigger terms")
        ax_trigger.set_xlim(t[0], t[-1])
        ax_trigger.set_ylim(0, 1.05 * trig_max)
        ax_trigger.legend()
        ax_trigger.grid(True)

    def draw_control_panel(k_idx: int):
        ax_control.cla()
        i = indices[k_idx]

        ax_control.step(t, u[:, 0], where="post", label=r"$u_1$")
        ax_control.step(t, u[:, 1], where="post", label=r"$u_2$")
        ax_control.step(t, u[:, 2], where="post", label=r"$u_3$")

        ax_control.plot(t[i], u[i, 0], "o", color="tab:blue")
        ax_control.plot(t[i], u[i, 1], "o", color="tab:orange")
        ax_control.plot(t[i], u[i, 2], "o", color="tab:green")
        ax_control.axvline(t[i], color="k", linestyle="--", alpha=0.35)

        ax_control.set_title("Hybrid Control Input (ZOH)")
        ax_control.set_xlabel("t")
        ax_control.set_ylabel("u(t)")
        ax_control.set_xlim(t[0], t[-1])
        ax_control.set_ylim(-1.1 * u_abs_max, 1.1 * u_abs_max)
        ax_control.legend()
        ax_control.grid(True)

    def update(frame_idx: int):
        i = indices[frame_idx]
        t_now = t[i]

        draw_cube_panel(
            ax3d_top,
            q_c[i],
            t_now,
            label="Continuous PD",
            verts0=verts0,
            event_times=None,
        )

        draw_cube_panel(
            ax3d_bot,
            q_h[i],
            t_now,
            label="Hybrid",
            verts0=verts0,
            event_times=event_times,
        )

        draw_error_panel(frame_idx)
        draw_omega_panel(frame_idx)
        draw_trigger_panel(frame_idx)
        draw_control_panel(frame_idx)

        fig.tight_layout()

    ani = FuncAnimation(
        fig,
        update,
        frames=len(indices),
        interval=100,
        repeat=True,
    )

    plt.show()
    return ani


def main():
    cfg = SimulationConfig()

    continuous = run_continuous_sim(cfg)
    hybrid = run_hybrid_sim(cfg)

    animate_dashboard(
        continuous,
        hybrid,
        stride=100,
        cube_size=0.8,
    )


if __name__ == "__main__":
    main()