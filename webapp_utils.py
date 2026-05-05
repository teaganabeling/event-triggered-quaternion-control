import numpy as np


def axis_angle_to_quaternion(axis: str, angle_deg: float) -> np.ndarray:
    """
    Convert an axis-angle initial attitude into a scalar-first unit quaternion.

    q = [cos(theta/2), sin(theta/2) * axis_vector]
    """
    theta = np.deg2rad(angle_deg)

    if axis == "x":
        a = np.array([1.0, 0.0, 0.0])
    elif axis == "y":
        a = np.array([0.0, 1.0, 0.0])
    elif axis == "z":
        a = np.array([0.0, 0.0, 1.0])
    elif axis == "diagonal":
        a = np.array([1.0, 1.0, 1.0])
        a = a / np.linalg.norm(a)
    else:
        raise ValueError(f"Unknown axis option: {axis}")

    q0 = np.cos(theta / 2.0)
    q_vec = np.sin(theta / 2.0) * a

    q = np.concatenate(([q0], q_vec))
    return q / np.linalg.norm(q)


def summarize_results(continuous: dict, hybrid: dict, dt: float) -> dict:
    """
    Compute simple poster/webapp-facing metrics.
    """
    continuous_steps = len(continuous["t"])
    hybrid_updates_including_initial = int(hybrid["num_events"])
    hybrid_triggered_updates = max(hybrid_updates_including_initial - 1, 0)

    update_reduction = 100.0 * (
        1.0 - hybrid_updates_including_initial / max(continuous_steps, 1)
    )

    inter_event_times = hybrid.get("inter_event_times", np.array([]))

    if len(inter_event_times) > 0:
        mean_inter_event = float(np.mean(inter_event_times))
        min_inter_event = float(np.min(inter_event_times))
    else:
        mean_inter_event = float("nan")
        min_inter_event = float("nan")

    return {
        "continuous_steps": continuous_steps,
        "hybrid_updates_including_initial": hybrid_updates_including_initial,
        "hybrid_triggered_updates": hybrid_triggered_updates,
        "update_reduction_percent": update_reduction,
        "final_continuous_error": float(continuous["e_norm"][-1]),
        "final_hybrid_error": float(hybrid["e_norm"][-1]),
        "final_continuous_omega": float(continuous["omega_norm"][-1]),
        "final_hybrid_omega": float(hybrid["omega_norm"][-1]),
        "mean_inter_event_time": mean_inter_event,
        "min_inter_event_time": min_inter_event,
        "dt": dt,
    }