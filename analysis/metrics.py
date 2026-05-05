# metrics.py: compute performance metrics for the continuous and hybrid simulations
import numpy as np

def summarize_continuous(results: dict) -> dict:
    return {
        "final_e_norm": float(results["e_norm"][-1]),
        "final_omega_norm": float(results["omega_norm"][-1]),
        "num_updates": int(len(results["t"])),
    }

def summarize_hybrid(results: dict) -> dict:
    iet = results["inter_event_times"]
    return {
        "final_e_norm": float(results["e_norm"][-1]),
        "final_omega_norm": float(results["omega_norm"][-1]),
        "num_events": int(results["num_events"]),
        "min_inter_event_time": float(np.min(iet)) if len(iet) > 0 else None,
        "max_inter_event_time": float(np.max(iet)) if len(iet) > 0 else None,
        "mean_inter_event_time": float(np.mean(iet)) if len(iet) > 0 else None,
    }