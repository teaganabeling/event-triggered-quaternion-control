# main.py: main entry point to run the continuous and hybrid simulations, generate summaries, and create plots
from config import SimulationConfig
from config import save_results_to_csv
from simulation.continuous_sim import run_continuous_sim
from simulation.hybrid_sim import run_hybrid_sim
from analysis.metrics import summarize_continuous, summarize_hybrid
from analysis.plots import plot_error_comparison, plot_omega_comparison, plot_hybrid_control, plot_trigger_terms, plot_inter_event_histogram

import os
def ensure_results_dir(path="results"):
    os.makedirs(path, exist_ok=True)
ensure_results_dir()

def print_summary(cont_summary: dict, hybrid_summary: dict) -> None:
    print("\n==== Comparison Summary ====\n")

    print("Continuous-time baseline (PD):")
    for key, value in cont_summary.items():
        print(f"  {key}: {value}")

    print("\nHybrid event-triggered execution:")
    for key, value in hybrid_summary.items():
        print(f"  {key}: {value}")
    reduction = 1.0 - (hybrid_summary["num_events"] / cont_summary["num_updates"])
    print(f"\nUpdate reduction: {100.0 * reduction:.2f}% fewer control updates")

def main() -> None:
    print("\nRunning simulations...")

    cfg = SimulationConfig()

    cont_results = run_continuous_sim(cfg)
    hybrid_results = run_hybrid_sim(cfg)

    cont_summary = summarize_continuous(cont_results)
    hybrid_summary = summarize_hybrid(hybrid_results)

    print_summary(cont_summary, hybrid_summary)

    print("\nGenerating plots...")
    plot_error_comparison(cont_results, hybrid_results)
    plot_omega_comparison(cont_results, hybrid_results)
    plot_hybrid_control(hybrid_results)
    plot_trigger_terms(hybrid_results)
    plot_inter_event_histogram(hybrid_results)

    save_results_to_csv(cont_results, "results/continuous.csv")
    save_results_to_csv(hybrid_results, "results/hybrid.csv")

    print("\nSimulation Complete.")

if __name__ == "__main__":
    main()