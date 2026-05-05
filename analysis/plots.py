# plots.py: plotting functions for comparing continuous and hybrid simulation results
import matplotlib.pyplot as plt

def plot_error_comparison(cont_results: dict, hybrid_results: dict) -> None:
    plt.figure()
    plt.plot(cont_results["t"], cont_results["e_norm"], label="Continuous")
    plt.plot(hybrid_results["t"], hybrid_results["e_norm"], label="Hybrid")

    plt.xlabel("t")
    plt.ylabel(r"$\|e_R\|$")
    plt.title("Attitude Error Norm")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("results/error_plot.png", dpi=300)
    plt.show()


def plot_omega_comparison(cont_results: dict, hybrid_results: dict) -> None:
    plt.figure()
    plt.plot(cont_results["t"], cont_results["omega_norm"], label="Continuous")
    plt.plot(hybrid_results["t"], hybrid_results["omega_norm"], label="Hybrid")

    plt.xlabel("t")
    plt.ylabel(r"$\|\omega\|$")
    plt.title("Angular Velocity Norm")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("results/omega_norm.png", dpi=300)
    plt.show()


def plot_hybrid_control(hybrid_results: dict) -> None:
    plt.figure()

    t = hybrid_results["t"]
    u = hybrid_results["u"]

    plt.plot(t, u[:, 0], label=r"$u_1$")
    plt.plot(t, u[:, 1], label=r"$u_2$")
    plt.plot(t, u[:, 2], label=r"$u_3$")

    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("Hybrid Control Input (Zero-Order Hold)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("results/hybrid_control.png", dpi=300)
    plt.show()


def plot_trigger_terms(hybrid_results: dict) -> None:
    plt.figure()
    plt.plot(hybrid_results["t"], hybrid_results["mismatch"], label=r"$\|u_c - u\|$")
    plt.plot(hybrid_results["t"], hybrid_results["sigma"], label=r"$\sigma(\|e_R\|,\|\omega\|)$")

    plt.xlabel("t")
    plt.ylabel("Trigger terms")
    plt.title("Trigger Condition")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("results/trigger_terms.png", dpi=300)
    plt.show()


def plot_inter_event_histogram(hybrid_results: dict) -> None:
    if len(hybrid_results["inter_event_times"]) == 0:
        print("No inter-event times to plot.")
        return

    plt.figure()
    plt.hist(hybrid_results["inter_event_times"], bins=20)
    plt.xlabel("Inter-event time")
    plt.ylabel("Count")
    plt.title("Histogram of Inter-Event Times")
    plt.grid()
    plt.tight_layout()
    plt.savefig("results/inter_event_histogram.png", dpi=300)
    plt.show()