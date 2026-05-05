# config.py: configuration dataclass and helper functions for the rigid-body quaternion attitude control simulation
from dataclasses import dataclass, field
import numpy as np

# normalize a quaternion to unit length
def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must always be positive.")
    return q / norm

# pack state into the flat hybrid state vector: x = [q0, q1, q2, q3, wx, wy, wz, ux, uy, uz]
def pack_state(q: np.ndarray, omega: np.ndarray, u: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    omega = np.asarray(omega, dtype=float)
    u = np.asarray(u, dtype=float)

    if q.shape != (4,):
        raise ValueError(f"q must have shape (4,), instead got {q.shape}")
    if omega.shape != (3,):
        raise ValueError(f"omega must have shape (3,), instead got {omega.shape}")
    if u.shape != (3,):
        raise ValueError(f"u must have shape (3,), instead got {u.shape}")
    return np.concatenate([q, omega, u])

# unpack the flat hybrid state vector into quaternion, angular velocity, and held control torque
def unpack_state(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.shape != (10,):
        raise ValueError(f"x must have shape (10,), instead got {x.shape}")

    q = x[0:4]
    omega = x[4:7]
    u = x[7:10]
    return q, omega, u

# define main configuration dataclass for the simulation
@dataclass
class SimulationConfig:

    """
    Central configuration dataclass for the rigid-body quaternion attitude control simulation.
    All physical parameters, controller gains, simulation settings, and trigger settings are defined here.
    """

    # Physical parameters
    J: np.ndarray = field(default_factory=lambda: np.diag([0.8, 1.0, 1.2]))
    J_inv: np.ndarray = field(init=False)

    # Controller gains
    K_R: np.ndarray = field(default_factory=lambda: np.diag([4.0, 4.0, 4.0]))
    K_omega: np.ndarray = field(default_factory=lambda: np.diag([2.0, 2.0, 2.0]))

    # Desired and Initial States
    # Quaternion format: q = [q0, q1, q2, q3], where q0 is the scalar part
    
    # Desired state
    q_d: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    omega_d: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Initial state
    q_0: np.ndarray = field(default_factory=lambda: np.array([0.92387953, 0.38268343, 0.0, 0.0]))
    omega_0: np.ndarray = field(default_factory=lambda: np.array([0.3, -0.2, 0.15]))
    u_0: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Simulation settings
    t0: float = 0.0 # initial time
    tf: float = 20.0 # final time
    dt: float = 1e-3 # nominal timestep for fixed-step integration + event checking

    # Optional solver tolerances for solve_ivp later
    rtol: float = 1e-8 # relative tolerance
    atol: float = 1e-10 # absolute tolerance

    # Event-trigger parameters
    # phi(q, omega, u) = ||u_c(q, omega) - u|| - sigma(||e_R||, ||omega||)
    # phi is the trigger function, sigma is the threshold function 
    
    sigma_e_gain: float = 0.5 # gain for the error norm in the trigger threshold
    sigma_w_gain: float = 0.1 # gain for the angular velocity norm in the trigger threshold
    sigma_floor: float = 1e-3 # minimum threshold to prevent Zeno behavior

    # Minimum inter-event time, the minimum time between events to prevent Zeno behavior
    tau_min: float = 0.05

    # Post-initialization to validate shapes, normalize quaternions, and compute J_inv
    def __post_init__(self) -> None:
        self.J = np.asarray(self.J, dtype=float)
        self.K_R = np.asarray(self.K_R, dtype=float)
        self.K_omega = np.asarray(self.K_omega, dtype=float)
        self.q_d = np.asarray(self.q_d, dtype=float)
        self.omega_d = np.asarray(self.omega_d, dtype=float)
        self.q_0 = np.asarray(self.q_0, dtype=float)
        self.omega_0 = np.asarray(self.omega_0, dtype=float)
        self.u_0 = np.asarray(self.u_0, dtype=float)

        self._validate_shapes()

        self.q_d = normalize_quaternion(self.q_d)
        self.q_0 = normalize_quaternion(self.q_0)

        self.J_inv = np.linalg.inv(self.J)

    # Validate the shapes of all parameters to ensure they are correct for the simulation
    def _validate_shapes(self) -> None:
        if self.J.shape != (3, 3):
            raise ValueError(f"J must be shape (3,3), got {self.J.shape}")
        if self.K_R.shape != (3, 3):
            raise ValueError(f"K_R must be shape (3,3), got {self.K_R.shape}")
        if self.K_omega.shape != (3, 3):
            raise ValueError(f"K_omega must be shape (3,3), got {self.K_omega.shape}")
        if self.q_d.shape != (4,):
            raise ValueError(f"q_d must be shape (4,), got {self.q_d.shape}")
        if self.omega_d.shape != (3,):
            raise ValueError(f"omega_d must be shape (3,), got {self.omega_d.shape}")
        if self.q_0.shape != (4,):
            raise ValueError(f"q_0 must be shape (4,), got {self.q_0.shape}")
        if self.omega_0.shape != (3,):
            raise ValueError(f"omega_0 must be shape (3,), got {self.omega_0.shape}")
        if self.u_0.shape != (3,):
            raise ValueError(f"u_0 must be shape (3,), got {self.u_0.shape}")

        if self.tf <= self.t0:
            raise ValueError("Final time tf must be greater than initial time t0.")
        if self.dt <= 0.0:
            raise ValueError("Timestep dt must be positive.")
        if self.tau_min < 0.0:
            raise ValueError("Minimum inter-event time tau_min must be nonnegative.")

    # Threshold function for event triggering, depends on the norm of the error and the norm of the angular velocity
    def sigma(self, e_norm: float, omega_norm: float) -> float:
        sigma_e = self.sigma_e_gain * e_norm # proportional to the error norm, so larger errors allow for larger control deviations before triggering
        sigma_w = self.sigma_w_gain * omega_norm # proportional to the angular velocity norm, so faster spinning allows for larger control deviations before triggering
        
        # Combine the contributions from the error norm and the angular velocity norm
        sigma_raw = sigma_e + sigma_w # the raw threshold before applying the floor
        sigma_value = max(sigma_raw, self.sigma_floor) # apply the floor to ensure the threshold never goes below a certain minimum value, preventing Zeno behavior

        return sigma_value # return the computed threshold value for the event trigger

# Helper function to build the initial state vector from the configuration
def initial_state(config: SimulationConfig) -> np.ndarray:
    return pack_state(config.q_0, config.omega_0, config.u_0)

# Helper function to save results to a CSV file using pandas, flattening vector data into separate columns for easier analysis and plotting later
import pandas as pd
def save_results_to_csv(results: dict, filename: str):
    df = pd.DataFrame()

    # Flatten vector data
    df["t"] = results["t"]

    df["e_norm"] = results["e_norm"]
    df["omega_norm"] = results["omega_norm"]

    # quaternion
    for i in range(4):
        df[f"q{i}"] = results["q"][:, i]

    # angular velocity
    for i in range(3):
        df[f"omega{i}"] = results["omega"][:, i]

    # control
    for i in range(3):
        df[f"u{i}"] = results["u"][:, i]

    # optional hybrid-only signals
    if "phi" in results:
        df["phi"] = results["phi"]
    if "sigma" in results:
        df["sigma"] = results["sigma"]
    if "mismatch" in results:
        df["mismatch"] = results["mismatch"]

    df.to_csv(filename, index=False)