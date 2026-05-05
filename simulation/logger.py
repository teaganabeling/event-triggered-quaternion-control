# logger.py: simple logger class to store simulation data for later analysis and plotting
import numpy as np

class SimulationLogger:
    def __init__(self):
        self.t = []
        self.q = []
        self.omega = []
        self.u = []

        self.e_R = []
        self.e_norm = []
        self.omega_norm = []

        self.phi = []
        self.sigma = []
        self.mismatch = []

        self.event_times = []
        self.inter_event_times = []

    def log_step(
        self,
        t,
        q,
        omega,
        u,
        e_R,
        e_norm,
        omega_norm,
        phi=None,
        sigma=None,
        mismatch=None,
    ):
        self.t.append(t)
        self.q.append(q.copy())
        self.omega.append(omega.copy())
        self.u.append(u.copy())

        self.e_R.append(e_R.copy())
        self.e_norm.append(e_norm)
        self.omega_norm.append(omega_norm)

        if phi is not None:
            self.phi.append(phi)
        if sigma is not None:
            self.sigma.append(sigma)
        if mismatch is not None:
            self.mismatch.append(mismatch)

    def log_event(self, t, last_event_time):
        self.event_times.append(t)

        if last_event_time is not None:
            self.inter_event_times.append(t - last_event_time)

    def to_dict(self):
        return {
            "t": np.array(self.t),
            "q": np.array(self.q),
            "omega": np.array(self.omega),
            "u": np.array(self.u),
            "e_R": np.array(self.e_R),
            "e_norm": np.array(self.e_norm),
            "omega_norm": np.array(self.omega_norm),
            "phi": np.array(self.phi),
            "sigma": np.array(self.sigma),
            "mismatch": np.array(self.mismatch),
            "event_times": np.array(self.event_times),
            "inter_event_times": np.array(self.inter_event_times),
            "num_events": len(self.event_times),
        }