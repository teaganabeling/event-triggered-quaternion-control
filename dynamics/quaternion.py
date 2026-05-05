# quaternion.py: quaternion operations for the rigid-body quaternion attitude control simulation
import numpy as np

# Normalize a quaternion to unit length
# Quaternion convention: q = [q0, q1, q2, q3], where q0 is the scalar part
def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {q.shape}")

    norm = np.linalg.norm(q)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive.")

    return q / norm

# Quaternion conjugate: q* = [q0, -q1, -q2, -q3]
def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {q.shape}")

    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

# Quaternion inverse: q^{-1} = q* / ||q||^2
def quat_inverse(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {q.shape}")

    norm_sq = np.dot(q, q)
    if norm_sq <= 0.0:
        raise ValueError("Quaternion norm squared must be positive.")

    return quat_conjugate(q) / norm_sq

# Hamilton product of two quaternions
def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # q1 ⊗ q2 = [s1, v1] ⊗ [s2, v2] = [s1*s2 - v1⋅v2, s1*v2 + s2*v1 + v1×v2]
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    if q1.shape != (4,):
        raise ValueError(f"q1 must have shape (4,), got {q1.shape}")
    if q2.shape != (4,):
        raise ValueError(f"q2 must have shape (4,), got {q2.shape}")

    s1 = q1[0]
    v1 = q1[1:4]

    s2 = q2[0]
    v2 = q2[1:4]

    scalar = s1 * s2 - np.dot(v1, v2)
    vector = s1 * v2 + s2 * v1 + np.cross(v1, v2)

    return np.concatenate(([scalar], vector))

# Convert angular velocity vector omega in R^3 to the pure quaternion [0, omega]
def omega_to_quat(omega: np.ndarray) -> np.ndarray:
    omega = np.asarray(omega, dtype=float)
    if omega.shape != (3,):
        raise ValueError(f"omega must have shape (3,), got {omega.shape}")

    return np.concatenate(([0.0], omega))

# Compute the attitude error quaternion q_e = q_d^{-1} ⊗ q
# If shortest=True, enforce the shortest-rotation convention by flipping the sign so that the scalar part is nonnegative
def quat_error(q: np.ndarray, q_d: np.ndarray, shortest: bool = True) -> np.ndarray:
    q = quat_normalize(q)
    q_d = quat_normalize(q_d)

    q_e = quat_multiply(quat_inverse(q_d), q)
    q_e = quat_normalize(q_e)

    if shortest and q_e[0] < 0.0:
        q_e = -q_e

    return q_e

# Return the vector part of a quaternion, s.t. if q = [q0, q1, q2, q3], then the vector part is [q1, q2, q3]
def quat_vector_part(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {q.shape}")

    return q[1:4].copy()

# Returns attitude error vector: e_R = vec(q_e), where q_e = q_d^{-1} ⊗ q
def attitude_error_vector(q: np.ndarray, q_d: np.ndarray, shortest: bool = True) -> np.ndarray:
    q_e = quat_error(q, q_d, shortest=shortest)
    return quat_vector_part(q_e)

# Returns quaternion kinematics: q_dot = 0.5 * q ⊗ [0, omega]
def quaternion_kinematics(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    omega_quat = omega_to_quat(omega)
    return 0.5 * quat_multiply(q, omega_quat)