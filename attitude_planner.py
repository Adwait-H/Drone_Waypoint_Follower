import numpy as np

def attitude_planner(a_des: np.ndarray, psi_des: float, g: float):
    """
    Take desired world-frame acceleration and desired yaw,
    and return desired Euler angles [roll, pitch, yaw] (ZYX convention).
    """
    ez = np.array([0.0, 0.0, 1.0], dtype=float)  # world up

    # figure out where we want thrust to point (body z-axis)
    # b3d is along a_des + g*ez so hover maps to +Z
    f = a_des + g * ez
    fn = np.linalg.norm(f)
    if fn < 1e-6:
        b3d = ez.copy()  # if accel is tiny, just keep it upright
    else:
        b3d = f / fn

    # set the heading we want in world frame from yaw
    cpsi = np.array([np.cos(psi_des), np.sin(psi_des), 0.0], dtype=float)

    # project that heading onto the plane orthogonal to b3d
    # gives the desired body x-axis direction before orthonormalizing
    b1c = cpsi - np.dot(cpsi, b3d) * b3d
    n1 = np.linalg.norm(b1c)
    b1c /= n1
    b2d = np.cross(b3d, b1c)
    n2 = np.linalg.norm(b2d)

    # finish the right-handed frame
    b2d /= max(n2, 1e-9)
    b1d = np.cross(b2d, b3d)

    # rotation matrix columns are the desired body axes in world frame
    Rd = np.column_stack((b1d, b2d, b3d))

    # get ZYX euler: roll (phi), pitch (theta), yaw (psi)
    theta_d = -np.arcsin(np.clip(Rd[2, 0], -1.0, 1.0))       # pitch
    phi_d   =  np.arctan2(Rd[2, 1], Rd[2, 2])                # roll
    return np.array([float(phi_d), float(theta_d), float(psi_des)], dtype=float)
