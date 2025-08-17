import numpy as np

def _euler_to_R(phi: float, theta: float, psi: float) -> np.ndarray:
    """ZYX (yaw-pitch-roll) rotation matrix from Euler angles."""
    c = np.cos; s = np.sin
    Rz = np.array([[c(psi), -s(psi), 0],
                   [s(psi),  c(psi), 0],
                   [0,       0,      1]], dtype=float)
    Ry = np.array([[ c(theta), 0, s(theta)],
                   [ 0,        1, 0       ],
                   [-s(theta), 0, c(theta)]], dtype=float)
    Rx = np.array([[1, 0,       0      ],
                   [0, c(phi), -s(phi)],
                   [0, s(phi),  c(phi)]], dtype=float)
    return Rz @ Ry @ Rx

def _T_euler(phi: float, theta: float) -> np.ndarray:
    """
    Mapping from body rates [p,q,r] to ZYX Euler rates [phi_dot, theta_dot, psi_dot]:
        eul_dot = T(phi,theta) @ omega
    """
    sphi, cphi = np.sin(phi), np.cos(phi)
    sth,  cth  = np.sin(theta), np.cos(theta)

    # preserves sign near the singularity
    if abs(cth) < 1e-6:
        cth = np.sign(cth) * 1e-6
        if cth == 0.0:
            cth = 1e-6

    tth = sth / cth
    return np.array([
        [1.0, sphi * tth,  cphi * tth],
        [0.0,      cphi,        -sphi],
        [0.0, sphi / cth,  cphi / cth]
    ], dtype=float)

def dynamics(params: dict, state: np.ndarray, F: float, M: np.ndarray, rpm_dot: np.ndarray) -> np.ndarray:
    """
    Rigid body 6DOF dynamics with ZYX Euler angles and linear drag.
    State (16,): [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r, rpm1..rpm4]
    """
    m = float(params['mass'])
    g = float(params['g'])
    J_arr = params.get('J')
    J = np.array(J_arr, dtype=float)
    Jinv = np.linalg.inv(J)

    # Linear drag in world frame 
    dxy = float(params.get('drag_lin_xy', 0.0))
    dz  = float(params.get('drag_lin_z', 0.0))
    C_lin = np.diag([dxy, dxy, dz])

    # Get state
    x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, rpm1, rpm2, rpm3, rpm4 = state.astype(float)
    vel   = np.array([vx, vy, vz], dtype=float)
    omega = np.array([p, q, r], dtype=float)

    # Forces
    R = _euler_to_R(phi, theta, psi)
    ez_b = R[:, 2]                         # body z in world
    thrust_w = F * ez_b                    # thrust along +b3
    grav = np.array([0.0, 0.0, -g * m], dtype=float)
    drag = -C_lin @ vel * m                # linear velocity drag
    force_w = thrust_w + grav + drag
    acc = force_w / m

    # Rotational dynamics
    Jw = J @ omega
    omega_dot = Jinv @ (M - np.cross(omega, Jw))

    # Euler kinematics
    eul_dot = _T_euler(phi, theta) @ omega

    # State derivative
    state_dot = np.zeros(16, dtype=float)
    state_dot[0:3]   = vel
    state_dot[3:6]   = acc
    state_dot[6:9]   = eul_dot
    state_dot[9:12]  = omega_dot
    state_dot[12:16] = rpm_dot
    return state_dot
