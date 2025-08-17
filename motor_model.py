import numpy as np

def motor_model(current_rpm: np.ndarray, F_cmd: float, M_cmd: np.ndarray, params: dict, dt: float):
    """
    Split a desired total thrust + body torques into motor thrusts,
    convert thrust to rpm commands, then apply a simple first order motor dynamic.
    Returns rpm, rpm_dot, delivered total thrust, and delivered body torques.
    Motor order: 0=front(+x), 1=right(+y), 2=back(-x), 3=left(-y).
    Spin: 0 & 2 CCW, 1 & 3 CW
    """

    # parameters
    k_f     = float(params.get('k_f', params.get('kf', 1e-7)))              # thrust coeff 
    arm     = float(params.get('arm', params.get('l_arm', 0.046)))          # arm length [m]
    rpm_max = float(params.get('rpm_max', params.get('max_rpm', 32000)))    # hard rpm cap
    tau     = float(params.get('motor_tau', params.get('rpm_tau', 0.035)))  # motor time constant
    yaw_k   = float(params.get('yaw_k', params.get('k_m_over_k_f', 0.016))) # yaw torque per N of thrust

    # desired moment
    F = float(F_cmd)
    M = np.asarray(M_cmd, dtype=float).flatten()
    Mx, My, Mz = float(M[0]), float(M[1]), float(M[2])

    # mixing matrix: maps motor forces -> [F, Mx, My, Mz]
    A = np.array([
        [ 1.0,    1.0,   1.0,    1.0   ],
        [ 0.0,   +arm,   0.0,   -arm   ],
        [-arm,    0.0,  +arm,    0.0   ],
        [ yaw_k, -yaw_k, yaw_k, -yaw_k ],
    ], dtype=float)

    wrench = np.array([F, Mx, My, Mz], dtype=float)

    # initial allocation 
    A_pinv = np.linalg.pinv(A)
    f_i = A_pinv @ wrench  # per-motor thrusts [N]

    # if any motor is negative, set to 0 and rescale the rest
    neg = f_i < 0.0
    if np.any(neg):
        deficit = -np.sum(f_i[neg])
        f_i[neg] = 0.0
        pos = ~neg
        pos_sum = np.sum(f_i[pos])
        if pos_sum > 1e-9:
            f_i[pos] *= (pos_sum + deficit) / pos_sum

    # cap at what rpm_max can produce
    f_max = k_f * (rpm_max ** 2)
    f_i = np.clip(f_i, 0.0, f_max)

    # thrust -> rpm: f = k_f * rpm^2  =>  rpm = sqrt(f/k_f)
    rpm_cmd = np.sqrt(np.maximum(f_i / max(k_f, 1e-12), 0.0))
    rpm_cmd = np.clip(rpm_cmd, 0.0, rpm_max)

    # first order rpm dynamics
    rpm = current_rpm.astype(float)
    rpm_dot = (rpm_cmd - rpm) / max(tau, 1e-3)
    rpm = np.clip(rpm + rpm_dot * dt, 0.0, rpm_max)

    # calculate wrench from realized rpm
    f_i_actual = k_f * rpm**2
    Mx_act = arm * (f_i_actual[1] - f_i_actual[3])
    My_act = arm * (f_i_actual[2] - f_i_actual[0])
    Mz_act = yaw_k * (f_i_actual[0] - f_i_actual[1] + f_i_actual[2] - f_i_actual[3])
    F_act = float(np.sum(f_i_actual))
    M_actual = np.array([Mx_act, My_act, Mz_act], dtype=float)

    return rpm.astype(float), rpm_dot.astype(float), F_act, M_actual
