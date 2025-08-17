import numpy as np

def _wrap_pi(a: float) -> float:
    # wrap any angle to [-pi, pi] used for yaw error
    return float((a + np.pi) % (2*np.pi) - np.pi)

def attitude_controller(current_state: np.ndarray, desired_euler: np.ndarray, desired_rates: np.ndarray, params: dict):

    # read current angles and body rates from state
    phi, theta, psi = current_state[6:9].astype(float)   # roll, pitch, yaw [rad]
    p, q, r = current_state[9:12].astype(float)          # body rates
    eul = np.array([phi, theta, psi], dtype=float)
    rates = np.array([p, q, r], dtype=float)

    # desired angles and desired body rates
    eul_d   = np.asarray(desired_euler, dtype=float).flatten()
    rates_d = np.asarray(desired_rates, dtype=float).flatten()

    # angle error with yaw wrapped to shortest path
    e_ang = eul_d - eul
    e_ang[2] = _wrap_pi(eul_d[2] - eul[2])

    # rate error
    e_rates = rates_d - rates

    # torque limits from params
    tau_max = np.array(params.get('tau_max', [0.02, 0.02, 0.01]), dtype=float) 

    # direct torque gains 
    kp_m = np.array(params.get('kp_m', [0.010, 0.010, 0.004]), dtype=float)
    kd_m = np.array(params.get('kd_m', [0.002, 0.002, 0.0015]), dtype=float)

    # PD on attitude with wrapped yaw
    M = kp_m * e_ang + kd_m * e_rates


    # clamp torques to what the actuators can realistically produce
    M = np.clip(M, -tau_max, tau_max)
    return M.astype(float)
