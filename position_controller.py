import numpy as np

def _get_vec3(params, key_xyz, default):
    """Read a gain or limit. Accepts a scalar, (xy, z), or vector"""
    val = params.get(key_xyz, None)
    if val is None:
        return np.array(default, dtype=float)
    arr = np.asarray(val, dtype=float).flatten()
    if arr.size == 1:  return np.array([arr[0], arr[0], arr[0]], dtype=float)
    if arr.size == 2:  return np.array([arr[0], arr[0], arr[1]], dtype=float)
    return np.array(arr[:3], dtype=float)

def position_controller(current_state: np.ndarray, desired_traj: np.ndarray, params: dict):
    """
    Cascaded controller in world frame:
    position -> velocity (P), then velocity -> acceleration (PI) + accel feedforward.
    Returns desired thrust and world-frame acceleration.
    """

    # basic constants
    m  = float(params.get('mass', 0.027))            # kg
    g  = float(params.get('g', 9.80665))             
    dt = float(params.get('ctrl_dt', 1.0/240.0))     # controller step

    # limits
    tilt_max = float(params.get('tilt_max', np.deg2rad(28.0)))  # max tilt [rad]
    az_max   = float(params.get('az_max',   8.0))               # max vertical accel 
    a_xy_max = float(params.get('a_xy_max', 3.0))               # max lateral accel
    acc_ff_gain = float(params.get('acc_ff_gain', 1.0))         # scale on accel feedforward

    # define when drone is stopped 
    hold_xy_tol_pos = float(params.get('hold_xy_tol_pos', 0.02))  
    hold_xy_tol_vel = float(params.get('hold_xy_tol_vel', 0.05))  
    accel_deadband  = float(params.get('accel_deadband_xy', 0.0)) # 0 means off

    # outer loop gains (pos -> vel).
    kp_pos = _get_vec3(params, 'kp_pos', [float(params.get('kp_xy', 1.4)), float(params.get('kp_xy', 1.4)), float(params.get('kp_z',  3.0))])
    kd_pos_outer = _get_vec3(params, 'kd_pos_outer', [0.0, 0.0, 0.0])  # usually keep at 0

    # inner loop gains (vel -> acc, PI)
    kd_pos_default = _get_vec3(params, 'kd_pos', [float(params.get('kd_xy', 9.5)), float(params.get('kd_xy', 9.5)), float(params.get('kd_z',  6.5))])
    kv_vel = _get_vec3(params, 'kv_vel', kd_pos_default)        # P on velocity error
    ki_vel = _get_vec3(params, 'ki_vel', [0.9, 0.9, 0.15])      # I on velocity error

    # integrator maintenance
    i_leak      = float(params.get('vel_i_leak', 0.02))         # decay
    i_lim_xy    = float(params.get('i_accel_limit_xy', 1.5))    # clamp XY I 
    i_lim_z     = float(params.get('i_accel_limit_z',  3.0))    # clamp Z  I 
    awu_gain_xy = float(params.get('awu_gain_xy', 0.8))         # anti windup strength

    # current state (world frame)
    x, y, z, vx, vy, vz = current_state[0:6].astype(float)
    pos = np.array([x, y, z], dtype=float)
    vel = np.array([vx, vy, vz], dtype=float)

    # desired sample from the planner
    pos_d  = desired_traj[0:3].astype(float)
    vel_d  = desired_traj[3:6].astype(float)
    acc_ff = desired_traj[6:9].astype(float)

    # check if we are basically at the point in XY
    e_pos  = pos_d - pos
    hold_xy = (np.linalg.norm(e_pos[:2]) < hold_xy_tol_pos) and (np.linalg.norm(vel[:2]) < hold_xy_tol_vel)

    # outer loop: turn position error into a velocity request
    v_cmd = vel_d + kp_pos * e_pos + kd_pos_outer * (vel_d - vel)

    # inner loop: track that velocity and add feedforward accel
    e_vel = v_cmd - vel

    # keep a small velocity-integral across steps 
    if not hasattr(position_controller, "_vel_i"):
        position_controller._vel_i = np.zeros(3, dtype=float)
    I = position_controller._vel_i

    # leak old XY bias while moving, always leak Z. freeze XY while holding still
    if not hold_xy:
        I[:2] *= (1.0 - np.clip(i_leak*dt, 0.0, 1.0))
    I[2]  *= (1.0 - np.clip(i_leak*dt, 0.0, 1.0))

    # accel command before limits
    a_cmd = kv_vel * e_vel + I + acc_ff_gain * acc_ff

    # vertical accel limit
    a_cmd[2] = np.clip(a_cmd[2], -az_max, az_max)

    # XY accel limit by magnitude
    n_xy = float(np.linalg.norm(a_cmd[:2]))
    if n_xy > max(1e-9, a_xy_max):
        a_cmd[:2] *= (a_xy_max / n_xy)

    # update the integral: Z always, XY only when not holding
    I[2] += ki_vel[2] * e_vel[2] * dt

    # back calculation if XY saturates
    raw_xy = kv_vel[:2]*e_vel[:2] + I[:2] + acc_ff_gain*acc_ff[:2]
    n_raw = float(np.linalg.norm(raw_xy))
    sat_xy = n_raw > max(1e-9, a_xy_max)

    if not hold_xy:
        if sat_xy:
            sat_xy_vec = raw_xy * (a_xy_max / n_raw)
            I[:2] += ki_vel[:2] * e_vel[:2] * dt + awu_gain_xy * (sat_xy_vec - raw_xy)
        else:
            I[:2] += ki_vel[:2] * e_vel[:2] * dt

    # clamp the integral and store it back
    I[:2] = np.clip(I[:2], -i_lim_xy, i_lim_xy)
    I[2]  = float(np.clip(I[2], -i_lim_z,  i_lim_z))
    position_controller._vel_i = I

    # rebuild accel after I update and limits
    a_cmd = kv_vel * e_vel + I + acc_ff_gain * acc_ff
    a_cmd[2] = np.clip(a_cmd[2], -az_max, az_max)
    n_xy = float(np.linalg.norm(a_cmd[:2]))
    if n_xy > max(1e-9, a_xy_max):
        a_cmd[:2] *= (a_xy_max / n_xy)

    # when parked and accel is tiny, set XY to zero to stop small twitching
    if hold_xy and accel_deadband > 0.0 and np.linalg.norm(a_cmd[:2]) < accel_deadband:
        a_cmd[:2] = 0.0

    # map accel to specific force and respect tilt limit: |f_xy| <= |f_z| * tan(tilt_max)
    f_vec = a_cmd + g * np.array([0.0, 0.0, 1.0])
    fz = float(f_vec[2])
    f_xy = f_vec[:2]
    f_xy_max = abs(fz) * np.tan(tilt_max)
    n_xy = float(np.linalg.norm(f_xy))
    if n_xy > max(1e-9, f_xy_max):
        f_xy *= (f_xy_max / n_xy)
        f_vec = np.array([f_xy[0], f_xy[1], fz], dtype=float)

    # final outputs
    a_des = f_vec - g * np.array([0.0, 0.0, 1.0])               # world accel for attitude planner
    F_des = m * float(np.linalg.norm(f_vec))                     # total thrust 

    desired_state = desired_traj.copy().astype(float)            # pass through desired, update accel slot
    desired_state[6:9] = a_des
    return float(F_des), a_des.astype(float), desired_state.astype(float)

# Clear the saved velocity integral
def reset_position_controller():
    if hasattr(position_controller, "_vel_i"):
        position_controller._vel_i[:] = 0.0
