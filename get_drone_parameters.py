import pybullet as p
import numpy as np

def get_drone_parameters(drone_id):

    # read mass and inertia from the loaded body
    dyn = p.getDynamicsInfo(drone_id, -1)
    mass = float(dyn[0])
    J_diag = np.array(dyn[2], dtype=float)                    
    J = np.diag(J_diag)

    # gravity
    g = 9.80665

    # arm length from COM to each motor [m]
    arm = 0.046

    # motor constants for a f = k_f * rpm^2 model
    k_f = 1.2e-7              
    rpm_max = 32000.0          
    motor_tau = 0.035         
    yaw_k = 0.016             

    # torque clamp per axis for the attitude controller [N*m]
    tau_max = np.array([0.020, 0.020, 0.010], dtype=float)

    # attitude controller gains
    kp_m = np.array([0.010, 0.010, 0.004], dtype=float)
    kd_m = np.array([0.002, 0.002, 0.0015], dtype=float)
    use_direct_torque = True

    # small linear drag in world frame for extra damping 
    drag_lin_xy = 0.05   
    drag_lin_z  = 0.08  

    # position/velocity controller limits
    tilt_max = np.deg2rad(28.0)  
    az_max   = 8.0              
    a_xy_max = 3.0             
    acc_ff_gain = 1.0            
    accel_lpf_tau_xy = 0.0      

    # checks for XY hold
    hold_xy_tol_pos = 0.02     
    hold_xy_tol_vel = 0.05       
    accel_deadband_xy = 0.0     

    # cascaded controller gains
    # outer: pos -> vel (P). using separate XY and Z makes tuning easier
    kp_xy = 1.4
    kp_z  = 3.0
    kd_pos_outer = np.array([0.0, 0.0, 0.0], dtype=float)  # usually keep at 0

    # inner: vel -> acc (PI).
    kv_vel = np.array([9.5, 9.5, 6.5], dtype=float)
    ki_vel = np.array([0.90, 0.90, 0.15], dtype=float)
    vel_i_leak = 0.02
    i_accel_limit_xy = 1.5
    i_accel_limit_z  = 3.0
    awu_gain_xy = 0.8   # anti windup back calculation strength

    # controller tick 
    ctrl_dt = 1.0 / 240.0

    # pack everything into a dict the rest of the code expects
    return {
        # physical constants and geometry
        "mass": mass,
        "g": g,
        "J": J,
        "arm": arm,

        # motor model
        "k_f": k_f,
        "rpm_max": rpm_max,
        "motor_tau": motor_tau,
        "yaw_k": yaw_k,        

        # attitude controller
        "kp_m": kp_m,
        "kd_m": kd_m,
        "tau_max": tau_max,

        # simple linear drag in world frame
        "drag_lin_xy": drag_lin_xy,
        "drag_lin_z":  drag_lin_z,

        # cascaded pos/vel control (outer loop)
        "kp_xy": kp_xy,
        "kp_z": kp_z,
        "kd_pos_outer": kd_pos_outer,

        # cascaded pos/vel control (inner loop)
        "kv_vel": kv_vel,
        "ki_vel": ki_vel,
        "vel_i_leak": vel_i_leak,
        "i_accel_limit_xy": i_accel_limit_xy,
        "i_accel_limit_z":  i_accel_limit_z,
        "awu_gain_xy": awu_gain_xy,

        # limits
        "tilt_max": tilt_max,
        "az_max":   az_max,
        "a_xy_max": a_xy_max,
        "acc_ff_gain": acc_ff_gain,
        "accel_lpf_tau_xy": accel_lpf_tau_xy,

        # hold behavior
        "hold_xy_tol_pos": hold_xy_tol_pos,
        "hold_xy_tol_vel": hold_xy_tol_vel,
        "accel_deadband_xy": accel_deadband_xy,

        # timing
        "ctrl_dt": ctrl_dt,
    }
