import os
import time
import numpy as np
import pybullet as p

from setup_pybullet import setup_pybullet
from load_drone import load_drone
from get_drone_parameters import get_drone_parameters
from run_sim_with_arrow_camera import update_camera_from_keyboard
from trajectory_planner import trajectory_planner
from position_controller import position_controller, reset_position_controller
from attitude_planner import attitude_planner
from attitude_controller import attitude_controller
from motor_model import motor_model
from dynamics import dynamics
from scipy.integrate import solve_ivp

def main():
    gui = True
    dt = 1.0 / 240.0
    setup_pybullet(gui=gui, dt=dt)

    # URDF path 
    project_root = r"C:\Users\adwai\gym-pybullet-drones\gym_pybullet_drones\assets"
    urdf_filename = "cf2x.urdf"
    urdf_path = os.path.join(project_root, urdf_filename)
    drone_id = load_drone(urdf_path)

    # Get params
    params = get_drone_parameters(drone_id)
    g = float(params["g"])
    m = float(params["mass"])
    k_f = float(params["k_f"])

    # Initialize state: [x y z vx vy vz phi theta psi p q r rpm1 rpm2 rpm3 rpm4]
    state = np.zeros(16, dtype=float)
    state[2] = 0.02  # small start height

    # Initialize motors
    state[12:16] = 0

    # Waypoints (x, y, z, yaw)
    waypoints = [
        (0.0, 0.0, 5.0, 0.0),
        (0.0, 3.0, 3.0, 0.0),
        (1.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    ]

    # Set total sim time
    T = 16 # [s]
    steps = int(T / dt) + 1
    t_vec = np.linspace(0.0, T, steps) # initialize time vector

    v_nom = {
        "v_min": 0.40,      # base cruise speed in meters per second
        "k": 0.40,          # how much speed grows with segment length in meters
        # "v_max": 0.80,    # optional hard cap in meters per second
        "safety": 0.80,     # shave a bit off for braking room
        "a_xy_meas": 2.22,  # measured lateral acceleration the controller can track
        "params": params,   # lets the planner read limits such as g and tilt
    }

    # Build the trajectory using the cruise speeds and safe timing
    traj = trajectory_planner(
        state,
        waypoints,
        t_vec,
        v_nom,
        params,         
    )


    # Camera settings
    cam = {"yaw": 30, "pitch": -30, "dist": 3.0, "target": [0, 0, 1.0]}
    p.resetDebugVisualizerCamera(cam["dist"], cam["yaw"], cam["pitch"], cam["target"])

    reset_position_controller()  # clear I term in position controller

    last_time = time.time()

    # Simulation loop
    for k in range(steps - 1):
        t0 = t_vec[k]
        t1 = t_vec[k + 1]
        desired_traj = traj[:, k] # slice desired trajectory at current time step

        # 1) Position controller gives desired accel and thrust
        F_des, a_des, desired_state = position_controller(state, desired_traj, params)

        # 2) Attitude planner gives desired euler angles
        psi_des = desired_state[11]  # desired yaw
        eul_des = attitude_planner(a_des, psi_des, g)
        rates_des = np.zeros(3, dtype=float) # set desired angular rates to 0

        # 3) Attitude controller returns moments
        M_cmd = attitude_controller(state, eul_des, rates_des, params)

        # 4) Motor model finds realistic force and moments
        rpm = state[12:16].copy()
        rpm, rpm_dot, F_act, M_act = motor_model(rpm, F_des, M_cmd, params, dt)

        # 5) Integrate dynamics
        sol = solve_ivp(
            fun=lambda t, s: dynamics(params, s, F_act, M_act, rpm_dot),
            t_span=(t0, t1),
            y0=state,
            method="RK45",
            t_eval=[t1],
            rtol=1e-6,
            atol=1e-8
        )
        state = sol.y[:, -1]
        state[12:16] = rpm

        # Update in PyBullet
        pos = state[0:3].tolist()
        eul = state[6:9].tolist()
        quat = p.getQuaternionFromEuler(eul)
        p.resetBasePositionAndOrientation(drone_id, pos, quat)
        p.resetBaseVelocity(drone_id, linearVelocity=state[3:6].tolist(), angularVelocity=state[9:12].tolist())

        if gui:
            cam["target"] = [pos[0], pos[1], max(0.3, pos[2])]
            cam = update_camera_from_keyboard(cam)
        p.stepSimulation()

        # Real time simulation
        now = time.time()
        sleep_time = dt - (now - last_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_time = time.time()

if __name__ == "__main__":
    main()
