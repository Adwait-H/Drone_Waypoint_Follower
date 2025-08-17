import numpy as np

# Quintic minimum jerk helpers

def _quintic_coeffs(x0, v0, a0, xf, vf, af, T):
    """
    Solve for the coefficients of a fifth order polynomial
    x(t) = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5
    that meets position, velocity, and acceleration at the start and end.
    Start is at t = 0. End is at t = T.
    """
    a0c = float(x0)
    a1c = float(v0)
    a2c = 0.5 * float(a0)

    T2, T3, T4, T5 = T*T, T**3, T**4, T**5

    # Solve the three unknowns [a3 a4 a5] so end constraints hold
    M = np.array([
        [   T3,    T4,     T5],
        [ 3*T2,  4*T3,   5*T4],
        [ 6*T,  12*T2,  20*T3],
    ], dtype=float)

    b = np.array([
        xf - (a0c + a1c*T + a2c*T2),
        vf - (a1c + 2*a2c*T),
        af - (2*a2c),
    ], dtype=float)

    a3, a4, a5 = np.linalg.solve(M, b)
    return np.array([a0c, a1c, a2c, a3, a4, a5], dtype=float)


def _poly_eval(coeffs, t):
    """
    Evaluate position, velocity, and acceleration of a quintic at time t.
    coeffs has length 6 for a0 through a5.
    """
    a0, a1, a2, a3, a4, a5 = coeffs
    t2, t3, t4, t5 = t*t, t**3, t**4, t**5
    p = a0 + a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5
    v = a1 + 2*a2*t + 3*a3*t2 + 4*a4*t3 + 5*a5*t4
    a = 2*a2 + 6*a3*t + 12*a4*t2 + 20*a5*t3
    return p, v, a


def _shortest_angle(a0, a1):
    """
    Return an angle that is equivalent to a1 but chosen near a0.
    Helpful for smooth yaw moves without spinning the long way around.
    """
    return a0 + ((a1 - a0 + np.pi) % (2*np.pi) - np.pi)


def _as_waypoint_list(waypoints):
    """
    Normalize input into a list of (x, y, z, psi) tuples.
    Accepts a single 4 vector or a list of 4 vectors.
    """
    w = np.asarray(waypoints, dtype=float)
    if w.ndim == 1 and w.size == 4:
        return [tuple(w.tolist())]
    return [tuple(np.asarray(wi, dtype=float).tolist()) for wi in waypoints]


# Segment speed policy and helpers

_CV = 15.0 / 8.0          # peak of s' for quintic minimum jerk
_CA = 5.773502692          # max magnitude of s'' for quintic minimum jerk
_EPS = 1e-9                # small threshold to avoid divide by zero


def _resolve_policy(raw_v, d, idx):
    """
    Convert v_nom into a desired segment speed before limits.

    Supports:
      float            -> treat as v_max with a simple linear rule
      list or array    -> per segment speeds
      dict             -> keys: v_min, v_max, k, policy

    Policies:
      linear  : v = v_min + k * d
      sqrt    : v = v_min + k * sqrt(d)
      tanh    : smooth ramp that approaches v_max if provided
      const   : always return v_max if provided else v_min
    """
    v_min = 0.4
    v_max = None
    k = 0.4
    policy = 'linear'

    # Per segment array wins
    if isinstance(raw_v, (list, tuple, np.ndarray)):
        arr = np.asarray(raw_v, dtype=float).ravel()
        use = arr[min(idx, arr.size - 1)]
        return max(0.0, float(use))

    if isinstance(raw_v, (int, float)):
        v_max = float(raw_v)
    elif isinstance(raw_v, dict):
        v_min = float(raw_v.get('v_min', v_min))
        v_max = raw_v.get('v_max', None)
        if v_max is not None:
            v_max = float(v_max)
        k = float(raw_v.get('k', k))
        policy = str(raw_v.get('policy', policy)).lower()

    d = float(max(d, 0.0))

    if policy == 'sqrt':
        v = v_min + k * np.sqrt(d)
    elif policy == 'tanh':
        # Smooth start, levels off as d grows
        if v_max is not None:
            v = v_min + (v_max - v_min) * np.tanh(k * d)
        else:
            v = v_min + k * d
    elif policy == 'const':
        v = v_max if v_max is not None else v_min
    else:  # linear
        v = v_min + k * d

    # Final clamp
    if v_max is not None:
        v = min(v, v_max)
    v = max(v, 0.0)
    return float(v)



def _pull_limit(params, key, default):
    """
    Read a float limit from params with a safe fallback.
    Keeps the planner resilient if a field is missing.
    """
    try:
        return float(params.get(key, default))
    except Exception:
        return float(default)


def trajectory_planner(current_state, waypoints, t_vec, v_nom, params):
    """
    Build a multi segment minimum jerk path that stops at each waypoint.
    Each segment gets a nominal speed from a simple policy.
    That speed is then capped by the controller and basic physics.
    The whole plan is sampled on t_vec.

    Inputs:
    current_state uses only the first nine values x y z vx vy vz phi theta psi.
    waypoints is a list of x y z psi. Can also pass a single 4 vector.
    t_vec is the global time samples for the plan. It should start at the current time.
    v_nom can be a float a list or a dict. A dict can hold policy fields and an optional params field.
    params can be passed directly or inside v_nom. Limits read are g a_xy_max az_max tilt_max and psi_rate_max.

    Output:
    traj has shape 15 by N where rows are
    position then velocity then acceleration then Euler angles then Euler rates.
    Only yaw and yaw rate are populated in the Euler blocks. Roll and pitch stay at zero here.
    """


    p = params

    # Gravity and basic caps
    g = float(p.get('g', 9.80665))
    a_xy_max = float(p.get('a_xy_max', np.inf))
    az_max = float(p.get('az_max', np.inf))
    tilt_max = float(p.get('tilt_max', np.deg2rad(28.0)))
    psi_rate_max = float(p.get('psi_rate_max', np.inf))


    # Optional global caps and a safety factor that gives the controller room to brake
    v_abs_max = None
    safety = 0.85
    a_xy_meas = None

    if isinstance(v_nom, dict):
        safety = float(v_nom.get('safety', safety))
        a_xy_meas = v_nom.get('a_xy_meas', None)
        if a_xy_meas is not None:
            a_xy_meas = float(a_xy_meas)
        v_abs_max = v_nom.get('v_abs_max', None)
        if v_abs_max is not None:
            v_abs_max = float(v_abs_max)

    # Effective lateral acceleration from tilt and any explicit caps
    a_tilt = g * np.tan(tilt_max)
    a_xy_eff_list = [a_xy_max, a_tilt]
    if a_xy_meas is not None:
        a_xy_eff_list.append(a_xy_meas)
    a_xy_eff = min(a_xy_eff_list) if len(a_xy_eff_list) else np.inf

    # Waypoint list and initial state
    wp_list = _as_waypoint_list(waypoints)
    x0, y0, z0 = [float(x) for x in current_state[0:3]]
    vx0, vy0, vz0 = [float(v) for v in current_state[3:6]]
    psi0 = float(current_state[8])

    # Output buffer
    N = int(len(t_vec))
    out = np.zeros((15, N), dtype=float)

    # Segment builder
    segments = []
    t_cursor = 0.0
    total_T = 0.0

    p_start = np.array([x0, y0, z0], dtype=float)
    v_start = np.array([vx0, vy0, vz0], dtype=float)
    a_start = np.zeros(3, dtype=float)
    psi_start = psi0

    # Floor on segment time to reduce jerk spikes and numeric noise 
    T_floor = 0.30

    for i, (x, y, z, psi_des) in enumerate(wp_list):
        p_end = np.array([x, y, z], dtype=float)
        psi_end = float(psi_des)

        # Pick the yaw target that turns the short way
        psi_end = _shortest_angle(psi_start, psi_end)

        dp = p_end - p_start
        d = float(np.linalg.norm(dp))
        dxy = float(np.linalg.norm(dp[:2]))
        dz = float(abs(dp[2]))
        dpsi = float(abs(_shortest_angle(psi_start, psi_end) - psi_start))

        # Propose a nominal speed from the policy
        v_raw = _resolve_policy(v_nom, d, i)

        # Collect caps that translate limits into a per segment speed bound
        v_caps = []

        # Lateral acceleration cap when there is lateral motion
        if dxy > _EPS and np.isfinite(a_xy_eff):
            v_cap_xy = np.sqrt((a_xy_eff * d * d) / (_CA * dxy))
            v_caps.append(v_cap_xy)

        # Vertical acceleration cap when there is vertical motion
        if dz > _EPS and np.isfinite(az_max):
            v_cap_z = np.sqrt((az_max * d * d) / (_CA * dz))
            v_caps.append(v_cap_z)

        # Yaw rate cap when yaw moves and the limit is finite
        if dpsi > _EPS and np.isfinite(psi_rate_max):
            v_cap_psi = (psi_rate_max * d) / (_CV * dpsi)
            v_caps.append(v_cap_psi)

        # Optional absolute speed cap
        if v_abs_max is not None:
            v_caps.append(float(v_abs_max))

        # Choose the final speed with a safety factor
        v_cap = min(v_caps) if len(v_caps) else np.inf
        v_nom_i = safety * min(v_raw, v_cap) if np.isfinite(v_cap) else safety * v_raw

        # Guard against zero if the segment has distance
        if d > _EPS:
            v_nom_i = max(1e-6, v_nom_i)

        # Segment time from distance and speed
        T = 0.0
        if d > _EPS and v_nom_i > 0.0:
            T = d / v_nom_i

        # Ensure enough time to turn the yaw
        if dpsi > _EPS and np.isfinite(psi_rate_max):
            T_yaw = (_CV * dpsi) / psi_rate_max
            T = max(T, T_yaw)

        # Keep a minimum time for stability
        T = max(T, T_floor)

        # Build stop to stop quintics for x y z and yaw
        cx = _quintic_coeffs(p_start[0], v_start[0], a_start[0], p_end[0], 0.0, 0.0, T)
        cy = _quintic_coeffs(p_start[1], v_start[1], a_start[1], p_end[1], 0.0, 0.0, T)
        cz = _quintic_coeffs(p_start[2], v_start[2], a_start[2], p_end[2], 0.0, 0.0, T)
        cpsi = _quintic_coeffs(psi_start, 0.0, 0.0, psi_end, 0.0, 0.0, T)

        segments.append({"t0": t_cursor, "T": T, "cx": cx, "cy": cy, "cz": cz, "cpsi": cpsi})
        t_cursor += T
        total_T += T

        # Next segment starts from rest at the new waypoint
        p_start = p_end
        v_start[:] = 0.0
        a_start[:] = 0.0
        psi_start = psi_end

    # If there are no segments just hold the current pose and yaw
    if not segments:
        out[0:3, :] = np.array([x0, y0, z0])[:, None]
        out[9:12, :] = np.array([0.0, 0.0, psi0])[:, None]
        return out

    # Sample the active segment at each time in t_vec
    t0_global = float(t_vec[0]) if len(t_vec) else 0.0
    for k, tk_abs in enumerate(t_vec):
        tk = float(tk_abs - t0_global)

        # After the plan ends hold the final point
        if tk >= total_T:
            seg = segments[-1]
            tau = seg["T"]
        else:
            # Find the segment that covers the current time
            seg = segments[0]
            for s in segments:
                if s["t0"] <= tk < (s["t0"] + s["T"]):
                    seg = s
                    break
            tau = float(tk - seg["t0"])
            tau = np.clip(tau, 0.0, seg["T"])

        px, vx, ax = _poly_eval(seg["cx"], tau)
        py, vy, ay = _poly_eval(seg["cy"], tau)
        pz, vz, az = _poly_eval(seg["cz"], tau)
        ppsi, vpsi, _apsi = _poly_eval(seg["cpsi"], tau)

        out[0:3,  k] = [px, py, pz]
        out[3:6,  k] = [vx, vy, vz]
        out[6:9,  k] = [ax, ay, az]
        out[9:12, k] = [0.0, 0.0, ppsi]
        out[12:15, k] = [0.0, 0.0, vpsi]

    return out
