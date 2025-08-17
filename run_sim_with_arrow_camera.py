import pybullet as p
import numpy as np

def update_camera_from_keyboard(cam_state):
    """
    cam_state: dict with keys {"yaw","pitch","dist","target"}
    Arrow keys: left/right yaw, up/down pitch. +/- change distance.
    """
    keys = p.getKeyboardEvents()
    yaw_delta = 0
    pitch_delta = 0
    dist_delta = 0

    if p.B3G_LEFT_ARROW in keys:  yaw_delta -= 1.5
    if p.B3G_RIGHT_ARROW in keys: yaw_delta += 1.5
    if p.B3G_UP_ARROW in keys:    pitch_delta += 1.0
    if p.B3G_DOWN_ARROW in keys:  pitch_delta -= 1.0
    if ord('=') in keys or ord('+') in keys: dist_delta -= 0.01
    if ord('-') in keys:                      dist_delta += 0.01

    cam_state["yaw"] += yaw_delta
    cam_state["pitch"] = np.clip(cam_state["pitch"] + pitch_delta, -89, 89)
    cam_state["dist"] = max(0.05, cam_state["dist"] + dist_delta)

    p.resetDebugVisualizerCamera(cam_state["dist"], cam_state["yaw"], cam_state["pitch"], cam_state["target"])
    return cam_state
