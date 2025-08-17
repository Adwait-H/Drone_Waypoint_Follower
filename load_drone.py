import pybullet as p

def load_drone(urdf_path: str, start_pos=(0, 0, 0.02), start_orn=(0, 0, 0)):
    drone_id = p.loadURDF(
        urdf_path,
        start_pos,
        p.getQuaternionFromEuler(start_orn),
        useFixedBase=False
    )
    p.changeDynamics(drone_id, -1, linearDamping=0.01, angularDamping=0.01)
    return drone_id
